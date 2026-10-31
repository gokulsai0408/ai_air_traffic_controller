"""LLMAgent (BUILD_PLAN 5.5): one Claude call per step, with its own role system prompt and its own context.

The LLM reads ops notes, proposes and votes. Feasibility and cost always come from the deterministic
assessor (AssessingAgent); the LLM only ever sees them as given numbers. Output must validate against a
pydantic model (extra="forbid"): on failure retry once, then REJECT/MALFORMED and switch to RuleAgent
for the rest of the scenario. Every call is recorded and handed to the coordinator as an LLM_CALL
ledger entry, so a replay never needs the model.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cdm.agents.base import AssessingAgent, CostFn, DeriveFn, merge_facts, parse_ops_rules
from cdm.agents.llm_client import MODEL, LLMClient, LLMError, complete
from cdm.agents.rule import RuleAgent
from cdm.schemas import OpsFacts, OptionDraft, Reason, Role, Vote

__all__ = ["LLMAgent", "complete", "LLMError"]

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
ATTEMPTS = 2                     # first try + one retry
MAX_DRAFTS = 2
M = TypeVar("M", bound=BaseModel)


# ---- what the LLM may emit (agent and round are filled in by code, so it cannot impersonate) ------

class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drafts: list[OptionDraft] = Field(min_length=1, max_length=MAX_DRAFTS)


class BallotItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    option_id: str
    vote: Literal["ACCEPT", "REJECT"]
    reason: Reason
    rationale: str = Field(default="", max_length=300)
    counter_proposal: Optional[OptionDraft] = None


class Ballot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    votes: list[BallotItem]


@lru_cache(maxsize=None)
def _prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def system_prompt(role: Role) -> str:
    return f"{_prompt(role)}\n\n{_prompt('common')}"


_LONE_NUMBER = re.compile(r"(?<![\w\-.:])\d+(?:[.,]\d+)?(?![\w\-:])")


def redact_numbers(text: str) -> str:
    """Rationales are display-only, but must not leak private numbers. IDs such as G-017 or S-14 survive."""
    return _LONE_NUMBER.sub("#", text)


def _dump(obj: Any) -> Any:
    from cdm.coordinator import jsonable
    return jsonable(obj)


def _table(options: list, assessments: list) -> list[dict]:
    own = {a.option_id: a for a in assessments}
    rows = []
    for o in options:
        a = own.get(o.option_id)
        if a is None:
            continue
        rows.append({"option_id": o.option_id, "proposed_by": o.proposed_by, "action_type": o.action_type.value,
                     "hold_minutes": o.hold_minutes, "slot_id": o.slot_id, "gate_id": o.gate_id,
                     "divert_to": o.divert_to, "feasible_for_me": a.feasible_for_me,
                     "my_cost_usd": round(a.cost_usd, 2), "my_reasons": [r.value for r in a.reasons]})
    return rows


class LLMAgent(AssessingAgent):
    kind = "llm"

    def __init__(self, role: Role, client: Optional[LLMClient] = None, derive_fn: Optional[DeriveFn] = None,
                 cost_fn: Optional[CostFn] = None):
        super().__init__(role, derive_fn, cost_fn)
        self.client = client or complete
        self.fallback = RuleAgent(role, derive_fn, cost_fn)
        self.degraded = False
        self._events: list[tuple[str, dict]] = []

    # ---- coordinator hook ---------------------------------------------------------------------
    def pop_ledger_events(self) -> list[tuple[str, dict]]:
        events, self._events = self._events, []
        return events

    # ---- the one place an LLM is called -------------------------------------------------------
    def _ask(self, purpose: str, system: str, payload: dict, model: Type[M]) -> Optional[M]:
        user = json.dumps(payload, sort_keys=True, default=str)
        schema = model.model_json_schema()
        for attempt in range(1, ATTEMPTS + 1):
            started = time.perf_counter()
            raw, error, parsed = None, None, None
            try:
                raw = self.client(system, user, schema)
                parsed = model.model_validate_json(raw)
            except (LLMError, ValidationError, ValueError) as e:
                error = f"{type(e).__name__}: {e}"[:1000]
            self._events.append(("LLM_CALL", {
                "agent": self.role, "purpose": purpose, "attempt": attempt, "model": MODEL,
                "system": system, "user": user, "schema": model.__name__, "response": raw,
                "ok": parsed is not None, "error": error, "latency_s": round(time.perf_counter() - started, 3)}))
            if parsed is not None:
                return parsed
        return None

    def _degrade(self, purpose: str) -> None:
        if not self.degraded:
            self.degraded = True
            self._events.append(("AGENT_FALLBACK", {"agent": self.role, "from": "llm", "to": "rule",
                                                    "after": purpose, "reason": Reason.MALFORMED.value}))

    def _context(self, view: dict) -> dict:
        return {"role": self.role, "round": view["round"], "public_scenario": _dump(view.get("public")),
                "my_private_state": _dump(view.get("private")), "menus": _dump(view.get("menus")),
                "ops_facts": _dump(self.ops_facts), "my_walkaway_cost_usd": round(view["walkaway_usd"], 2)}

    # ---- Agent protocol -----------------------------------------------------------------------
    def read_ops(self, notes: str) -> OpsFacts:
        rules = parse_ops_rules(notes)
        if not (notes or "").strip():
            self.ops_facts = rules
            return rules
        facts = self._ask("read_ops", _prompt("ops_extraction"), {"ops_notes": notes}, OpsFacts)
        # tighten-only: the LLM can add facts on top of the regex reading, never remove them
        self.ops_facts = merge_facts(rules, facts) if facts else rules.model_copy(update={"uncertain": True})
        return self.ops_facts

    def propose(self, view: dict) -> list[OptionDraft]:
        if self.degraded:
            return self.fallback.propose(view)
        payload = {**self._context(view),
                   "task": ("Round 1. Propose 1 or 2 complete plans. Prefer copying a row from my_assessment_table "
                            "that is feasible for you; you may also combine IDs from the menus."),
                   "my_assessment_table": _table(view["candidates"], view["own_assessments"])}
        out = self._ask("propose", system_prompt(self.role), payload, Proposal)
        if out is None:
            self._degrade("propose")
            return self.fallback.propose(view)
        return [d.model_copy(update={"rationale": redact_numbers(d.rationale)}) for d in out.drafts]

    def vote(self, view: dict) -> list[Vote]:
        if self.degraded:
            return self.fallback.vote(view)
        payload = {**self._context(view),
                   "task": ("Vote ACCEPT or REJECT on every option in shortlist_assessment_table, each with one "
                            "reason code. Optionally attach a counter_proposal to a REJECT."),
                   "shortlist_assessment_table": _table(view["shortlist"], view["own_assessments"]),
                   "other_agents_last_votes": view.get("other_votes", [])}
        out = self._ask("vote", system_prompt(self.role), payload, Ballot)
        if out is None:
            self._degrade("vote")
            return [Vote(agent=self.role, round=view["round"], option_id=o.option_id, vote="REJECT",
                         reason=Reason.MALFORMED, rationale="LLM output invalid after retry")
                    for o in view["shortlist"]]
        return [Vote(agent=self.role, round=view["round"], option_id=b.option_id, vote=b.vote, reason=b.reason,
                     rationale=redact_numbers(b.rationale),
                     counter_proposal=b.counter_proposal and b.counter_proposal.model_copy(
                         update={"rationale": redact_numbers(b.counter_proposal.rationale)}))
                for b in out.votes]
