"""Negotiation coordinator (BUILD_PLAN 5.6-5.7). Pure code: no LLM, no randomness.

propose -> kernel filter -> vote -> re-propose -> ... -> CONSENSUS | binding COMPROMISE (final round)
| FALLBACK | ESCALATED. The round cap is hard, so the loop always terminates.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, ValidationError

from cdm.agents.base import ROLES, Agent
from cdm.schemas import (ActionType, Assessment, CClass, Decision, Escalation, KernelReport, OpsFacts, Option,
                         OptionDraft, Reason, Vote)

DEFAULT_MAX_ROUNDS = 3
DEFAULT_SHORTLIST_CAP = 6
DEFAULT_MAX_DRAFTS_PER_AGENT = 2
COST_TOLERANCE_USD = 1e-6   # float-comparison tolerance only
ID_PREFIX = {"pilot": "P", "ground": "G", "atc": "A"}   # e.g. "P-2-03" = pilot, round 2, draft 3
HARD_CLASSES = (CClass.SAFETY, CClass.PHYSICAL)


class UnsafeDecisionError(RuntimeError):
    """Raised if anything tries to finalise an option the kernel marked unsafe (golden rule 2)."""


@dataclass(frozen=True)
class Deps:
    """B's kernel/options/fallback. Injected so the coordinator can be tested before they land."""
    check: Callable[[Option, Any], KernelReport]
    seed_options: Callable[[Any], list[Option]]
    do_nothing: Callable[[Any], Option]
    fallback_plan: Callable[[Any], Option]

    @classmethod
    def default(cls) -> "Deps":
        from cdm.fallback import fallback_plan
        from cdm.kernel import check
        from cdm.options import do_nothing, seed_options
        return cls(check=check, seed_options=seed_options, do_nothing=do_nothing, fallback_plan=fallback_plan)


class ListLedger:
    """Stand-in with the same append() signature as cdm.ledger.Ledger (no hash chain)."""

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def append(self, type: str, actor: str, payload: Any, constraint_ids: Optional[list[str]] = None) -> dict:
        entry = {"seq": len(self.entries), "type": type, "actor": actor,
                 "payload": jsonable(payload), "constraint_ids": list(constraint_ids or [])}
        self.entries.append(entry)
        return entry


def jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    return obj


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _ops_notes(world: Any) -> str:
    return _get(_get(world, "public"), "ops_notes") or _get(world, "ops_notes") or ""


def _plan_key(o: OptionDraft) -> tuple:
    return (o.action_type, o.hold_minutes, o.slot_id, o.gate_id, o.divert_to)


def _ids(*reports: KernelReport) -> list[str]:
    return sorted({r.id for rep in reports for r in rep.results})


def _hard_fails(rep: KernelReport) -> list:
    return [r for r in rep.results if r.cls in HARD_CLASSES and r.status == "FAIL"]


def _policy_fails(rep: KernelReport) -> list:
    return [r for r in rep.results if r.cls == CClass.POLICY and r.status == "FAIL"]


def min_safety_margin(rep: KernelReport) -> float:
    margins = [r.margin for r in rep.results if r.cls == CClass.SAFETY and r.margin is not None]
    return min(margins) if margins else math.inf


def describe(o: Option) -> str:
    if o.action_type == ActionType.DIVERT:
        return f"divert to {o.divert_to} after {o.hold_minutes} min hold"
    return f"{o.action_type.value}: hold {o.hold_minutes} min, slot {o.slot_id}, gate {o.gate_id}"


def _dominates(a: tuple, b: tuple) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_shortlist(feasible: list[Option], assess: dict[str, dict[str, Assessment]], cap: int) -> list[Option]:
    """Drop options that are worse for every agent; keep at most `cap`, cheapest total first."""
    costs = {o.option_id: tuple(assess[r][o.option_id].cost_usd for r in ROLES) for o in feasible}
    front = [o for o in feasible
             if not any(_dominates(costs[p.option_id], costs[o.option_id]) for p in feasible if p is not o)]
    front.sort(key=lambda o: (sum(costs[o.option_id]), o.option_id))
    return front[:cap]


def total_cost(assess: dict[str, dict[str, Assessment]], oid: str) -> float:
    return sum(assess[r][oid].cost_usd for r in ROLES)


def view_for(agent: Agent, world: Any, rnd: int, *, candidates: list[Option], own: list[Assessment],
             walkaway_usd: float, shortlist: list[Option], other_votes: list[Vote]) -> dict:
    """Everything one agent may see: public scenario, ITS OWN private state and assessments, and other
    agents' votes as reason codes only (rationales are display-only and never forwarded)."""
    return {
        "role": agent.role, "round": rnd,
        "public": _get(world, "public"),
        "private": _get(world, agent.role),
        "menus": _get(world, "menus"),
        "ops_facts": getattr(agent, "ops_facts", OpsFacts()),
        "candidates": candidates, "own_assessments": own, "walkaway_usd": walkaway_usd,
        "shortlist": shortlist,
        "other_votes": [{"agent": v.agent, "option_id": v.option_id, "vote": v.vote, "reason": v.reason.value}
                        for v in other_votes if v.agent != agent.role],
    }


class _Negotiation:
    def __init__(self, world: Any, agents: list[Agent], cfg: Any, ledger: Any, deps: Deps):
        self.world, self.ledger, self.deps = world, ledger, deps
        self.agents = {a.role: a for a in agents}
        if len(agents) != len(ROLES) or set(self.agents) != set(ROLES):
            raise ValueError("need exactly one agent per role: pilot, ground, atc")
        self.max_rounds = int(_get(cfg, "max_rounds", DEFAULT_MAX_ROUNDS))
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.cap = int(_get(cfg, "shortlist_cap", DEFAULT_SHORTLIST_CAP))
        self.max_drafts = int(_get(cfg, "max_drafts_per_agent", DEFAULT_MAX_DRAFTS_PER_AGENT))
        self.cfg_hash = _get(cfg, "hash")
        self.pool: dict[str, Option] = {}
        self.by_plan: dict[tuple, str] = {}
        self.reports: dict[str, KernelReport] = {}
        self.drafted: list[str] = []
        self.all_votes: list[Vote] = []
        self.draft_seq: dict[int, int] = {}

    # ---- helpers -------------------------------------------------------------------------------
    def log(self, type: str, actor: str, payload: Any, constraint_ids: Optional[list[str]] = None) -> None:
        self.ledger.append(type, actor, jsonable(payload), constraint_ids or [])

    def kcheck(self, o: Option) -> KernelReport:
        if o.option_id not in self.reports:
            rep = self.deps.check(o, self.world)
            if rep.option_id != o.option_id:
                raise ValueError(f"kernel returned report for {rep.option_id}, expected {o.option_id}")
            self.reports[o.option_id] = rep
        return self.reports[o.option_id]

    def ok(self, oid: str) -> bool:
        rep = self.reports[oid]
        return rep.safe and rep.policy_ok

    def add(self, o: Option) -> str:
        key = _plan_key(o)
        if key in self.by_plan:
            return self.by_plan[key]
        if o.option_id in self.pool:
            raise ValueError(f"duplicate option_id {o.option_id}")
        self.pool[o.option_id] = o
        self.by_plan[key] = o.option_id
        return o.option_id

    def add_draft(self, raw: Any, role: str, rnd: int) -> None:
        try:
            draft = raw if isinstance(raw, OptionDraft) else OptionDraft.model_validate(raw)
        except ValidationError as e:
            self.log("DRAFT_REJECTED", "coordinator", {"agent": role, "round": rnd, "reason": Reason.MALFORMED,
                                                       "error": str(e)[:500]}, ["C-SCHEMA-01"])
            return
        n = self.draft_seq.get(rnd, 0) + 1
        self.draft_seq[rnd] = n
        option = Option(**draft.model_dump(), option_id=f"{ID_PREFIX[role]}-{rnd}-{n:02d}", round=rnd,
                        proposed_by=role)
        oid = self.add(option)
        if oid not in self.drafted:
            self.drafted.append(oid)

    def safe_call(self, role: str, what: str, fn: Callable, *args: Any) -> Any:
        try:
            return fn(*args)
        except Exception as e:   # an agent failure must never stop the protocol
            self.log("AGENT_ERROR", "coordinator", {"agent": role, "call": what, "error": repr(e)[:500]})
            return None

    # ---- protocol ------------------------------------------------------------------------------
    def run(self) -> Decision:
        w = self.world
        self.log("SCENARIO", "coordinator", {
            "flight_key": _get(w, "flight_key") or _get(_get(w, "public"), "flight_key"),
            "public": _get(w, "public"), "cfg_hash": self.cfg_hash, "max_rounds": self.max_rounds,
            "agents": {r: type(a).__name__ for r, a in self.agents.items()}})

        notes = _ops_notes(w)
        for role, a in self.agents.items():
            facts = self.safe_call(role, "read_ops", a.read_ops, notes)
            if not isinstance(facts, OpsFacts):
                facts = OpsFacts(uncertain=True)
            self.log("OPS_FACTS", role, facts)

        dn, fb = self.deps.do_nothing(w), self.deps.fallback_plan(w)
        dn_rep, fb_rep = self.kcheck(dn), self.kcheck(fb)
        self.fallback = fb
        self.log("BASELINE", "coordinator", {"do_nothing": {"option": dn, "kernel": dn_rep},
                                             "fallback": {"option": fb, "kernel": fb_rep}}, _ids(dn_rep, fb_rep))
        self.walkaway = {r: a.assess(fb, w).cost_usd for r, a in self.agents.items()}
        self.log("WALKAWAY", "coordinator", self.walkaway)

        for o in self.deps.seed_options(w):
            self.add(o)
        self.add(dn)
        self.add(fb)

        last_votes: list[Vote] = []
        for rnd in range(1, self.max_rounds + 1):
            # 1) PROPOSE (round 1) or COUNTER-PROPOSE (from the previous round's votes)
            if rnd == 1:
                for o in list(self.pool.values()):
                    self.kcheck(o)
                candidates = [o for o in self.pool.values() if self.ok(o.option_id)]
                for role, a in self.agents.items():
                    own = [a.assess(o, w) for o in candidates]
                    view = view_for(a, w, rnd, candidates=candidates, own=own, walkaway_usd=self.walkaway[role],
                                    shortlist=[], other_votes=[])
                    drafts = self.safe_call(role, "propose", a.propose, view) or []
                    drafts = list(drafts)[:self.max_drafts]
                    self.log("PROPOSALS", role, {"round": rnd, "drafts": drafts})
                    for d in drafts:
                        self.add_draft(d, role, rnd)
            else:
                for v in last_votes:
                    if v.counter_proposal is not None:
                        self.add_draft(v.counter_proposal, v.agent, rnd)

            # 2) FILTER: the kernel is the only gate; unsafe options never reach a vote
            new = [o for o in self.pool.values() if o.option_id not in self.reports]
            for o in new:
                self.kcheck(o)
            checked = new if rnd > 1 else list(self.pool.values())
            self.log("KERNEL", "kernel", {"round": rnd, "reports": [self.reports[o.option_id] for o in checked]},
                     _ids(*(self.reports[o.option_id] for o in checked)))
            feasible = [o for o in self.pool.values() if self.ok(o.option_id)]
            if not feasible:
                return self.escalate(rnd)

            # 3) ASSESS: each agent's own code; only feasibility, cost and reason codes leave it
            assess = {r: {o.option_id: a.assess(o, w) for o in feasible} for r, a in self.agents.items()}
            self.log("ASSESSMENTS", "coordinator", {"round": rnd, "assessments": assess})

            # 4) SHORTLIST: Pareto front (cap) + every feasible agent draft
            shortlist = pareto_shortlist(feasible, assess, self.cap)
            listed = {o.option_id for o in shortlist}
            shortlist += [self.pool[oid] for oid in self.drafted if oid not in listed and self.ok(oid)]
            self.log("SHORTLIST", "coordinator", {"round": rnd, "option_ids": [o.option_id for o in shortlist]})

            if rnd < self.max_rounds:
                # 5) VOTE
                votes = self.collect_votes(rnd, feasible, shortlist, assess, last_votes)
                self.all_votes += votes
                self.log("VOTES", "coordinator", {"round": rnd, "votes": votes})
                unanimous = [o for o in shortlist
                             if all(any(v.agent == r and v.option_id == o.option_id and v.vote == "ACCEPT"
                                        for v in votes) for r in ROLES)]
                if unanimous:
                    best = min(unanimous, key=lambda o: (total_cost(assess, o.option_id),
                                                         -min_safety_margin(self.reports[o.option_id]), o.option_id))
                    return self.finalise(best, "CONSENSUS", rnd)
                last_votes = votes
            else:
                # 6) BINDING COMPROMISE: individually rational for all; blockable only by a verifiable
                #    objection (an agent's deterministic assessor marks it infeasible), never by preference
                ir = [o for o in feasible
                      if all(assess[r][o.option_id].cost_usd <= self.walkaway[r] + COST_TOLERANCE_USD for r in ROLES)]
                for o in sorted(ir, key=lambda o: (total_cost(assess, o.option_id), o.option_id)):
                    objections = [r for r in ROLES if not assess[r][o.option_id].feasible_for_me]
                    self.log("COMPROMISE_CHECK", "coordinator",
                             {"option_id": o.option_id, "verified_objections": objections},
                             _ids(self.reports[o.option_id]))
                    if not objections:
                        return self.finalise(o, "COMPROMISE", rnd)
                if self.ok(fb.option_id):
                    return self.finalise(fb, "FALLBACK", rnd)
                return self.escalate(rnd)
        raise AssertionError("unreachable: the final round always returns")

    def collect_votes(self, rnd: int, feasible: list[Option], shortlist: list[Option],
                      assess: dict[str, dict[str, Assessment]], last_votes: list[Vote]) -> list[Vote]:
        listed = {o.option_id for o in shortlist}
        out: list[Vote] = []
        for role, a in self.agents.items():
            view = view_for(a, self.world, rnd, candidates=feasible, own=[assess[role][o.option_id] for o in shortlist],
                            walkaway_usd=self.walkaway[role], shortlist=shortlist, other_votes=last_votes)
            raw = self.safe_call(role, "vote", a.vote, view)
            seen: set[str] = set()
            for rv in raw or []:
                try:
                    v = rv if isinstance(rv, Vote) else Vote.model_validate(rv)
                except ValidationError as e:
                    self.log("VOTE_DISCARDED", "coordinator", {"agent": role, "why": "MALFORMED", "error": str(e)[:500]})
                    continue
                if v.agent != role or v.round != rnd or v.option_id not in listed or v.option_id in seen:
                    self.log("VOTE_DISCARDED", "coordinator", {"agent": role, "vote": v,
                                                               "why": "wrong agent/round, unlisted or duplicate"})
                    continue
                seen.add(v.option_id)
                mine = assess[role][v.option_id]
                if v.vote == "ACCEPT" and not mine.feasible_for_me:
                    reason = next((r for r in mine.reasons if r != Reason.ACCEPTABLE), Reason.UNKNOWN)
                    self.log("GUARDRAIL_OVERRIDE", "coordinator",
                             {"agent": role, "option_id": v.option_id, "from": "ACCEPT", "to": "REJECT", "reason": reason})
                    v = v.model_copy(update={"vote": "REJECT", "reason": reason})
                out.append(v)
            for oid in sorted(listed - seen):
                out.append(Vote(agent=role, round=rnd, option_id=oid, vote="REJECT",
                                reason=Reason.MALFORMED if raw is None else Reason.UNKNOWN,
                                rationale="no valid vote cast (coordinator default)"))
        return out

    def finalise(self, option: Option, outcome: str, rnd: int, escalation: Optional[Escalation] = None) -> Decision:
        rep = self.kcheck(option)
        if not rep.safe:
            raise UnsafeDecisionError(f"{option.option_id} failed the kernel and cannot reach a Decision")
        costs = {r: a.assess(option, self.world).cost_usd for r, a in self.agents.items()}
        d = Decision(outcome=outcome, option=option, kernel=rep, rounds_used=rnd, votes=self.all_votes,
                     escalation=escalation, costs_by_agent=costs)
        self.log("DECISION", "coordinator", d, _ids(rep))
        return d

    def escalate(self, rnd: int) -> Decision:
        """5.7: smallest single POLICY relaxation if one exists, else DECLINE. SAFETY is never relaxed."""
        relaxable = []
        for oid, o in self.pool.items():
            rep = self.reports[oid]
            pf = _policy_fails(rep)
            if rep.safe and len(pf) == 1:
                size = abs(pf[0].margin) if pf[0].margin is not None else math.inf
                relaxable.append((size, oid, o, pf[0]))
        if relaxable:
            _, oid, o, c = min(relaxable, key=lambda t: (t[0], t[1]))
            esc = Escalation(kind="POLICY_RELAXATION", required_human_decision=policy_text(c, o),
                             relaxed_constraint_id=c.id, enabled_option_id=oid, blocking_constraints=[c.id])
            self.log("ESCALATION", "coordinator", esc, [c.id])
            return self.finalise(o, "ESCALATED", rnd, esc)

        closest = min(self.pool.values(), key=lambda o: (len(_hard_fails(self.reports[o.option_id])),
                                                         -min_safety_margin(self.reports[o.option_id]), o.option_id))
        blocking = _hard_fails(self.reports[closest.option_id])
        esc = Escalation(kind="DECLINE", required_human_decision=decline_text(blocking),
                         blocking_constraints=[c.id for c in blocking])
        self.log("ESCALATION", "coordinator", esc, esc.blocking_constraints)
        d = Decision(outcome="ESCALATED", option=None, kernel=None, rounds_used=rnd, votes=self.all_votes,
                     escalation=esc)
        self.log("DECISION", "coordinator", d, esc.blocking_constraints)
        return d


# ---- deterministic escalation templates (never an LLM) -------------------------------------------

def _fmt_margin(c) -> str:
    return f"{abs(c.margin):g} {c.unit}".strip() if c.margin is not None else "an unknown amount"


POLICY_TEMPLATES: dict[str, str] = {
    "C-POL-OT-01": "approve {amount} of ground-crew overtime beyond the policy cap",
}


def policy_text(c, o: Option) -> str:
    action = POLICY_TEMPLATES.get(c.id, "relax {cid} by {amount}").format(amount=_fmt_margin(c), cid=c.id)
    detail = f"; {c.detail}" if c.detail else ""
    return (f"No plan satisfies all constraints. Minimum change: {action} ({c.id}{detail}). "
            f"This enables option {o.option_id}: {describe(o)}. Approve / Reject.")


def decline_text(blocking: list) -> str:
    parts = [f"{c.id} (margin {c.margin:+g} {c.unit})".replace(" )", ")") if c.margin is not None else c.id
             for c in blocking]
    return ("No safe plan within the model. PIC and controller decision required: priority handling / fuel advisory. "
            f"Constraints blocking: {', '.join(parts) or 'UNKNOWN'}.")


def negotiate(world: Any, agents: list[Agent], cfg: Any, ledger: Any, deps: Optional[Deps] = None) -> Decision:
    """BUILD_PLAN 6.3 entry point. `deps` defaults to B's real kernel/options/fallback modules."""
    return _Negotiation(world, agents, cfg, ledger, deps or Deps.default()).run()
