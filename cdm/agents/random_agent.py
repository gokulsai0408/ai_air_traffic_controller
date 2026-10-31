"""RandomAgent (BUILD_PLAN 5.5, E1 kernel ablation): proposes random menu combinations with random field
values (independently chosen, so many are contradictory or unsafe) and votes at random. Seeded, so a
run is reproducible. Its assess() is still the deterministic one, so guardrails apply to its votes."""
from __future__ import annotations

import random
from typing import Any, Optional

from cdm.agents.base import AssessingAgent, CostFn, DeriveFn
from cdm.schemas import ActionType, OptionDraft, Reason, Role, Vote

MAX_HOLD_MIN = 120          # OptionDraft schema bound
P_INVENTED_ID = 0.1         # chance an ID is replaced by one that is not in any menu
P_FIELD_EMPTY = 0.15        # chance a field is left None regardless of action type
P_COUNTER = 0.3

_MENU_KEYS = {"slot_id": ("slot",), "gate_id": ("gate", "stand"), "divert_to": ("altern", "divert")}
_ID_KEYS = ("id", "slot_id", "gate_id", "iata", "code", "airport")


def _ids_from(entries: Any) -> list[str]:
    out = []
    for e in entries if isinstance(entries, (list, tuple)) else (entries.keys() if isinstance(entries, dict) else []):
        if isinstance(e, str):
            out.append(e)
        elif isinstance(e, dict):
            out += [str(e[k]) for k in _ID_KEYS if e.get(k)][:1]
        else:
            out += [str(getattr(e, k)) for k in _ID_KEYS if getattr(e, k, None)][:1]
    return out


def menu_ids(view: dict) -> dict[str, list[str]]:
    """Every slot / gate / alternate ID visible to the agent: from the menus and from candidate options."""
    found: dict[str, set[str]] = {f: set() for f in _MENU_KEYS}
    menus = view.get("menus") or {}
    items = menus.items() if isinstance(menus, dict) else vars(menus).items() if hasattr(menus, "__dict__") else []
    for key, entries in items:
        for field, hints in _MENU_KEYS.items():
            if any(h in str(key).lower() for h in hints):
                found[field].update(_ids_from(entries))
    for o in view.get("candidates", []) + view.get("shortlist", []):
        for field in _MENU_KEYS:
            if getattr(o, field, None):
                found[field].add(getattr(o, field))
    return {f: sorted(v) for f, v in found.items()}


class RandomAgent(AssessingAgent):
    kind = "random"

    def __init__(self, role: Role, seed: int = 0, derive_fn: Optional[DeriveFn] = None,
                 cost_fn: Optional[CostFn] = None):
        super().__init__(role, derive_fn, cost_fn)
        self.rng = random.Random(f"{role}:{seed}")

    def _pick(self, ids: list[str], invented_prefix: str) -> Optional[str]:
        if self.rng.random() < P_FIELD_EMPTY:
            return None
        if not ids or self.rng.random() < P_INVENTED_ID:
            return f"{invented_prefix}{self.rng.randint(900, 999)}"
        return self.rng.choice(ids)

    def random_draft(self, view: dict) -> OptionDraft:
        ids = menu_ids(view)
        return OptionDraft(action_type=self.rng.choice(list(ActionType)), hold_minutes=self.rng.randint(0, MAX_HOLD_MIN),
                           slot_id=self._pick(ids["slot_id"], "S-"), gate_id=self._pick(ids["gate_id"], "Z"),
                           divert_to=self._pick(ids["divert_to"], "X"), rationale="random agent")

    def propose(self, view: dict) -> list[OptionDraft]:
        return [self.random_draft(view) for _ in range(2)]

    def vote(self, view: dict) -> list[Vote]:
        votes = []
        for o in view["shortlist"]:
            counter = self.random_draft(view) if self.rng.random() < P_COUNTER else None
            votes.append(Vote(agent=self.role, round=view["round"], option_id=o.option_id,
                              vote=self.rng.choice(["ACCEPT", "REJECT"]), reason=self.rng.choice(list(Reason)),
                              rationale="random agent", counter_proposal=counter))
        return votes
