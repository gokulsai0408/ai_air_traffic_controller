"""Agent interface (BUILD_PLAN 6.3) and the deterministic per-role assessor shared by every agent kind.

Feasibility and cost always come from code here (derive() + the role's cost function), never from an LLM.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from cdm.schemas import ActionType, Assessment, Derived, OpsFacts, Option, OptionDraft, Reason, Role, Vote

ROLES: tuple[Role, ...] = ("pilot", "ground", "atc")
DRAFT_FIELDS = frozenset(OptionDraft.model_fields)

DeriveFn = Callable[[Option, Any], Derived]
CostFn = Callable[[Option, Derived, Any], float]


@runtime_checkable
class Agent(Protocol):
    role: Role
    def assess(self, option: Option, world: Any) -> Assessment: ...
    def read_ops(self, notes: str) -> OpsFacts: ...
    def propose(self, view: dict) -> list[OptionDraft]: ...
    def vote(self, view: dict) -> list[Vote]: ...


def _default_derive() -> DeriveFn:
    from cdm.world import derive
    return derive


def _default_cost(role: Role) -> CostFn:
    import cdm.costs as costs
    return getattr(costs, f"cost_{role}")


def role_feasibility(role: Role, option: Option, d: Derived, facts: OpsFacts) -> tuple[bool, list[Reason]]:
    """Deterministic feasibility + reason codes from one role's point of view. Sign checks only:
    thresholds live in the kernel / constraints.yaml, not here."""
    feasible, reasons = True, []
    if role == "pilot":
        if d.fuel_margin_min < 0:
            feasible = False; reasons.append(Reason.FUEL_RESERVE)
        if d.crew_margin_min < 0:
            feasible = False; reasons.append(Reason.CREW_FDP)
        if d.missed_pax > 0:
            reasons.append(Reason.CONNECTIONS_MISSED)
        if facts.turbulence_in_hold and option.hold_minutes > 0:
            reasons.append(Reason.TURBULENCE)
        if option.action_type == ActionType.DIVERT:
            reasons.append(Reason.DIVERT_COST)
    elif role == "ground":
        if d.gate_overlap_min > 0 or (option.gate_id is not None and option.gate_id in facts.gates_unavailable):
            feasible = False; reasons.append(Reason.GATE_CONFLICT)
        if d.ground_overtime_min > 0:
            reasons.append(Reason.GROUND_OVERTIME)
    else:
        if option.hold_minutes > 0:
            reasons.append(Reason.HOLDING_WORKLOAD)
        if d.wake_gap_waste_min > 0:
            reasons.append(Reason.RUNWAY_CAPACITY)
        if option.action_type == ActionType.DIVERT:
            reasons.append(Reason.DIVERT_COST)
    return feasible, reasons or [Reason.ACCEPTABLE]


_TWY = re.compile(r"\bTWY\s+([A-Z]{1,2}\d{0,2})\b[^.]*?\bCLOSED\b", re.I)
_GATE = re.compile(r"\bGATE\s+([A-Z]{1,2}\d{1,3})\b[^.]*?\b(CLOSED|UNAVAILABLE|U/S|OUT OF SERVICE)\b", re.I)
_TURB = re.compile(r"\b(MOD|SEV|MODERATE|SEVERE)\s+TURB", re.I)
_RWY = re.compile(r"\b(RWY|RUNWAY)\s+CONFIG[^.]*", re.I)


def parse_ops_rules(notes: str) -> OpsFacts:
    """Regex extraction of ops notes into OpsFacts. Deterministic, tighten-only: anything it can't
    map to a fixed field is ignored, so instructions in the text have nowhere to go."""
    notes = notes or ""
    rwy = _RWY.search(notes)
    return OpsFacts(
        taxiway_closures=sorted({m.group(1).upper() for m in _TWY.finditer(notes)}),
        gates_unavailable=sorted({m.group(1).upper() for m in _GATE.finditer(notes)}),
        turbulence_in_hold=bool(_TURB.search(notes)),
        runway_config_change=rwy.group(0).strip() if rwy else None,
    )


def merge_facts(a: OpsFacts, b: OpsFacts) -> OpsFacts:
    """Tighten-only union of two extractions."""
    return OpsFacts(
        taxiway_closures=sorted(set(a.taxiway_closures) | set(b.taxiway_closures)),
        gates_unavailable=sorted(set(a.gates_unavailable) | set(b.gates_unavailable)),
        turbulence_in_hold=a.turbulence_in_hold or b.turbulence_in_hold,
        runway_config_change=a.runway_config_change or b.runway_config_change,
        uncertain=a.uncertain or b.uncertain,
    )


class AssessingAgent:
    """Base for every agent kind: owns the role's deterministic assess(). Subclasses add propose/vote."""
    kind = "base"

    def __init__(self, role: Role, derive_fn: Optional[DeriveFn] = None, cost_fn: Optional[CostFn] = None):
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        self.role: Role = role
        self._derive_fn = derive_fn
        self._cost_fn = cost_fn
        self.ops_facts = OpsFacts()

    def assess(self, option: Option, world: Any) -> Assessment:
        derive = self._derive_fn or _default_derive()
        cost = self._cost_fn or _default_cost(self.role)
        d = derive(option, world)
        feasible, reasons = role_feasibility(self.role, option, d, self.ops_facts)
        return Assessment(agent=self.role, option_id=option.option_id, feasible_for_me=feasible,
                          cost_usd=float(cost(option, d, world)), reasons=reasons)

    def read_ops(self, notes: str) -> OpsFacts:
        self.ops_facts = parse_ops_rules(notes)
        return self.ops_facts


def as_draft(option: Option, rationale: str = "") -> OptionDraft:
    data = option.model_dump(include=DRAFT_FIELDS)
    data["rationale"] = rationale
    return OptionDraft(**data)
