"""Deterministic escalation texts (BUILD_PLAN 5.7). Templates only: no LLM ever writes these.

POLICY_RELAXATION names the smallest single POLICY change a human may approve and the plan it enables.
DECLINE names what blocks the nearest plan, who must decide, and what blocks the whole pool.
SAFETY and PHYSICAL constraints are never offered for relaxation.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Optional

from cdm.schemas import ActionType, ConstraintResult, KernelReport, Option

# Who must act when a hard constraint blocks every plan. Grouped by the party that owns the problem.
DECLINE_ACTIONS: dict[str, str] = {
    "C-FUEL-01": "PIC and arrival controller: priority handling or a fuel advisory",
    "C-FUEL-02": "PIC and arrival controller: priority handling or a fuel advisory",
    "C-CREW-01": "airline crew control: approve a duty extension or arrange a relief crew",
    "C-RWY-01": "arrival controller: priority sequencing or a slot swap",
    "C-WAKE-01": "arrival controller: priority sequencing or a slot swap",
    "C-GATE-01": "ramp control: free or tow a suitable gate",
    "C-GATE-02": "ramp control: free or tow a suitable gate",
    "C-TAXI-01": "ramp control and tower: reopen a taxi route or assign a reachable gate",
}
DEFAULT_DECLINE_ACTION = "duty manager: review the scenario data; no valid plan could be built"

POLICY_ACTIONS: dict[str, str] = {
    "C-POL-OT-01": "approve {amount} of ground-crew overtime beyond the policy cap",
}
DEFAULT_POLICY_ACTION = "relax {cid} by {amount}"


def _amount(c: ConstraintResult) -> str:
    """Rounded up, so approving the stated amount is always enough."""
    if c.margin is None:
        return "an unknown amount"
    return f"{math.ceil(abs(c.margin) - 1e-9):d} {c.unit}".strip()


def describe(option: Option, landing_time: Optional[str] = None) -> str:
    if option.action_type == ActionType.DIVERT:
        hold = f" after a {option.hold_minutes} min hold" if option.hold_minutes else ""
        return f"divert to {option.divert_to}{hold}"
    at = f" at {landing_time}" if landing_time else ""
    hold = f"hold {option.hold_minutes} min, " if option.hold_minutes else ""
    regate = option.action_type in (ActionType.REGATE, ActionType.HOLD_AND_REGATE)
    land = "land as planned" if option.action_type == ActionType.LAND_AS_PLANNED else "land"
    gate = f"re-gate to {option.gate_id}" if regate else f"gate {option.gate_id}"
    return f"{hold}{land} in slot {option.slot_id}{at}, {gate}"


def policy_text(c: ConstraintResult, option: Option, landing_time: Optional[str] = None) -> str:
    action = POLICY_ACTIONS.get(c.id, DEFAULT_POLICY_ACTION).format(amount=_amount(c), cid=c.id)
    detail = f": {c.detail}" if c.detail else ""
    return (f"No plan satisfies all constraints. Minimum change: {action} ({c.id}{detail}). "
            f"This enables option {option.option_id}: {describe(option, landing_time)}. Approve / Reject.")


def hard_failures(report: KernelReport) -> list[ConstraintResult]:
    return [r for r in report.results if r.status == "FAIL" and r.cls.value != "POLICY"]


def blocking_counts(reports: Iterable[KernelReport]) -> Counter:
    """How many plans each hard constraint blocks."""
    return Counter(c.id for rep in reports for c in hard_failures(rep))


def decline_text(nearest: Optional[Option], blockers: list[ConstraintResult], counts: Counter,
                 n_options: int, landing_time: Optional[str] = None) -> str:
    def why(c: ConstraintResult) -> str:
        return f"{c.id} (short by {_amount(c)})" if c.margin is not None else c.id

    actions = list(dict.fromkeys(DECLINE_ACTIONS.get(c.id, DEFAULT_DECLINE_ACTION) for c in blockers)) \
        or [DEFAULT_DECLINE_ACTION]
    nearest_part = (f" Nearest plan {nearest.option_id} ({describe(nearest, landing_time)}) is blocked by "
                    f"{', '.join(why(c) for c in blockers)}." if nearest is not None and blockers else "")
    pool_part = (f" Across all {n_options} plans: "
                 + ", ".join(f"{cid} blocks {n}" for cid, n in counts.most_common()) + "." if counts else "")
    return (f"No safe plan within the model.{nearest_part} Human decision required: {'; '.join(actions)}."
            f"{pool_part} Safety constraints are never relaxed by this system.")
