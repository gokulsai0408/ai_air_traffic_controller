"""Deterministic fallback plan and escalation search helpers (BUILD_PLAN 5.7)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from cdm import kernel
from cdm.options import divert_option, free_slots_from_eta, landing_option
from cdm.safety_view import REMOTE_GATE_ID, SafetyView, Slot
from cdm.schemas import CClass, ConstraintResult, KernelReport, Option

FALLBACK_ID = "FB-000"
_GATE_IDS = ("C-GATE-01", "C-GATE-02", "C-TAXI-01")


def _first_wake_legal_slot(view: SafetyView) -> Optional[Slot]:
    free = free_slots_from_eta(view)
    for slot in free:
        margin = kernel.wake_margin_min(slot, view)
        if margin is None or margin >= 0:
            return slot
    return free[0] if free else None   # none is legal: the kernel will report this one as unsafe


def _first_usable_gate(slot: Slot, view: SafetyView, world: Any) -> str:
    for gate in sorted((g for g in view.gates if g.contact), key=lambda g: (g.gate_id != view.original_gate_id, g.gate_id)):
        report = kernel.check(landing_option(FALLBACK_ID, slot, gate.gate_id, view, "fallback"), world)
        if not set(kernel.failing_ids(report)) & set(_GATE_IDS):
            return gate.gate_id
    return REMOTE_GATE_ID


def fallback_plan(world: Any) -> Option:
    """Always returns an Option. Run it through check(): an unsafe result triggers escalation."""
    view: SafetyView = world.safety_view
    alternates = sorted(view.alternates, key=lambda a: (a.minutes_to, a.iata))
    slot = _first_wake_legal_slot(view)

    plan = None
    if slot is not None:
        plan = landing_option(FALLBACK_ID, slot, _first_usable_gate(slot, view, world), view, "fallback")
        if "C-FUEL-01" not in kernel.failing_ids(kernel.check(plan, world)):
            return plan

    for alternate in alternates:
        candidate = divert_option(FALLBACK_ID, alternate.iata, "fallback")
        if "C-FUEL-02" not in kernel.failing_ids(kernel.check(candidate, world)):
            return candidate

    if plan is not None:
        return plan
    # No free slot and no reachable alternate: an explicitly unsafe plan rather than an exception.
    return divert_option(FALLBACK_ID, alternates[0].iata, "fallback") if alternates \
        else landing_option(FALLBACK_ID, None, view.original_gate_id, view, "fallback")


@dataclass(frozen=True)
class PolicyRelaxation:
    report: KernelReport
    constraint: ConstraintResult

    @property
    def relaxation(self) -> float:
        return abs(self.constraint.margin or 0.0)


def find_policy_relaxation(reports: Iterable[KernelReport]) -> Optional[PolicyRelaxation]:
    """The option needing the smallest relaxation of exactly one POLICY constraint.

    Requires report.safe, so a SAFETY or PHYSICAL failure can never be offered to the human.
    """
    candidates = []
    for report in reports:
        failed = [r for r in report.results if r.status == "FAIL"]
        if report.safe and len(failed) == 1 and failed[0].cls is CClass.POLICY:
            candidates.append(PolicyRelaxation(report, failed[0]))
    return min(candidates, key=lambda c: (c.relaxation, c.report.option_id), default=None)


def blocking_constraints(reports: Iterable[KernelReport]) -> list[str]:
    """Constraint IDs behind a DECLINE, most common first, then by ID."""
    counts: dict[str, int] = {}
    for report in reports:
        for cid in kernel.failing_ids(report):
            counts[cid] = counts.get(cid, 0) + 1
    return sorted(counts, key=lambda cid: (-counts[cid], cid))
