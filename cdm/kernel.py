"""Deterministic safety kernel (BUILD_PLAN 5.3).

check() is pure: no I/O, randomness or LLM calls. constraints.yaml is read once at import.
Reads only world.safety_view (via view_of) and derive(). Anything it cannot evaluate fails closed.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from cdm.safety_view import Gate, SafetyView, Slot, view_of
from cdm.schemas import ActionType, CClass, ConstraintResult, Derived, KernelReport, Option, OpsFacts

CONSTRAINTS_PATH = Path(__file__).resolve().parents[1] / "config" / "constraints.yaml"
_RAW_CONSTRAINTS = CONSTRAINTS_PATH.read_bytes()
CONSTRAINTS: dict[str, dict[str, Any]] = yaml.safe_load(_RAW_CONSTRAINTS)
_WAKE_CATEGORIES = ("J", "H", "M", "L")


def constraint_params(constraint_id: str) -> dict[str, Any]:
    return dict(CONSTRAINTS[constraint_id].get("params") or {})


def constraints_hash() -> str:
    return hashlib.sha256(_RAW_CONSTRAINTS).hexdigest()


def _derive(option: Option, world: Any) -> Derived:
    from cdm.world import derive  # A's module; imported late so B's code loads without it

    return derive(option, world)


def _result(cid: str, passed: bool, margin: Optional[float] = None, unit: str = "", detail: str = "") -> ConstraintResult:
    cfg = CONSTRAINTS[cid]
    return ConstraintResult(
        id=cid, cls=CClass(cfg["class"]), status="PASS" if passed else "FAIL",
        margin=margin, unit=unit, detail=detail, source=cfg["source"], tag=cfg["tag"],
    )


def _margin_result(cid: str, margin: float, unit: str, detail: str) -> ConstraintResult:
    # `margin >= 0` is False for NaN, so a bad number fails closed.
    return _result(cid, bool(margin >= 0), margin, unit, detail)


def _not_applicable(cid: str, action: ActionType) -> ConstraintResult:
    return _result(cid, True, detail=f"N/A for {action.value}")


def _report(option_id: str, results: list[ConstraintResult], evaluated: bool = True) -> KernelReport:
    passed = lambda *classes: all(r.status == "PASS" for r in results if r.cls in classes)
    return KernelReport(
        option_id=option_id, results=results,
        safe=passed(CClass.SAFETY, CClass.PHYSICAL),
        policy_ok=evaluated and passed(CClass.POLICY),   # not evaluated is not compliant
    )


def _schema_problem(opt: Option, view: SafetyView) -> Optional[str]:
    action = opt.action_type
    if action is ActionType.DIVERT:
        if opt.divert_to is None or opt.slot_id is not None or opt.gate_id is not None:
            return "DIVERT needs divert_to and no slot_id or gate_id"
        return None
    if opt.slot_id is None or opt.gate_id is None or opt.divert_to is not None:
        return f"{action.value} needs slot_id and gate_id and no divert_to"
    holds = action in (ActionType.HOLD, ActionType.HOLD_AND_REGATE)
    if holds != (opt.hold_minutes > 0):
        return f"{action.value} is inconsistent with hold_minutes={opt.hold_minutes}"
    regates = action in (ActionType.REGATE, ActionType.HOLD_AND_REGATE)
    if regates != (opt.gate_id != view.original_gate_id):
        return f"{action.value} is inconsistent with gate_id={opt.gate_id}"
    return None


def _unknown_refs(opt: Option, view: SafetyView) -> list[str]:
    unknown = []
    if opt.slot_id is not None and opt.slot_id not in {s.slot_id for s in view.slots}:
        unknown.append(f"slot_id={opt.slot_id}")
    if opt.gate_id is not None and opt.gate_id not in {g.gate_id for g in view.gates}:
        unknown.append(f"gate_id={opt.gate_id}")
    if opt.divert_to is not None and opt.divert_to not in {a.iata for a in view.alternates}:
        unknown.append(f"divert_to={opt.divert_to}")
    return unknown


def _required_gap_min(leader: Optional[str], follower: Optional[str]) -> float:
    params = constraint_params("C-WAKE-01")
    pairs = params["pairs_nm"]
    # An unknown category takes the most restrictive value it could be.
    leaders = (leader,) if leader in _WAKE_CATEGORIES else _WAKE_CATEGORIES
    followers = (follower,) if follower in _WAKE_CATEGORIES else _WAKE_CATEGORIES
    nm = max(pairs.get(f"{a}-{b}", params["default_nm"]) for a in leaders for b in followers)
    return nm / params["approach_speed_kt"] * 60.0


def wake_margin_min(slot: Slot, view: SafetyView) -> Optional[float]:
    """Smallest (gap - required gap) to any other traffic, in minutes. None if no traffic."""
    margins = []
    for other in view.slots:
        if not other.occupied or other.slot_id == slot.slot_id:
            continue
        gap = (slot.time - other.time).total_seconds() / 60.0
        if gap >= 0:
            margins.append(gap - _required_gap_min(other.wake, view.wake_cat))
        if gap <= 0:
            margins.append(-gap - _required_gap_min(view.wake_cat, other.wake))
    return min(margins) if margins else None


def _taxiway_token(text: str) -> str:
    """'TWY B closed between B3-B7' -> 'B'. Segments map to their taxiway, the conservative reading."""
    upper = text.upper()
    named = re.search(r"\b(?:TWY|TAXIWAY)\s+([A-Z]+)", upper)
    if named:
        return named.group(1)
    first = re.search(r"[A-Z]+", upper)
    return first.group(0) if first else ""


def _check_slot(opt: Option, slot: Slot, view: SafetyView) -> list[ConstraintResult]:
    tolerance = constraint_params("C-CONSIST-01")["tolerance_min"]
    expected_hold = max(0.0, (slot.time - view.eta).total_seconds() / 60.0)
    consist = _margin_result(
        "C-CONSIST-01", tolerance - abs(opt.hold_minutes - expected_hold), "min",
        f"hold_minutes={opt.hold_minutes}, slot {slot.slot_id} implies {expected_hold:.1f}",
    )
    in_grid = view.grid_start <= slot.time <= view.grid_end
    rwy = _result(
        "C-RWY-01", not slot.occupied and in_grid,
        detail=f"slot {slot.slot_id}: {'occupied' if slot.occupied else 'free'}, {'in' if in_grid else 'outside'} grid",
    )
    margin = wake_margin_min(slot, view)
    if margin is None:
        wake = _result("C-WAKE-01", True, unit="min", detail="no adjacent traffic")
    else:
        wake = _margin_result("C-WAKE-01", margin, "min", f"tightest wake gap margin at slot {slot.slot_id}")
    return [consist, rwy, wake]


def apply_facts(world: Any, facts: OpsFacts) -> Any:
    """A copy of world whose safety_view also holds these ops facts. Tighten-only: it only adds."""
    if not isinstance(world.safety_view, SafetyView):
        from cdm.world import apply_ops_facts   # A's worlds carry their own tighten-only version

        return apply_ops_facts(world, facts)
    view: SafetyView = world.safety_view
    tightened = replace(
        view,
        taxiway_closures=tuple(sorted(set(view.taxiway_closures) | set(facts.taxiway_closures))),
        gates_unavailable=tuple(sorted(set(view.gates_unavailable) | set(facts.gates_unavailable))),
    )
    clone = copy.copy(world)
    object.__setattr__(clone, "safety_view", tightened)   # also works if World is a frozen dataclass
    return clone


def _check_gate(gate: Gate, derived: Derived, view: SafetyView) -> list[ConstraintResult]:
    size = _margin_result(
        "C-GATE-01", float(gate.size_class - view.aircraft_size_class), "class",
        f"gate {gate.gate_id} class {gate.size_class}, aircraft class {view.aircraft_size_class}",
    )
    overlap = _margin_result("C-GATE-02", -derived.gate_overlap_min, "min", f"gate {gate.gate_id} occupancy overlap")

    closed = {t for t in map(_taxiway_token, view.taxiway_closures) if t}
    blocked = sorted(closed & {_taxiway_token(t) for t in gate.taxi_route})
    is_unavailable = gate.gate_id.upper() in {g.strip().upper() for g in view.gates_unavailable}
    detail = f"gate {gate.gate_id} unavailable" if is_unavailable else (
        f"route blocked by taxiway {', '.join(blocked)}" if blocked else f"gate {gate.gate_id} reachable"
    )
    return [size, overlap, _result("C-TAXI-01", not blocked and not is_unavailable, detail=detail)]


def check(option: Any, world: Any) -> KernelReport:
    """Check one option. Ops facts reach it through apply_facts(), so they can only tighten."""
    view: SafetyView = view_of(world)
    raw_id = option.get("option_id") if isinstance(option, dict) else getattr(option, "option_id", None)
    option_id = str(raw_id) if raw_id else "UNKNOWN"

    # Re-validate rather than trust the object: model_construct() can bypass validation.
    try:
        dumped = option if isinstance(option, dict) else option.model_dump()
        opt = Option.model_validate(dumped)
    except (ValidationError, AttributeError, TypeError, ValueError) as exc:
        return _report(option_id, [_result("C-SCHEMA-01", False, detail=f"invalid message: {type(exc).__name__}")], False)
    problem = _schema_problem(opt, view)
    unknown = _unknown_refs(opt, view)
    # Both always report, so an invented ID is attributed to C-REF-01 even if the action also disagrees.
    results = [
        _result("C-SCHEMA-01", problem is None, detail=problem or "valid"),
        _result("C-REF-01", not unknown, detail="unknown " + ", ".join(unknown) if unknown else "all IDs in menus"),
    ]
    if problem or unknown:
        return _report(opt.option_id, results, False)

    try:
        derived = _derive(opt, world)
    except Exception as exc:  # fail closed: an option we cannot evaluate is not safe
        results.append(_result("C-CONSIST-01", False, detail=f"derive failed: {type(exc).__name__}"))
        return _report(opt.option_id, results, False)

    action = opt.action_type
    if action is ActionType.DIVERT:
        results += [_not_applicable(c, action) for c in ("C-CONSIST-01", "C-RWY-01", "C-WAKE-01", "C-FUEL-01")]
        results.append(_margin_result(
            "C-FUEL-02", derived.fuel_margin_min - constraint_params("C-FUEL-02")["min_margin_min"], "min",
            f"fuel above final reserve after diverting to {opt.divert_to}",
        ))
    else:
        slot = next(s for s in view.slots if s.slot_id == opt.slot_id)
        results += _check_slot(opt, slot, view)
        results.append(_margin_result(
            # extra_fuel_min is already above final reserve + alternate; the reserve is not subtracted again.
            "C-FUEL-01", view.extra_fuel_min - opt.hold_minutes - constraint_params("C-FUEL-01")["min_margin_min"], "min",
            "fuel above reserve + alternate after hold",
        ))
        results.append(_not_applicable("C-FUEL-02", action))

    results.append(_margin_result(
        "C-CREW-01", derived.crew_margin_min - constraint_params("C-CREW-01")["min_margin_min"], "min",
        "crew FDP end minus on-block time",
    ))

    if action is ActionType.DIVERT:
        results += [_not_applicable(c, action) for c in ("C-GATE-01", "C-GATE-02", "C-TAXI-01")]
    else:
        gate = next(g for g in view.gates if g.gate_id == opt.gate_id)
        results += _check_gate(gate, derived, view)

    cap = constraint_params("C-POL-OT-01")["max_overtime_min"]
    results.append(_margin_result(
        "C-POL-OT-01", cap - derived.ground_overtime_min, "min",
        f"overtime {derived.ground_overtime_min:.0f} min against cap {cap} min",
    ))
    return _report(opt.option_id, results)


def failing_ids(report: KernelReport) -> list[str]:
    return [r.id for r in report.results if r.status == "FAIL"]


def min_safety_margin(report: KernelReport) -> float:
    """Tie-break value for CONSENSUS (5.6): the smallest margin on a SAFETY constraint."""
    margins = [r.margin for r in report.results if r.cls is CClass.SAFETY and r.margin is not None]
    return min(margins) if margins else math.inf
