"""Hand-built world and a reference derive() for B's tests, independent of cdm.world."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdm.kernel import constraint_params
from cdm.safety_view import Alternate, Gate, SafetyView, Slot
from cdm.schemas import ActionType, Derived, Option

ETA = datetime(2015, 6, 15, 20, 0)
TAXI_MIN = 5


def minutes(n: float) -> datetime:
    return ETA + timedelta(minutes=n)


@dataclass
class FakeWorld:
    safety_view: SafetyView
    extra_fuel_min: float = 30.0
    alternate_reserve_min: float = 30.0
    fdp_end: datetime = field(default_factory=lambda: minutes(120))
    shift_end: datetime = field(default_factory=lambda: minutes(120))
    gate_busy: dict[str, list[tuple[datetime, datetime]]] = field(default_factory=dict)
    occupy_min: int = 40


def reference_derive(option: Option, world: FakeWorld) -> Derived:
    """Implements the field meanings documented in cdm/safety_view.py."""
    view = world.safety_view
    if option.action_type is ActionType.DIVERT:
        alternate = next(a for a in view.alternates if a.iata == option.divert_to)
        landing = view.eta + timedelta(minutes=option.hold_minutes + alternate.minutes_to)
        on_block = landing + timedelta(minutes=TAXI_MIN)
        fuel = world.extra_fuel_min + world.alternate_reserve_min - option.hold_minutes - alternate.minutes_to
        overlap = 0.0
    else:
        landing = next(s for s in view.slots if s.slot_id == option.slot_id).time
        on_block = landing + timedelta(minutes=TAXI_MIN)
        fuel = world.extra_fuel_min - option.hold_minutes
        buffer = timedelta(minutes=constraint_params("C-GATE-02")["buffer_min"])
        end = on_block + timedelta(minutes=world.occupy_min)
        overlap = 0.0
        for start_busy, end_busy in world.gate_busy.get(option.gate_id, []):
            shared = min(end + buffer, end_busy + buffer) - max(on_block - buffer, start_busy - buffer)
            overlap = max(overlap, shared.total_seconds() / 60.0)
    overtime = 0.0
    if option.action_type is not ActionType.DIVERT:
        overtime = max(0.0, (on_block + timedelta(minutes=world.occupy_min) - world.shift_end).total_seconds() / 60.0)
    return Derived(
        landing_time=landing.strftime("%H:%M"), on_block_time=on_block.strftime("%H:%M"),
        fuel_margin_min=fuel, crew_margin_min=(world.fdp_end - on_block).total_seconds() / 60.0,
        gate_overlap_min=overlap, ground_overtime_min=overtime, missed_pax=0, wake_gap_waste_min=0.0,
    )


def make_view(**overrides) -> SafetyView:
    """Grid: S-00 (ETA-4, M), S-02 (ETA free), S-04 (+4 free), S-08 (+8, H traffic),
    S-10 (+10 free, 2 min behind the Heavy), S-14, S-20, S-40 free, S-60 (M traffic)."""
    slots = (
        Slot("S-00", minutes(-4), True, "M"),
        Slot("S-02", minutes(0), False),
        Slot("S-04", minutes(4), False),
        Slot("S-08", minutes(8), True, "H"),
        Slot("S-10", minutes(10), False),
        Slot("S-14", minutes(14), False),
        Slot("S-20", minutes(20), False),
        Slot("S-40", minutes(40), False),
        Slot("S-60", minutes(60), True, "M"),
    )
    gates = (
        Gate("B1", 2, True, ("A",)),
        Gate("B2", 2, True, ("B",)),
        Gate("B3", 1, True, ("A",)),
        Gate("B4", 3, True, ("C",)),
        Gate("REMOTE-1", 3, False, ()),
    )
    alternates = (Alternate("BHM", 25.0), Alternate("CHA", 45.0), Alternate("MEM", 90.0))
    base = SafetyView(eta=ETA, wake_cat="M", aircraft_size_class=2, original_gate_id="B1",
                      grid_start=minutes(-10), grid_end=minutes(60), slots=slots, gates=gates, alternates=alternates,
                      extra_fuel_min=30.0)
    return replace(base, **overrides)


def make_world(view: SafetyView | None = None, **overrides) -> FakeWorld:
    view = view or make_view()
    extra = overrides.pop("extra_fuel_min", view.extra_fuel_min)
    return FakeWorld(safety_view=replace(view, extra_fuel_min=extra), extra_fuel_min=extra, **overrides)


def land(slot_id: str = "S-02", gate_id: str = "B1", hold: int = 0, action: ActionType | None = None,
         option_id: str = "T-1") -> Option:
    if action is None:
        if hold > 0:
            action = ActionType.HOLD if gate_id == "B1" else ActionType.HOLD_AND_REGATE
        else:
            action = ActionType.LAND_AS_PLANNED if gate_id == "B1" else ActionType.REGATE
    return Option(option_id=option_id, round=1, proposed_by="pilot", action_type=action,
                  hold_minutes=hold, slot_id=slot_id, gate_id=gate_id)


def divert(iata: str = "BHM", hold: int = 0, option_id: str = "T-D") -> Option:
    return Option(option_id=option_id, round=1, proposed_by="pilot", action_type=ActionType.DIVERT,
                  hold_minutes=hold, divert_to=iata)


def result(report, cid: str):
    return next(r for r in report.results if r.id == cid)
