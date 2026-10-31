"""What the kernel may see: the contract between world.py (A) and kernel.py (B).

build_world must set `world.safety_view` to a SafetyView. derive(option, world) must fill
the Derived fields the kernel reads with these meanings:

    fuel_margin_min      non-DIVERT: extra_fuel_min - hold_minutes (C-FUEL-01 computes this itself)
                         DIVERT: fuel above final reserve - hold_minutes - minutes to alternate
                         Both are already above the 45-min reserve: never subtract it again.
    crew_margin_min      crew FDP end - on-block time, in minutes
    gate_overlap_min     minutes on-block occupancy overlaps other occupancy, with each interval
                         widened by constraint_params("C-GATE-02")["buffer_min"]; 0 if none
    ground_overtime_min  minutes past the ground shift end; 0 if none

derive is only called on options that passed C-SCHEMA-01 and C-REF-01, so IDs resolve.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

REMOTE_GATE_ID = "REMOTE-1"


@dataclass(frozen=True)
class Slot:
    slot_id: str
    time: datetime
    occupied: bool
    wake: Optional[str] = None      # J/H/M/L of the occupying traffic; None is treated as worst case


@dataclass(frozen=True)
class Gate:
    gate_id: str
    size_class: int                 # 1 regional, 2 narrowbody, 3 widebody
    contact: bool = True            # False for REMOTE-1
    taxi_route: tuple[str, ...] = ()


@dataclass(frozen=True)
class Alternate:
    iata: str
    minutes_to: float


_EPOCH = datetime(2000, 1, 1)   # only differences between times matter to the kernel


def _at(minutes: float) -> datetime:
    return _EPOCH + timedelta(minutes=minutes)


@dataclass(frozen=True)
class SafetyView:
    eta: datetime
    wake_cat: str
    aircraft_size_class: int
    original_gate_id: str
    grid_start: datetime
    grid_end: datetime
    slots: tuple[Slot, ...]         # the WHOLE grid including occupied slots, sorted by time
    gates: tuple[Gate, ...]         # includes REMOTE-1
    alternates: tuple[Alternate, ...]
    extra_fuel_min: float           # holding fuel ABOVE final reserve + alternate, in minutes
    taxiway_closures: tuple[str, ...] = ()
    gates_unavailable: tuple[str, ...] = ()


def view_of(world: Any) -> SafetyView:
    """The kernel's view of a world, whichever form the world's safety_view takes.

    A SafetyView is returned as is. Person A's cdm.world.SafetyView (times in minutes, sizes as
    names) is converted. Nothing is added or dropped, except that a closure's end time is ignored:
    a closed taxiway counts as closed for the whole scenario, which can only tighten the result.
    """
    raw = world.safety_view
    if isinstance(raw, SafetyView):
        return raw
    from cdm.world import SIZE_RANK   # A's module; imported late so B's code loads without it

    slots = tuple(Slot(s.slot_id, _at(s.time_min), s.occupied, s.wake) for s in raw.slots)
    times = [s.time for s in slots]
    return SafetyView(
        eta=_at(raw.eta_min), wake_cat=raw.wake, aircraft_size_class=SIZE_RANK[raw.aircraft_category],
        original_gate_id=world.ground.original_gate,
        grid_start=min(times) if times else _at(raw.eta_min), grid_end=max(times) if times else _at(raw.eta_min),
        slots=slots,
        gates=tuple(Gate(g.gate_id, SIZE_RANK[g.size_class], g.gate_id != REMOTE_GATE_ID,
                         (g.taxiway,) if g.taxiway else ()) for g in raw.gates),
        alternates=tuple(Alternate(a.iata, a.time_min) for a in raw.alternates),
        extra_fuel_min=raw.extra_fuel_min,
        taxiway_closures=tuple(c.taxiway for c in raw.taxiway_closures),
        gates_unavailable=tuple(raw.gates_unavailable),
    )
