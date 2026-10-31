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
from datetime import datetime
from typing import Optional

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
