"""Option menus (BUILD_PLAN 5.2): complete plans built only from IDs that exist in the world."""
from __future__ import annotations

import math
from typing import Any, Optional

from cdm.safety_view import Gate, SafetyView, Slot
from cdm.schemas import ActionType, Option

MAX_SEED_SLOTS = 6
EARLIEST_SEED_SLOTS = 3
MAX_HOLD_MIN = 120


def hold_for_slot(slot: Slot, view: SafetyView) -> int:
    return max(0, math.ceil((slot.time - view.eta).total_seconds() / 60.0))


def landing_option(option_id: str, slot: Optional[Slot], gate_id: Optional[str], view: SafetyView,
                   proposed_by: str = "generator", round_: int = 0) -> Option:
    """A non-DIVERT option whose action_type and hold_minutes agree with its slot and gate.

    A missing slot produces an option that fails C-SCHEMA-01, so "no slot" stays explicitly unsafe.
    """
    hold = hold_for_slot(slot, view) if slot else 0
    regate = gate_id != view.original_gate_id
    if hold > 0:
        action = ActionType.HOLD_AND_REGATE if regate else ActionType.HOLD
    else:
        action = ActionType.REGATE if regate else ActionType.LAND_AS_PLANNED
    return Option(option_id=option_id, round=round_, proposed_by=proposed_by, action_type=action,
                  hold_minutes=min(hold, MAX_HOLD_MIN), slot_id=slot.slot_id if slot else None, gate_id=gate_id)


def divert_option(option_id: str, iata: str, proposed_by: str = "generator", round_: int = 0) -> Option:
    return Option(option_id=option_id, round=round_, proposed_by=proposed_by,
                  action_type=ActionType.DIVERT, hold_minutes=0, divert_to=iata)


def free_slots_from_eta(view: SafetyView) -> list[Slot]:
    # Slots before ETA stay in the menu for agents, but we never seed a plan that lands early.
    return sorted((s for s in view.slots if not s.occupied and s.time >= view.eta), key=lambda s: (s.time, s.slot_id))


def do_nothing(world: Any) -> Option:
    """FCFS baseline: original gate, first free slot >= ETA, no coordination."""
    view: SafetyView = world.safety_view
    free = free_slots_from_eta(view)
    return landing_option("DN-000", free[0] if free else None, view.original_gate_id, view, proposed_by="do_nothing")


def _spread(slots: list[Slot]) -> list[Slot]:
    """The earliest slots (least holding) plus evenly spaced later ones (gates that free up)."""
    if len(slots) <= MAX_SEED_SLOTS:
        return slots
    rest = slots[EARLIEST_SEED_SLOTS:]
    extra = MAX_SEED_SLOTS - EARLIEST_SEED_SLOTS
    return slots[:EARLIEST_SEED_SLOTS] + [rest[(i + 1) * len(rest) // extra - 1] for i in range(extra)]


def _seed_gates(view: SafetyView) -> list[Gate]:
    return sorted(
        (g for g in view.gates if g.size_class >= view.aircraft_size_class or g.gate_id == view.original_gate_id),
        key=lambda g: (g.gate_id != view.original_gate_id, not g.contact, g.gate_id),
    )


def seed_options(world: Any) -> list[Option]:
    """Slots x gates x alternates, roughly 30-80 options, deterministic order.

    Not filtered for safety or policy: the kernel decides, and escalation needs the
    options that only fail on policy.
    """
    view: SafetyView = world.safety_view
    options: list[Option] = []
    for slot in _spread(free_slots_from_eta(view)):
        for gate in _seed_gates(view):
            options.append(landing_option(f"G-{len(options) + 1:03d}", slot, gate.gate_id, view))
    for alternate in sorted(view.alternates, key=lambda a: (a.minutes_to, a.iata)):
        options.append(divert_option(f"G-{len(options) + 1:03d}", alternate.iata))
    return options
