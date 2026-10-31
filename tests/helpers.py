"""Test support. flights_sample.csv and airports_sample.csv are HAND-MADE, not the DOT data.

find_feasible() roughly mirrors the kernel so Person A can calibrate the generator before Person B's
kernel lands. It is not the safety kernel, and nothing should rely on it for safety claims.
"""
from pathlib import Path

from cdm.data import load_config, load_flights
from cdm.schemas import ActionType, Option
from cdm.world import SIZE_RANK, World, derive

FIXTURES = Path(__file__).parent / "fixtures"


def sample_cfg():
    return load_config(airports_path=FIXTURES / "airports_sample.csv")


def sample_flights():
    return load_flights(FIXTURES / "flights_sample.csv")


def option_for(world: World, slot_id=None, gate_id=None, divert_to=None, option_id="T-1") -> Option:
    if divert_to:
        return Option(action_type=ActionType.DIVERT, hold_minutes=0, divert_to=divert_to,
                      option_id=option_id, round=1, proposed_by="generator")
    slot = next(s for s in world.safety_view.slots if s.slot_id == slot_id)
    hold = max(0, round(slot.time_min - world.public.eta_min))
    return Option(action_type=ActionType.HOLD_AND_REGATE, hold_minutes=hold, slot_id=slot_id, gate_id=gate_id,
                  option_id=option_id, round=1, proposed_by="generator")


def _wake_legal(world: World, slot) -> bool:
    for other in world.safety_view.slots:
        if not other.occupied:
            continue
        if other.time_min < slot.time_min and slot.time_min - other.time_min < world.wake_gap(other.wake, world.public.wake):
            return False
        if other.time_min > slot.time_min and other.time_min - slot.time_min < world.wake_gap(world.public.wake, other.wake):
            return False
    return True


def _gate_reachable(world: World, gate, on_block: float) -> bool:
    if gate.gate_id in world.safety_view.gates_unavailable:
        return False
    return not any(c.taxiway == gate.taxiway and (c.until_min is None or on_block < c.until_min)
                   for c in world.safety_view.taxiway_closures)


def find_feasible(world: World, cfg) -> list[Option]:
    """Menu options passing approximate SAFETY, PHYSICAL and POLICY checks."""
    cap = cfg.constraints["C-POL-OT-01"]["params"]["max_overtime_min"]
    category = SIZE_RANK[world.public.aircraft_category]
    slots = {s.slot_id: s for s in world.safety_view.slots}
    gates = {g.gate_id: g for g in world.safety_view.gates}
    feasible = []
    for slot_id in world.menus.slots:
        if not _wake_legal(world, slots[slot_id]):
            continue
        for gate_id in world.menus.gates:
            gate = gates[gate_id]
            if SIZE_RANK[gate.size_class] < category:
                continue
            option = option_for(world, slot_id, gate_id)
            d = derive(option, world)
            on_block = slots[slot_id].time_min + world.public.taxi_in_min
            if (d.fuel_margin_min >= 0 and d.crew_margin_min >= 0 and d.gate_overlap_min == 0
                    and d.ground_overtime_min <= cap and _gate_reachable(world, gate, on_block)):
                feasible.append(option)
    for iata in world.menus.alternates:
        option = option_for(world, divert_to=iata)
        d = derive(option, world)
        if d.fuel_margin_min >= 0 and d.crew_margin_min >= 0:
            feasible.append(option)
    return feasible
