"""One deterministic cost function per agent, in USD-equivalent (plan section 5.4).

Delay is the extra delay this decision adds beyond the ETA. Delay already incurred is the same for
every option, and including it would hide the differences the agents negotiate over.
"""
from __future__ import annotations

from .schemas import Derived, Option
from .world import REMOTE_GATE, World, timeline


def cost_pilot(option: Option, derived: Derived, world: World) -> float:
    line = timeline(option, world)
    delay = line.on_block_min - (world.public.eta_min + world.public.taxi_in_min)
    cost = (delay * world.cost("cost_per_delay_min")
            + line.hold_min * world.pilot.hold_flow_kg_per_min * world.cost("fuel_price_usd_per_kg")
            + derived.missed_pax * world.cost("cost_per_missed_pax"))
    if line.diverted:
        cost += world.cost("divert_fixed")
    if world.pilot.turbulence_in_hold:
        cost += line.hold_min * world.cost("turbulence_hold_penalty_per_min")
    return float(cost)


def cost_ground(option: Option, derived: Derived, world: World) -> float:
    line = timeline(option, world)
    min_turn = world.ground.min_turn_min
    if line.diverted:
        # The aircraft isn't there to turn, so the next departure loses its whole turnaround
        short_turn = min_turn
    else:
        # Capped: once the next departure is already blown, landing later squeezes nothing further
        short_turn = min(min_turn, max(0.0, min_turn - (world.ground.next_departure_min - line.on_block_min)))
    cost = (derived.ground_overtime_min * world.cost("overtime_rate_per_min")
            + short_turn * world.cost("turn_penalty_per_min"))
    if not line.diverted and option.gate_id != world.ground.original_gate:
        cost += world.cost("gate_change_fixed")
    if option.gate_id == REMOTE_GATE:
        cost += world.cost("remote_stand_cost")
    return float(cost)


def cost_atc(option: Option, derived: Derived, world: World) -> float:
    line = timeline(option, world)
    if line.diverted:
        return float(line.hold_min * world.cost("holding_workload_per_min") + world.cost("divert_coordination_cost"))
    stacked = sum(1 for s in world.atc.slots
                  if s.occupied and line.landing_min - 10 <= s.time_min < line.landing_min)
    return float(line.hold_min * world.cost("holding_workload_per_min")
                 + stacked * world.cost("stack_congestion_per_aircraft")
                 + derived.wake_gap_waste_min * world.cost("runway_minute_value"))
