"""Test support. flights_sample.csv and airports_sample.csv are HAND-MADE, not the DOT data.

find_feasible() runs Person B's real kernel, so the calibration numbers are the real ones.
"""
from pathlib import Path

from cdm.data import load_config, load_flights
from cdm.schemas import ActionType, Option
from cdm.kernel import check
from cdm.options import seed_options
from cdm.world import World

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


def find_feasible(world: World, cfg=None) -> list[Option]:
    """Seed options the real kernel accepts as safe and policy-compliant."""
    return [o for o in seed_options(world) if (r := check(o, world)).safe and r.policy_ok]
