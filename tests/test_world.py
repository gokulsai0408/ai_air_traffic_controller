import statistics
import subprocess
import sys
import time

import pytest

from cdm.costs import cost_atc, cost_ground, cost_pilot
from cdm.schemas import ActionType, OpsFacts, Option
from cdm.world import REMOTE_GATE, World, apply_ops_facts, build_world, derive
from tests.helpers import find_feasible, option_for, sample_cfg, sample_flights

CFG = sample_cfg()
FLIGHTS = sample_flights()
ATL = FLIGHTS.query("FLIGHT_NUMBER == 1234").iloc[0]


def calm_world(seed_start=0) -> World:
    return next(w for s in range(seed_start, 500) if not (w := build_world(ATL, s, CFG)).stress)


def worlds(n=200):
    return [build_world(FLIGHTS.iloc[i % len(FLIGHTS)], i, CFG) for i in range(n)]


def test_same_input_same_world():
    assert build_world(ATL, 7, CFG).model_dump() == build_world(ATL, 7, CFG).model_dump()


def test_different_seed_different_world():
    assert build_world(ATL, 1, CFG).model_dump() != build_world(ATL, 2, CFG).model_dump()


def test_world_reproducible_across_processes():
    # Guards against hash(): its value changes per process, which would break verify replay
    code = ("from tests.helpers import *; from cdm.world import build_world;"
            "r=sample_flights().query('FLIGHT_NUMBER == 1234').iloc[0];"
            "print(build_world(r,7,sample_cfg()).model_dump_json())")
    outputs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                              env={"PYTHONHASHSEED": str(s), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}).stdout
               for s in (1, 2)}
    assert len(outputs) == 1


def test_eta_excludes_taxi_in():
    world = build_world(ATL, 1, CFG)
    assert world.public.eta_min == 19 * 60 + 50 + 45 - 9


def test_do_nothing_on_block_matches_recorded_arrival():
    world = calm_world()
    line_on_block = world.public.eta_min + world.public.taxi_in_min
    assert world.fmt_time(line_on_block) == "2015-06-12T20:35"


def test_every_view_field_is_tagged():
    world = build_world(ATL, 1, CFG)
    for view in ("public", "pilot", "ground", "atc"):
        for field in type(getattr(world, view)).model_fields:
            assert f"{view}.{field}" in world.tags, f"{view}.{field} has no tag"


def test_real_next_leg_tagged_real_otherwise_synthetic():
    assert build_world(ATL, 1, CFG).tags["ground.next_departure_min"] == "REAL"
    other = FLIGHTS.query("FLIGHT_NUMBER == 1500").iloc[0]
    assert build_world(other, 1, CFG).tags["ground.next_departure_min"] == "SYNTHETIC"


def test_safety_view_holds_no_commercial_data():
    fields = set(type(build_world(ATL, 1, CFG).safety_view).model_fields)
    assert not fields & {"connections", "connecting_pax", "shift_end_min", "hold_flow_kg_per_min", "next_departure_min"}


def test_public_view_holds_no_private_numbers():
    fields = set(type(build_world(ATL, 1, CFG).public).model_fields)
    assert not fields & {"extra_fuel_min", "fdp_end_min", "connecting_pax", "shift_end_min"}


def test_menu_slots_are_free_and_not_before_eta():
    for world in worlds(30):
        slots = {s.slot_id: s for s in world.safety_view.slots}
        assert all(not slots[i].occupied and slots[i].time_min >= world.public.eta_min for i in world.menus.slots)


def test_remote_stand_always_available():
    for world in worlds(30):
        assert REMOTE_GATE in world.menus.gates


def test_alternates_exclude_near_airports_and_stress_closes_them():
    for world in worlds(60):
        if world.public.destination == "ATL":
            alternates = [a for a in world.safety_view.alternates]
            assert all(a.distance_nm > 50 for a in alternates)
            assert (not alternates) == bool(world.stress)


def test_calibration_no_safe_option_at_least_15_percent():
    sample = worlds(200)
    share = sum(not find_feasible(w, CFG) for w in sample) / len(sample)
    assert share >= 0.15, share


def test_calibration_not_everything_is_stuck():
    sample = worlds(200)
    assert sum(not find_feasible(w, CFG) for w in sample) / len(sample) <= 0.45


def test_calibration_original_gate_conflict_at_least_half():
    sample = worlds(200)
    conflicts = 0
    for world in sample:
        gate = next(g for g in world.ground.gates if g.gate_id == world.ground.original_gate)
        option = Option(action_type=ActionType.LAND_AS_PLANNED, hold_minutes=0, slot_id=None, gate_id=gate.gate_id,
                        option_id="D", round=1, proposed_by="do_nothing")
        conflicts += derive(option, world).gate_overlap_min > 0
    assert conflicts / len(sample) >= 0.5


def test_derive_fuel_margin_does_not_subtract_reserve_again():
    world = calm_world()
    slot = world.menus.slots[0]
    option = option_for(world, slot, REMOTE_GATE)
    assert derive(option, world).fuel_margin_min == pytest.approx(world.pilot.extra_fuel_min - option.hold_minutes, abs=1)


def test_derive_uses_real_wait_when_option_understates_hold():
    world = calm_world()
    late = max((s for s in world.safety_view.slots if not s.occupied), key=lambda s: s.time_min)
    lying = Option(action_type=ActionType.HOLD, hold_minutes=0, slot_id=late.slot_id, gate_id=REMOTE_GATE,
                   option_id="X", round=1, proposed_by="adversary")
    expected = world.pilot.extra_fuel_min - (late.time_min - world.public.eta_min)
    assert derive(lying, world).fuel_margin_min == pytest.approx(expected)


def test_derive_survives_unknown_references():
    world = build_world(ATL, 1, CFG)
    for option in (
        Option(action_type=ActionType.HOLD, hold_minutes=5, slot_id="S-99", gate_id="Z99", option_id="X", round=1, proposed_by="adversary"),
        Option(action_type=ActionType.DIVERT, hold_minutes=0, divert_to="XXX", option_id="Y", round=1, proposed_by="adversary"),
    ):
        derive(option, world)


def test_wake_table_values():
    world = build_world(ATL, 1, CFG)
    assert world.wake_gap("M", "M") == pytest.approx(3 / 140 * 60)
    assert world.wake_gap("H", "M") == pytest.approx(5 / 140 * 60)
    assert world.wake_gap("H", "H") == pytest.approx(4 / 140 * 60)


def test_divert_misses_all_connections_and_has_no_overtime():
    world = calm_world()
    derived = derive(option_for(world, divert_to=world.menus.alternates[0]), world)
    assert derived.missed_pax == world.pilot.connecting_pax
    assert derived.ground_overtime_min == 0


def test_ops_facts_only_tighten():
    world = calm_world()
    before = list(world.safety_view.taxiway_closures)
    empty = apply_ops_facts(world, OpsFacts())
    assert empty.safety_view.taxiway_closures == before
    assert empty.pilot.turbulence_in_hold == world.pilot.turbulence_in_hold
    tighter = apply_ops_facts(world, OpsFacts(taxiway_closures=["Q"], gates_unavailable=["A1"], turbulence_in_hold=True))
    assert any(c.taxiway == "Q" and c.until_min is None for c in tighter.safety_view.taxiway_closures)
    assert "A1" in tighter.safety_view.gates_unavailable and tighter.pilot.turbulence_in_hold


def test_costs_rise_with_hold():
    world = calm_world()
    free = [s for s in world.safety_view.slots if not s.occupied and s.time_min >= world.public.eta_min]
    early, late = option_for(world, free[0].slot_id, REMOTE_GATE), option_for(world, free[-1].slot_id, REMOTE_GATE)
    if free[-1].time_min - free[0].time_min < 10:
        pytest.skip("slots too close together")
    assert cost_pilot(late, derive(late, world), world) > cost_pilot(early, derive(early, world), world)


def test_divert_costs_more_for_pilot_than_landing_now():
    world = calm_world()
    land = option_for(world, world.menus.slots[0], REMOTE_GATE)
    divert = option_for(world, divert_to=world.menus.alternates[0])
    assert cost_pilot(divert, derive(divert, world), world) > cost_pilot(land, derive(land, world), world)


def test_remote_stand_costs_ground_more_than_original_gate():
    world = calm_world()
    slot = world.menus.slots[0]
    remote, original = option_for(world, slot, REMOTE_GATE), option_for(world, slot, world.ground.original_gate)
    gap = cost_ground(remote, derive(remote, world), world) - cost_ground(original, derive(original, world), world)
    assert gap == pytest.approx(CFG.cost("gate_change_fixed") + CFG.cost("remote_stand_cost"))


def test_short_turn_capped_so_divert_never_beats_landing_on_a_blown_next_leg():
    world = calm_world()
    world.ground.next_departure_min = world.public.eta_min - 300
    gate = world.ground.original_gate
    land = option_for(world, world.menus.slots[0], gate)
    divert = option_for(world, divert_to=world.menus.alternates[0])
    land_cost = cost_ground(land, derive(land, world), world) - derive(land, world).ground_overtime_min * CFG.cost("overtime_rate_per_min")
    assert land_cost == pytest.approx(world.ground.min_turn_min * CFG.cost("turn_penalty_per_min"))
    assert cost_ground(divert, derive(divert, world), world) == pytest.approx(land_cost)


def test_costs_never_negative():
    for world in worlds(40):
        for slot in world.menus.slots[:3]:
            option = option_for(world, slot, REMOTE_GATE)
            derived = derive(option, world)
            for fn in (cost_pilot, cost_ground, cost_atc):
                assert fn(option, derived, world) >= 0


def test_world_round_trips_through_json():
    world = build_world(ATL, 1, CFG)
    assert World.model_validate_json(world.model_dump_json()).model_dump() == world.model_dump()


def test_config_override_changes_hash():
    assert CFG.with_costs(divert_fixed=1).hash != CFG.hash
    assert CFG.with_synthetic(aar_weather_per_hour=20).hash != CFG.hash


def test_build_world_is_fast():
    times = []
    for seed in range(50):
        start = time.perf_counter()
        build_world(ATL, seed, CFG)
        times.append(time.perf_counter() - start)
    assert statistics.median(times) < 0.05
