from __future__ import annotations

import pytest

from kernel_support import divert, land, make_view, make_world, minutes, reference_derive

from cdm import kernel
from cdm.attacks import kernel_attacks
from cdm.fallback import blocking_constraints, fallback_plan, find_policy_relaxation
from cdm.options import do_nothing, seed_options
from cdm.safety_view import Alternate, Gate, Slot
from cdm.schemas import ActionType, CClass


@pytest.fixture(autouse=True)
def fake_derive(monkeypatch):
    monkeypatch.setattr(kernel, "_derive", reference_derive)


# --- options
def test_do_nothing_is_original_gate_first_free_slot():
    option = do_nothing(make_world())
    assert (option.slot_id, option.gate_id, option.proposed_by) == ("S-02", "B1", "do_nothing")
    assert kernel.check(option, make_world()).safe


def test_do_nothing_with_no_free_slot_is_explicitly_unsafe():
    view = make_view(slots=(Slot("S-00", minutes(0), True, "M"),))
    assert kernel.failing_ids(kernel.check(do_nothing(make_world(view)), make_world(view))) == ["C-SCHEMA-01"]


def test_seed_options_only_use_menu_ids_and_are_schema_consistent():
    world = make_world()
    options = seed_options(world)
    assert len({o.option_id for o in options}) == len(options) >= 10
    for option in options:
        failing = kernel.failing_ids(kernel.check(option, world))
        assert not {"C-SCHEMA-01", "C-REF-01", "C-CONSIST-01"} & set(failing)
        assert option.proposed_by == "generator"


def test_seed_options_are_deterministic_and_never_land_early():
    assert seed_options(make_world()) == seed_options(make_world())
    assert all(o.slot_id != "S-00" for o in seed_options(make_world()))


def test_seed_option_count_on_a_full_size_world():
    slots = tuple(Slot(f"S-{i}", minutes(2 * i), i % 3 == 0, "M" if i % 3 == 0 else None) for i in range(35))
    gates = tuple(Gate(f"G{i}", 2, True, ("A",)) for i in range(12)) + (Gate("REMOTE-1", 3, False, ()),)
    view = make_view(slots=slots, gates=gates, original_gate_id="G0")
    assert 30 <= len(seed_options(make_world(view))) <= 85


def test_seed_includes_policy_failing_options_for_escalation():
    world = make_world(shift_end=minutes(20))
    reports = {o.option_id: kernel.check(o, world) for o in seed_options(world)}
    assert any(r.safe and not r.policy_ok for r in reports.values())


# --- fallback
def test_fallback_takes_earliest_wake_legal_slot_and_original_gate():
    plan = fallback_plan(make_world())
    assert (plan.slot_id, plan.gate_id, plan.proposed_by) == ("S-02", "B1", "fallback")


def test_fallback_skips_wake_illegal_slot():
    view = make_view(slots=(Slot("S-H", minutes(-1), True, "H"), Slot("S-1", minutes(0), False),
                            Slot("S-2", minutes(4), False)))
    plan = fallback_plan(make_world(view))
    assert plan.slot_id == "S-2" and plan.hold_minutes == 4


def test_fallback_moves_off_busy_gate_to_next_usable():
    plan = fallback_plan(make_world(gate_busy={"B1": [(minutes(0), minutes(60))]}))
    assert plan.gate_id == "B2" and plan.action_type is ActionType.REGATE


def test_fallback_uses_remote_when_no_contact_gate_is_usable():
    busy = [(minutes(0), minutes(60))]
    plan = fallback_plan(make_world(gate_busy={"B1": busy, "B2": busy, "B4": busy}))
    assert plan.gate_id == "REMOTE-1"


def test_fallback_respects_ops_facts():
    from cdm.schemas import OpsFacts
    assert fallback_plan(kernel.apply_facts(make_world(), OpsFacts(taxiway_closures=["TWY A"]))).gate_id == "B2"


def test_fallback_diverts_when_fuel_fails():
    view = make_view(slots=(Slot("S-L", minutes(20), False),))
    plan = fallback_plan(make_world(view, extra_fuel_min=5.0))
    assert plan.action_type is ActionType.DIVERT and plan.divert_to == "BHM"
    assert kernel.check(plan, make_world(view, extra_fuel_min=5.0)).safe


def test_fallback_is_explicitly_unsafe_when_nothing_works():
    view = make_view(slots=(Slot("S-L", minutes(40), False),), alternates=(Alternate("MEM", 90.0),))
    world = make_world(view, extra_fuel_min=5.0)
    report = kernel.check(fallback_plan(world), world)
    assert not report.safe and "C-FUEL-01" in kernel.failing_ids(report)


@pytest.mark.parametrize("view", [
    make_view(slots=(), alternates=()),
    make_view(slots=(Slot("S-0", minutes(0), True, "M"),), alternates=()),
    make_view(alternates=()),
])
def test_fallback_always_returns_an_option_and_never_raises(view):
    assert kernel.check(fallback_plan(make_world(view)), make_world(view)).option_id == "FB-000"


# --- escalation helpers
def test_policy_relaxation_picks_smallest_of_exactly_one_policy_failure():
    world = make_world(shift_end=minutes(20))
    options = [land("S-02", option_id="near"), land("S-20", "B2", 20, option_id="far")]
    found = find_policy_relaxation(kernel.check(o, world) for o in options)
    assert found.report.option_id == "near"
    assert found.constraint.id == "C-POL-OT-01" and found.constraint.cls is CClass.POLICY
    assert found.relaxation == 10.0


def test_policy_relaxation_never_offers_an_unsafe_option():
    world = make_world(shift_end=minutes(20), extra_fuel_min=1.0)
    report = kernel.check(land("S-20", hold=20), world)
    assert not report.safe and not report.policy_ok
    assert find_policy_relaxation([report]) is None


def test_blocking_constraints_lists_most_common_first():
    world = make_world(extra_fuel_min=1.0, fdp_end=minutes(0))
    options = [land("S-20", hold=20, option_id="a"), land("S-40", hold=40, option_id="b"), divert("MEM")]
    blocked = blocking_constraints(kernel.check(o, world) for o in options)
    assert blocked[0] in {"C-CREW-01"} and "C-FUEL-01" in blocked and "C-FUEL-02" in blocked


# --- attacks
def attack_world():
    view = make_view(taxiway_closures=("TWY C",))
    return make_world(view, extra_fuel_min=10.0, fdp_end=minutes(30),
                      gate_busy={"B2": [(minutes(0), minutes(60))]})


def test_every_kernel_attack_is_rejected_by_its_expected_constraint():
    world = attack_world()
    attacks = kernel_attacks(world)
    for attack in attacks:
        report = kernel.check(attack.payload, world)
        assert not report.safe, attack.name
        assert attack.expected in kernel.failing_ids(report), attack.name


def test_attack_suite_covers_every_kernel_row_of_e2():
    expected = {a.expected for a in kernel_attacks(attack_world())}
    assert expected == {"C-WAKE-01", "C-RWY-01", "C-FUEL-01", "C-FUEL-02", "C-CREW-01", "C-GATE-01", "C-GATE-02",
                        "C-TAXI-01", "C-REF-01", "C-CONSIST-01", "C-SCHEMA-01"}


def test_attacks_are_omitted_rather_than_faked_when_the_world_cannot_express_them():
    view = make_view(slots=(Slot("S-1", minutes(0), False),), alternates=(Alternate("BHM", 1.0),))
    names = {a.name for a in kernel_attacks(make_world(view))}
    assert "slot-already-occupied" not in names and "wake-gap-too-small" not in names
    assert "divert-too-far" not in names and "invented-gate" in names
