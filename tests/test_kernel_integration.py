"""Runs B's modules against A's and C's real code. Each test skips until that code is on the branch."""
from __future__ import annotations

from pathlib import Path

import kernel_support  # noqa: F401  (puts the repo root on sys.path)
import pytest
from kernel_support import make_view, make_world, minutes, reference_derive

from cdm import kernel
from cdm.attacks import kernel_attacks
from cdm.fallback import fallback_plan
from cdm.ledger import Ledger
from cdm.options import do_nothing, landing_option, seed_options
from cdm.safety_view import SafetyView, Slot, view_of
from cdm.schemas import ActionType
from cdm.verify import verify


@pytest.fixture(scope="module")
def real_worlds():
    world_module = pytest.importorskip("cdm.world")
    data_module = pytest.importorskip("cdm.data")
    flights = data_module.load_flights(Path(__file__).resolve().parents[1] / "data" / "flights.csv")
    cfg = data_module.load_config()
    rows = flights[flights["DESTINATION_AIRPORT"] == "ATL"].sample(25, random_state=1)
    return [world_module.build_world(row, seed, cfg) for _, row in rows.iterrows() for seed in (0, 1)]


def test_real_worlds_convert_to_the_kernel_view(real_worlds):
    for world in real_worlds:
        view = view_of(world)
        assert isinstance(view, SafetyView)
        assert view.original_gate_id in {g.gate_id for g in view.gates}
        assert view.extra_fuel_min == world.safety_view.extra_fuel_min
        assert len(view.slots) == len(world.safety_view.slots) and len(view.gates) == len(world.safety_view.gates)
        assert [s.slot_id for s in view.slots] == [s.slot_id for s in world.safety_view.slots]
        # Same slot gaps in minutes, whatever the epoch.
        first, last = world.safety_view.slots[0], world.safety_view.slots[-1]
        assert (view.slots[-1].time - view.slots[0].time).total_seconds() / 60 == pytest.approx(last.time_min - first.time_min)


def test_real_worlds_options_are_checked_and_deterministic(real_worlds):
    world_module = pytest.importorskip("cdm.world")
    for world in real_worlds:
        for option in [do_nothing(world), fallback_plan(world), *seed_options(world)]:
            assert kernel.check(option, world) == kernel.check(option, world)
            if option.action_type is not ActionType.DIVERT and option.slot_id:
                # Costs and the dashboard read derive(): it must not double-count the reserve either.
                derived = world_module.derive(option, world)
                assert derived.fuel_margin_min == pytest.approx(world.safety_view.extra_fuel_min - option.hold_minutes)
                report = kernel.check(option, world)
                assert next(r for r in report.results if r.id == "C-FUEL-01").margin == pytest.approx(derived.fuel_margin_min)


def test_real_worlds_seed_options_pass_the_reference_checks(real_worlds):
    # seed_options only builds consistent plans from menu IDs, so nothing may fail C-SCHEMA/REF/CONSIST.
    for world in real_worlds:
        for option in seed_options(world):
            failing = set(kernel.failing_ids(kernel.check(option, world)))
            assert not failing & {"C-SCHEMA-01", "C-REF-01", "C-CONSIST-01"}, (option.option_id, failing)


def test_real_world_attacks_are_all_caught(real_worlds):
    ran = 0
    for world in real_worlds:
        for attack in kernel_attacks(world):
            report = kernel.check(attack.payload, world)
            ran += 1
            assert not report.safe and attack.expected in kernel.failing_ids(report), attack.name
    assert ran > 0


def test_real_world_ops_facts_tighten_through_a_own_function(real_worlds):
    from cdm.schemas import OpsFacts

    checked = 0
    for world in real_worlds:
        view = view_of(world)
        slot = next(sl for sl in view.slots if sl.slot_id == do_nothing(world).slot_id)
        for gate in (g for g in view.gates if g.gate_id != "REMOTE-1"):
            option = landing_option("T-1", slot, gate.gate_id, view)
            if "C-TAXI-01" in kernel.failing_ids(kernel.check(option, world)):
                continue                                     # already blocked by the world's own closures
            tightened = kernel.apply_facts(world, OpsFacts(gates_unavailable=[gate.gate_id]))
            assert "C-TAXI-01" in kernel.failing_ids(kernel.check(option, tightened))
            assert gate.gate_id not in world.safety_view.gates_unavailable   # the original is untouched
            checked += 1
            break
    assert checked > 0


def test_real_worlds_include_both_clean_and_unsafe_do_nothing_plans(real_worlds):
    """Reported, not asserted for a rate: shows the kernel is discriminating on A's worlds."""
    safe = [kernel.check(do_nothing(w), w).safe for w in real_worlds]
    assert any(safe) and not all(safe), f"do-nothing safe in {sum(safe)}/{len(safe)} worlds"


@pytest.mark.parametrize("max_rounds", [1, 2, 3])
@pytest.mark.parametrize("world_args,outcome", [
    ({}, None),
    ({"view": make_view(alternates=()), "shift_end": minutes(20)}, "ESCALATED"),   # every plan breaks the overtime policy
    ({"extra_fuel_min": 0.0, "alternate_reserve_min": 0.0, "fdp_end": minutes(0)}, "ESCALATED"),   # nothing is safe
])
def test_verify_reproduces_real_coordinator_logs(tmp_path, monkeypatch, max_rounds, world_args, outcome):
    coordinator = pytest.importorskip("cdm.coordinator")
    rule = pytest.importorskip("cdm.agents.rule")
    monkeypatch.setattr(kernel, "_derive", reference_derive)

    world = make_world(**world_args)
    agents = [rule.RuleAgent(role, derive_fn=reference_derive,
                             cost_fn=lambda option, derived, w: float(option.hold_minutes + derived.ground_overtime_min))
              for role in ("pilot", "ground", "atc")]
    ledger = Ledger("k", 1, "c")
    decision = coordinator.negotiate(world, agents, {"max_rounds": max_rounds}, ledger)
    if outcome:
        assert decision.outcome == outcome

    report = verify(ledger.save(tmp_path / "r.jsonl"), lambda *args: world)
    assert report.complete, (decision.outcome, report.summary(), report.problems)


def test_real_coordinator_cannot_finalise_an_unsafe_fallback(monkeypatch):
    coordinator = pytest.importorskip("cdm.coordinator")
    rule = pytest.importorskip("cdm.agents.rule")
    monkeypatch.setattr(kernel, "_derive", reference_derive)

    view = make_view(slots=(Slot("S-L", minutes(40), False),), alternates=())
    world = make_world(view, extra_fuel_min=5.0)
    agents = [rule.RuleAgent(role, derive_fn=reference_derive, cost_fn=lambda o, d, w: 1.0)
              for role in ("pilot", "ground", "atc")]
    decision = coordinator.negotiate(world, agents, {"max_rounds": 2}, Ledger("k", 1, "c"))
    assert decision.outcome == "ESCALATED" and decision.option is None


def test_verify_rebuilds_a_real_world_from_the_log_alone(tmp_path):
    from cdm.schemas import Assessment, Decision

    world_module = pytest.importorskip("cdm.world")
    data_module = pytest.importorskip("cdm.data")
    flights = data_module.load_flights(Path(__file__).resolve().parents[1] / "data" / "flights.csv")
    cfg = data_module.load_config()
    roles = ("pilot", "ground", "atc")
    for _, row in flights[flights["DESTINATION_AIRPORT"] == "ATL"].sample(20, random_state=3).iterrows():
        world = world_module.build_world(row, 5, cfg)
        reports = {o.option_id: (o, kernel.check(o, world)) for o in seed_options(world)}
        feasible = [o for o, r in reports.values() if r.safe and r.policy_ok]
        if feasible:
            break
    else:
        pytest.skip("no sampled world had a feasible seed option")

    fb = fallback_plan(world)
    chosen = feasible[0]
    ledger = Ledger(world.flight_key, 5, world.config_hash)
    ledger.append("SCENARIO", "coordinator", {"flight_key": world.flight_key, "max_rounds": 1})
    ledger.append("BASELINE", "coordinator", {"do_nothing": {"option": do_nothing(world), "kernel": kernel.check(do_nothing(world), world)},
                                              "fallback": {"option": fb, "kernel": kernel.check(fb, world)}})
    ledger.append("WALKAWAY", "coordinator", {r: 1e9 for r in roles})
    ledger.append("KERNEL", "kernel", {"round": 1, "options": [chosen], "reports": [kernel.check(chosen, world)]})
    ledger.append("ASSESSMENTS", "coordinator", {"round": 1, "assessments": {
        r: {chosen.option_id: Assessment(agent=r, option_id=chosen.option_id, feasible_for_me=True, cost_usd=1.0, reasons=[])}
        for r in roles}})
    ledger.append("DECISION", "coordinator", Decision(outcome="COMPROMISE", option=chosen,
                                                      kernel=kernel.check(chosen, world), rounds_used=1))
    report = verify(ledger.save(tmp_path / "real.jsonl"))          # default loader: rebuilds from data/ and config/
    assert report.complete, (report.summary(), report.problems)
    assert report.kernel_checked >= 3          # do-nothing, fallback and the chosen option, at least
