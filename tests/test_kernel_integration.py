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
from cdm.options import do_nothing, seed_options
from cdm.safety_view import SafetyView, Slot
from cdm.schemas import ActionType
from cdm.verify import verify


def test_real_world_satisfies_the_kernel_contract():
    world_module = pytest.importorskip("cdm.world")
    data_module = pytest.importorskip("cdm.data")
    flights = data_module.load_flights(Path(__file__).resolve().parents[1] / "data" / "flights.csv")
    for _, row in flights[flights["DESTINATION_AIRPORT"] == "ATL"].head(20).iterrows():
        world = world_module.build_world(row, 7, None)
        assert isinstance(world.safety_view, SafetyView)
        for option in [do_nothing(world), fallback_plan(world), *seed_options(world)]:
            assert kernel.check(option, world) == kernel.check(option, world)
            if option.action_type is not ActionType.DIVERT and option.slot_id:
                # Costs and the dashboard read derive(): it must not double-count the reserve either.
                expected = world.safety_view.extra_fuel_min - option.hold_minutes
                assert world_module.derive(option, world).fuel_margin_min == pytest.approx(expected)
        for attack in kernel_attacks(world):
            report = kernel.check(attack.payload, world)
            assert not report.safe and attack.expected in kernel.failing_ids(report), attack.name


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
