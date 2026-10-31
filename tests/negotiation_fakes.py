"""Stand-ins for A's world/derive/costs and B's kernel/options/fallback, so the coordinator can be tested
before those modules land. Each option's Derived values, per-role costs and kernel failures are scripted."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cdm.agents import make_agents
from cdm.coordinator import Deps
from cdm.schemas import CClass, ConstraintResult, Derived, KernelReport, Option, OptionDraft

CONSTRAINTS = {
    "C-SCHEMA-01": CClass.PHYSICAL, "C-REF-01": CClass.PHYSICAL, "C-CONSIST-01": CClass.PHYSICAL,
    "C-RWY-01": CClass.SAFETY, "C-WAKE-01": CClass.SAFETY, "C-FUEL-01": CClass.SAFETY, "C-FUEL-02": CClass.SAFETY,
    "C-CREW-01": CClass.SAFETY, "C-GATE-01": CClass.PHYSICAL, "C-GATE-02": CClass.PHYSICAL,
    "C-TAXI-01": CClass.PHYSICAL, "C-POL-OT-01": CClass.POLICY,
}
OK_DERIVED = dict(landing_time="20:52", on_block_time="21:00", fuel_margin_min=10.0, crew_margin_min=60.0,
                  gate_overlap_min=0.0, ground_overtime_min=0.0, missed_pax=0, wake_gap_waste_min=0.0)
UNKNOWN_COST = 1e9


def opt(oid: str, action: str = "HOLD", hold: int = 0, slot: Optional[str] = "S-1", gate: Optional[str] = "B1",
        divert: Optional[str] = None, by: str = "generator") -> Option:
    return Option(action_type=action, hold_minutes=hold, slot_id=slot, gate_id=gate, divert_to=divert,
                  option_id=oid, round=0, proposed_by=by)


@dataclass
class Spec:
    option: Option
    costs: tuple[float, float, float]                          # pilot, ground, atc
    derived: dict = field(default_factory=dict)                # overrides on OK_DERIVED
    fails: list[tuple[str, float]] = field(default_factory=list)   # (constraint id, margin)


def _key(o: OptionDraft) -> tuple:
    return (o.action_type, o.hold_minutes, o.slot_id, o.gate_id, o.divert_to)


@dataclass
class FakeWorld:
    specs: list[Spec]
    fallback_id: str = "F"
    do_nothing_id: str = "DN"
    ops_notes: str = ""
    pilot: dict = field(default_factory=lambda: {"extra_fuel_min": 20, "connecting_pax": 31})
    ground: dict = field(default_factory=lambda: {"shift_end": "21:00"})
    atc: dict = field(default_factory=lambda: {"aar": 30})
    menus: dict = field(default_factory=dict)

    @property
    def public(self) -> dict:
        return {"flight_key": "2015-07-01_DL_1234_TEST", "ops_notes": self.ops_notes}

    def spec_for(self, o: OptionDraft) -> Optional[Spec]:
        return next((s for s in self.specs if _key(s.option) == _key(o)), None)

    def by_id(self, oid: str) -> Option:
        return next(s.option for s in self.specs if s.option.option_id == oid)


def fake_derive(option: Option, world: FakeWorld) -> Derived:
    spec = world.spec_for(option)
    return Derived(**{**OK_DERIVED, **(spec.derived if spec else {})})


def fake_cost(i: int):
    def cost(option: Option, derived: Derived, world: FakeWorld) -> float:
        spec = world.spec_for(option)
        return spec.costs[i] if spec else UNKNOWN_COST
    return cost


def fake_check(option: Option, world: FakeWorld) -> KernelReport:
    spec = world.spec_for(option)
    fails = dict(spec.fails) if spec else {"C-REF-01": -1.0}
    results = [ConstraintResult(id=cid, cls=cls, status="FAIL" if cid in fails else "PASS",
                                margin=fails.get(cid, 1.0), unit="min", source="fake")
               for cid, cls in CONSTRAINTS.items()]
    safe = all(r.status == "PASS" for r in results if r.cls != CClass.POLICY)
    policy_ok = all(r.status == "PASS" for r in results if r.cls == CClass.POLICY)
    return KernelReport(option_id=option.option_id, results=results, safe=safe, policy_ok=policy_ok)


def fake_deps() -> Deps:
    return Deps(check=fake_check,
                seed_options=lambda w: [s.option for s in w.specs
                                        if s.option.option_id not in (w.fallback_id, w.do_nothing_id)],
                do_nothing=lambda w: w.by_id(w.do_nothing_id),
                fallback_plan=lambda w: w.by_id(w.fallback_id))


def rule_agents():
    return make_agents("rule", derive_fn=fake_derive, cost_fns={r: fake_cost(i) for i, r in
                                                                enumerate(("pilot", "ground", "atc"))})
