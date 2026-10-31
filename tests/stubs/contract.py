"""Throwaway stand-in for A, B and C's modules, shaped to the Section 6 contracts.

It lets Part D run end to end before they land. Its worlds and costs are made up: nothing it
produces may be cited, and every output says STUB. Delete this once the real pipeline lands.
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from eval.integration import ROLES, flight_key
from tests.stubs import schemas_copy as S

SLOT_PREFIX = "S-"
REMOTE = "REMOTE-1"


@dataclass
class World:
    flight_key: str; seed: int; eta_min: int; category: str
    public: dict; pilot: dict; ground: dict; atc: dict
    safety_view: dict; menus: dict; tags: dict = field(default_factory=dict)


def clock(minutes: float) -> str:
    whole = int(round(minutes)) % 1440
    return f"{whole // 60:02d}:{whole % 60:02d}"


def build_world(row: pd.Series, seed: int, cfg) -> World:
    key = flight_key(row)
    rng = np.random.default_rng(zlib.crc32(key.encode()) ^ seed)
    sched = str(row["SCHEDULED_ARRIVAL"]).split(".")[0].zfill(4)
    eta = int(sched[:2]) * 60 + int(sched[2:]) + int(row["ARRIVAL_DELAY"])
    fuel_low, fuel_high = cfg.synthetic.get("extra_fuel_min", [4, 30])
    offsets = list(range(0, 60, 2))
    occupied = rng.random(len(offsets)) < 0.65
    slots = {f"{SLOT_PREFIX}{i}": off for i, off in enumerate(offsets)}
    free = [sid for (sid, _), busy in zip(slots.items(), occupied) if not busy]
    gates = [f"G{i}" for i in range(1, 7)]
    gate_free_at = {g: float(rng.uniform(-20, 40)) for g in gates} | {REMOTE: -999.0}
    safety = {
        "free_slots": free, "slots": slots, "extra_fuel_min": float(rng.uniform(fuel_low, fuel_high)),
        "crew_margin_at_eta": float(rng.uniform(10, 120)), "gate_free_at": gate_free_at,
        "shift_margin_at_eta": float(rng.uniform(-15, 60)), "alternates": {"ALT1": float(rng.uniform(8, 30))},
    }
    return World(
        flight_key=key, seed=seed, eta_min=eta, category="Narrowbody",
        public={"airline": row["AIRLINE"], "flight": int(row["FLIGHT_NUMBER"]), "tail": row["TAIL_NUMBER"],
                "origin": row["ORIGIN_AIRPORT"], "destination": row["DESTINATION_AIRPORT"],
                "scheduled_arrival": sched, "arrival_delay": float(row["ARRIVAL_DELAY"]),
                "ops_notes": "STUB: TWY B closed until 2130Z."},
        pilot={"extra_fuel_min": safety["extra_fuel_min"], "crew_margin_at_eta": safety["crew_margin_at_eta"],
               "connecting_pax": int(rng.integers(20, 70))},
        ground={"gate_free_at": gate_free_at, "shift_margin_at_eta": safety["shift_margin_at_eta"]},
        atc={"slots": slots, "free_slots": free},
        safety_view=safety,
        menus={"slots": free, "gates": gates + [REMOTE], "alternates": list(safety["alternates"])},
        tags={"public": "REAL", "pilot": "SYNTHETIC", "ground": "SYNTHETIC", "atc": "SYNTHETIC"},
    )


def derive(option, world: World) -> S.Derived:
    view = world.safety_view
    hold = option.hold_minutes
    if option.action_type == S.ActionType.DIVERT or option.action_type == "DIVERT":
        leg = view["alternates"].get(option.divert_to, 0.0)
        return S.Derived(landing_time=None, on_block_time=None, fuel_margin_min=view["extra_fuel_min"] - hold - leg,
                         crew_margin_min=view["crew_margin_at_eta"] - hold - leg, gate_overlap_min=0,
                         ground_overtime_min=0, missed_pax=world.pilot["connecting_pax"], wake_gap_waste_min=0)
    offset = view["slots"].get(option.slot_id, hold)
    on_block = offset + 8
    overlap = max(0.0, view["gate_free_at"].get(option.gate_id, 0.0) + 10 - on_block)
    return S.Derived(
        landing_time=clock(world.eta_min + offset), on_block_time=clock(world.eta_min + on_block),
        fuel_margin_min=view["extra_fuel_min"] - hold, crew_margin_min=view["crew_margin_at_eta"] - on_block,
        gate_overlap_min=overlap, ground_overtime_min=max(0.0, on_block + 35 - view["shift_margin_at_eta"]),
        missed_pax=int(world.pilot["connecting_pax"] * min(1.0, offset / 60)), wake_gap_waste_min=0.0)


def cost_pilot(option, derived, world) -> float:
    divert = 8000.0 if derived.on_block_time is None else 0.0
    return option.hold_minutes * 100.0 + derived.missed_pax * 150.0 + divert


def cost_ground(option, derived, world) -> float:
    return derived.ground_overtime_min * 20.0 + (600.0 if option.gate_id == REMOTE else 0.0)


def cost_atc(option, derived, world) -> float:
    return option.hold_minutes * 15.0 + (500.0 if derived.on_block_time is None else 0.0)


def _result(cid: str, cls: S.CClass, ok: bool, margin: float | None = None) -> S.ConstraintResult:
    return S.ConstraintResult(id=cid, cls=cls, status="PASS" if ok else "FAIL", margin=margin,
                              source="STUB", tag="ASSUMED_CONST")


def check(option, world: World) -> S.KernelReport:
    view, is_divert = world.safety_view, str(getattr(option.action_type, "value", option.action_type)) == "DIVERT"
    d = derive(option, world)
    if is_divert:
        ref = option.divert_to in view["alternates"]
        results = [_result("C-REF-01", S.CClass.PHYSICAL, ref),
                   _result("C-FUEL-02", S.CClass.SAFETY, d.fuel_margin_min >= 0, d.fuel_margin_min),
                   _result("C-CREW-01", S.CClass.SAFETY, d.crew_margin_min >= 0, d.crew_margin_min)]
    else:
        ref = option.slot_id in view["slots"] and option.gate_id in view["gate_free_at"]
        offset = view["slots"].get(option.slot_id, -1)
        results = [_result("C-REF-01", S.CClass.PHYSICAL, ref),
                   _result("C-CONSIST-01", S.CClass.PHYSICAL, abs(option.hold_minutes - offset) <= 1),
                   _result("C-RWY-01", S.CClass.SAFETY, option.slot_id in view["free_slots"]),
                   _result("C-FUEL-01", S.CClass.SAFETY, d.fuel_margin_min >= 0, d.fuel_margin_min),
                   _result("C-CREW-01", S.CClass.SAFETY, d.crew_margin_min >= 0, d.crew_margin_min),
                   _result("C-GATE-02", S.CClass.PHYSICAL, d.gate_overlap_min <= 0, -d.gate_overlap_min),
                   _result("C-POL-OT-01", S.CClass.POLICY, d.ground_overtime_min <= 15, 15 - d.ground_overtime_min)]
    safe = all(r.status == "PASS" for r in results if r.cls != S.CClass.POLICY)
    policy_ok = all(r.status == "PASS" for r in results if r.cls == S.CClass.POLICY)
    return S.KernelReport(option_id=option.option_id, results=results, safe=safe, policy_ok=policy_ok)


def _land(world, slot, gate, oid, origin, rnd=0) -> S.Option:
    hold = world.safety_view["slots"][slot]
    action = "HOLD_AND_REGATE" if hold and gate != "G1" else "HOLD" if hold else "REGATE" if gate != "G1" else "LAND_AS_PLANNED"
    return S.Option(action_type=action, hold_minutes=hold, slot_id=slot, gate_id=gate,
                    option_id=oid, round=rnd, proposed_by=origin)


def seed_options(world) -> list[S.Option]:
    options, index = [], 0
    for slot in world.menus["slots"][:8]:
        for gate in world.menus["gates"]:
            options.append(_land(world, slot, gate, f"G-{index:03d}", "generator"))
            index += 1
    for alt in world.menus["alternates"]:
        options.append(S.Option(action_type="DIVERT", hold_minutes=0, divert_to=alt, option_id=f"G-{index:03d}",
                                round=0, proposed_by="generator"))
        index += 1
    return options


def do_nothing(world) -> S.Option:
    slot = world.menus["slots"][0] if world.menus["slots"] else "S-0"
    return _land(world, slot, "G1", "DN-1", "do_nothing")


def fallback_plan(world) -> S.Option:
    view = world.safety_view
    slot = world.menus["slots"][0] if world.menus["slots"] else "S-0"
    if view["extra_fuel_min"] < view["slots"][slot]:
        return S.Option(action_type="DIVERT", hold_minutes=0, divert_to="ALT1", option_id="FB-1",
                        round=0, proposed_by="fallback")
    for gate in world.menus["gates"]:
        candidate = _land(world, slot, gate, "FB-1", "fallback")
        if derive(candidate, world).gate_overlap_min <= 0:
            return candidate
    return _land(world, slot, REMOTE, "FB-1", "fallback")


class Ledger:
    def __init__(self, key: str, seed: int, cfg_hash: str):
        self.prev = hashlib.sha256(f"{key}{seed}{cfg_hash}".encode()).hexdigest()
        self.entries: list[dict] = []

    def append(self, type: str, actor: str, payload, constraint_ids=()) -> dict:
        entry = {"seq": len(self.entries), "ts": "STUB", "type": type, "actor": actor, "payload": payload,
                 "constraint_ids": list(constraint_ids), "prev_hash": self.prev}
        entry["hash"] = hashlib.sha256((canonical(entry) + self.prev).encode()).hexdigest()
        self.prev = entry["hash"]
        self.entries.append(entry)
        return entry

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def canonical(entry: dict) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class VerifyReport:
    ok: bool; first_broken_seq: int | None; message: str


def verify(path) -> VerifyReport:
    previous = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            stored = entry.pop("hash")
            if (previous is not None and entry["prev_hash"] != previous) or \
                    hashlib.sha256((canonical(entry) + entry["prev_hash"]).encode()).hexdigest() != stored:
                return VerifyReport(False, entry["seq"], f"CHAIN BROKEN at seq {entry['seq']}")
            previous = stored
    return VerifyReport(True, None, "CHAIN OK (stub verify: chain only, no kernel or decision replay)")


class RuleAgent:
    def __init__(self, role: str):
        self.role = role

    def assess(self, option, world) -> S.Assessment:
        d = derive(option, world)
        fn = {"pilot": cost_pilot, "ground": cost_ground, "atc": cost_atc}[self.role]
        return S.Assessment(agent=self.role, option_id=option.option_id, feasible_for_me=True,
                            cost_usd=fn(option, d, world), reasons=[S.Reason.ACCEPTABLE])

    def read_ops(self, notes: str) -> S.OpsFacts:
        return S.OpsFacts()

    def propose(self, view: dict) -> list[S.OptionDraft]:
        return []

    def vote(self, view: dict) -> list[S.Vote]:
        votes = []
        for oid, cost in view["my_costs"].items():
            accept = cost <= view["walkaway"]
            votes.append(S.Vote(agent=self.role, round=view["round"], option_id=oid,
                                vote="ACCEPT" if accept else "REJECT",
                                reason=S.Reason.ACCEPTABLE if accept else S.Reason.COST_ABOVE_WALKAWAY))
        return votes


class RandomAgent(RuleAgent):
    def __init__(self, role: str, seed: int):
        super().__init__(role)
        self.rng = np.random.default_rng(seed)

    def propose(self, view: dict) -> list[S.OptionDraft]:
        menus = view["menus"]
        slots = [f"S-{i}" for i in range(30)]
        return [S.OptionDraft(action_type="HOLD", hold_minutes=int(self.rng.integers(0, 60)),
                              slot_id=str(self.rng.choice(slots)), gate_id=str(self.rng.choice(menus["gates"])))
                for _ in range(2)]


class AdversarialAgent(RandomAgent):
    def propose(self, view: dict) -> list[S.OptionDraft]:
        return [S.OptionDraft(action_type="HOLD", hold_minutes=119, slot_id="S-999", gate_id="Z99"),
                S.OptionDraft(action_type="DIVERT", hold_minutes=0, divert_to="XXX")]


def attacks(world) -> list[dict]:
    base = do_nothing(world)
    return [{"name": "hold longer than fuel", "category": "Fuel", "expected": "C-CONSIST-01",
             "payload": base.model_copy(update={"hold_minutes": 119})}]


def negotiate(world, agents, cfg, ledger) -> S.Decision:
    """Section 5.6, cut down. The real one is C's cdm/coordinator.py."""
    ledger.append("SCENARIO", "coordinator", world.public)
    fb = fallback_plan(world)
    ledger.append("BASELINE", "coordinator", {
        "do_nothing": check(do_nothing(world), world).model_dump(mode="json"),
        "fallback": check(fb, world).model_dump(mode="json")})
    walkaway = {a.role: a.assess(fb, world).cost_usd for a in agents}
    pool = seed_options(world)
    views = {"menus": world.menus, "round": 1}
    for agent in agents:
        for i, draft in enumerate(agent.propose(views)):
            pool.append(S.Option(**draft.model_dump(), option_id=f"{agent.role[0].upper()}-1-{i:02d}",
                                 round=1, proposed_by=agent.role))
    reports = {o.option_id: check(o, world) for o in pool}
    ledger.append("KERNEL", "kernel", {k: v.model_dump(mode="json") for k, v in reports.items()})
    feasible = [o for o in pool if reports[o.option_id].safe and reports[o.option_id].policy_ok]
    if not feasible:
        relaxable = [o for o in pool if reports[o.option_id].safe]
        if relaxable:
            esc = S.Escalation(kind="POLICY_RELAXATION", relaxed_constraint_id="C-POL-OT-01",
                               enabled_option_id=relaxable[0].option_id,
                               required_human_decision=f"STUB: approve overtime above the cap for {relaxable[0].option_id}.")
        else:
            esc = S.Escalation(kind="DECLINE", blocking_constraints=["C-FUEL-01"],
                               required_human_decision="STUB: no safe plan. PIC and controller decision required.")
        ledger.append("ESCALATION", "coordinator", esc.model_dump())
        return S.Decision(outcome="ESCALATED", option=None, kernel=None, rounds_used=1, escalation=esc)
    costs = {o.option_id: {a.role: a.assess(o, world).cost_usd for a in agents} for o in feasible}
    shortlist = sorted(feasible, key=lambda o: sum(costs[o.option_id].values()))[:6]
    votes: list[S.Vote] = []
    for agent in agents:
        votes += agent.vote({"round": 1, "walkaway": walkaway[agent.role],
                             "my_costs": {o.option_id: costs[o.option_id][agent.role] for o in shortlist}})
    ledger.append("VOTES", "coordinator", [v.model_dump(mode="json") for v in votes])
    unanimous = [o for o in shortlist
                 if all(v.vote == "ACCEPT" for v in votes if v.option_id == o.option_id)]
    best, outcome = (unanimous[0], "CONSENSUS") if unanimous else (fb, "FALLBACK")
    report = check(best, world)
    if not report.safe:
        esc = S.Escalation(kind="DECLINE", required_human_decision="STUB: fallback unsafe. Human decision required.",
                           blocking_constraints=[r.id for r in report.results if r.status == "FAIL"])
        ledger.append("ESCALATION", "coordinator", esc.model_dump())
        return S.Decision(outcome="ESCALATED", option=None, kernel=None, rounds_used=2, escalation=esc, votes=votes)
    per_agent = {a.role: a.assess(best, world).cost_usd for a in agents}
    ledger.append("DECISION", "coordinator", {"outcome": outcome, "option": best.model_dump(mode="json"),
                                              "costs_by_agent": per_agent, "walkaway": walkaway})
    return S.Decision(outcome=outcome, option=best, kernel=report, rounds_used=2 if outcome == "CONSENSUS" else 3,
                      votes=votes, costs_by_agent=per_agent)


_EXPORTS = {
    "cdm.world": dict(World=World, build_world=build_world, derive=derive, IS_STUB=True),
    "cdm.costs": dict(cost_pilot=cost_pilot, cost_ground=cost_ground, cost_atc=cost_atc),
    "cdm.kernel": dict(check=check),
    "cdm.options": dict(seed_options=seed_options, do_nothing=do_nothing),
    "cdm.fallback": dict(fallback_plan=fallback_plan),
    "cdm.ledger": dict(Ledger=Ledger),
    "cdm.verify": dict(verify=verify, VerifyReport=VerifyReport),
    "cdm.coordinator": dict(negotiate=negotiate),
    "cdm.agents": dict(__path__=[]),
    "cdm.agents.rule": dict(RuleAgent=RuleAgent),
    "cdm.agents.random_agent": dict(RandomAgent=RandomAgent),
    "cdm.agents.adversarial": dict(AdversarialAgent=AdversarialAgent, attacks=attacks),
}


def install() -> None:
    """All or nothing: a real kernel judging a stub world would mean nothing."""
    sys.modules["cdm.schemas"] = S
    for name, attrs in _EXPORTS.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module
