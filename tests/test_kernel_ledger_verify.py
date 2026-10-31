from __future__ import annotations

import json

import pytest

from kernel_support import land, make_world, minutes, reference_derive

from cdm import kernel, verify as verify_module
from cdm.fallback import fallback_plan
from cdm.ledger import Ledger, canonical_json, entry_hash, first_broken_seq, genesis_hash, read_entries
from cdm.options import do_nothing
from cdm.schemas import Assessment, Decision, Escalation, OpsFacts, Reason, Vote
from cdm.verify import verify

ROLES = ("pilot", "ground", "atc")


@pytest.fixture(autouse=True)
def fake_derive(monkeypatch):
    monkeypatch.setattr(kernel, "_derive", reference_derive)


def loader(world=None):
    return lambda flight_key, seed, config_hash: world or make_world()


def assess(option_ids, costs=None, round_=1, infeasible_for=None):
    costs = costs or {}
    return {"round": round_, "assessments": {
        role: {oid: Assessment(agent=role, option_id=oid, feasible_for_me=role != infeasible_for,
                               cost_usd=costs.get(oid, 100.0), reasons=[]) for oid in option_ids}
        for role in ROLES}}


def votes_for(option_id, round_=1, reject_by=None):
    return {"round": round_, "votes": [
        Vote(agent=role, round=round_, option_id=option_id, vote="REJECT" if role == reject_by else "ACCEPT",
             reason=Reason.ACCEPTABLE) for role in ROLES]}


def start_log(world, max_rounds=2, walkaway=500.0):
    """Opens a log with the entries C's coordinator writes before round 1."""
    ledger = Ledger("2015-06-15-DL-1", 7, "cfg", clock=lambda: "2026-10-31T00:00:00Z")
    dn, fb = do_nothing(world), fallback_plan(world)
    ledger.append("SCENARIO", "coordinator", {"flight_key": "2015-06-15-DL-1", "max_rounds": max_rounds})
    ledger.append("BASELINE", "coordinator", {"do_nothing": {"option": dn, "kernel": kernel.check(dn, world)},
                                              "fallback": {"option": fb, "kernel": kernel.check(fb, world)}})
    ledger.append("WALKAWAY", "coordinator", {r: walkaway for r in ROLES})
    return ledger, fb


def kernel_entry(ledger, world, checked, round_=1, **extra):
    ledger.append("KERNEL", "kernel", {"round": round_, "reports": [kernel.check(o, world) for o in checked], **extra})


def run_log(path, world=None, outcome="CONSENSUS", votes=None, chosen="a", costs=None, walkaway=500.0, max_rounds=2,
            infeasible_for=None):
    world = world or make_world()
    options = [land("S-02", option_id="a"), land("S-04", hold=4, option_id="b")]
    ledger, fb = start_log(world, max_rounds, walkaway)
    kernel_entry(ledger, world, options, options=options)
    ledger.append("ASSESSMENTS", "coordinator", assess(["a", "b"], costs, infeasible_for=infeasible_for))
    if votes is not False:
        ledger.append("VOTES", "coordinator", votes if votes is not None else votes_for("a"))
    chosen_option = fb if outcome == "FALLBACK" else next(o for o in options if o.option_id == chosen)
    ledger.append("DECISION", "coordinator", Decision(
        outcome=outcome, option=chosen_option, kernel=kernel.check(chosen_option, world), rounds_used=1))
    return ledger.save(path)


# --- ledger
def test_chain_links_and_genesis():
    ledger = Ledger("k", 1, "c")
    ledger.append("A", "x", {"n": 1})
    assert ledger.entries[0]["prev_hash"] == genesis_hash("k", 1, "c")
    assert ledger.entries[1]["prev_hash"] == ledger.entries[0]["hash"]
    assert first_broken_seq(ledger.entries) is None


def test_append_accepts_models_and_datetimes_and_returns_entry():
    entry = Ledger("k", 1, "c").append("A", "x", {"v": Vote(agent="atc", round=1, option_id="o", vote="ACCEPT",
                                                            reason=Reason.ACCEPTABLE), "t": minutes(0)}, ["C-RWY-01"])
    assert entry["payload"]["v"]["reason"] == "ACCEPTABLE" and entry["constraint_ids"] == ["C-RWY-01"]
    assert isinstance(entry["payload"]["t"], str)


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1}) == '{"a":2,"b":1}'


def test_save_load_round_trip_keeps_chain_and_extends(tmp_path):
    ledger = Ledger("k", 1, "c")
    ledger.append("A", "x", {"f": 0.1 + 0.2, "s": "Zürich"})
    path = ledger.save(tmp_path / "sub" / "run.jsonl")
    reloaded = Ledger.load(path)
    assert reloaded.entries == ledger.entries
    reloaded.append("HUMAN", "duty_manager", {"action": "APPROVED"})
    assert first_broken_seq(read_entries(reloaded.save(path))) is None


def test_same_inputs_same_hashes():
    make = lambda: Ledger("k", 1, "c", clock=lambda: "T").append("A", "x", {"n": 1})["hash"]
    assert make() == make()


# --- verify: happy paths
def test_verify_consensus_reproduced(tmp_path):
    report = verify(run_log(tmp_path / "r.jsonl"), loader())
    assert report.complete, report.problems
    assert report.summary() == "CHAIN OK · KERNEL REPLAY 4/4 MATCH · DECISION REPRODUCED"
    assert report.message == report.summary()


def test_verify_does_not_import_any_llm_module(tmp_path, monkeypatch):
    import sys
    is_llm = lambda name: name.startswith("cdm.agents.llm")   # cdm.agents.llm and cdm.agents.llm_client
    # Other test modules import the LLM agents during collection; hide them so this checks verify alone.
    for name in [m for m in sys.modules if is_llm(m)]:
        monkeypatch.delitem(sys.modules, name)
    report = verify(run_log(tmp_path / "r.jsonl"), loader())
    assert report.complete and not any(is_llm(m) for m in sys.modules)


def test_verify_compromise_reproduced(tmp_path):
    path = run_log(tmp_path / "r.jsonl", outcome="COMPROMISE", votes=False, max_rounds=1, costs={"a": 50.0, "b": 90.0})
    report = verify(path, loader())
    assert report.complete, report.problems


def test_verify_fallback_when_nothing_rational(tmp_path):
    path = run_log(tmp_path / "r.jsonl", outcome="FALLBACK", votes=False, max_rounds=1, walkaway=1.0)
    report = verify(path, loader())
    assert report.complete, report.problems


def test_verify_follows_max_rounds_from_the_log(tmp_path):
    late = run_log(tmp_path / "r.jsonl", outcome="COMPROMISE", votes=False, max_rounds=3)
    assert verify(late, loader()).decision_status == "MISMATCH"   # round 1 of 3 is not the final round


def test_verified_objection_allows_fallback(tmp_path):
    path = run_log(tmp_path / "r.jsonl", outcome="FALLBACK", votes=False, max_rounds=1, infeasible_for="pilot")
    assert verify(path, loader()).complete


def test_apply_facts_in_the_log_is_replayed(tmp_path):
    world = make_world()
    facts = OpsFacts(taxiway_closures=["TWY A"])
    tightened = kernel.apply_facts(world, facts)
    option = land("S-02", "B2", option_id="a")
    ledger, _ = start_log(tightened, max_rounds=2)
    kernel_entry(ledger, tightened, [option], options=[option], facts=facts)
    ledger.append("ASSESSMENTS", "coordinator", assess(["a"]))
    ledger.append("VOTES", "coordinator", votes_for("a"))
    ledger.append("DECISION", "coordinator", Decision(outcome="CONSENSUS", option=option,
                                                      kernel=kernel.check(option, tightened), rounds_used=1))
    report = verify(ledger.save(tmp_path / "r.jsonl"), loader(world))
    assert report.complete, report.problems


# --- verify: tamper detection
@pytest.mark.parametrize("target", ["payload", "constraint_ids", "ts"])
def test_tampering_any_field_reports_that_entry(tmp_path, target):
    path = run_log(tmp_path / "r.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[3])
    entry[target] = {"forged": True} if target == "payload" else ["X"] if target == "constraint_ids" else "1999"
    lines[3] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = verify(path, loader())
    assert not report.chain_ok and report.first_broken_seq == 3 and not report.ok
    assert "CHAIN BROKEN AT ENTRY 3" in report.summary()


def test_one_byte_edit_is_detected(tmp_path):
    path = run_log(tmp_path / "r.jsonl")
    data = path.read_bytes()
    index = data.index(b'"outcome":"CONSENSUS"') + len(b'"outcome":"')
    path.write_bytes(data[:index] + b"X" + data[index + 1:])
    report = verify(path, loader())
    assert not report.ok and report.first_broken_seq is not None


def test_deleting_a_middle_entry_is_detected(tmp_path):
    path = run_log(tmp_path / "r.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[3]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert verify(path, loader()).first_broken_seq == 3


def test_forger_who_recomputes_the_entry_hash_is_caught_at_the_next_link():
    ledger = Ledger("k", 1, "c")
    ledger.append("A", "x", {"n": 1})
    ledger.append("B", "x", {"n": 2})
    forged = ledger.entries[1]
    forged["payload"] = {"n": 999}
    forged["hash"] = entry_hash(forged)
    assert first_broken_seq(ledger.entries) == 2


def test_genesis_swap_is_detected(tmp_path):
    path = run_log(tmp_path / "r.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    entry["payload"]["seed"] = 8
    lines[0] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert verify(path, loader()).first_broken_seq == 0


def test_corrupt_json_line_is_a_failure_not_a_crash(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    assert not verify(path, loader()).ok


def test_empty_file_is_a_failure(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text("", encoding="utf-8")
    assert not verify(path, loader()).ok


def test_truncating_the_decision_is_not_reported_as_verified(tmp_path):
    path = run_log(tmp_path / "r.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    report = verify(path, loader())
    assert not report.complete and report.decision_status == "SKIPPED"


# --- verify: replay catches a correctly chained but wrong log
def test_kernel_replay_catches_a_different_world(tmp_path):
    path = run_log(tmp_path / "r.jsonl")
    report = verify(path, loader(make_world(extra_fuel_min=1.0)))
    assert report.kernel_matched < report.kernel_checked and not report.ok


def test_replay_catches_forged_report_in_a_valid_chain(tmp_path):
    world = make_world(extra_fuel_min=1.0)
    option = land("S-20", hold=20, option_id="a")
    forged = kernel.check(option, world).model_copy(update={"safe": True, "policy_ok": True})
    ledger, _ = start_log(world)
    ledger.append("KERNEL", "kernel", {"round": 1, "reports": [forged], "options": [option]})
    ledger.append("ASSESSMENTS", "coordinator", assess(["a"]))
    ledger.append("VOTES", "coordinator", votes_for("a"))
    ledger.append("DECISION", "coordinator", Decision(outcome="CONSENSUS", option=option, kernel=forged, rounds_used=1))
    report = verify(ledger.save(tmp_path / "r.jsonl"), loader(world))
    assert report.chain_ok and report.decision_status == "MISMATCH" and not report.ok
    assert any("C-FUEL-01" in p for p in report.problems)


def test_report_without_its_option_is_not_counted_as_verified(tmp_path):
    world = make_world()
    option = land("S-04", "B2", 4, option_id="P-1-01")
    ledger, fb = start_log(world)
    kernel_entry(ledger, world, [option])                 # no "options", as C logs today
    ledger.append("ASSESSMENTS", "coordinator", assess([fb.option_id]))
    ledger.append("VOTES", "coordinator", votes_for(fb.option_id))
    ledger.append("DECISION", "coordinator", Decision(outcome="CONSENSUS", option=fb, kernel=kernel.check(fb, world),
                                                      rounds_used=1))
    report = verify(ledger.save(tmp_path / "r.jsonl"), loader(world))
    assert report.ok and not report.complete and report.kernel_unreplayable == 1
    assert "1 NOT REPLAYABLE" in report.summary()


def test_consensus_without_unanimity_is_a_mismatch(tmp_path):
    path = run_log(tmp_path / "r.jsonl", votes=votes_for("a", reject_by="pilot"))
    report = verify(path, loader())
    assert report.chain_ok and report.decision_status == "MISMATCH"


def test_consensus_on_a_costlier_unanimous_option_is_a_mismatch(tmp_path):
    votes = votes_for("a")
    votes["votes"] += votes_for("b")["votes"]
    path = run_log(tmp_path / "r.jsonl", votes=votes, chosen="b", costs={"a": 10.0, "b": 90.0})
    assert verify(path, loader()).decision_status == "MISMATCH"


def test_consensus_in_the_binding_round_is_a_mismatch(tmp_path):
    path = run_log(tmp_path / "r.jsonl", max_rounds=1)
    assert verify(path, loader()).decision_status == "MISMATCH"


def test_equal_cost_tie_is_not_a_mismatch(tmp_path):
    world = make_world()
    votes = votes_for("a")
    votes["votes"] += votes_for("b")["votes"]
    # Compare margins so we pick whichever option the documented tie-break prefers.
    margin = lambda o: kernel.min_safety_margin(kernel.check(o, world))
    best = max([land("S-02", option_id="a"), land("S-04", hold=4, option_id="b")], key=margin).option_id
    assert verify(run_log(tmp_path / "r.jsonl", votes=votes, chosen=best), loader()).complete


def test_compromise_on_costlier_option_is_a_mismatch(tmp_path):
    path = run_log(tmp_path / "r.jsonl", outcome="COMPROMISE", votes=False, max_rounds=1, chosen="b",
                   costs={"a": 50.0, "b": 90.0})
    assert verify(path, loader()).decision_status == "MISMATCH"


def test_fallback_despite_available_compromise_is_a_mismatch(tmp_path):
    path = run_log(tmp_path / "r.jsonl", outcome="FALLBACK", votes=False, max_rounds=1)
    assert verify(path, loader()).decision_status == "MISMATCH"


def test_earlier_unanimous_round_is_a_mismatch(tmp_path):
    world = make_world()
    option = land(option_id="a")
    ledger, _ = start_log(world, max_rounds=3)
    kernel_entry(ledger, world, [option], options=[option])
    ledger.append("VOTES", "coordinator", votes_for("a", 1))
    ledger.append("ASSESSMENTS", "coordinator", assess(["a"], round_=2))
    ledger.append("VOTES", "coordinator", votes_for("a", 2))
    ledger.append("DECISION", "coordinator", Decision(outcome="CONSENSUS", option=option,
                                                      kernel=kernel.check(option, world), rounds_used=2))
    assert verify(ledger.save(tmp_path / "r.jsonl"), loader(world)).decision_status == "MISMATCH"


def test_exceeding_the_round_cap_is_a_mismatch(tmp_path):
    world = make_world()
    option = land(option_id="a")
    ledger, _ = start_log(world, max_rounds=2)
    kernel_entry(ledger, world, [option], options=[option])
    ledger.append("DECISION", "coordinator", Decision(outcome="COMPROMISE", option=option,
                                                      kernel=kernel.check(option, world), rounds_used=3))
    assert verify(ledger.save(tmp_path / "r.jsonl"), loader(world)).decision_status == "MISMATCH"


# --- verify: escalation
def escalation_log(path, world, escalation, option_list=None, attach=None):
    options = option_list or [land(option_id="a")]
    ledger, _ = start_log(world)
    kernel_entry(ledger, world, options, options=options)
    ledger.append("DECISION", "coordinator", Decision(
        outcome="ESCALATED", option=attach, kernel=kernel.check(attach, world) if attach else None,
        rounds_used=1, escalation=escalation))
    return ledger.save(path)


def no_slots_world(**overrides):
    """Baselines are unsafe too, so escalating is legitimate whatever the listed options do."""
    return make_world(gate_busy={g: [(minutes(-60), minutes(200))] for g in ("B1", "B2", "B3", "B4")}, **overrides)


def test_policy_escalation_reproduced(tmp_path):
    world = make_world(shift_end=minutes(20))
    option = land(option_id="a")
    escalation = Escalation(kind="POLICY_RELAXATION", required_human_decision="Approve overtime",
                            relaxed_constraint_id="C-POL-OT-01", enabled_option_id="a")
    report = verify(escalation_log(tmp_path / "r.jsonl", world, escalation, [option], attach=option), loader(world))
    assert report.complete, report.problems


def test_offering_to_relax_a_safety_constraint_is_a_mismatch(tmp_path):
    world = make_world(extra_fuel_min=1.0, alternate_reserve_min=0.0)
    option = land("S-20", hold=20, option_id="a")
    escalation = Escalation(kind="POLICY_RELAXATION", required_human_decision="Relax fuel",
                            relaxed_constraint_id="C-FUEL-01", enabled_option_id="a")
    path = escalation_log(tmp_path / "r.jsonl", world, escalation, [option])
    assert verify(path, loader(world)).decision_status == "MISMATCH"


def test_escalating_with_feasible_options_is_a_mismatch(tmp_path):
    world = make_world()
    escalation = Escalation(kind="DECLINE", required_human_decision="PIC decision")
    assert verify(escalation_log(tmp_path / "r.jsonl", world, escalation), loader(world)).decision_status == "MISMATCH"


def test_offering_a_larger_relaxation_than_needed_is_a_mismatch(tmp_path):
    world = make_world(shift_end=minutes(20))
    options = [land("S-02", option_id="near"), land("S-20", "B2", 20, option_id="far")]
    escalation = Escalation(kind="POLICY_RELAXATION", required_human_decision="Approve overtime",
                            relaxed_constraint_id="C-POL-OT-01", enabled_option_id="far")
    path = escalation_log(tmp_path / "r.jsonl", world, escalation, options)
    assert verify(path, loader(world)).decision_status == "MISMATCH"


def test_decline_reproduced_when_nothing_is_relaxable(tmp_path):
    world = make_world(extra_fuel_min=1.0, alternate_reserve_min=0.0, fdp_end=minutes(0))
    option = land("S-20", hold=20, option_id="a")
    escalation = Escalation(kind="DECLINE", required_human_decision="PIC decision", blocking_constraints=["C-FUEL-01"])
    report = verify(escalation_log(tmp_path / "r.jsonl", world, escalation, [option]), loader(world))
    assert report.complete, report.problems


def test_decline_when_relaxation_exists_is_a_mismatch(tmp_path):
    world = make_world(shift_end=minutes(20))
    escalation = Escalation(kind="DECLINE", required_human_decision="PIC decision")
    assert verify(escalation_log(tmp_path / "r.jsonl", world, escalation), loader(world)).decision_status == "MISMATCH"


# --- verify: skipping is visible
def test_missing_world_builder_is_skipped_not_passed(tmp_path):
    def unavailable(*args):
        raise ImportError("cdm.world not ready")
    report = verify(run_log(tmp_path / "r.jsonl"), unavailable)
    assert report.ok and not report.complete
    assert "KERNEL REPLAY SKIPPED" in report.summary()


# --- CLI
def test_cli_exit_codes(tmp_path, monkeypatch, capsys):
    path = run_log(tmp_path / "r.jsonl")
    monkeypatch.setattr(verify_module, "verify", lambda p: verify(p, loader()))
    assert verify_module.main([str(path)]) == 0
    assert "DECISION REPRODUCED" in capsys.readouterr().out

    data = path.read_bytes()
    path.write_bytes(data.replace(b"CONSENSUS", b"COMPROMISE", 1))
    assert verify_module.main([str(path)]) == 1
    assert "CHAIN BROKEN" in capsys.readouterr().out
