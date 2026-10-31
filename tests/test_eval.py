import json

import pandas as pd
import pytest

from cdm import baseline_single_prompt as bsp
from cdm.solver import central_solve, costs_of
from eval import ablation, integration, redteam, report, sensitivity, whatif
from eval.results import read_notes, read_results, write_results
from eval.run_batch import gate_delay_min, run_batch, summarise_ledger
from eval.sample import sample_rows, stratum_mask, top_hubs


@pytest.fixture(scope="module")
def batch(rows, cfg):
    return run_batch(rows, "rule", cfg, range(3))


@pytest.fixture(scope="module")
def world(rows, cfg):
    from cdm.world import build_world
    return build_world(rows.iloc[0], 0, cfg.config)


def test_sample_is_balanced_reproducible_and_disjoint(flights):
    one = sample_rows(flights, 100, seed=7)
    assert len(one) == 100
    assert one["stratum"].value_counts().to_dict() == {"weather": 50, "late_aircraft": 50}
    assert set(one["DESTINATION_AIRPORT"]) == set(top_hubs(flights))
    assert one.groupby("DESTINATION_AIRPORT").size().tolist() == [10] * 10
    pd.testing.assert_frame_equal(one, sample_rows(flights, 100, seed=7))
    assert not sample_rows(flights, 100, seed=8).equals(one)
    assert not (stratum_mask(flights, "weather") & stratum_mask(flights, "late_aircraft")).any()


def test_results_notes_round_trip(tmp_path):
    write_results(pd.DataFrame({"a": [1]}), "x", ("first", "second"), tmp_path)
    assert read_notes("x", tmp_path) == ["first", "second"]
    assert read_results("x", tmp_path)["a"].tolist() == [1]


@pytest.mark.parametrize("on_block,sched,recorded,expected", [
    ("10:30", "1000", 30, 30),
    ("00:30", "2330", 60, 60),      # delay crosses midnight
    ("10:30", "1000", 1470, 1470),  # delay over a day, disambiguated by the recorded value
    (None, "1000", 30, None),       # diverted, never reaches the gate
])
def test_gate_delay(on_block, sched, recorded, expected):
    result = gate_delay_min(on_block, pd.Series({"SCHEDULED_ARRIVAL": sched, "ARRIVAL_DELAY": recorded}))
    assert pd.isna(result) if expected is None else result == expected


# ---- round limit ----------------------------------------------------------------------------------
def test_protocol_config_sets_two_rounds(cfg):
    assert integration.load_max_rounds() == 2 == cfg.max_rounds


def test_max_rounds_reaches_negotiate_and_the_ledger(tmp_path, rows, cfg):
    """A's Config has no max_rounds, so the value must travel through RunCfg.negotiation() to the SCENARIO entry."""
    (tmp_path / "protocol.yaml").write_text("max_rounds: 1\n")
    one = integration.load_cfg(tmp_path / "protocol.yaml")
    assert one.max_rounds == 1 and one.hash == cfg.hash   # the round cap is protocol, not world config
    for run_cfg, cap in ((cfg, 2), (one, 1)):
        result = run_batch(rows.head(6), "rule", run_cfg)
        assert set(result["logged_max_rounds"]) == {cap}
        assert (result["rounds_used"] <= cap).all()
    table = report.summarise_negotiation(run_batch(rows.head(6), "rule", cfg))
    assert table.loc[0, "max_rounds"] == 2 and table.loc[0, "logged_max_rounds"] == "2"


def test_negotiation_cfg_is_readable_by_the_coordinator(world, cfg):
    assert cfg.negotiation(world) == {"max_rounds": 2, "hash": world.config_hash}


# ---- baselines ------------------------------------------------------------------------------------
def test_central_never_returns_unsafe_and_is_minimal(rows, cfg):
    from cdm.kernel import check
    from cdm.options import seed_options
    from cdm.world import build_world
    for _, row in rows.iterrows():
        world = build_world(row, 1, cfg.config)
        best, best_costs = central_solve(world)
        safe = [o for o in seed_options(world) if check(o, world).safe and check(o, world).policy_ok]
        if best is None:
            assert not safe
            continue
        assert check(best, world).safe and check(best, world).policy_ok
        assert sum(best_costs.values()) == pytest.approx(min(sum(costs_of(o, world).values()) for o in safe))


def test_central_bound_holds_on_the_negotiation_pool(batch):
    assert (batch["error"] == "").all(), batch.loc[batch["error"] != "", "error"].tolist()
    compared = batch["central_bound_ok"].dropna()
    assert len(compared) and compared.astype(bool).all()
    assert batch["costs_match_cost_functions"].dropna().astype(bool).all()


def test_no_unsafe_decision_any_agent_kind(rows, cfg, batch):
    assert not batch["unsafe_accepted"].astype(bool).any()
    assert batch["outcome"].nunique() > 1
    for kind in ("random", "adversarial"):
        attacked = run_batch(rows.head(5), kind, cfg, range(2))
        assert (attacked["error"] == "").all(), attacked["error"].tolist()
        assert not attacked["unsafe_accepted"].astype(bool).any()


def test_every_plan_has_total_and_per_agent_costs(batch):
    for prefix in ("neg", "fb", "dn"):
        legal = batch[batch[f"{prefix}_safe"].eq(True)]
        totals = legal[[f"{prefix}_cost_{r}" for r in integration.ROLES]].sum(axis=1)
        assert totals.round(6).tolist() == legal[f"{prefix}_total_cost"].round(6).tolist()


def test_e4_headline_numbers_are_labelled_and_consistent(batch):
    table = report.headline(batch).set_index("metric")
    assert set(table["label"]) <= {"measured", "modelled", "modelled (SYNTHETIC context)"}
    assert table.loc["do_nothing_unsafe_rate", "label"] == "modelled (SYNTHETIC context)"
    assert table.loc["unsafe_accepted", "value"] == 0
    assert table.loc["central_bound_violations", "value"] == 0
    for metric in ("neg_beats_fallback_rate", "conflict_rate", "escalation_no_compliant_option_rate",
                   "do_nothing_unsafe_rate", "decided_rate"):
        value = table.loc[metric, "value"]
        assert pd.isna(value) or 0 <= value <= 1
    assert table.loc["price_of_privacy_pct", "value"] >= -1e-9


def test_slide_text_never_claims_savings_it_does_not_have():
    def table(beats: float) -> pd.DataFrame:
        rows = {"neg_beats_fallback_rate": beats, "neg_equals_fallback_rate": 1 - beats,
                "mean_saving_vs_fallback_usd": 0.0, "saving_vs_fallback_pct": 0.0, "price_of_privacy_pct": 0.0,
                "conflict_rate": 0.0, "do_nothing_unsafe_rate": 0.74, "unsafe_accepted": 0,
                "escalation_no_compliant_option_rate": 1.0}
        return pd.DataFrame([{"metric": k, "value": v, "n": 50} for k, v in rows.items()])
    failures = pd.DataFrame([{"constraint_id": "C-GATE-02", "share_of_unsafe_do_nothing": 0.6}])
    flat = report.slide_text(table(0.0), failures, 2)
    assert "did not beat the fallback" in flat and "We do not claim savings" in flat
    assert "unsafe in 74%" in flat and "C-GATE-02" in flat and "SYNTHETIC" in flat
    better = report.slide_text(table(0.3), failures, 2)
    assert "did not beat" not in better and "30%" in better


def test_report_writes_e4_files(tmp_path, batch):
    write_results(batch, "batch_rule", (), tmp_path)
    skipped = report.build_report(tmp_path)
    assert "batch_llm" in skipped and "e6_sensitivity" in skipped
    for name in ("e3_negotiation", "e4_baselines", "e4_headline", "e4_per_scenario", "e4_do_nothing_failures"):
        assert (tmp_path / f"{name}.csv").exists()
    assert (tmp_path / "e4_slide.md").exists() and (tmp_path / "baseline_comparison.png").exists()
    assert read_results("e4_baselines", tmp_path)["label_cost"].eq("modelled").all()
    per = read_results("e4_per_scenario", tmp_path)
    assert {"dn_unsafe", "cs_cost_pilot", "fb_cost_ground", "neg_cost_atc"} <= set(per.columns)


# ---- single prompt, ablation, red team, sensitivity -------------------------------------------------
def test_single_prompt_retries_once_then_fails(world):
    calls = []

    def garbage(system, user, schema):
        calls.append(1)
        return "not json"

    with pytest.raises(bsp.SinglePromptFailure):
        bsp.single_prompt_plan(world, garbage)
    assert len(calls) == 2


def test_single_prompt_is_one_call_with_all_private_data_and_no_kernel_filter(world):
    from cdm.kernel import check
    seen = []

    def unsafe(system, user, schema):
        seen.append(user)
        return json.dumps({"action_type": "HOLD", "hold_minutes": 119, "slot_id": "S-999", "gate_id": "Z99"})

    option = bsp.single_prompt_plan(world, unsafe)
    assert len(seen) == 1
    for section in ("public", "pilot", "ground", "atc"):
        assert f'"{section}"' in seen[0]
    assert option.option_id == "SP-1"
    assert not check(option, world).safe  # returned as is, so the eval can count the violation


def test_ablation_summary(rows, cfg):
    batch = run_batch(rows.head(3), "adversarial", cfg)
    summary, breakdown = ablation.summarise(batch)
    assert summary.loc[0, "options_proposed"] > 0
    assert summary.loc[0, "unsafe_accepted"] == 0
    assert "C-REF-01" in set(breakdown["constraint_id"])


def test_redteam_reports_a_miss_as_a_miss(world):
    from cdm.options import do_nothing
    harmless = {"name": "not an attack", "category": "x", "expected": "C-REF-01", "payload": do_nothing(world)}
    assert redteam.attack_caught(harmless, world)[0] is False


def test_redteam_table_states_the_known_limitation(rows, cfg):
    table = redteam.run_redteam(rows.head(2), cfg)
    assert (table["result"] == "CAUGHT").sum() >= 7
    assert "KNOWN_LIMITATION" in set(table["result"])
    assert table["unsafe_accepted"].sum() == 0


def test_sensitivity_skips_absent_parameter_openly(rows, cfg):
    table = sensitivity.run_sensitivity(rows.head(4), cfg, params={
        "extra_fuel_min": ("synthetic", "extra_fuel_min"), "gone": ("synthetic", "not_a_param")})
    assert set(table.loc[table["parameter"] == "gone", "status"]) == {"SKIPPED_NOT_IN_CONFIG"}
    assert table.loc[table["parameter"] == "extra_fuel_min", "unsafe_accepted"].sum() == 0
    assert (table["parameter"] == "BASELINE").sum() == 1


def test_all_escalated_batch_does_not_crash_sensitivity():
    empty = pd.DataFrame({"neg_total_cost": [float("nan")], "dn_total_cost": [1.0], "dn_safe": [True]})
    assert pd.isna(sensitivity.beats_fcfs_rate(empty))


def test_ground_drafts_count_as_agent_options_and_seeds_do_not(tmp_path):
    def kernel_report(oid, safe):
        status = "PASS" if safe else "FAIL"
        return {"option_id": oid, "safe": safe, "results": [{"id": "C-REF-01", "status": status}]}
    entry = {"type": "KERNEL", "payload": {"round": 1, "reports": [
        kernel_report("G-017", True), kernel_report("G-1-01", False), kernel_report("P-1-01", False),
        kernel_report("A-2-01", True)]}}
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(entry) + "\n")
    summary = summarise_ledger(path)
    assert (summary["agent_options"], summary["agent_options_unsafe"]) == (3, 2)
    assert summary["kernel_entries_unparsed"] == 0


# ---- what-if (dashboard sliders) --------------------------------------------------------------------
def test_untouched_sliders_change_nothing(cfg):
    synthetic = {s.key: whatif.default_value(cfg, "synthetic", s.key)
                 for s in whatif.available(whatif.SYNTHETIC_SLIDERS, cfg, "synthetic")}
    costs = {s.key: whatif.default_value(cfg, "costs", s.key) for s in whatif.available(whatif.COST_SLIDERS, cfg, "costs")}
    overrides = whatif.changed(cfg, synthetic, costs)
    assert whatif.is_empty(overrides)
    assert whatif.apply(cfg, overrides).hash == cfg.hash


def test_whatif_rebuilds_world_with_overrides(cfg, rows):
    from cdm.world import build_world
    overrides = {"synthetic": {"shift_end_after_eta_min": [10, 60], "transfer_min.far": [20, 22]},
                 "costs": {"overtime_rate_per_min": 30.0}}
    tight = whatif.apply(cfg, overrides)
    assert tight.hash != cfg.hash and tight.max_rounds == cfg.max_rounds
    assert tight.synthetic["transfer_min"]["far"] == [20, 22]
    assert tight.synthetic["transfer_min"]["same"] == cfg.synthetic["transfer_min"]["same"]
    assert tight.costs["overtime_rate_per_min"]["value"] == 30.0
    world = build_world(rows.iloc[0], 0, tight.config)
    assert world.ground.shift_end_min - world.public.eta_min <= 60 + 1e-6
    assert world.cost("overtime_rate_per_min") == 30.0
    assert {"ground.shift_end_min", "pilot.transfer_min"} <= whatif.affected_fields(overrides)


def test_whatif_run_is_logged_and_verifies_with_its_own_config(rows, cfg, tmp_path):
    from cdm.ledger import read_entries
    from cdm.verify import verify
    from eval.run_batch import run_scenario
    overrides = {"synthetic": {"shift_end_after_eta_min": [10, 60]}, "costs": {}}
    record = run_scenario(rows.iloc[1], 0, "rule", whatif.apply(cfg, overrides), tmp_path, what_if=overrides)
    entries = read_entries(record["ledger"])
    assert whatif.overrides_in(entries)["synthetic"] == {"shift_end_after_eta_min": [10, 60]}
    assert next(e for e in entries if e["type"] == whatif.WHAT_IF_ENTRY)["payload"]["tag"] == "SYNTHETIC"
    assert verify(record["ledger"], whatif.world_loader_for(record["ledger"])).ok


def test_tight_shift_raises_escalations(rows, cfg):
    tight = whatif.apply(cfg, {"synthetic": {"shift_end_after_eta_min": [10, 60]}})
    base = run_batch(rows, "rule", cfg, range(2))
    squeezed = run_batch(rows, "rule", tight, range(2))
    assert (squeezed["outcome"] == "ESCALATED").mean() > (base["outcome"] == "ESCALATED").mean()
    assert not squeezed["unsafe_accepted"].astype(bool).any()
