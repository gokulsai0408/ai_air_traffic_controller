import json

import pandas as pd
import pytest

from cdm import baseline_single_prompt as bsp
from cdm.solver import central_solve, costs_of
from eval import ablation, integration, redteam, report, sensitivity
from eval.results import STUB_NOTE, read_notes, read_results, write_results
from eval.run_batch import gate_delay_min, run_batch
from eval.sample import sample_rows, stratum_mask, top_hubs


@pytest.fixture(scope="module")
def cfg():
    return integration.load_cfg()


@pytest.fixture(scope="module")
def batch(rows, cfg):
    return run_batch(rows, "rule", cfg, range(3))


@pytest.fixture(scope="module")
def world(rows, cfg):
    from cdm.world import build_world
    return build_world(rows.iloc[0], 0, cfg)


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


def test_central_never_returns_unsafe_and_is_minimal(rows, cfg):
    from cdm.kernel import check
    from cdm.options import seed_options
    from cdm.world import build_world
    for _, row in rows.iterrows():
        world = build_world(row, 1, cfg)
        best, best_costs = central_solve(world)
        safe = [o for o in seed_options(world) if check(o, world).safe and check(o, world).policy_ok]
        if best is None:
            assert not safe
            continue
        assert check(best, world).safe and check(best, world).policy_ok
        assert sum(best_costs.values()) == pytest.approx(min(sum(costs_of(o, world).values()) for o in safe))


def test_central_bound_holds_in_batch(batch):
    assert (batch["error"] == "").all()
    compared = batch["central_bound_ok"].dropna()
    assert len(compared) and compared.astype(bool).all()


def test_no_unsafe_decision_any_agent_kind(rows, cfg, batch):
    assert not batch["unsafe_accepted"].astype(bool).any()
    assert batch["outcome"].nunique() > 1
    for kind in ("random", "adversarial"):
        attacked = run_batch(rows.head(5), kind, cfg, range(2))
        assert (attacked["error"] == "").all()
        assert not attacked["unsafe_accepted"].astype(bool).any()


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
    cfg2 = integration.Cfg(synthetic={"extra_fuel_min": [4, 30]})
    table = sensitivity.run_sensitivity(rows.head(4), cfg2, params={
        "extra_fuel_min": ("synthetic", "extra_fuel_min"), "gone": ("synthetic", "not_a_param")})
    assert set(table.loc[table["parameter"] == "gone", "status"]) == {"SKIPPED_NOT_IN_CONFIG"}
    assert table.loc[table["parameter"] == "extra_fuel_min", "unsafe_accepted"].sum() == 0
    assert (table["parameter"] == "BASELINE").sum() == 1


def test_all_escalated_batch_does_not_crash_sensitivity():
    empty = pd.DataFrame({"neg_total_cost": [float("nan")], "dn_total_cost": [1.0], "dn_safe": [True]})
    assert pd.isna(sensitivity.beats_fcfs_rate(empty))


def test_report_builds_and_marks_stub_results(tmp_path, batch):
    write_results(batch, "batch_rule", (STUB_NOTE,), tmp_path)
    skipped = report.build_report(tmp_path)
    assert "batch_llm" in skipped and "e6_sensitivity" in skipped
    for name in ("e3_negotiation", "e4_baselines"):
        assert STUB_NOTE in read_notes(name, tmp_path)
    assert (tmp_path / "baseline_comparison.png").exists()
    assert read_results("e4_baselines", tmp_path)["label_cost"].str.contains("ASSUMED").all()
