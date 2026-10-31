"""E6 sensitivity: scale each assumed parameter by +/-30% and see what the conclusions do.

Paths point into config/*.yaml. A path that is not in the config is reported as SKIPPED,
so a renamed parameter cannot look like a stable one.
"""
from __future__ import annotations

import argparse

import pandas as pd

from eval import integration
from eval.results import write_results
from eval.run_batch import load_sample, notes_for, run_batch

PARAMS = {
    "extra_fuel_min": ("synthetic", "extra_fuel_min"),
    "slot_occupancy": ("synthetic", "slot_occupancy"),
    "gate_occupancy": ("synthetic", "gate_occupancy"),
    "connecting_pax": ("synthetic", "connecting_pax_fraction"),
    "aar_weather": ("synthetic", "aar_weather"),
    "cost_per_min": ("costs", "cost_per_min"),
    "missed_pax_cost": ("costs", "cost_per_missed_pax"),
    "overtime_rate": ("costs", "overtime_rate"),
}
FACTORS = (0.7, 1.3)


def plan_signature(batch: pd.DataFrame) -> pd.Series:
    """What the duty manager would see: the outcome and the plan's action and hold."""
    columns = ["flight_key", "seed", "outcome", "neg_action", "neg_hold_min"]
    frame = batch.reindex(columns=columns).astype(str)
    return frame.drop(columns=["flight_key", "seed"]).agg("|".join, axis=1).set_axis(
        pd.MultiIndex.from_frame(frame[["flight_key", "seed"]]))


def beats_fcfs_rate(batch: pd.DataFrame) -> float:
    """Only where do-nothing is safe: beating an illegal plan on cost would say nothing."""
    both = batch[batch["dn_safe"] == True].dropna(subset=["neg_total_cost", "dn_total_cost"])  # noqa: E712
    return float((both["neg_total_cost"] <= both["dn_total_cost"]).mean()) if len(both) else float("nan")


def run_sensitivity(rows: pd.DataFrame, cfg, seeds=(0,), params: dict = PARAMS) -> pd.DataFrame:
    base = run_batch(rows, "rule", cfg, seeds)
    base_plans = plan_signature(base)
    records = [{"parameter": "BASELINE", "factor": 1.0, "status": "OK", "scenarios": len(base),
                "decision_stability": 1.0, "unsafe_accepted": int(base["unsafe_accepted"].sum()),
                "neg_beats_fcfs_rate": beats_fcfs_rate(base), "mean_total_cost": base["neg_total_cost"].mean(),
                "label": "modelled"}]
    for name, (section, path) in params.items():
        for factor in FACTORS:
            try:
                scaled = cfg.scaled(section, path, factor)
            except KeyError:
                records.append({"parameter": name, "factor": factor, "status": "SKIPPED_NOT_IN_CONFIG",
                                "label": "modelled"})
                continue
            batch = run_batch(rows, "rule", scaled, seeds)
            plans = plan_signature(batch)
            records.append({
                "parameter": name, "factor": factor, "status": "OK", "scenarios": len(batch),
                "decision_stability": float((plans.reindex(base_plans.index) == base_plans).mean()),
                "unsafe_accepted": int(batch["unsafe_accepted"].sum()),
                "neg_beats_fcfs_rate": beats_fcfs_rate(batch),
                "mean_total_cost": batch["neg_total_cost"].mean(), "label": "modelled",
            })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--allow-stubs", action="store_true")
    args = parser.parse_args()

    stubbed = integration.use_pipeline(args.allow_stubs)
    table = run_sensitivity(load_sample(args.n), integration.load_cfg(), range(args.seeds))
    write_results(table, "e6_sensitivity", notes_for(stubbed))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
