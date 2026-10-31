"""E5 cascade. The validation is measured on real legs; the before/after is modelled."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cdm import cascade
from eval.results import RESULTS_DIR, read_results, write_results


def run_validation(out_dir: Path | str = RESULTS_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pairs, stats = cascade.chain_legs(cascade.load_legs())
    validation = cascade.validate_model(pairs)
    best = validation[validation["predictor"].str.startswith("model_fitted")]
    turn = int(best.loc[best["target"] == "next_departure_delay", "fitted_min_turn"].iloc[0])
    notes = (cascade.LIMITATION,
             "min_turn is fitted on months 1-6 and scored on months 7-12. Label: measured.")
    write_results(validation, "e5_validation", notes, out_dir)
    write_results(pd.DataFrame([{**stats, "label": "measured"}]), "e5_continuity", (cascade.LIMITATION,), out_dir)
    held_out = pairs[pairs["month"] > 6]  # the chart quotes held-out scores, so it plots held-out legs
    points = held_out.sample(n=min(5000, len(held_out)), random_state=2015).assign(
        predicted_departure_delay=lambda p: cascade.predict(p, turn))
    write_results(points[["arrival_delay", "sched_gap_min", "predicted_departure_delay", "next_departure_delay"]],
                  "e5_points", notes, out_dir)
    return validation, pairs, stats


def counterfactual(batch: pd.DataFrame, pairs: pd.DataFrame, min_turn: int) -> pd.DataFrame:
    """Next-leg delay per scenario: recorded, modelled from the recorded arrival, modelled after our plan.

    Compare modelled with modelled. Comparing "after" with "recorded" would mix in the model's own error.
    """
    keys = ["tail", "month", "day", "flight"]
    joined = batch.merge(pairs.drop_duplicates(keys), on=keys, how="inner")
    joined = joined.dropna(subset=["neg_gate_delay_min"])
    slack = joined["sched_gap_min"] - min_turn
    return pd.DataFrame({
        "flight_key": joined["flight_key"], "seed": joined["seed"], "outcome": joined["outcome"],
        "next_delay_recorded": joined["next_departure_delay"],
        "next_delay_modelled_before": (joined["arrival_delay"] - slack).clip(lower=0),
        "next_delay_modelled_after": (joined["neg_gate_delay_min"] - slack).clip(lower=0),
        "label_recorded": "measured", "label_before": "modelled", "label_after": "modelled",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default=None, help="results name of a run_batch CSV for the before/after")
    args = parser.parse_args()

    validation, pairs, stats = run_validation()
    print(stats)
    print(validation.round(2).to_string(index=False))
    if args.batch:
        fitted = validation.loc[validation["target"] == "next_departure_delay", "fitted_min_turn"].iloc[0]
        table = counterfactual(read_results(args.batch), pairs, int(fitted))
        write_results(table, "e5_counterfactual", (cascade.LIMITATION,
                      "The after figure is modelled, never measured."))
        print(table[["next_delay_recorded", "next_delay_modelled_before", "next_delay_modelled_after"]].mean().round(1))


if __name__ == "__main__":
    main()
