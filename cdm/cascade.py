"""Cascade on real aircraft rotations (BUILD_PLAN 5.11). Needs only flights.csv.

The file holds only delayed legs, so a next leg that recovered is missing and
absorption is invisible. Every result written from here states that bias.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LIMITATION = (
    "File holds only delayed legs: next legs that recovered are missing, "
    "so results are biased toward delay that persisted."
)
COLUMNS = [
    "MONTH", "DAY", "AIRLINE", "FLIGHT_NUMBER", "TAIL_NUMBER", "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL", "SCHEDULED_TIME",
    "ARRIVAL_DELAY", "DEPARTURE_DELAY", "LATE_AIRCRAFT_DELAY",
]
SORT_KEYS = ["TAIL_NUMBER", "MONTH", "DAY", "_dep_min"]


def load_legs(path: Path | str = DATA_DIR / "flights.csv") -> pd.DataFrame:
    """Read the columns the cascade uses, keeping HHMM times as strings."""
    dtype = {"SCHEDULED_DEPARTURE": str, "SCHEDULED_ARRIVAL": str}
    return pd.read_csv(path, usecols=COLUMNS, dtype=dtype)


def hhmm_to_min(values: pd.Series) -> pd.Series:
    """HHMM to minutes after midnight. 2400 stays 1440, so it sorts last in its day."""
    text = values.astype(str).str.split(".").str[0].str.zfill(4)
    return text.str[:2].astype(int) * 60 + text.str[2:4].astype(int)


def _arrival_local_min(dep_min: pd.Series, arr_min: pd.Series, block_min: pd.Series) -> pd.Series:
    """Arrival in destination-local minutes counted from the departure day's midnight.

    The two clocks can be in different time zones, so "arrival < departure" is not a
    reliable overnight test. The offset is recovered from the scheduled block time.
    """
    tz_offset = ((arr_min - dep_min - block_min + 720) % 1440) - 720
    return dep_min + block_min + tz_offset


def propagate(arr_delay_min: float, slack_min: float) -> float:
    """next_dep_delay = max(0, arrival delay at gate - slack)."""
    return max(0.0, float(arr_delay_min) - float(slack_min))


def next_leg(df: pd.DataFrame, row: pd.Series) -> Optional[pd.Series]:
    """The same tail's next same-day leg, only if it departs from this leg's destination."""
    same_day = df[
        (df["TAIL_NUMBER"] == row["TAIL_NUMBER"])
        & (df["MONTH"] == row["MONTH"])
        & (df["DAY"] == row["DAY"])
    ]
    dep = hhmm_to_min(same_day["SCHEDULED_DEPARTURE"])
    this_dep = hhmm_to_min(pd.Series([row["SCHEDULED_DEPARTURE"]])).iloc[0]
    later = same_day[dep > this_dep]
    if later.empty:
        return None
    candidate = later.loc[hhmm_to_min(later["SCHEDULED_DEPARTURE"]).idxmin()]
    return candidate if candidate["ORIGIN_AIRPORT"] == row["DESTINATION_AIRPORT"] else None


def chain_legs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Pair each leg with its next same-day leg. Returns chained pairs and continuity stats.

    `sched_gap_min` is next scheduled departure minus this scheduled arrival. Both clocks
    are local to the connecting airport, so the gap needs no time-zone correction.
    """
    legs = df.assign(_dep_min=hhmm_to_min(df["SCHEDULED_DEPARTURE"]))
    legs = legs.sort_values(SORT_KEYS, kind="mergesort").reset_index(drop=True)
    arr_min = hhmm_to_min(legs["SCHEDULED_ARRIVAL"])
    legs["_arr_local"] = _arrival_local_min(legs["_dep_min"], arr_min, legs["SCHEDULED_TIME"])

    keys = ["TAIL_NUMBER", "MONTH", "DAY"]
    nxt = legs.groupby(keys, sort=False).shift(-1)
    consecutive = nxt["_dep_min"].notna()
    connects = consecutive & (nxt["ORIGIN_AIRPORT"] == legs["DESTINATION_AIRPORT"])
    gap = nxt["_dep_min"] - legs["_arr_local"]
    usable = connects & (gap >= 0)

    pairs = pd.DataFrame({
        "tail": legs["TAIL_NUMBER"], "month": legs["MONTH"], "day": legs["DAY"],
        "flight": legs["FLIGHT_NUMBER"], "airport": legs["DESTINATION_AIRPORT"],
        "arrival_delay": legs["ARRIVAL_DELAY"],
        "sched_gap_min": gap,
        "next_flight": nxt["FLIGHT_NUMBER"],
        "next_late_aircraft_delay": nxt["LATE_AIRCRAFT_DELAY"],
        "next_departure_delay": nxt["DEPARTURE_DELAY"],
    })[usable].reset_index(drop=True)
    stats = {
        "consecutive_same_day_legs": int(consecutive.sum()),
        "connecting_legs": int(connects.sum()),
        "continuity_rate": float(connects.sum() / consecutive.sum()) if consecutive.any() else 0.0,
        "dropped_negative_gap": int((connects & (gap < 0)).sum()),
        "chained_pairs": int(usable.sum()),
    }
    return pairs, stats


def predict(pairs: pd.DataFrame, min_turn: float) -> pd.Series:
    """Vectorised propagate(): slack = gap - min_turn. Slack may go negative on a tight turn."""
    return (pairs["arrival_delay"] - (pairs["sched_gap_min"] - min_turn)).clip(lower=0.0)


def fit_min_turn(pairs: pd.DataFrame, target: str, grid: range = range(0, 121)) -> int:
    """The constant min_turn with the lowest MAE on `pairs`, by grid search."""
    actual = pairs[target].to_numpy()
    errors = {m: float(np.abs(predict(pairs, m).to_numpy() - actual).mean()) for m in grid}
    return min(errors, key=errors.get)


def score(predicted: pd.Series, actual: pd.Series) -> dict[str, float]:
    """MAE, Pearson correlation and median error of a prediction."""
    error = (predicted - actual).abs()
    corr = float(np.corrcoef(predicted, actual)[0, 1]) if predicted.std() > 0 else float("nan")
    return {"n": int(len(actual)), "mae_min": float(error.mean()),
            "median_ae_min": float(error.median()), "pearson_r": corr}


def validate_model(pairs: pd.DataFrame, fit_months: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
                   assumed_min_turn: int = 35) -> pd.DataFrame:
    """Measured model validation, tidy with one row per target and predictor.

    min_turn is fitted on `fit_months` and scored on the other months. Each target also
    gets two simple comparators, so the model's error can be read against something:
    the plan's assumed min_turn, and "the next leg inherits the whole arrival delay".
    """
    fit = pairs[pairs["month"].isin(fit_months)]
    test = pairs[~pairs["month"].isin(fit_months)]
    rows = []
    for target in ("next_departure_delay", "next_late_aircraft_delay"):
        fitted = fit_min_turn(fit, target)
        predictors = {
            f"model_fitted_min_turn_{fitted}": predict(test, fitted),
            f"model_assumed_min_turn_{assumed_min_turn}": predict(test, assumed_min_turn),
            "naive_inherit_full_delay": test["arrival_delay"].clip(lower=0.0),
        }
        for name, predicted in predictors.items():
            rows.append({"target": target, "predictor": name, "fitted_min_turn": fitted,
                         **score(predicted, test[target]), "label": "measured"})
    return pd.DataFrame(rows)
