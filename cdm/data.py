"""Load the DOT flights file, airports and configs.

Quirks handled (plan section 2): HHMM times read as strings, 2400 = midnight, overnight arrivals.
The file only holds delayed legs, so a next leg that recovered is missing.
Times are local. "Arrival before departure means next day" is the plan's rule, but a short westbound
flight across time zones can trip it (dep 0500, arr 0450), so SCHEDULED_TIME settles those cases.
"""
from __future__ import annotations
import hashlib
import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
TIME_COLS = ("SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL", "DEPARTURE_TIME", "ARRIVAL_TIME")
NUM_COLS = ("ARRIVAL_DELAY", "TAXI_IN", "DISTANCE", "WEATHER_DELAY", "LATE_AIRCRAFT_DELAY", "AIR_SYSTEM_DELAY")


def parse_hhmm(value) -> int | None:
    """'0510' -> 310, 659 -> 419, '2400' -> 1440, blank -> None. Minutes after midnight."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text:
        return None
    text = text.zfill(4)
    hours, minutes = int(text[:-2]), int(text[-2:])
    if hours > 24 or minutes > 59 or (hours == 24 and minutes != 0):
        raise ValueError(f"Bad HHMM time: {value!r}")
    return hours * 60 + minutes


def load_flights(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={c: str for c in TIME_COLS}, low_memory=False)
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("ORIGIN_AIRPORT", "DESTINATION_AIRPORT"):
        df[col] = df[col].astype(str)

    year = df["YEAR"] if "YEAR" in df else 2015
    date = pd.to_datetime(pd.DataFrame({"year": year, "month": df["MONTH"], "day": df["DAY"]}))
    dep_min = df["SCHEDULED_DEPARTURE"].map(parse_hhmm).astype(float)
    arr_min = df["SCHEDULED_ARRIVAL"].map(parse_hhmm).astype(float)
    df["SCHED_DEP_DT"] = date + pd.to_timedelta(dep_min, unit="min")
    overnight = arr_min < dep_min
    if "SCHEDULED_TIME" in df:
        block = pd.to_numeric(df["SCHEDULED_TIME"], errors="coerce")
        overnight &= ~((arr_min - dep_min - block).abs() < (arr_min + 1440 - dep_min - block).abs())
    df["SCHED_ARR_DT"] = date + pd.to_timedelta(arr_min + overnight.astype(float) * 1440, unit="min")

    df["FLIGHT_KEY"] = (
        date.dt.strftime("%Y%m%d") + "-" + df["AIRLINE"].astype(str) + df["FLIGHT_NUMBER"].astype(str)
        + "-" + df["ORIGIN_AIRPORT"] + "-" + df["DESTINATION_AIRPORT"] + "-" + df["SCHEDULED_DEPARTURE"].str.zfill(4)
    )
    return _add_next_leg(df)


def _add_next_leg(df: pd.DataFrame) -> pd.DataFrame:
    """NEXT_SCHED_DEP_DT: same tail, same day, departing from this leg's destination, after it arrives."""
    ordered = df.sort_values(["TAIL_NUMBER", "SCHED_DEP_DT"])
    next_row = ordered[["TAIL_NUMBER", "MONTH", "DAY", "ORIGIN_AIRPORT", "SCHED_DEP_DT"]].shift(-1)
    connects = (
        (next_row["TAIL_NUMBER"] == ordered["TAIL_NUMBER"])
        & (next_row["MONTH"] == ordered["MONTH"])
        & (next_row["DAY"] == ordered["DAY"])
        & (next_row["ORIGIN_AIRPORT"] == ordered["DESTINATION_AIRPORT"])
        & (next_row["SCHED_DEP_DT"] >= ordered["SCHED_ARR_DT"])
    )
    df["NEXT_SCHED_DEP_DT"] = next_row["SCHED_DEP_DT"].where(connects).reindex(df.index)
    return df


def load_airports(path) -> pd.DataFrame:
    """Airports with coordinates only (ECP, PBG and UST have none and are dropped)."""
    return pd.read_csv(path).dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)


@dataclass(frozen=True, eq=False)
class Config:
    constraints: dict
    costs: dict
    synthetic: dict
    airports: pd.DataFrame
    hash: str

    def cost(self, key: str) -> float:
        return self.costs[key]["value"]

    def with_synthetic(self, **overrides) -> "Config":
        """Copy with changed generator values; used for sensitivity runs (E6)."""
        synthetic = {**self.synthetic, **overrides}
        return replace(self, synthetic=synthetic, hash=_config_hash(self.constraints, self.costs, synthetic))

    def with_costs(self, **values) -> "Config":
        """Copy with changed cost values; used for sensitivity runs (E6)."""
        costs = {**self.costs, **{k: {**self.costs[k], "value": v} for k, v in values.items()}}
        return replace(self, costs=costs, hash=_config_hash(self.constraints, costs, self.synthetic))


def _config_hash(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_config(config_dir=ROOT / "config", airports_path=ROOT / "data" / "airports.csv") -> Config:
    config_dir = Path(config_dir)
    constraints, costs, synthetic = (
        yaml.safe_load((config_dir / f"{name}.yaml").read_text(encoding="utf-8"))
        for name in ("constraints", "costs", "synthetic")
    )
    if Path(airports_path).exists():
        airports = load_airports(airports_path)
    else:
        warnings.warn(f"{airports_path} not found: worlds will have no alternates")
        airports = pd.DataFrame(columns=["IATA_CODE", "LATITUDE", "LONGITUDE"])
    return Config(constraints, costs, synthetic, airports, _config_hash(constraints, costs, synthetic))
