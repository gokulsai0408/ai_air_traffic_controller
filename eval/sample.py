"""Fixed-seed scenario sample: weather and late-aircraft strata across the top hubs."""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

TOP10 = ("ORD", "ATL", "DFW", "DEN", "LAX", "SFO", "IAH", "LAS", "PHX", "MCO")


def top_hubs(df: pd.DataFrame, n: int = 10) -> tuple[str, ...]:
    return tuple(df["DESTINATION_AIRPORT"].value_counts().head(n).index)


def stratum_mask(df: pd.DataFrame, stratum: str) -> pd.Series:
    """The strata are disjoint: a leg with weather delay > 15 always counts as weather."""
    if stratum == "weather":
        return df["WEATHER_DELAY"] > 15
    if stratum == "late_aircraft":
        return (df["LATE_AIRCRAFT_DELAY"] > 20) & ~(df["WEATHER_DELAY"] > 15)
    raise ValueError(f"unknown stratum {stratum!r}")


def sample_rows(df: pd.DataFrame, n: int, seed: int,
                strata: Sequence[str] = ("weather", "late_aircraft"),
                airports: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Sample n rows split evenly by stratum, and evenly by hub within each stratum."""
    hubs = list(airports) if airports is not None else list(top_hubs(df))
    pool = df[df["DESTINATION_AIRPORT"].isin(hubs)]
    picked = []
    for s_index, stratum in enumerate(strata):
        quota = n // len(strata) + (s_index < n % len(strata))
        candidates = pool[stratum_mask(pool, stratum)]
        for h_index, hub in enumerate(hubs):
            take = quota // len(hubs) + (h_index < quota % len(hubs))
            at_hub = candidates[candidates["DESTINATION_AIRPORT"] == hub]
            chosen = at_hub.sample(n=min(take, len(at_hub)), random_state=seed + s_index * 100 + h_index)
            picked.append(chosen.assign(stratum=stratum))
    return pd.concat(picked).reset_index(drop=True)
