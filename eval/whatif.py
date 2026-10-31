"""What-if parameters for the dashboard sliders.

A slider changes a SYNTHETIC generator range (config/synthetic.yaml) or a cost constant
(config/costs.yaml). The world is rebuilt with A's cfg.with_synthetic(...) / cfg.with_costs(...)
and the negotiation reruns. The overrides are written into the ledger as a WHAT_IF_CONFIG entry,
so the log says what was changed and verify can rebuild the same world.

Everything a slider touches is SYNTHETIC from then on, including cost constants that were
sourced (REAL) in the repo config: a what-if value is not an observation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from eval import integration

WHAT_IF_ENTRY = "WHAT_IF_CONFIG"


@dataclass(frozen=True)
class Slider:
    key: str            # synthetic.yaml key, dotted for nested ("transfer_min.far"), or costs.yaml key
    label: str
    lo: float
    hi: float
    step: float
    affects: tuple[str, ...] = ()   # world fields that change, for the SYNTHETIC badge


SYNTHETIC_SLIDERS = (
    Slider("shift_end_after_eta_min", "Ground shift ends after ETA (min)", 0, 300, 5, ("ground.shift_end_min",)),
    Slider("original_gate_conflict_prob", "Chance the original gate is taken", 0.0, 1.0, 0.05,
           ("ground.gates", "ground.original_gate")),
    Slider("concourses", "Concourses the gates spread over", 1, 5, 1, ("ground.gates", "pilot.connections")),
    Slider("transfer_min.adjacent", "Transfer time, adjacent concourse (min)", 0, 40, 1, ("pilot.transfer_min",)),
    Slider("transfer_min.far", "Transfer time, two concourses apart (min)", 0, 40, 1, ("pilot.transfer_min",)),
    Slider("transfer_min.remote_bus", "Transfer time from REMOTE-1 by bus (min)", 0, 40, 1, ("pilot.transfer_min",)),
    Slider("connecting_pax_fraction", "Share of passengers connecting", 0.0, 1.0, 0.05,
           ("pilot.connecting_pax", "pilot.connections")),
    Slider("extra_fuel_min", "Holding fuel above reserve (min)", 0, 60, 1, ("pilot.extra_fuel_min",)),
    Slider("slot_occupancy", "Share of runway slots already taken", 0.0, 0.95, 0.05, ("atc.slots",)),
)
COST_SLIDERS = (
    Slider("cost_per_delay_min", "Delay cost (USD/min)", 0, 200, 1),
    Slider("cost_per_missed_pax", "Missed connection (USD/pax)", 0, 1000, 10),
    Slider("overtime_rate_per_min", "Ground overtime (USD/min)", 0, 100, 1),
    Slider("gate_change_fixed", "Gate change (USD)", 0, 2000, 50),
    Slider("remote_stand_cost", "Remote stand (USD)", 0, 5000, 100),
    Slider("gate_size_mismatch_per_class", "Oversized gate (USD/class)", 0, 1500, 25),
    Slider("tight_fit_per_min", "Tight gate fit (USD/min short)", 0, 100, 1),
    Slider("turn_penalty_per_min", "Short turnaround (USD/min)", 0, 200, 5),
    Slider("holding_workload_per_min", "ATC holding workload (USD/min)", 0, 100, 1),
    Slider("runway_minute_value", "Runway minute (USD)", 0, 200, 5),
)
PRESETS = {
    "Tight ramp shift [10, 60] (shows escalation)": {"synthetic": {"shift_end_after_eta_min": [10, 60]}, "costs": {}},
}


def _get(tree: dict, dotted: str) -> Any:
    node: Any = tree
    for part in dotted.split("."):
        node = node[part]
    return node


def available(sliders: tuple[Slider, ...], cfg: integration.RunCfg, section: str) -> list[Slider]:
    """Only sliders whose key is in today's config, so a renamed parameter disappears instead of failing."""
    out = []
    for slider in sliders:
        try:
            value = _get(cfg.synthetic, slider.key) if section == "synthetic" else cfg.costs[slider.key]["value"]
        except (KeyError, TypeError):
            continue
        if isinstance(value, (int, float, list)):
            out.append(slider)
    return out


def default_value(cfg: integration.RunCfg, section: str, key: str) -> Any:
    return copy.deepcopy(_get(cfg.synthetic, key) if section == "synthetic" else cfg.costs[key]["value"])


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return a == b


def changed(cfg: integration.RunCfg, synthetic: dict[str, Any], costs: dict[str, float]) -> dict:
    """Keep only values that differ from the repo config, so an untouched slider leaves the hash unchanged."""
    def plain(value: Any) -> Any:
        return list(value) if isinstance(value, tuple) else value
    return {
        "synthetic": {k: plain(v) for k, v in synthetic.items() if not _same(v, default_value(cfg, "synthetic", k))},
        "costs": {k: float(v) for k, v in costs.items() if not _same(v, default_value(cfg, "costs", k))},
    }


def _keep_type(template: Any, value: Any) -> Any:
    """A slider hands back floats; integer ranges (gate counts, concourses) stay integers for the generator."""
    if isinstance(template, list):
        values = value if isinstance(value, (list, tuple)) else [value] * len(template)   # a point = [v, v]
        return [_keep_type(t, v) for t, v in zip(template, values)]
    if isinstance(template, int) and not isinstance(template, bool):
        return int(round(float(value)))
    return value


def apply(cfg: integration.RunCfg, overrides: Optional[dict]) -> integration.RunCfg:
    """RunCfg with the overrides applied through A's with_synthetic / with_costs."""
    if not overrides:
        return cfg
    synthetic: dict[str, Any] = {}
    for dotted, value in (overrides.get("synthetic") or {}).items():
        top, *rest = dotted.split(".")
        if rest:
            holder = synthetic.setdefault(top, copy.deepcopy(cfg.synthetic[top]))
            for part in rest[:-1]:
                holder = holder[part]
            holder[rest[-1]] = _keep_type(holder[rest[-1]], value)
        else:
            synthetic[top] = _keep_type(cfg.synthetic[top], value)
    out = cfg.with_synthetic(**synthetic) if synthetic else cfg
    costs = overrides.get("costs") or {}
    return out.with_costs(**costs) if costs else out


def is_empty(overrides: Optional[dict]) -> bool:
    return not overrides or not (overrides.get("synthetic") or overrides.get("costs"))


def ledger_payload(overrides: dict) -> dict:
    return {"synthetic": overrides.get("synthetic") or {}, "costs": overrides.get("costs") or {}, "tag": "SYNTHETIC",
            "note": "What-if values set on the dashboard. Not observations."}


def overrides_in(entries: list[dict]) -> dict:
    entry = next((e for e in entries if e["type"] == WHAT_IF_ENTRY), None)
    return {"synthetic": entry["payload"].get("synthetic", {}), "costs": entry["payload"].get("costs", {})} if entry \
        else {"synthetic": {}, "costs": {}}


def affected_fields(overrides: Optional[dict]) -> set[str]:
    keys = set((overrides or {}).get("synthetic") or {})
    return {field for s in SYNTHETIC_SLIDERS if s.key in keys for field in s.affects}


def world_loader_for(path: Path | str):
    """verify() world loader that rebuilds with the what-if config the ledger recorded."""
    from cdm.ledger import read_entries
    from cdm.world import build_world

    overrides = overrides_in(read_entries(path))

    def load(flight_key: str, seed: int, config_hash: str):
        cfg = apply(integration.load_cfg(), overrides)
        flights = integration.load_flights()
        match = flights[flights["flight_key"] == flight_key]
        if match.empty:
            raise LookupError(f"flight {flight_key} is not in data/flights.csv")
        return build_world(match.iloc[0], seed, cfg.config)
    return load
