"""The one place Part D assumes anything the frozen contracts leave open.

A, B and C have landed, so everything here runs the real pipeline:

  * cfg: RunCfg = A's cdm.data.Config (worlds, costs, constraints) + max_rounds from config/protocol.yaml.
    A's Config has no max_rounds field, so negotiate() gets `run_cfg.negotiation(world)`, a dict with
    "max_rounds" and "hash" that C's coordinator reads.
  * flights: cdm.data.load_flights (adds FLIGHT_KEY, SCHED_ARR_DT, NEXT_SCHED_DEP_DT that build_world needs)
  * agents: RuleAgent(role), LLMAgent(role), RandomAgent(role, seed), AdversarialAgent(role, seed)
  * ledger: Ledger(flight_key, seed, cfg.hash)
  * attacks: cdm.attacks.kernel_attacks(world) (B) + cdm.agents.adversarial.attacks(world) (C)
  * LLM: cdm.agents.llm.complete(system, user, json_schema) -> raw text
"""
from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "config" / "protocol.yaml"
ROLES = ("pilot", "ground", "atc")
DEFAULT_MAX_ROUNDS = 2


@dataclass(frozen=True, eq=False)
class RunCfg:
    """A's Config plus the protocol settings it doesn't carry."""
    config: Any                      # cdm.data.Config
    max_rounds: int = DEFAULT_MAX_ROUNDS

    @property
    def hash(self) -> str:
        return self.config.hash

    @property
    def synthetic(self) -> dict:
        return self.config.synthetic

    @property
    def costs(self) -> dict:
        return self.config.costs

    def negotiation(self, world) -> dict:
        """What negotiate() reads. The hash is the world's, so the ledger names the config the world came from."""
        return {"max_rounds": self.max_rounds, "hash": world.config_hash}

    def with_synthetic(self, **overrides) -> "RunCfg":
        return dataclasses.replace(self, config=self.config.with_synthetic(**overrides))

    def with_costs(self, **values) -> "RunCfg":
        return dataclasses.replace(self, config=self.config.with_costs(**values))

    def with_rounds(self, max_rounds: int) -> "RunCfg":
        return dataclasses.replace(self, max_rounds=int(max_rounds))

    def scaled(self, section: str, key: str, factor: float) -> "RunCfg":
        """A copy with one synthetic range or cost value scaled. KeyError if the key is absent."""
        if section == "synthetic":
            return self.with_synthetic(**{key: _scale(self.synthetic[key], factor)})
        if section == "costs":
            return self.with_costs(**{key: _scale(self.costs[key]["value"], factor)})
        raise KeyError(section)


def _scale(value: Any, factor: float) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value * factor
    if isinstance(value, list):
        return [_scale(v, factor) for v in value]
    if isinstance(value, dict):
        return {k: _scale(v, factor) for k, v in value.items()}
    return value


def load_max_rounds(protocol_path: Path | str = PROTOCOL) -> int:
    path = Path(protocol_path)
    data = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}
    return int(data.get("max_rounds", DEFAULT_MAX_ROUNDS))


def load_cfg(protocol_path: Path | str = PROTOCOL) -> RunCfg:
    from cdm.data import load_config
    return RunCfg(config=load_config(), max_rounds=load_max_rounds(protocol_path))


@lru_cache(maxsize=1)
def _flights() -> pd.DataFrame:
    from cdm.data import load_flights
    return load_flights(ROOT / "data" / "flights.csv")


def load_flights() -> pd.DataFrame:
    """A's loader, cached per process. Shared, so treat it as read-only (filter or .copy() before changing)."""
    return _flights()


def flight_key(row: pd.Series) -> str:
    return str(row["flight_key"])


def make_agents(kind: str, seed: int) -> list:
    if kind == "rule":
        from cdm.agents.rule import RuleAgent
        return [RuleAgent(role) for role in ROLES]
    if kind == "llm":
        from cdm.agents.llm import LLMAgent
        return [LLMAgent(role) for role in ROLES]
    if kind == "random":
        from cdm.agents.random_agent import RandomAgent
        return [RandomAgent(role, seed + i) for i, role in enumerate(ROLES)]
    if kind == "adversarial":
        from cdm.agents.adversarial import AdversarialAgent
        return [AdversarialAgent(role, seed + i) for i, role in enumerate(ROLES)]
    raise ValueError(f"unknown agent kind {kind!r}")


def make_ledger(key: str, seed: int, cfg: RunCfg):
    from cdm.ledger import Ledger
    return Ledger(key, seed, cfg.hash)


def attacks_for(world) -> list[dict]:
    from cdm.agents.adversarial import attacks
    from cdm.attacks import kernel_attacks
    kernel_level = [{"name": a.name, "category": a.category, "expected": a.expected, "payload": a.payload}
                    for a in kernel_attacks(world)]
    return kernel_level + attacks(world)


def single_prompt_client():
    return importlib.import_module("cdm.agents.llm").complete
