"""The one place Part D assumes anything the frozen contracts leave open.

Section 6.3 does not pin down constructors, the config object or the attack generators.
When A, B and C land, any mismatch should need a change here and nowhere else:

  * cfg: load_cfg() below (attributes max_rounds, hash, synthetic, costs, constraints)
  * agents: RuleAgent(role), RandomAgent(role, seed), AdversarialAgent(role, seed)
  * ledger: Ledger(flight_key, seed, cfg_hash)
  * attacks: cdm.agents.adversarial.attacks(world) -> list of dicts (see eval/redteam.py)
  * LLM: cdm.agents.llm.complete(system, user, json_schema) -> raw text
  * ledger entries: as C's coordinator writes them, plus LLM_CALL for each LLM call (not landed yet)
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("pilot", "ground", "atc")
PIPELINE_MODULES = (
    "cdm.schemas", "cdm.world", "cdm.costs", "cdm.kernel", "cdm.options", "cdm.fallback",
    "cdm.ledger", "cdm.verify", "cdm.coordinator", "cdm.agents.rule",
    "cdm.agents.random_agent", "cdm.agents.adversarial",
)


def missing_modules() -> list[str]:
    """Checks the files, because installed stubs would satisfy an import check."""
    def landed(name: str) -> bool:
        base = ROOT.joinpath(*name.split("."))
        return base.with_suffix(".py").exists() or (base / "__init__.py").exists()
    return [name for name in PIPELINE_MODULES if not landed(name)]


def real_pipeline_ready() -> bool:
    return not missing_modules() and not is_stubbed()


def is_stubbed() -> bool:
    return getattr(sys.modules.get("cdm.world"), "IS_STUB", False)


def use_pipeline(allow_stubs: bool) -> bool:
    """Make the pipeline importable. Returns True if it is the stub, never silently."""
    if is_stubbed():
        return True
    missing = missing_modules()
    if not missing:
        return False
    if not allow_stubs:
        raise RuntimeError(f"Pipeline modules not landed yet: {', '.join(missing)}")
    from eval.stubs.contract import install
    install()
    return True


@dataclass
class Cfg:
    max_rounds: int = 3
    synthetic: dict = field(default_factory=dict)
    costs: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)

    @property
    def hash(self) -> str:
        body = json.dumps([self.max_rounds, self.synthetic, self.costs, self.constraints],
                          sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def scaled(self, section: str, path: str, factor: float) -> "Cfg":
        """A copy with every number under section.path scaled. KeyError if the path is absent."""
        clone = copy.deepcopy(self)
        holder: Any = getattr(clone, section)
        parts = path.split(".")
        for part in parts[:-1]:
            holder = holder[part]
        holder[parts[-1]] = _scale(holder[parts[-1]], factor)
        return clone


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


def load_cfg(config_dir: Path = ROOT / "config") -> Cfg:
    def read(name: str) -> dict:
        path = config_dir / f"{name}.yaml"
        return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}
    return Cfg(max_rounds=int(read("protocol").get("max_rounds", 3)), synthetic=read("synthetic"),
               costs=read("costs"), constraints=read("constraints"))


def flight_key(row: pd.Series) -> str:
    if "flight_key" in row:
        return str(row["flight_key"])
    return (f"{row['AIRLINE']}{row['FLIGHT_NUMBER']}-{row['TAIL_NUMBER']}-"
            f"{int(row['MONTH']):02d}{int(row['DAY']):02d}-{row['SCHEDULED_DEPARTURE']}")


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


def make_ledger(key: str, seed: int, cfg: Cfg):
    from cdm.ledger import Ledger
    return Ledger(key, seed, cfg.hash)


def attacks_for(world) -> list[dict]:
    from cdm.agents.adversarial import attacks
    return attacks(world)


def single_prompt_client():
    return importlib.import_module("cdm.agents.llm").complete
