"""Pluggable agent kinds (BUILD_PLAN 5.5). Pick one with `make_agents(kind)`."""
from __future__ import annotations

from typing import Literal, Optional

from cdm.agents.base import ROLES, Agent, CostFn, DeriveFn
from cdm.agents.rule import RuleAgent

AgentKind = Literal["rule", "llm", "random", "adversarial"]


def make_agents(kind: AgentKind = "rule", seed: int = 0, derive_fn: Optional[DeriveFn] = None,
                cost_fns: Optional[dict[str, CostFn]] = None) -> list[Agent]:
    """One agent per role (pilot, ground, atc), each its own object with its own state."""
    cost_fns = cost_fns or {}
    kw = lambda r: {"derive_fn": derive_fn, "cost_fn": cost_fns.get(r)}
    if kind == "rule":
        return [RuleAgent(r, **kw(r)) for r in ROLES]
    if kind == "llm":
        from cdm.agents.llm import LLMAgent
        return [LLMAgent(r, **kw(r)) for r in ROLES]
    if kind == "random":
        from cdm.agents.random_agent import RandomAgent
        return [RandomAgent(r, seed + i, **kw(r)) for i, r in enumerate(ROLES)]
    if kind == "adversarial":
        from cdm.agents.adversarial import AdversarialAgent
        return [AdversarialAgent(r, seed + i, **kw(r)) for i, r in enumerate(ROLES)]
    raise ValueError(f"unknown agent kind {kind!r}")
