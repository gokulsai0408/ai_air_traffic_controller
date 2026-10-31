"""Pluggable agent kinds (BUILD_PLAN 5.5). Pick one with `make_agents(kind)`."""
from __future__ import annotations

from typing import Literal, Optional

from cdm.agents.base import ROLES, Agent, CostFn, DeriveFn
from cdm.agents.rule import RuleAgent

AgentKind = Literal["rule", "llm", "random", "adversarial"]


def make_agents(kind: AgentKind = "rule", derive_fn: Optional[DeriveFn] = None,
                cost_fns: Optional[dict[str, CostFn]] = None) -> list[Agent]:
    """One agent per role (pilot, ground, atc), each its own object with its own state."""
    cost_fns = cost_fns or {}
    if kind == "rule":
        return [RuleAgent(r, derive_fn=derive_fn, cost_fn=cost_fns.get(r)) for r in ROLES]
    raise NotImplementedError(f"agent kind {kind!r} lands in deliverable 3")
