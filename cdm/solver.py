"""Central solver baseline: full information, minimum total cost over kernel-safe options.

It is the upper bound the negotiation is compared against, and it is not deployable,
because the three organisations do not share commercial data. It uses B's check() and
A's cost functions, with no copies of its own.
"""
from __future__ import annotations

from typing import Iterable, Optional

from cdm import costs
from cdm.kernel import check
from cdm.options import seed_options
from cdm.schemas import Option
from cdm.world import derive


def costs_of(option: Option, world) -> dict[str, float]:
    derived = derive(option, world)
    return {
        "pilot": float(costs.cost_pilot(option, derived, world)),
        "ground": float(costs.cost_ground(option, derived, world)),
        "atc": float(costs.cost_atc(option, derived, world)),
    }


def central_solve(world, pool: Optional[Iterable[Option]] = None) -> tuple[Optional[Option], dict[str, float]]:
    """Returns (best option, cost per agent), or (None, {}) when nothing is safe and policy-compliant.

    Pass `pool` to compare on the same options a negotiation saw. Ties break on option_id,
    so the result is deterministic.
    """
    best: Optional[Option] = None
    best_costs: dict[str, float] = {}
    best_key: Optional[tuple[float, str]] = None
    for option in (seed_options(world) if pool is None else pool):
        report = check(option, world)
        if not (report.safe and report.policy_ok):
            continue
        option_costs = costs_of(option, world)
        key = (sum(option_costs.values()), option.option_id)
        if best_key is None or key < best_key:
            best, best_costs, best_key = option, option_costs, key
    if best is None:
        return None, {}
    return best.model_copy(update={"proposed_by": "central_solver"}), best_costs
