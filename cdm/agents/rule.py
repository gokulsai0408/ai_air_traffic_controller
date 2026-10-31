"""RuleAgent (BUILD_PLAN 5.5): deterministic policy. Proposes its own min-cost option and accepts any
option that is feasible for it and costs no more than its walk-away (fallback) cost.

Used for fast evals and as the agent-level non-AI fallback when an LLM call fails.
"""
from __future__ import annotations

from cdm.agents.base import AssessingAgent, as_draft
from cdm.schemas import OptionDraft, Reason, Vote

COST_TOLERANCE_USD = 1e-6   # float-comparison tolerance only, not a domain threshold


class RuleAgent(AssessingAgent):
    kind = "rule"

    def propose(self, view: dict) -> list[OptionDraft]:
        candidates = {o.option_id: o for o in view["candidates"]}
        mine = [a for a in view["own_assessments"] if a.feasible_for_me and a.option_id in candidates]
        if not mine:
            return []
        best = min(mine, key=lambda a: (a.cost_usd, a.option_id))
        return [as_draft(candidates[best.option_id], f"{self.role} rule agent: lowest own cost among kernel-safe options")]

    def vote(self, view: dict) -> list[Vote]:
        own = {a.option_id: a for a in view["own_assessments"]}
        walkaway = view["walkaway_usd"]
        votes = []
        for o in view["shortlist"]:
            a = own[o.option_id]
            if a.feasible_for_me and a.cost_usd <= walkaway + COST_TOLERANCE_USD:
                vote, reason = "ACCEPT", Reason.ACCEPTABLE
            elif not a.feasible_for_me:
                vote, reason = "REJECT", next((r for r in a.reasons if r != Reason.ACCEPTABLE), Reason.UNKNOWN)
            else:
                vote, reason = "REJECT", Reason.COST_ABOVE_WALKAWAY
            votes.append(Vote(agent=self.role, round=view["round"], option_id=o.option_id, vote=vote, reason=reason,
                              rationale=f"{self.role} rule agent: {reason.value}"))
        return votes
