"""BASELINE ONLY: one LLM call holding all three roles and all private data.

This is the design the brief bans. It exists so its kernel-violation rate can be measured,
and it must never be used as a planner. Nothing here checks legality: callers run the
returned option through the kernel and count the failures.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from pydantic import ValidationError

from cdm.schemas import Option, OptionDraft

SYSTEM_PROMPT = (
    "You are the arrival flow coordinator, the airline pilot and operations team, and the "
    "ground operations lead, all at once. Protect separation and runway capacity; fuel, "
    "crew duty, connections and comfort; gates, overtime and turnaround. Choose one plan "
    "using only IDs from the menus. Reply with JSON only, matching the schema."
)
LLMClient = Callable[[str, str, dict], str]  # (system, user, json_schema) -> raw text


class SinglePromptFailure(Exception):
    """The LLM gave no valid plan after one retry. Counted as a failure, not as a safe outcome."""


def build_prompt(world) -> str:
    sections = {name: getattr(world, name) for name in ("public", "pilot", "ground", "atc", "menus")}
    return json.dumps(sections, sort_keys=True, default=str)


def default_client() -> LLMClient:
    """C's shared wrapper. The function name is agreed in eval/integration.py."""
    from eval.integration import single_prompt_client
    return single_prompt_client()


def single_prompt_plan(world, client: Optional[LLMClient] = None) -> Option:
    """Temperature and timeout belong to the client wrapper, per the team conventions."""
    call = client or default_client()
    prompt = build_prompt(world)
    schema = OptionDraft.model_json_schema()
    error: Exception | None = None
    for _ in range(2):  # one retry, the same rule the real agents follow
        try:
            draft = OptionDraft.model_validate_json(call(SYSTEM_PROMPT, prompt, schema))
            return Option(**draft.model_dump(), option_id="SP-1", round=1, proposed_by="generator")
        except (ValidationError, ValueError) as exc:
            error = exc
    raise SinglePromptFailure(str(error))
