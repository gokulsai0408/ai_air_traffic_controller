"""The one shared Claude client wrapper (team convention). Every LLM call in the repo goes through
`complete()`: the three agents, and D's labelled single-prompt baseline.

- Structured output: the response is constrained to the caller's JSON schema (output_config.format);
  callers still validate it with pydantic, which enforces the constraints the API strips.
- No sampling parameters: the current API rejects temperature/top_p/top_k on every model, so runs are
  reproduced from the logged LLM I/O (ledger LLM_CALL entries), never by re-calling the model.
- Settings come from environment variables; the key is resolved by the SDK (ANTHROPIC_API_KEY etc.).
"""
from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any, Callable

LLMClient = Callable[[str, str, dict], str]   # (system, user, json_schema) -> raw JSON text

MODEL = os.environ.get("CDM_LLM_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("CDM_LLM_EFFORT", "low")            # short structured replies; raise for harder reasoning
TIMEOUT_S = float(os.environ.get("CDM_LLM_TIMEOUT_S", "20"))
MAX_TOKENS = int(os.environ.get("CDM_LLM_MAX_TOKENS", "16000"))
USE_FALLBACKS = os.environ.get("CDM_LLM_FALLBACKS", "1") != "0"   # server-side refusal fallback
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# JSON-schema keywords structured outputs does not accept; pydantic re-checks them client-side.
_UNSUPPORTED = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                "minLength", "maxLength", "minItems", "maxItems", "uniqueItems", "default"}


class LLMError(RuntimeError):
    """Any failed call: transport error, timeout, refusal, truncation or no text."""


class LLMAuthError(LLMError):
    """Missing/invalid credentials or no permission: retrying cannot help."""


def api_schema(schema: dict) -> dict:
    """Copy of a pydantic JSON schema reduced to what structured outputs accepts."""
    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            out = {k: clean(v) for k, v in node.items() if k not in _UNSUPPORTED}
            if out.get("type") == "object" or "properties" in out:
                out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node
    return clean(copy.deepcopy(schema))


_client = None
_lock = threading.Lock()
_fallbacks_ok = USE_FALLBACKS


ENV_FILE = Path(__file__).resolve().parents[2] / ".env"   # gitignored; never commit keys


def _load_env_file() -> None:
    """Fill missing ANTHROPIC_* variables from the repo's .env, if there is one."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key.startswith("ANTHROPIC_") and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _get_client():
    global _client
    with _lock:
        if _client is None:
            _load_env_file()
            import anthropic
            _client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=0)   # callers own the retry
        return _client


def complete(system: str, user: str, json_schema: dict) -> str:
    """One Claude call constrained to `json_schema`. Returns the raw JSON text; raises LLMError."""
    global _fallbacks_ok
    import anthropic

    client = _get_client()
    params = dict(model=MODEL, max_tokens=MAX_TOKENS, system=system,
                  messages=[{"role": "user", "content": user}],
                  output_config={"effort": EFFORT,
                                 "format": {"type": "json_schema", "schema": api_schema(json_schema)}})
    try:
        if _fallbacks_ok:
            try:
                resp = client.beta.messages.create(**params, betas=[FALLBACK_BETA], fallbacks="default")
            except anthropic.BadRequestError:
                _fallbacks_ok = False        # endpoint/proxy without the fallback beta: carry on without it
                resp = client.messages.create(**params)
        else:
            resp = client.messages.create(**params)
    except anthropic.APITimeoutError as e:
        raise LLMError(f"timeout after {TIMEOUT_S}s") from e
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        raise LLMAuthError(f"API error {e.status_code}: check ANTHROPIC_API_KEY") from e
    except anthropic.RateLimitError as e:
        raise LLMError("rate limited") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("connection error") from e

    if resp.stop_reason == "refusal":
        category = getattr(getattr(resp, "stop_details", None), "category", None)
        raise LLMError(f"refusal ({category})")
    if resp.stop_reason == "max_tokens":
        raise LLMError("truncated at max_tokens")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        raise LLMError("no text block in response")
    return text
