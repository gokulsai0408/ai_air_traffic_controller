"""Hash-chained, append-only log (BUILD_PLAN 5.9).

Payloads the verifier replays (all other entry types are chained but not interpreted):

    SCENARIO    {"max_rounds": n, ...}
    BASELINE    {"do_nothing": {"option", "kernel"}, "fallback": {"option", "kernel"}}
    WALKAWAY    {role: usd}
    KERNEL      {"round": n, "reports": [KernelReport], "options": [Option], "facts": OpsFacts}
                # reports new this round; options and facts are optional but needed to replay
                # agent drafts and any ops facts passed through kernel.apply_facts
    ASSESSMENTS {"round": n, "assessments": {role: {option_id: Assessment}}}
    VOTES       {"round": n, "votes": [Vote]}       # after guardrails
    DECISION    Decision.model_dump()
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from pydantic import BaseModel


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def genesis_hash(flight_key: str, seed: int, config_hash: str) -> str:
    return hashlib.sha256(f"{flight_key}{seed}{config_hash}".encode("utf-8")).hexdigest()


def entry_hash(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256((canonical_json(body) + entry["prev_hash"]).encode("utf-8")).hexdigest()


def _plain(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return _plain(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_plain(v) for v in obj]
    return obj


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Ledger:
    """Entry 0 is a GENESIS record naming the flight_key, seed and config_hash to replay."""

    def __init__(self, flight_key: str = "", seed: int = 0, config_hash: str = "",
                 clock: Callable[[], str] = _utc_now) -> None:
        self.flight_key, self.seed, self.config_hash = flight_key, seed, config_hash
        self.entries: list[dict[str, Any]] = []
        self._clock = clock
        self._prev_hash = genesis_hash(flight_key, seed, config_hash)
        self.append("GENESIS", "ledger", {"flight_key": flight_key, "seed": seed, "config_hash": config_hash})

    def append(self, type: str, actor: str, payload: Any, constraint_ids: Iterable[str] = ()) -> dict[str, Any]:
        # The round trip stores exactly what save() will write, so the hash survives reloading.
        entry = {
            "seq": len(self.entries), "ts": self._clock(), "type": type, "actor": actor,
            "payload": json.loads(json.dumps(_plain(payload), default=str)),
            "constraint_ids": list(constraint_ids), "prev_hash": self._prev_hash,
        }
        entry["hash"] = entry_hash(entry)
        self.entries.append(entry)
        self._prev_hash = entry["hash"]
        return entry

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for entry in self.entries:
                handle.write(canonical_json(entry) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path, clock: Callable[[], str] = _utc_now) -> "Ledger":
        """Reopen a saved log to extend its chain, e.g. to record the human's approval."""
        entries = read_entries(path)
        if not entries or entries[0].get("type") != "GENESIS":
            raise ValueError(f"{path}: no GENESIS entry")
        genesis = entries[0]["payload"]
        ledger = cls.__new__(cls)
        ledger.flight_key, ledger.seed, ledger.config_hash = genesis["flight_key"], genesis["seed"], genesis["config_hash"]
        ledger.entries, ledger._clock, ledger._prev_hash = entries, clock, entries[-1]["hash"]
        return ledger


def read_entries(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_broken_seq(entries: list[dict[str, Any]]) -> Optional[int]:
    """Recompute the chain. Returns the first bad position, or None if it is intact.

    Dropping entries from the very end is not detectable from the chain alone, which is
    why verify also requires a DECISION entry.
    """
    if not entries:
        return 0
    for position, entry in enumerate(entries):
        try:
            if position == 0:
                genesis = entry["payload"]
                expected_prev = genesis_hash(genesis["flight_key"], genesis["seed"], genesis["config_hash"])
            else:
                expected_prev = entries[position - 1]["hash"]
            if entry["seq"] != position or entry["prev_hash"] != expected_prev or entry["hash"] != entry_hash(entry):
                return position
        except (KeyError, TypeError):
            return position
    return None
