"""Replay a ledger without calling any LLM (BUILD_PLAN 5.9).

    python -m cdm.verify runs/<id>.jsonl

Exit codes: 0 fully verified, 1 tampered or mismatched, 2 clean but some steps were skipped.
A step that cannot run is reported as SKIPPED, never as passed.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import ValidationError

from cdm import kernel
from cdm.fallback import find_policy_relaxation
from cdm.ledger import canonical_json, first_broken_seq, read_entries
from cdm.options import seed_options
from cdm.schemas import KernelReport, OpsFacts

ROLES = ("pilot", "ground", "atc")
COST_TOLERANCE_USD = 1e-6   # same float-comparison tolerance as the coordinator
WorldLoader = Callable[[str, int, str], Any]


@dataclass
class VerifyReport:
    chain_ok: bool = False
    first_broken_seq: Optional[int] = None
    kernel_checked: int = 0
    kernel_matched: int = 0
    kernel_unreplayable: int = 0           # logged reports whose option the log does not contain
    kernel_skipped: bool = False
    decision_status: str = "SKIPPED"       # REPRODUCED | MISMATCH | SKIPPED
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.chain_ok and self.kernel_matched == self.kernel_checked and self.decision_status != "MISMATCH"

    @property
    def complete(self) -> bool:
        return (self.ok and not self.kernel_skipped and not self.kernel_unreplayable
                and self.decision_status == "REPRODUCED")

    def summary(self) -> str:
        chain = "CHAIN OK" if self.chain_ok else f"CHAIN BROKEN AT ENTRY {self.first_broken_seq}"
        if self.kernel_skipped:
            replay = "KERNEL REPLAY SKIPPED"
        else:
            verdict = "MATCH" if self.kernel_matched == self.kernel_checked else "MISMATCH"
            replay = f"KERNEL REPLAY {self.kernel_matched}/{self.kernel_checked} {verdict}"
            if self.kernel_unreplayable:
                replay += f" ({self.kernel_unreplayable} NOT REPLAYABLE)"
        return f"{chain} · {replay} · DECISION {self.decision_status}"

    @property
    def message(self) -> str:
        return self.summary()


def default_world_loader(flight_key: str, seed: int, config_hash: str) -> Any:
    """Rebuild the world from the flight row. Raises ImportError until A's modules are on the branch."""
    from cdm.data import load_flights
    from cdm.world import build_world

    flights = load_flights(Path(__file__).resolve().parents[1] / "data" / "flights.csv")
    return build_world(flights[flights["flight_key"] == flight_key].iloc[0], seed, None)


def _of_type(entries: list[dict[str, Any]], entry_type: str) -> list[dict[str, Any]]:
    return [e for e in entries if e.get("type") == entry_type]


def _logged_reports(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every kernel report in the log by option ID. KERNEL entries hold only each round's new ones."""
    reports: dict[str, dict[str, Any]] = {}
    for entry in _of_type(entries, "BASELINE"):
        for plan in entry["payload"].values():
            reports[plan["option"]["option_id"]] = plan["kernel"]
    for entry in _of_type(entries, "KERNEL"):
        for report in entry["payload"]["reports"]:
            reports[report["option_id"]] = report
    return reports


def _replay_world(entries: list[dict[str, Any]], world: Any) -> Any:
    for entry in _of_type(entries, "KERNEL"):
        if entry["payload"].get("facts"):
            world = kernel.apply_facts(world, OpsFacts.model_validate(entry["payload"]["facts"]))
    return world


def _known_options(entries: list[dict[str, Any]], world: Any) -> dict[str, Any]:
    known: dict[str, Any] = {o.option_id: o for o in seed_options(world)}
    for entry in _of_type(entries, "BASELINE"):
        for plan in entry["payload"].values():
            known[plan["option"]["option_id"]] = plan["option"]
    for entry in _of_type(entries, "KERNEL"):
        for option in entry["payload"].get("options", []):
            known[option["option_id"]] = option
    for entry in _of_type(entries, "DECISION"):
        option = entry["payload"].get("option")
        if option:
            known[option["option_id"]] = option
    return known


def _replay_kernel(entries: list[dict[str, Any]], world: Any, report: VerifyReport) -> None:
    known = _known_options(entries, world)
    for option_id, logged in _logged_reports(entries).items():
        if option_id not in known:
            report.kernel_unreplayable += 1
            report.problems.append(f"{option_id}: the log has its report but not the option, so it cannot be replayed")
            continue
        report.kernel_checked += 1
        replayed = kernel.check(known[option_id], world).model_dump(mode="json")
        if canonical_json(logged) == canonical_json(replayed):
            report.kernel_matched += 1
        else:
            report.problems.append(f"kernel replay differs for {option_id}")


class _Skip(Exception):
    pass


def _for_round(entries: list[dict[str, Any]], entry_type: str, round_: int) -> dict[str, Any]:
    matching = [e["payload"] for e in _of_type(entries, entry_type) if e["payload"].get("round") == round_]
    if not matching:
        raise _Skip(f"no {entry_type} entry for round {round_}")
    return matching[-1]


def _unanimous(votes: list[dict[str, Any]]) -> list[str]:
    """Option IDs every role ACCEPTed. One REJECT from a role outweighs any ACCEPT from it."""
    by_option: dict[str, dict[str, list[str]]] = {}
    for vote in votes:
        by_option.setdefault(vote["option_id"], {}).setdefault(vote["agent"], []).append(vote["vote"])
    return [oid for oid, roles in by_option.items()
            if all(role in roles and "REJECT" not in roles[role] for role in ROLES)]


def _is_feasible(report: dict[str, Any]) -> bool:
    return bool(report["safe"] and report["policy_ok"])


def _total_costs(assessments: dict[str, dict[str, dict[str, Any]]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for by_option in assessments.values():
        for option_id, assessment in by_option.items():
            totals[option_id] = totals.get(option_id, 0.0) + assessment["cost_usd"]
    return totals


def _compromise_candidates(assessments: dict[str, dict[str, dict[str, Any]]], walkaway: dict[str, float],
                           reports: dict[str, dict[str, Any]]) -> list[str]:
    """Feasible, individually rational and with no verifiable objection from any role."""
    candidates = []
    for option_id in _total_costs(assessments):
        by_role = [assessments[role][option_id] for role in ROLES]
        if (_is_feasible(reports[option_id]) and all(a["feasible_for_me"] for a in by_role)
                and all(a["cost_usd"] <= walkaway[a["agent"]] + COST_TOLERANCE_USD for a in by_role)):
            candidates.append(option_id)
    return candidates


def _check_escalation(decision: dict[str, Any], reports: dict[str, dict[str, Any]],
                      fallback_ok: bool, final_round: bool) -> Optional[str]:
    # 5.6 also escalates at the final round when no compromise holds and the fallback is unusable.
    if any(_is_feasible(r) for r in reports.values()) and (fallback_ok or not final_round):
        return "ESCALATED although feasible options exist"
    escalation = decision.get("escalation")
    if not escalation:
        return "ESCALATED decision has no escalation record"
    parsed = {k: KernelReport.model_validate(v) for k, v in reports.items()}
    found = find_policy_relaxation(parsed.values())
    if escalation["kind"] == "DECLINE":
        if decision.get("option") is not None:
            return "DECLINE decision carries an option"
        return "DECLINE although a POLICY relaxation was available" if found else None

    if found is None:
        return "offered a relaxation but no option needs only a POLICY relaxation"
    if escalation.get("relaxed_constraint_id") != found.constraint.id:
        return "relaxed constraint is not the POLICY constraint the search found"
    enabled = parsed.get(escalation.get("enabled_option_id") or "")
    if enabled is None or not enabled.safe:
        return "relaxation enables an unsafe option"
    chosen = next((r for r in enabled.results if r.id == found.constraint.id), None)
    if chosen is None or chosen.status != "FAIL" or abs(chosen.margin or 0.0) != found.relaxation:
        return "offered relaxation is not the smallest available"
    option = decision.get("option")
    if option is not None and option["option_id"] != escalation["enabled_option_id"]:
        return "ESCALATED option is not the one the relaxation enables"
    return None


def replay_decision(entries: list[dict[str, Any]]) -> Optional[str]:
    """Re-apply the 5.6 decision rule to the logged votes and costs. Returns a mismatch or None.

    Raises _Skip if the log lacks the entries needed. Candidates are compared by their cost
    key, not identity, so equal-cost ties do not read as tampering.
    """
    decisions = _of_type(entries, "DECISION")
    if not decisions:
        raise _Skip("no DECISION entry")
    if not _of_type(entries, "KERNEL"):
        raise _Skip("no KERNEL entry")
    baselines = _of_type(entries, "BASELINE")
    if not baselines:
        raise _Skip("no BASELINE entry")
    decision = decisions[-1]["payload"]
    outcome, option, rounds_used = decision["outcome"], decision.get("option"), decision["rounds_used"]
    reports = _logged_reports(entries)
    baseline = baselines[-1]["payload"]

    scenarios = _of_type(entries, "SCENARIO")
    max_rounds = scenarios[-1]["payload"].get("max_rounds") if scenarios else None
    if max_rounds is None:
        raise _Skip("no max_rounds in SCENARIO")
    if rounds_used > max_rounds:
        return f"used {rounds_used} rounds, cap is {max_rounds}"
    final_round = rounds_used == max_rounds

    if option is not None:
        logged = decision.get("kernel") or reports.get(option["option_id"])
        if not logged or not logged.get("safe"):
            return f"decided option {option['option_id']} is not logged as safe"

    # Consensus ends a round, so no earlier round may have had a unanimous option.
    last_open_round = rounds_used if outcome == "CONSENSUS" else rounds_used + 1
    for vote_entry in _of_type(entries, "VOTES"):
        vote_round = vote_entry["payload"]["round"]
        if vote_round < last_open_round and _unanimous(vote_entry["payload"]["votes"]):
            return f"round {vote_round} had a unanimous option but the protocol continued"

    if outcome == "ESCALATED":
        fallback_ok = _is_feasible(baseline["fallback"]["kernel"])
        return _check_escalation(decision, reports, fallback_ok, final_round)
    if option is None:
        return f"{outcome} decision has no option"
    option_id = option["option_id"]

    if outcome == "CONSENSUS":
        if final_round:
            return "CONSENSUS in the final round, which is binding and takes no votes"
        unanimous = _unanimous(_for_round(entries, "VOTES", rounds_used)["votes"])
        if option_id not in unanimous:
            return f"CONSENSUS on {option_id} without unanimous ACCEPT"
        totals = _total_costs(_for_round(entries, "ASSESSMENTS", rounds_used)["assessments"])
        key = lambda oid: (totals[oid], -kernel.min_safety_margin(KernelReport.model_validate(reports[oid])))
        if key(option_id) != min(key(oid) for oid in unanimous):
            return f"CONSENSUS on {option_id} is not the best unanimous option"
        return None

    if outcome not in ("COMPROMISE", "FALLBACK"):
        return f"unknown outcome {outcome}"
    if not final_round:
        return f"{outcome} before the final round"
    walkaways = _of_type(entries, "WALKAWAY")
    if not walkaways:
        raise _Skip("no WALKAWAY entry")
    assessments = _for_round(entries, "ASSESSMENTS", rounds_used)["assessments"]
    candidates = _compromise_candidates(assessments, walkaways[-1]["payload"], reports)
    if outcome == "COMPROMISE":
        if option_id not in candidates:
            return f"COMPROMISE on {option_id} is not individually rational and unobjected"
        totals = _total_costs(assessments)
        if totals[option_id] != min(totals[oid] for oid in candidates):
            return f"COMPROMISE on {option_id} is not the cheapest candidate"
        return None
    if canonical_json(option) != canonical_json(baseline["fallback"]["option"]):
        return "FALLBACK decision is not the logged fallback plan"
    if candidates:
        return "FALLBACK although a compromise was available"
    return None


def _replay_final_option(entries: list[dict[str, Any]], world: Any) -> Optional[str]:
    """The invariant that matters most: the decided option must be safe on a fresh kernel run."""
    decisions = _of_type(entries, "DECISION")
    option = decisions[-1]["payload"].get("option") if decisions else None
    if option is None:
        return None
    report = kernel.check(option, world)
    return None if report.safe else f"decided option {report.option_id} fails {kernel.failing_ids(report)}"


def verify(path: str | Path, world_loader: WorldLoader = default_world_loader) -> VerifyReport:
    report = VerifyReport()
    try:
        entries = read_entries(path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        report.first_broken_seq = getattr(exc, "lineno", 1) - 1 if isinstance(exc, json.JSONDecodeError) else 0
        report.problems.append(f"cannot read log: {exc}")
        report.kernel_skipped = True
        return report

    report.first_broken_seq = first_broken_seq(entries)
    report.chain_ok = report.first_broken_seq is None
    if not report.chain_ok:
        # Nothing downstream of a broken chain can be trusted.
        report.kernel_skipped = True
        report.problems.append(f"hash chain broken at entry {report.first_broken_seq}")
        return report

    genesis = entries[0]["payload"]
    try:
        world = world_loader(genesis["flight_key"], genesis["seed"], genesis["config_hash"])
    except ImportError as exc:
        world = None
        report.problems.append(f"world rebuild unavailable: {exc}")
    except Exception as exc:
        world = None
        report.problems.append(f"world rebuild failed: {type(exc).__name__}: {exc}")

    try:
        if world is None:
            report.kernel_skipped = True
        else:
            world = _replay_world(entries, world)
            _replay_kernel(entries, world, report)
            unsafe = _replay_final_option(entries, world)
            if unsafe:
                report.problems.append(unsafe)
                report.decision_status = "MISMATCH"
                return report
        mismatch = replay_decision(entries)
        report.decision_status = "MISMATCH" if mismatch else "REPRODUCED"
        if mismatch:
            report.problems.append(mismatch)
    except _Skip as exc:
        report.problems.append(f"decision replay skipped: {exc}")
    except (KeyError, TypeError, AttributeError, ValidationError) as exc:
        report.decision_status = "MISMATCH"
        report.problems.append(f"log does not match the ledger payload contract: {type(exc).__name__}: {exc}")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m cdm.verify runs/<id>.jsonl", file=sys.stderr)
        return 2
    report = verify(args[0])
    print(report.summary())
    for problem in report.problems:
        print(f"  - {problem}")
    if not report.ok:
        return 1
    return 0 if report.complete else 2


if __name__ == "__main__":
    sys.exit(main())
