"""Run scenarios and collect one wide row per scenario (E3 and E4 read this).

Column prefixes: neg_ negotiated, dn_ do-nothing, fb_ fallback, cs_ central solver,
sp_ single-prompt baseline. All USD figures rest on ASSUMED cost constants.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from eval import integration
from eval.results import STUB_NOTE, write_results
from eval.sample import sample_rows

REQUIRED_COLUMNS = [
    "outcome", "rounds_used", "unsafe_accepted", "price_of_privacy_usd", "central_bound_ok",
    *(f"{p}_{m}" for p in ("neg", "dn", "fb", "cs") for m in (
        "has_plan", "safe", "policy_ok", "failed_constraints", "total_cost", "action",
        "hold_min", "gate_delay_min", "missed_pax", "overtime_min", "hold_fuel_kg")),
]


def gate_delay_min(on_block: Optional[str], row: pd.Series) -> float:
    """On-block delay against schedule. The HH:MM clock wraps, so take the wrap nearest the recorded delay."""
    if not on_block:
        return float("nan")
    hours, minutes = on_block.split(":")
    block = int(hours) * 60 + int(minutes)
    sched = str(row["SCHEDULED_ARRIVAL"]).split(".")[0].zfill(4)
    delay = block - (int(sched[:2]) * 60 + int(sched[2:]))
    recorded = float(row["ARRIVAL_DELAY"])
    return float(delay + 1440 * round((recorded - delay) / 1440))


def plan_metrics(prefix: str, option, world, row: pd.Series) -> dict:
    from cdm import costs as cost_module
    from cdm.kernel import check
    from cdm.solver import costs_of
    from cdm.world import derive

    if option is None:
        return {f"{prefix}_has_plan": False}
    report = check(option, world)
    derived = derive(option, world)
    option_costs = costs_of(option, world)
    fuel = getattr(cost_module, "hold_fuel_kg", None)  # optional helper from A; NaN until it lands
    return {
        f"{prefix}_has_plan": True,
        f"{prefix}_safe": bool(report.safe),
        f"{prefix}_policy_ok": bool(report.policy_ok),
        f"{prefix}_failed_constraints": ";".join(r.id for r in report.results if r.status == "FAIL"),
        f"{prefix}_total_cost": sum(option_costs.values()),
        f"{prefix}_action": getattr(option.action_type, "value", str(option.action_type)),
        f"{prefix}_hold_min": option.hold_minutes,
        f"{prefix}_gate_delay_min": gate_delay_min(derived.on_block_time, row),
        f"{prefix}_missed_pax": derived.missed_pax,
        f"{prefix}_overtime_min": derived.ground_overtime_min,
        f"{prefix}_hold_fuel_kg": fuel(option, derived, world) if fuel else float("nan"),
    }


def summarise_ledger(path: Path) -> dict:
    """Count what a saved ledger shows. KERNEL payloads must be {option_id: report dict}."""
    latest: dict[str, dict] = {}
    types: Counter = Counter()
    unparsed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            types[entry["type"]] += 1
            if entry["type"] == "KERNEL":
                payload = entry["payload"]
                if isinstance(payload, dict) and all(isinstance(v, dict) for v in payload.values()):
                    latest.update(payload)
                else:
                    unparsed += 1
    proposed = {k: v for k, v in latest.items() if not k.startswith("G-")}
    failures: Counter = Counter(
        result["id"] for report in proposed.values()
        for result in report["results"] if result["status"] == "FAIL"
    )
    return {
        "options_checked": len(latest),
        "agent_options": len(proposed),
        "agent_options_unsafe": sum(not r["safe"] for r in proposed.values()),
        "agent_failures_by_constraint": json.dumps(dict(sorted(failures.items()))),
        "guardrail_overrides": types["GUARDRAIL_OVERRIDE"],
        "llm_calls": types["LLM_CALL"],
        "kernel_entries_unparsed": unparsed,
    }


def run_scenario(row: pd.Series, seed: int, agent_kind: str, cfg, ledger_dir: Path,
                 single_prompt: bool = False) -> dict:
    from cdm.baseline_single_prompt import SinglePromptFailure, single_prompt_plan
    from cdm.coordinator import negotiate
    from cdm.fallback import fallback_plan
    from cdm.kernel import check
    from cdm.options import do_nothing, seed_options
    from cdm.solver import central_solve, costs_of
    from cdm.world import build_world

    key = integration.flight_key(row)
    record = {
        "flight_key": key, "seed": seed, "agent_kind": agent_kind,
        "airport": row["DESTINATION_AIRPORT"], "stratum": row.get("stratum", ""),
        "tail": row["TAIL_NUMBER"], "month": int(row["MONTH"]), "day": int(row["DAY"]),
        "flight": int(row["FLIGHT_NUMBER"]), "recorded_arrival_delay": float(row["ARRIVAL_DELAY"]),
    }
    world = build_world(row, seed, cfg)
    ledger = integration.make_ledger(key, seed, cfg)
    started = time.perf_counter()
    decision = negotiate(world, integration.make_agents(agent_kind, seed), cfg, ledger)
    record["latency_s"] = time.perf_counter() - started

    safe_name = "".join(c if c.isalnum() else "_" for c in key)
    ledger_path = ledger_dir / f"{safe_name}_s{seed}.jsonl"
    ledger.save(ledger_path)
    record.update(summarise_ledger(ledger_path))

    record.update(outcome=decision.outcome, rounds_used=decision.rounds_used,
                  escalation_kind=decision.escalation.kind if decision.escalation else "")
    record.update(plan_metrics("neg", decision.option, world, row))
    # The headline safety metric: re-check the decision here rather than trusting the coordinator.
    record["unsafe_accepted"] = bool(decision.option is not None and not check(decision.option, world).safe)

    record.update(plan_metrics("dn", do_nothing(world), world, row))
    record.update(plan_metrics("fb", fallback_plan(world), world, row))

    pool = list(seed_options(world))
    if decision.option is not None and all(o.option_id != decision.option.option_id for o in pool):
        pool.append(decision.option)
    central, central_costs = central_solve(world, pool)
    record.update(plan_metrics("cs", central, world, row))
    if central is not None and decision.option is not None and record["neg_policy_ok"] and record["neg_safe"]:
        gap = record["neg_total_cost"] - sum(central_costs.values())
        record["price_of_privacy_usd"] = gap
        record["central_bound_ok"] = gap >= -1e-6
        # Agents report costs themselves, so a mismatch here flags a broken assessor.
        reported = decision.costs_by_agent
        expected = costs_of(decision.option, world)
        record["costs_match_cost_functions"] = all(
            abs(reported.get(r, float("nan")) - expected[r]) < 1e-6 for r in expected)

    if single_prompt:
        try:
            record.update(plan_metrics("sp", single_prompt_plan(world), world, row))
            record["sp_error"] = ""
        except SinglePromptFailure as exc:
            record.update(sp_has_plan=False, sp_error=str(exc)[:200])
    return record


def run_batch(rows: pd.DataFrame, agent_kind: str, cfg, seeds: Sequence[int] = (0,),
              ledger_dir: Optional[Path] = None, single_prompt: bool = False) -> pd.DataFrame:
    """One row per scenario. A scenario that crashes is recorded in `error`, never dropped."""
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(ledger_dir) if ledger_dir else Path(tmp)
        target.mkdir(parents=True, exist_ok=True)
        for _, row in rows.iterrows():
            for seed in seeds:
                try:
                    records.append({**run_scenario(row, seed, agent_kind, cfg, target, single_prompt), "error": ""})
                except Exception as exc:  # a crash is itself a result the report must show
                    records.append({"flight_key": integration.flight_key(row), "seed": seed,
                                    "agent_kind": agent_kind, "error": f"{type(exc).__name__}: {exc}"[:300]})
    batch = pd.DataFrame(records)
    # Plans can be absent from every scenario (all escalated, all crashed), so columns may be missing.
    for column in REQUIRED_COLUMNS:
        if column not in batch:
            batch[column] = float("nan")
    return batch


def load_sample(n: int, sample_seed: int = 2015) -> pd.DataFrame:
    dtype = {"SCHEDULED_DEPARTURE": str, "SCHEDULED_ARRIVAL": str}
    flights = pd.read_csv(integration.ROOT / "data" / "flights.csv", dtype=dtype)
    return sample_rows(flights, n, sample_seed)


def notes_for(stubbed: bool, *notes: str) -> tuple[str, ...]:
    return ((STUB_NOTE,) if stubbed else ()) + notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", default="rule", choices=["rule", "llm", "random", "adversarial"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--single-prompt", action="store_true", help="also run the banned-design baseline")
    parser.add_argument("--allow-stubs", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    stubbed = integration.use_pipeline(args.allow_stubs)
    batch = run_batch(load_sample(args.n), args.agents, integration.load_cfg(), range(args.seeds),
                      integration.ROOT / "runs" / f"batch_{args.agents}", args.single_prompt)
    path = write_results(batch, args.out or f"batch_{args.agents}", notes_for(
        stubbed, f"Sample: {args.n} rows x {args.seeds} seeds. Costs use ASSUMED constants."))
    errors = int((batch["error"] != "").sum())
    print(f"{len(batch)} scenarios, {errors} errors -> {path}")


if __name__ == "__main__":
    main()
