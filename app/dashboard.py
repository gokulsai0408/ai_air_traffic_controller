"""Duty-manager dashboard: `streamlit run app/dashboard.py`.

Every panel reads a ledger file, so a live run and a saved run render the same way.
Until the pipeline lands it shows a fixture; set CDM_ALLOW_STUBS=1 to run the stubs live.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import integration  # noqa: E402
from eval.results import RESULTS_DIR, STUB_NOTE, read_notes, read_results  # noqa: E402
from eval.sample import stratum_mask  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "ledger_example.jsonl"
RUNS = ROOT / "runs"
CAUSES = {"Any": None, "Weather": "weather", "Late aircraft": "late_aircraft"}
REAL_FIELDS = ("airline", "flight", "tail", "origin", "destination", "scheduled_arrival", "arrival_delay")
EVAL_TABLES = {
    "E1 Kernel ablation": ["e1_ablation", "e1_failures_by_constraint"], "E2 Red team": ["e2_redteam"],
    "E3 Negotiation": ["e3_negotiation"], "E4 Baselines": ["e4_baselines", "e4_price_of_privacy"],
    "E5 Cascade": ["e5_continuity", "e5_validation", "e5_counterfactual"], "E6 Sensitivity": ["e6_sensitivity"],
}
EVAL_CHARTS = {
    "E1 Kernel ablation": ["constraint_failures"], "E3 Negotiation": ["rounds_histogram", "outcome_mix"],
    "E4 Baselines": ["baseline_comparison"], "E5 Cascade": ["cascade_scatter"], "E6 Sensitivity": ["sensitivity_tornado"],
}


def read_ledger(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_of(entries: list[dict], kind: str) -> dict | None:
    return next((e for e in entries if e["type"] == kind), None)


def last_of(entries: list[dict], kind: str) -> dict | None:
    return next((e for e in reversed(entries) if e["type"] == kind), None)


def append_human_action(path: Path, action: str, note: str = "") -> dict:
    """Continue the hash chain with the BUILD_PLAN 5.9 formula, so B's verify still passes."""
    entries = read_ledger(path)
    previous = entries[-1]["hash"] if entries else ""
    entry = {"seq": len(entries), "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "type": "HUMAN_ACTION", "actor": "duty_manager", "payload": {"action": action, "note": note},
             "constraint_ids": [], "prev_hash": previous}
    canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)
    entry["hash"] = hashlib.sha256((canonical + previous).encode()).hexdigest()
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return entry


def tamper_copy(path: Path) -> Path:
    """A copy with one character changed in the middle entry, for the verify demo."""
    lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    middle = len(lines) // 2
    field = '"actor": "'
    position = lines[middle].index(field) + len(field)
    swapped = "X" if lines[middle][position] != "X" else "Y"
    lines[middle] = lines[middle][:position] + swapped + lines[middle][position + 1:]
    target = Path(tempfile.gettempdir()) / f"tampered_{Path(path).name}"
    target.write_text("".join(lines), encoding="utf-8")
    return target


def badge_row(report: dict) -> None:
    badges = []
    for result in report.get("results", []):
        icon = "✅" if result["status"] == "PASS" else "❌"
        margin = "" if result.get("margin") is None else f" ({result['margin']:.1f}{result.get('unit', '')})"
        badges.append(f"{icon} `{result['id']}`{margin}")
    st.markdown(" · ".join(badges) or "No checks recorded")


@st.cache_data(show_spinner="Loading flights")
def load_flights_cached() -> pd.DataFrame:
    dtype = {"SCHEDULED_DEPARTURE": str, "SCHEDULED_ARRIVAL": str}
    return pd.read_csv(ROOT / "data" / "flights.csv", dtype=dtype)


def run_live(row: pd.Series, seed: int) -> Path:
    from cdm.coordinator import negotiate
    from cdm.world import build_world

    cfg = integration.load_cfg()
    key = integration.flight_key(row)
    ledger = integration.make_ledger(key, seed, cfg)
    negotiate(build_world(row, seed, cfg), integration.make_agents(st.session_state.agent_kind, seed), cfg, ledger)
    RUNS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S%f")
    path = RUNS / f"{''.join(c if c.isalnum() else '_' for c in key)}_s{seed}_{stamp}.jsonl"
    ledger.save(path)
    return path


def scenario_picker(can_run: bool) -> None:
    st.sidebar.header("Scenario")
    st.session_state.agent_kind = st.sidebar.selectbox("Agents", ["rule", "llm"], disabled=not can_run)
    flights = load_flights_cached()
    hubs = flights["DESTINATION_AIRPORT"].value_counts().head(30).index.tolist()
    airport = st.sidebar.selectbox("Airport", ["Any"] + hubs, index=1 + hubs.index("ATL"), disabled=not can_run)
    cause = st.sidebar.selectbox("Cause", list(CAUSES), disabled=not can_run)
    seed = int(st.sidebar.number_input("World seed", value=0, step=1, disabled=not can_run))
    if st.sidebar.button("🎲 Random row", type="primary", disabled=not can_run):
        pool = flights if airport == "Any" else flights[flights["DESTINATION_AIRPORT"] == airport]
        if CAUSES[cause]:
            pool = pool[stratum_mask(pool, CAUSES[cause])]
        row = pool.sample(n=1, random_state=None).iloc[0]
        with st.spinner("Negotiating"):
            st.session_state.ledger_path = str(run_live(row, seed))
    if not can_run:
        st.sidebar.info("Live runs start when A, B and C land. Showing a saved example.")


def problem_panel(entries: list[dict]) -> None:
    scenario = first_of(entries, "SCENARIO")
    if scenario:
        public = scenario["payload"]
        left, right = st.columns(2)
        left.markdown("**Flight data** :green-badge[REAL]")
        left.table(pd.DataFrame({"value": {k: str(public[k]) for k in REAL_FIELDS if k in public}}))
        right.markdown("**Generated context** :orange-badge[SYNTHETIC]")
        right.table(pd.DataFrame({"value": {k: str(v) for k, v in public.items() if k not in REAL_FIELDS}}))
    baseline = first_of(entries, "BASELINE")
    if baseline:
        for name, title in (("do_nothing", "Do-nothing plan"), ("fallback", "Fallback plan")):
            report = baseline["payload"].get(name)
            if isinstance(report, dict):
                st.markdown(f"**{title}:** {'safe' if report.get('safe') else 'UNSAFE'}")
                badge_row(report)


def timeline_panel(entries: list[dict]) -> None:
    kernel = last_of(entries, "KERNEL")
    reports = kernel["payload"] if kernel and isinstance(kernel["payload"], dict) else {}
    vote_entries = [e for e in entries if e["type"] == "VOTES"]
    if not vote_entries:
        st.info("No vote took place. The kernel found nothing to vote on.")
    votes = [v for e in vote_entries for v in e["payload"]]
    for round_number in sorted({v["round"] for v in votes}):
        st.markdown(f"#### Round {round_number}")
        in_round = [v for v in votes if v["round"] == round_number]
        for option_id in dict.fromkeys(v["option_id"] for v in in_round):
            with st.container(border=True):
                st.markdown(f"**{option_id}**")
                if option_id in reports:
                    badge_row(reports[option_id])
                columns = st.columns(3)
                option_votes = [v for v in in_round if v["option_id"] == option_id]
                for column, vote in zip(columns, option_votes):
                    mark = "✅" if vote["vote"] == "ACCEPT" else "❌"
                    column.markdown(f"{mark} **{vote['agent']}** `{vote['reason']}`")
                    if vote.get("rationale"):
                        column.caption(vote["rationale"])


def decision_panel(entries: list[dict], path: Path, writable: bool) -> None:
    decision, escalation = last_of(entries, "DECISION"), last_of(entries, "ESCALATION")
    human = last_of(entries, "HUMAN_ACTION")
    if decision:
        payload = decision["payload"]
        st.subheader(f"Outcome: {payload.get('outcome', 'UNKNOWN')}")
        st.json(payload.get("option", {}), expanded=True)
        if payload.get("costs_by_agent"):
            rows = {role: {"plan cost (USD, ASSUMED constants)": cost,
                           "walk-away (USD)": payload.get("walkaway", {}).get(role)}
                    for role, cost in payload["costs_by_agent"].items()}
            st.table(pd.DataFrame(rows).T)
    if escalation:
        payload = escalation["payload"]
        st.subheader(f"Outcome: ESCALATED ({payload.get('kind', '')})")
        st.error(payload.get("required_human_decision", ""))
        if payload.get("blocking_constraints"):
            st.markdown("Blocking: " + ", ".join(f"`{c}`" for c in payload["blocking_constraints"]))
    if human:
        st.success(f"Duty manager action on record: {human['payload']['action']}")
    elif not writable:
        st.caption("This is a read-only example, so approval is off.")
    else:
        note = st.text_input("Note (optional)")
        approve, override = st.columns(2)
        if approve.button("Approve", type="primary"):
            append_human_action(path, "APPROVED", note)
            st.rerun()
        if override.button("Override"):
            append_human_action(path, "OVERRIDDEN", note)
            st.rerun()


def audit_panel(entries: list[dict], path: Path) -> None:
    st.dataframe(pd.DataFrame([{
        "seq": e["seq"], "type": e["type"], "actor": e["actor"],
        "constraints": ", ".join(e.get("constraint_ids", [])), "hash": e["hash"][:12],
    } for e in entries]), hide_index=True, use_container_width=True)
    from cdm.verify import verify
    verify_col, tamper_col = st.columns(2)
    if verify_col.button("Verify log"):
        report = verify(path)
        (st.success if report.ok else st.error)(report.message)
    if tamper_col.button("Tamper with a copy, then verify"):
        report = verify(tamper_copy(path))
        (st.success if report.ok else st.error)(report.message)


def evaluation_panel(results_dir: Path = RESULTS_DIR) -> None:
    st.caption("Each table says whether its numbers are measured, modelled or assumed.")
    for title, names in EVAL_TABLES.items():
        available = [n for n in names if (results_dir / f"{n}.csv").exists()]
        with st.expander(title, expanded=bool(available)):
            if not available:
                st.info("Not run yet.")
            for name in available:
                notes = read_notes(name, results_dir)
                if STUB_NOTE in notes:
                    st.warning("Placeholder numbers from the stubs. Do not cite.")
                st.dataframe(read_results(name, results_dir), hide_index=True, use_container_width=True)
                for note in notes:
                    if note != STUB_NOTE:
                        st.caption(note)
            for chart in EVAL_CHARTS.get(title, []):
                image = results_dir / f"{chart}.png"
                if image.exists():
                    st.image(str(image))


def main() -> None:
    st.set_page_config(page_title="Disruption Negotiation", layout="wide")
    st.title("Disruption negotiation: ATC × Pilot × Ground")

    allow_stubs = os.environ.get("CDM_ALLOW_STUBS") == "1"
    can_run = allow_stubs or not integration.missing_modules()
    stubbed = integration.use_pipeline(True)  # Verify needs a module, so fixture mode uses the stub's
    if stubbed:
        st.warning("STUB PIPELINE: the worlds and costs on this page are placeholders. Do not cite them.")
    scenario_picker(can_run)

    path = Path(st.session_state.get("ledger_path", FIXTURE))
    writable = path != FIXTURE
    entries = read_ledger(path)
    live, audit, evaluation = st.tabs(["Negotiation", "Audit", "Evaluation"])
    with live:
        problem_panel(entries)
        st.divider()
        timeline_panel(entries)
        st.divider()
        decision_panel(entries, path, writable)
    with audit:
        audit_panel(entries, path)
    with evaluation:
        evaluation_panel()


if __name__ == "__main__":
    main()
