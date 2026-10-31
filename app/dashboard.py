"""Duty-manager dashboard: `streamlit run app/dashboard.py`.

Runs the real pipeline (A's world, B's kernel, C's coordinator) on any row. The sidebar sliders change
SYNTHETIC generator ranges and cost constants, rebuild the world with cfg.with_synthetic / cfg.with_costs
and rerun the negotiation. Every panel reads the saved ledger, so what you see is what verify replays.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import integration, whatif  # noqa: E402
from eval.results import RESULTS_DIR, read_notes, read_results  # noqa: E402
from eval.sample import sample_rows, stratum_mask  # noqa: E402

RUNS = ROOT / "runs"
CAUSES = {"Any": None, "Weather": "weather", "Late aircraft": "late_aircraft"}
ROLES = integration.ROLES
SYN = ":orange-badge[SYNTHETIC]"
REAL = ":green-badge[REAL]"
ASSUMED = ":gray-badge[ASSUMED]"
TAG_BADGE = {"REAL": REAL, "SYNTHETIC": SYN, "ASSUMED_CONST": ASSUMED}
PLAN_LABELS = {"neg": "Negotiated", "fb": "Fallback rule", "cs": "Central optimum (same pool, full info)",
               "dn": "Do-nothing / FCFS (status quo)"}
EVAL_TABLES = {
    "E1 Kernel ablation": ["e1_ablation", "e1_failures_by_constraint"], "E2 Red team": ["e2_redteam"],
    "E3 Negotiation": ["e3_negotiation"],
    "E4 Baselines": ["e4_headline", "e4_baselines", "e4_do_nothing_failures", "e4_per_scenario"],
    "E5 Cascade": ["e5_continuity", "e5_validation", "e5_counterfactual"], "E6 Sensitivity": ["e6_sensitivity"],
}
EVAL_CHARTS = {
    "E1 Kernel ablation": ["constraint_failures"], "E3 Negotiation": ["rounds_histogram", "outcome_mix"],
    "E4 Baselines": ["baseline_comparison"], "E5 Cascade": ["cascade_scatter"], "E6 Sensitivity": ["sensitivity_tornado"],
}
SEQUENCE = """
**How a decision is made** (no human in the loop until the agents are done)

1. **Disruption**: a real delayed flight (DOT 2015 row, REAL) plus generated airport context (SYNTHETIC).
2. **Agents negotiate on their own**: Pilot, ATC and Ground each propose plans and vote with reason codes.
   They see only their own private data. The **safety kernel** (pure code) removes every unsafe plan
   *before* anyone votes. The **coordinator** (pure code) enforces the round cap of **{max_rounds}**.
3. **Consensus is reached**: unanimous vote in round 1, else a binding compromise in round {max_rounds},
   else the deterministic fallback.
4. **Then the human**: the duty manager **approves or overrides** the agreed plan. If no compliant plan
   exists, the system **escalates** and names the exact decision only a human may take
   (for example, approve ground-crew overtime). SAFETY limits are never offered for relaxation.
5. **Audit**: every step is in a hash-chained ledger that `verify` replays without any LLM.
"""


# ---- ledger helpers ------------------------------------------------------------------------------
def read_ledger(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_of(entries: list[dict], kind: str) -> dict | None:
    return next((e for e in entries if e["type"] == kind), None)


def last_of(entries: list[dict], kind: str) -> dict | None:
    return next((e for e in reversed(entries) if e["type"] == kind), None)


def append_human_action(path: Path, action: str, note: str = "") -> dict:
    """Extend the hash chain with B's Ledger, so verify still passes."""
    from cdm.ledger import Ledger
    ledger = Ledger.load(path)
    entry = ledger.append("HUMAN_ACTION", "duty_manager", {"action": action, "note": note})
    ledger.save(path)
    return entry


def tamper_copy(path: Path) -> Path:
    """A copy with one character of the middle entry's actor changed, for the verify demo."""
    lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    middle = len(lines) // 2
    match = re.search(r'"actor":\s*"', lines[middle])
    position = match.end()
    swapped = "X" if lines[middle][position] != "X" else "Y"
    lines[middle] = lines[middle][:position] + swapped + lines[middle][position + 1:]
    target = Path(tempfile.gettempdir()) / f"tampered_{Path(path).name}"
    target.write_text("".join(lines), encoding="utf-8")
    return target


def badge_row(report: dict) -> None:
    badges, sources = [], []
    for result in report.get("results", []):
        icon = "✅" if result["status"] == "PASS" else "❌"
        margin = "" if result.get("margin") is None else f" ({result['margin']:.1f}{result.get('unit', '')})"
        badges.append(f"{icon} `{result['id']}`{margin}")
        sources.append(f"{result['id']}: {result.get('source') or 'no source'} [{result.get('tag', '')}]")
    st.markdown(" · ".join(badges) or "No checks recorded", help="\n\n".join(sources) or None)


# ---- world context -------------------------------------------------------------------------------
def fmt_min(minutes: float) -> str:
    """Minutes after midnight of the flight's date -> HH:MM (+1d past midnight)."""
    m = int(round(minutes))
    day = "" if m < 1440 else f" (+{m // 1440}d)"
    return f"{(m // 60) % 24:02d}:{m % 60:02d}{day}"


def context_rows(world, overrides: dict | None) -> pd.DataFrame:
    """The duty manager's omniscient view of the world, each field with its tag and a what-if marker."""
    touched = whatif.affected_fields(overrides)
    p, pi, g, a = world.public, world.pilot, world.ground, world.atc
    free = sum(1 for s in a.slots if not s.occupied)
    fields = [
        ("public.airline", "Airline / flight", f"{p.airline} {p.flight_number} (tail {p.tail_number})"),
        ("public.origin", "Route", f"{p.origin} → {p.destination}, {p.date}"),
        ("public.sched_arrival_min", "Scheduled arrival", fmt_min(p.sched_arrival_min)),
        ("public.arrival_delay_min", "Recorded arrival delay", f"{p.arrival_delay_min:.0f} min"),
        ("public.weather_delay_min", "Weather / late-aircraft delay",
         f"{p.weather_delay_min:.0f} / {p.late_aircraft_delay_min:.0f} min"),
        ("public.eta_min", "ETA at runway", fmt_min(p.eta_min)),
        ("public.taxi_in_min", "Taxi-in", f"{p.taxi_in_min:.0f} min"),
        ("public.aircraft_category", "Aircraft category / wake", f"{p.aircraft_category} / {p.wake}"),
        ("public.ops_notes", "Ops notes", p.ops_notes),
        ("pilot.extra_fuel_min", "Holding fuel above reserve", f"{pi.extra_fuel_min:.0f} min"),
        ("pilot.fdp_end_min", "Crew duty (FDP) ends", fmt_min(pi.fdp_end_min)),
        ("pilot.connecting_pax", "Connecting passengers", f"{pi.connecting_pax} in {len(pi.connections)} banks"),
        ("pilot.transfer_min", "Transfer times same/adj/far/bus",
         f"{pi.transfer_min.same:.0f}/{pi.transfer_min.adjacent:.0f}/{pi.transfer_min.far:.0f}/"
         f"{pi.transfer_min.remote_bus:.0f} min"),
        ("pilot.turbulence_in_hold", "Turbulence in hold", "yes" if pi.turbulence_in_hold else "no"),
        ("ground.original_gate", "Original gate", g.original_gate),
        ("ground.gates", "Gates", f"{len(g.gates)} contact gates + REMOTE-1"),
        ("ground.shift_end_min", "Ground shift ends", fmt_min(g.shift_end_min)),
        ("ground.min_turn_min", "Minimum turnaround", f"{g.min_turn_min:.0f} min"),
        ("ground.next_departure_min", "Next departure", fmt_min(g.next_departure_min)),
        ("atc.aar_per_hour", "Arrival rate (weather)", f"{a.aar_per_hour:.0f}/h"),
        ("atc.slots", "Runway slots", f"{free} free of {len(a.slots)}"),
        ("atc.taxiway_closures", "Taxiway closures", ", ".join(c.taxiway for c in a.taxiway_closures) or "none"),
    ]
    rows = []
    for path, label, value in fields:
        tag = "SYNTHETIC" if path in touched else world.tags.get(path, "SYNTHETIC")
        rows.append({"field": label, "value": value, "tag": tag,
                     "what-if": "⚙ changed by slider" if path in touched else ""})
    return pd.DataFrame(rows)


# ---- running ------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading flights (A's loader)")
def flights_cached() -> pd.DataFrame:
    return integration.load_flights()


@st.cache_resource
def base_cfg() -> integration.RunCfg:
    return integration.load_cfg()


def demo_row(flights: pd.DataFrame) -> pd.Series:
    """The first ATL row of the fixed eval sample, so a fresh page always opens on the same flight."""
    sample = sample_rows(flights, 20, seed=2015)
    atl = sample[sample["DESTINATION_AIRPORT"] == "ATL"]
    return (atl if len(atl) else sample).iloc[0]


def row_for(flights: pd.DataFrame, key: str) -> pd.Series:
    return flights[flights["flight_key"] == key].iloc[0]


def run_live(row: pd.Series, seed: int, agent_kind: str, overrides: dict | None) -> dict:
    from eval.run_batch import run_scenario
    cfg = whatif.apply(base_cfg(), overrides)
    safe_name = "".join(c if c.isalnum() else "_" for c in integration.flight_key(row))
    stamp = datetime.now().strftime("%H%M%S%f")
    RUNS.mkdir(exist_ok=True)
    return run_scenario(row, seed, agent_kind, cfg, RUNS, what_if=None if whatif.is_empty(overrides) else overrides,
                        ledger_name=f"{safe_name}_s{seed}_{stamp}.jsonl")


def signature(key: str, seed: int, agent_kind: str, overrides: dict) -> str:
    return json.dumps([key, seed, agent_kind, overrides], sort_keys=True, default=str)


def negotiate_now(key: str, seed: int, agent_kind: str, overrides: dict) -> None:
    row = row_for(flights_cached(), key)
    with st.spinner(f"Negotiating with {agent_kind} agents"):
        record = run_live(row, seed, agent_kind, overrides)
    st.session_state.record = record
    st.session_state.ledger_path = record["ledger"]
    st.session_state.run_overrides = overrides
    st.session_state.run_signature = signature(key, seed, agent_kind, overrides)


# ---- sidebar ------------------------------------------------------------------------------------
def _slider(slider: whatif.Slider, section: str, cfg: integration.RunCfg):
    default = whatif.default_value(cfg, section, slider.key)
    values = default if isinstance(default, list) else [default]
    integer = all(isinstance(v, int) and not isinstance(v, bool) for v in values) and float(slider.step).is_integer()
    cast = int if integer else float
    lo, hi = cast(min([slider.lo] + values)), cast(max([slider.hi] + values))
    value = tuple(cast(v) for v in default) if isinstance(default, list) else cast(default)
    key = f"{section}::{slider.key}"
    if key not in st.session_state:
        st.session_state[key] = value
    return st.sidebar.slider(slider.label, lo, hi, step=cast(slider.step), key=key,
                             help=f"{section}.yaml `{slider.key}`, repo value {default}")


def _reset_sliders() -> None:
    for key in [k for k in st.session_state if "::" in k]:
        del st.session_state[key]


def _apply_preset(name: str) -> None:
    _reset_sliders()
    for section, values in whatif.PRESETS[name].items():
        for key, value in values.items():
            st.session_state[f"{section}::{key}"] = tuple(value) if isinstance(value, list) else value


def sidebar() -> tuple[str, int, str, dict, bool]:
    """Returns (flight_key, seed, agent kind, what-if overrides, run requested)."""
    st.sidebar.header("Scenario")
    flights = flights_cached()
    agent_kind = st.sidebar.selectbox("Agents", ["rule", "llm"],
                                      help="llm needs ANTHROPIC_API_KEY; a failed LLM call falls back to RuleAgent")
    hubs = flights["DESTINATION_AIRPORT"].value_counts().head(30).index.tolist()
    airport = st.sidebar.selectbox("Airport", ["Any"] + hubs, index=1 + hubs.index("ATL"))
    cause = st.sidebar.selectbox("Cause", list(CAUSES))
    seed = int(st.sidebar.number_input("World seed", value=0, step=1))
    run = False
    if "row_key" not in st.session_state:
        st.session_state.row_key = integration.flight_key(demo_row(flights))
    if st.sidebar.button("🎲 Random row", type="primary"):
        pool = flights if airport == "Any" else flights[flights["DESTINATION_AIRPORT"] == airport]
        if CAUSES[cause]:
            pool = pool[stratum_mask(pool, CAUSES[cause])]
        st.session_state.row_key = integration.flight_key(pool.sample(n=1).iloc[0])
        run = True
    run |= st.sidebar.button("▶ Re-run negotiation")

    cfg = base_cfg()
    st.sidebar.divider()
    st.sidebar.markdown(f"**What-if parameters** {SYN}")
    st.sidebar.caption("These change the generated context and cost constants, then rebuild the world "
                       "and rerun the negotiation. Anything they change is SYNTHETIC.")
    for name in whatif.PRESETS:
        st.sidebar.button(name, on_click=_apply_preset, args=(name,))
    st.sidebar.button("Reset to repo config", on_click=_reset_sliders)
    synthetic, costs = {}, {}
    st.sidebar.markdown("**Generated context (synthetic.yaml)**")
    for slider in whatif.available(whatif.SYNTHETIC_SLIDERS, cfg, "synthetic"):
        synthetic[slider.key] = _slider(slider, "synthetic", cfg)
    st.sidebar.markdown("**Cost weights (costs.yaml)**")
    for slider in whatif.available(whatif.COST_SLIDERS, cfg, "costs"):
        costs[slider.key] = _slider(slider, "costs", cfg)
    st.sidebar.caption(f"Round cap: {cfg.max_rounds} (config/protocol.yaml, the brief's \"under 3 turns\"). "
                       "Not a slider.")
    return st.session_state.row_key, seed, agent_kind, whatif.changed(cfg, synthetic, costs), run


# ---- panels -------------------------------------------------------------------------------------
def what_if_banner(overrides: dict | None) -> None:
    if whatif.is_empty(overrides):
        return
    cfg = base_cfg()
    parts = [f"`{k}` {v} (repo {whatif.default_value(cfg, 'synthetic', k)})"
             for k, v in overrides.get("synthetic", {}).items()]
    parts += [f"`{k}` {v:g} (repo {whatif.default_value(cfg, 'costs', k):g})" for k, v in overrides.get("costs", {}).items()]
    st.warning(f"{SYN} **What-if run.** These values were set with the sliders and are not observations: "
               + "; ".join(parts) + ". They are recorded in the ledger as `WHAT_IF_CONFIG`.")


def context_panel(world, overrides: dict | None) -> None:
    table = context_rows(world, overrides)
    left, right = st.columns(2)
    left.markdown(f"**Flight data** {REAL}")
    left.dataframe(table[table["tag"] == "REAL"][["field", "value"]], hide_index=True, use_container_width=True)
    right.markdown(f"**Generated context** {SYN} {ASSUMED}")
    right.dataframe(table[table["tag"] != "REAL"], hide_index=True, use_container_width=True)
    st.caption("Omniscient demo view for the duty manager. Each agent sees only its own private state.")


def problem_panel(entries: list[dict], record: dict) -> None:
    baseline = first_of(entries, "BASELINE")
    if not baseline:
        return
    for name, prefix, title in (("do_nothing", "dn", "Do-nothing plan (status quo)"), ("fallback", "fb", "Fallback plan")):
        part = baseline["payload"].get(name) or {}
        report, option = part.get("kernel"), part.get("option") or {}
        if isinstance(report, dict):
            cost = record.get(f"{prefix}_total_cost")
            cost_text = "" if cost is None or pd.isna(cost) else f" · total ${cost:,.0f} (modelled)"
            plan = f"{option.get('action_type', '')} slot {option.get('slot_id')} gate {option.get('gate_id')}"
            verdict = "safe" if report.get("safe") else ":red[UNSAFE]"
            st.markdown(f"**{title}:** {verdict}{'' if report.get('policy_ok', True) else ', breaks policy'}"
                        f" · {plan}{cost_text} {SYN}")
            badge_row(report)


def timeline_panel(entries: list[dict]) -> None:
    reports = {r["option_id"]: r for e in entries if e["type"] == "KERNEL" for r in e["payload"].get("reports", [])}
    proposals = [e for e in entries if e["type"] == "PROPOSALS"]
    if proposals:
        st.markdown("#### Round 1: proposals")
        columns = st.columns(len(proposals))
        for column, entry in zip(columns, proposals):
            drafts = entry["payload"].get("drafts", [])
            column.markdown(f"**{entry['actor']}** proposed {len(drafts)}")
            for draft in drafts:
                column.caption(f"{draft.get('action_type')} · hold {draft.get('hold_minutes')} · "
                               f"slot {draft.get('slot_id')} · gate {draft.get('gate_id') or draft.get('divert_to')}")
                if draft.get("rationale"):
                    column.caption(f"“{draft['rationale']}”")
    vote_entries = [e for e in entries if e["type"] == "VOTES"]
    if not vote_entries:
        st.info("No votes on record: the kernel left nothing to vote on, or the final round was binding.")
    votes = [v for e in vote_entries for v in e["payload"]["votes"]]
    assessments = {e["payload"]["round"]: e["payload"]["assessments"] for e in entries if e["type"] == "ASSESSMENTS"}
    for round_number in sorted({v["round"] for v in votes}):
        st.markdown(f"#### Round {round_number}: votes on the shortlist")
        in_round = [v for v in votes if v["round"] == round_number]
        for option_id in dict.fromkeys(v["option_id"] for v in in_round):
            with st.container(border=True):
                st.markdown(f"**{option_id}**")
                if option_id in reports:
                    badge_row(reports[option_id])
                columns = st.columns(3)
                for column, vote in zip(columns, [v for v in in_round if v["option_id"] == option_id]):
                    mark = "✅" if vote["vote"] == "ACCEPT" else "❌"
                    own = assessments.get(round_number, {}).get(vote["agent"], {}).get(option_id, {})
                    cost = f" · ${own['cost_usd']:,.0f}" if "cost_usd" in own else ""
                    column.markdown(f"{mark} **{vote['agent']}** `{vote['reason']}`{cost}")
                    if vote.get("rationale"):
                        column.caption(vote["rationale"])
    compromise = [e for e in entries if e["type"] == "COMPROMISE_CHECK"]
    if compromise:
        st.markdown("#### Final round: binding compromise")
        for entry in compromise:
            objections = entry["payload"].get("verified_objections") or []
            st.caption(f"{entry['payload'].get('option_id')}: "
                       + (f"blocked by verified objection from {', '.join(objections)}" if objections else "no objection"))


def decision_panel(entries: list[dict], path: Path) -> None:
    decision, escalation = last_of(entries, "DECISION"), last_of(entries, "ESCALATION")
    human = last_of(entries, "HUMAN_ACTION")
    if escalation:
        payload = escalation["payload"]
        st.subheader(f"Outcome: ESCALATED ({payload.get('kind', '')})")
        st.error(payload.get("required_human_decision", ""))
        if payload.get("blocking_constraints"):
            st.markdown("Blocking: " + ", ".join(f"`{c}`" for c in payload["blocking_constraints"]))
    if decision:
        payload = decision["payload"]
        if not escalation:
            st.subheader(f"Outcome: {payload.get('outcome', 'UNKNOWN')} in round {payload.get('rounds_used', '?')}")
        if payload.get("option"):
            st.json(payload["option"], expanded=False)
        if payload.get("costs_by_agent"):
            walkaway_entry = last_of(entries, "WALKAWAY")
            walkaway = walkaway_entry["payload"] if walkaway_entry else {}
            rows = {role: {"plan cost (USD, modelled)": cost, "walk-away = fallback cost (USD)": walkaway.get(role)}
                    for role, cost in payload["costs_by_agent"].items()}
            st.dataframe(pd.DataFrame(rows).T, use_container_width=True)
    st.markdown("**Duty manager** (acts only after the agents have finished)")
    if human:
        st.success(f"Duty manager action on record: {human['payload']['action']}")
        return
    note = st.text_input("Note (optional)")
    approve, override = st.columns(2)
    if approve.button("Approve", type="primary"):
        append_human_action(path, "APPROVED", note)
        st.rerun()
    if override.button("Override"):
        append_human_action(path, "OVERRIDDEN", note)
        st.rerun()


def baselines_panel(record: dict, overrides: dict | None) -> None:
    badge = f" {SYN}" if not whatif.is_empty(overrides) else ""
    st.markdown(f"**E4 for this run: every plan scored with the same cost functions and kernel**{badge}")
    rows = []
    for prefix, label in PLAN_LABELS.items():
        if not record.get(f"{prefix}_has_plan"):
            rows.append({"plan": label, "total USD": None, "safe": None, "policy ok": None, "failed": "no plan"})
            continue
        rows.append({
            "plan": label, "total USD": record.get(f"{prefix}_total_cost"),
            **{f"{r} USD": record.get(f"{prefix}_cost_{r}") for r in ROLES},
            "safe": record.get(f"{prefix}_safe"), "policy ok": record.get(f"{prefix}_policy_ok"),
            "failed": record.get(f"{prefix}_failed_constraints") or "",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    def money(key: str) -> str:
        value = record.get(key)
        return "n/a" if value is None or pd.isna(value) else f"${value:,.0f}"

    a, b, c, d = st.columns(4)
    a.metric("Saving vs fallback", money("neg_saving_vs_fallback_usd"), help="modelled; n/a unless decided and "
             "the fallback is also legal")
    b.metric("Price of privacy", money("price_of_privacy_usd"), help="negotiated minus central optimum; modelled")
    conflict = record.get("conflict")
    c.metric("Conflict of interest", "n/a" if conflict is None or pd.isna(conflict) else ("yes" if conflict else "no"),
             help="no single compliant option is cheapest for all three agents; modelled")
    d.metric("Compliant options in pool", int(record.get("compliant_options") or 0), help="measured")
    st.caption("Costs: modelled with config/costs.yaml (delay and fuel sourced, the rest ASSUMED). "
               "Safety verdicts: measured by the kernel on SYNTHETIC context. The central optimum sees every "
               "agent's private data, so it is a bound, not a deployable design.")


def audit_panel(entries: list[dict], path: Path) -> None:
    st.dataframe(pd.DataFrame([{
        "seq": e["seq"], "type": e["type"], "actor": e["actor"],
        "constraints": ", ".join(e.get("constraint_ids", [])), "hash": e["hash"][:12],
    } for e in entries]), hide_index=True, use_container_width=True)
    from cdm.verify import verify
    verify_col, tamper_col = st.columns(2)
    if verify_col.button("Verify log"):
        report = verify(path, whatif.world_loader_for(path))
        (st.success if report.ok else st.error)(report.message)
    if tamper_col.button("Tamper with a copy, then verify"):
        tampered = tamper_copy(path)
        report = verify(tampered, whatif.world_loader_for(tampered))
        (st.success if report.ok else st.error)(report.message)
    st.caption(f"Log file: `{path}`")


def quick_batch_panel(overrides: dict | None) -> None:
    from eval.report import headline, slide_text, do_nothing_failures
    from eval.run_batch import run_batch
    badge = f" {SYN}" if not whatif.is_empty(overrides) else ""
    st.markdown(f"**Quick E4 batch with the current sliders** (rule agents){badge}")
    n = st.slider("Rows from the fixed eval sample", 10, 100, 30, step=10)
    seeds = st.slider("Seeds per row", 1, 5, 2)
    if st.button("Run batch"):
        cfg = whatif.apply(base_cfg(), overrides)
        with st.spinner(f"Running {n * seeds} scenarios"):
            batch = run_batch(sample_rows(flights_cached(), n, 2015), "rule", cfg, range(seeds))
        st.session_state.quick = (headline(batch), slide_text(headline(batch), do_nothing_failures(batch),
                                                              cfg.max_rounds))
    if "quick" in st.session_state:
        table, text = st.session_state.quick
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.markdown(text)


def evaluation_panel(results_dir: Path = RESULTS_DIR) -> None:
    st.caption("Saved results from `python -m eval.report`. Each table says whether its numbers are measured, "
               "modelled or assumed.")
    slide = results_dir / "e4_slide.md"
    if slide.exists():
        with st.expander("E4 slide text", expanded=True):
            st.markdown(slide.read_text(encoding="utf-8"))
    for title, names in EVAL_TABLES.items():
        available = [n for n in names if (results_dir / f"{n}.csv").exists()]
        with st.expander(title, expanded=False):
            if not available:
                st.info("Not run yet.")
            for name in available:
                st.markdown(f"`{name}`")
                st.dataframe(read_results(name, results_dir), hide_index=True, use_container_width=True)
                for note in read_notes(name, results_dir):
                    st.caption(note)
            for chart in EVAL_CHARTS.get(title, []):
                image = results_dir / f"{chart}.png"
                if image.exists():
                    st.image(str(image))


def main() -> None:
    st.set_page_config(page_title="Disruption Negotiation", layout="wide")
    st.title("Disruption negotiation: ATC × Pilot × Ground")
    key, seed, agent_kind, overrides, run = sidebar()
    wanted = signature(key, seed, agent_kind, overrides)
    stale = st.session_state.get("run_signature") != wanted
    # Rule agents are instant, so slider moves rerun at once. LLM agents cost API calls, so they wait for the button.
    if run or "record" not in st.session_state or (stale and agent_kind == "rule"):
        negotiate_now(key, seed, agent_kind, overrides)
    elif stale:
        st.info("Settings changed. Press ▶ Re-run negotiation to run the LLM agents again.")

    with st.expander("How a decision is made", expanded=False):
        st.markdown(SEQUENCE.format(max_rounds=base_cfg().max_rounds))
    record = st.session_state.record
    path = Path(st.session_state.ledger_path)
    run_overrides = st.session_state.get("run_overrides")
    entries = read_ledger(path)
    what_if_banner(run_overrides)

    from cdm.world import build_world
    world = build_world(row_for(flights_cached(), record["flight_key"]), record["seed"],
                        whatif.apply(base_cfg(), run_overrides).config)
    live, baselines, audit, evaluation = st.tabs(["Negotiation", "Baselines (this run)", "Audit", "Evaluation"])
    with live:
        context_panel(world, run_overrides)
        st.divider()
        problem_panel(entries, record)
        st.divider()
        timeline_panel(entries)
        st.divider()
        decision_panel(entries, path)
    with baselines:
        baselines_panel(record, run_overrides)
    with audit:
        audit_panel(entries, path)
    with evaluation:
        quick_batch_panel(overrides)
        st.divider()
        evaluation_panel()


if __name__ == "__main__":
    main()
