"""Turn eval/results/*.csv into the E3 and E4 summary tables, the E4 slide text and the PNG charts.

Any experiment that has not run yet is skipped and named. Every number carries a label:
  measured  = counted from our system's own behaviour (outcomes, rounds, kernel re-checks)
  modelled  = depends on the SYNTHETIC world context or on cost constants (mostly ASSUMED)
  assumed   = a design choice or constant, not an observation
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eval.results import RESULTS_DIR, read_results, write_results

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3a9e6f"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
ROLES = ("pilot", "ground", "atc")
PLANS = {"neg": "Negotiated", "fb": "Fallback rule", "cs": "Central optimum (same pool)",
         "dn": "Do-nothing / FCFS", "sp": "Single-prompt LLM (banned design)"}
OUTCOMES = ["CONSENSUS", "COMPROMISE", "FALLBACK", "ESCALATED"]
DECIDED = ("CONSENSUS", "COMPROMISE", "FALLBACK")
TOL = 1e-6
SYNTHETIC_NOTE = ("Flight rows are REAL. Gates, slots, fuel, crew, ground shift and connections are SYNTHETIC "
                  "(seeded generator), so any rate that depends on them is labelled modelled.")
COST_NOTE = "Costs are modelled with config/costs.yaml constants: delay and fuel prices are sourced, the rest ASSUMED."


def _ok(batch: pd.DataFrame) -> pd.DataFrame:
    return batch[batch["error"].isna() | (batch["error"].astype(str) == "")]


def _true(series: pd.Series) -> pd.Series:
    return series.eq(True)   # NaN (not computed) counts as False


def _rate(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def decided_runs(batch: pd.DataFrame) -> pd.DataFrame:
    ok = _ok(batch)
    return ok[ok["outcome"].isin(DECIDED) & _true(ok["neg_safe"]) & _true(ok["neg_policy_ok"])]


def summarise_negotiation(batch: pd.DataFrame) -> pd.DataFrame:
    """E3: one row per agent kind."""
    rows = []
    for kind, group in batch.groupby("agent_kind"):
        ok = _ok(group)
        outcomes = ok["outcome"].value_counts(normalize=True)
        rows.append({
            "agent_kind": kind, "scenarios": len(group), "crashed": len(group) - len(ok),
            **{f"{o.lower()}_rate": float(outcomes.get(o, 0.0)) for o in OUTCOMES},
            "max_rounds": int(ok["max_rounds"].max()) if len(ok) else 0,
            "logged_max_rounds": ",".join(str(int(v)) for v in sorted(ok["logged_max_rounds"].dropna().unique())),
            "agreed_within_limit_rate": _rate(ok["outcome"].isin(["CONSENSUS", "COMPROMISE"])
                                              & (ok["rounds_used"] <= ok["max_rounds"])),
            "unsafe_accepted": int(_true(ok["unsafe_accepted"]).sum()),
            "guardrail_overrides": int(ok["guardrail_overrides"].sum()),
            "median_latency_s": float(ok["latency_s"].median()),
            "llm_calls_per_scenario": float(ok["llm_calls"].mean()), "label": "measured",
        })
    return pd.DataFrame(rows)


def per_scenario_costs(batch: pd.DataFrame) -> pd.DataFrame:
    """E4 per scenario: total and per-agent cost of every plan, and whether do-nothing was unsafe."""
    ok = _ok(batch)
    columns = ["flight_key", "seed", "agent_kind", "outcome", "escalation_kind", "compliant_options", "conflict"]
    for prefix in ("neg", "fb", "cs", "dn"):
        columns += [f"{prefix}_total_cost", *(f"{prefix}_cost_{r}" for r in ROLES), f"{prefix}_safe",
                    f"{prefix}_policy_ok"]
    columns += ["dn_failed_constraints", "neg_beats_fallback", "neg_saving_vs_fallback_usd", "price_of_privacy_usd",
                "escalation_no_compliant"]
    table = ok.reindex(columns=columns).copy()
    table.insert(len(table.columns), "dn_unsafe", ok["dn_safe"].eq(False))
    table["label_costs"] = "modelled"
    table["label_outcome"] = "measured"
    return table


def summarise_baselines(batch: pd.DataFrame) -> pd.DataFrame:
    """E4 per plan. Costs are means over `common_scenarios`, where every plan is safe and within policy.

    A plan that breaks a rule can look cheap, so including it would flatter it.
    """
    ok = _ok(batch)
    present = [p for p in PLANS if f"{p}_safe" in ok and ok[f"{p}_safe"].notna().any()]
    legal = pd.Series(True, index=ok.index)
    for prefix in ("neg", "dn", "fb", "cs"):
        legal &= _true(ok[f"{prefix}_safe"]) & _true(ok[f"{prefix}_policy_ok"])
    common = ok[legal]
    rows = []
    for prefix in present:
        has_plan = _true(ok[f"{prefix}_has_plan"])
        unsafe = has_plan & ok[f"{prefix}_safe"].eq(False)
        mean = lambda column: float(common[column].mean()) if column in common else float("nan")  # noqa: E731
        rows.append({
            "plan": PLANS[prefix], "scenarios": len(ok), "with_plan": int(has_plan.sum()),
            "unsafe_rate": float(unsafe.sum() / has_plan.sum()) if has_plan.any() else float("nan"),
            "common_scenarios": len(common),
            "mean_total_cost_usd": mean(f"{prefix}_total_cost"),
            **{f"mean_cost_{r}_usd": mean(f"{prefix}_cost_{r}") for r in ROLES},
            "mean_gate_delay_min": mean(f"{prefix}_gate_delay_min"),
            "mean_missed_pax": mean(f"{prefix}_missed_pax"),
            "mean_overtime_min": mean(f"{prefix}_overtime_min"),
            "mean_hold_fuel_kg": mean(f"{prefix}_hold_fuel_kg"),
            "label_unsafe_rate": "modelled (SYNTHETIC context)" if prefix in ("dn", "fb") else "measured",
            "label_cost": "modelled",
        })
    return pd.DataFrame(rows)


def headline(batch: pd.DataFrame) -> pd.DataFrame:
    """The E4 numbers the slide quotes, each with its n, its label and its definition."""
    ok = _ok(batch)
    decided = decided_runs(batch)
    rows: list[dict] = []

    def add(metric: str, value: float, unit: str, n: int, label: str, definition: str) -> None:
        rows.append({"metric": metric, "value": value, "unit": unit, "n": n, "label": label,
                     "definition": definition})

    add("scenarios", len(ok), "runs", len(ok), "measured", "scenarios that ran without crashing")
    add("unsafe_accepted", int(_true(ok["unsafe_accepted"]).sum()), "runs", len(ok), "measured",
        "decisions that fail a SAFETY or PHYSICAL constraint when the kernel re-checks them")
    add("decided_rate", len(decided) / len(ok) if len(ok) else float("nan"), "share", len(ok), "measured",
        "CONSENSUS, COMPROMISE or FALLBACK with a safe, policy-compliant plan")
    escalated = ok[ok["outcome"] == "ESCALATED"]
    add("escalation_rate", len(escalated) / len(ok) if len(ok) else float("nan"), "share", len(ok), "measured",
        "runs handed to the duty manager")
    add("escalation_no_compliant_option_rate", _rate(escalated["compliant_options"].eq(0)), "share",
        len(escalated), "measured",
        "escalations where no safe + policy-compliant option existed in the pool (necessary escalations)")
    dn = ok[_true(ok["dn_has_plan"])]
    add("do_nothing_unsafe_rate", _rate(dn["dn_safe"].eq(False)), "share", len(dn), "modelled (SYNTHETIC context)",
        "the status-quo FCFS plan (original gate, first free slot) fails a SAFETY or PHYSICAL constraint")

    vs_fb = decided[decided["neg_beats_fallback"].notna()]
    beats = _true(vs_fb["neg_beats_fallback"])
    add("neg_beats_fallback_rate", _rate(beats), "share", len(vs_fb), "modelled",
        "decided runs (fallback also legal) where the negotiated plan is strictly cheaper in total")
    add("neg_equals_fallback_rate", _rate(vs_fb["neg_saving_vs_fallback_usd"].abs() <= TOL), "share", len(vs_fb),
        "modelled", "decided runs where negotiation lands on a plan costing the same as the fallback")
    add("mean_saving_vs_fallback_usd", float(vs_fb["neg_saving_vs_fallback_usd"].mean()), "USD", len(vs_fb),
        "modelled", "mean (fallback total - negotiated total) over decided runs")
    fb_sum = float(vs_fb["fb_total_cost"].sum())
    add("saving_vs_fallback_pct", 100 * float(vs_fb["neg_saving_vs_fallback_usd"].sum()) / fb_sum if fb_sum else
        float("nan"), "%", len(vs_fb), "modelled", "summed saving / summed fallback cost")

    priv = decided[decided["price_of_privacy_usd"].notna()]
    cs_sum = float(priv["cs_total_cost"].sum())
    add("price_of_privacy_pct", 100 * float(priv["price_of_privacy_usd"].sum()) / cs_sum if cs_sum else float("nan"),
        "%", len(priv), "modelled", "summed (negotiated - central optimum) / summed central optimum, same option pool")
    add("price_of_privacy_mean_usd", float(priv["price_of_privacy_usd"].mean()), "USD", len(priv), "modelled",
        "mean (negotiated - central optimum)")
    add("central_strictly_better_rate", _rate(priv["price_of_privacy_usd"] > TOL), "share", len(priv), "modelled",
        "decided runs where the full-information solver finds a cheaper compliant plan")
    add("central_bound_violations", int((priv["price_of_privacy_usd"] < -TOL).sum()), "runs", len(priv), "measured",
        "runs where negotiation beat the central optimum on its own pool (must be 0, else a bug)")

    with_pool = ok[ok["compliant_options"].fillna(0) > 0]
    add("conflict_rate", _rate(_true(with_pool["conflict"])), "share", len(with_pool), "modelled",
        "runs where no single compliant option is cheapest for all three agents (a real negotiation exists)")
    return pd.DataFrame(rows)


def do_nothing_failures(batch: pd.DataFrame) -> pd.DataFrame:
    ok = _ok(batch)
    unsafe = ok[ok["dn_safe"].eq(False)]
    counts: Counter = Counter(cid for cell in unsafe["dn_failed_constraints"].fillna("") for cid in cell.split(";")
                              if cid)
    return pd.DataFrame([{"constraint_id": cid, "do_nothing_runs_failing": n,
                          "share_of_unsafe_do_nothing": n / len(unsafe), "label": "modelled (SYNTHETIC context)"}
                         for cid, n in counts.most_common()],
                        columns=["constraint_id", "do_nothing_runs_failing", "share_of_unsafe_do_nothing", "label"])


def _value(table: pd.DataFrame, metric: str) -> tuple[float, int]:
    row = table[table["metric"] == metric].iloc[0]
    return float(row["value"]), int(row["n"])


def necessity_line(table: pd.DataFrame) -> str:
    rate, n = _value(table, "escalation_no_compliant_option_rate")
    if pd.isna(rate):
        return "- No escalations in this sample. *measured*"
    if rate >= 1 - TOL:
        return f"- Every escalation (n={n}) had no compliant option at all: escalation only when needed. *measured*"
    unnecessary = round((1 - rate) * n)
    return (f"- {100 * rate:.0f}% of escalations (n={n}) had no compliant option at all. The other {unnecessary} "
            "escalated although a compliant option existed (it cost some agent more than its walk-away and the "
            "fallback broke policy); see escalation_no_compliant in e4_per_scenario.csv. *measured*")


def slide_text(table: pd.DataFrame, failures: pd.DataFrame, max_rounds: int) -> str:
    """The E4 slide, worded from the numbers. It never claims a saving the data doesn't show."""
    def pct(metric: str) -> str:
        value, n = _value(table, metric)
        return "n/a" if pd.isna(value) else f"{100 * value:.0f}% (n={n})"

    beats, n_beats = _value(table, "neg_beats_fallback_rate")
    saving, _ = _value(table, "mean_saving_vs_fallback_usd")
    saving_pct, _ = _value(table, "saving_vs_fallback_pct")
    privacy, n_priv = _value(table, "price_of_privacy_pct")
    unsafe, n_runs = _value(table, "unsafe_accepted")
    top = ", ".join(f"{r.constraint_id} {100 * r.share_of_unsafe_do_nothing:.0f}%"
                    for r in failures.head(3).itertuples()) or "none"
    lines = ["# E4: baselines (generated by eval/report.py, do not edit by hand)", ""]
    if pd.isna(beats) or beats <= 0:
        lines += [
            f"**Negotiation did not beat the fallback on cost** in any decided run (n={n_beats}). "
            "We do not claim savings.",
            "Our claim is safety, necessary-only escalation and auditability:",
        ]
    else:
        lines += [
            f"- Negotiation beat the fallback rule in **{pct('neg_beats_fallback_rate')}** of decided runs; "
            f"mean saving ${saving:,.0f} ({saving_pct:.1f}% of fallback cost). *modelled, cost constants mostly ASSUMED*",
            f"- It tied the fallback in {pct('neg_equals_fallback_rate')} of decided runs. *modelled*",
            f"- Price of privacy: the full-information central solver was {privacy:.1f}% cheaper in total "
            f"(n={n_priv}); it is not deployable because the three organisations don't share commercial data. *modelled*",
            f"- A real conflict of interest (no option cheapest for all three) existed in {pct('conflict_rate')} "
            "of runs. *modelled*",
        ]
    lines += [
        f"- Status quo (do-nothing/FCFS) was **unsafe in {pct('do_nothing_unsafe_rate')}** of scenarios; "
        f"top blocking constraints: {top}. *modelled on SYNTHETIC gates/slots/fuel/crew*",
        f"- Unsafe plans accepted: **{int(unsafe)} of {n_runs}** runs (kernel re-check). *measured*",
        necessity_line(table),
        f"- Agreement within {max_rounds} rounds (\"under 3 turns\"); every decision is in a hash-chained log that "
        "`verify` replays. *measured*",
    ]
    return "\n".join(lines) + "\n"


def _axes(title: str, size=(7.5, 4.2)):
    fig, ax = plt.subplots(figsize=size, dpi=160)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", facecolor="white")
    plt.close(fig)


def _hbars(ax, labels, values, fmt="{:,.0f}") -> None:
    positions = range(len(labels))
    ax.barh(list(positions), values, color=BLUE, height=0.55)
    ax.set_yticks(list(positions), labels)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    top = max([v for v in values if pd.notna(v)] + [1])
    for pos, value in zip(positions, values):
        if pd.notna(value):
            ax.text(value + top * 0.01, pos, fmt.format(value), va="center", fontsize=9, color=INK)
    ax.set_xlim(0, top * 1.15)


def chart_rounds(batch, out) -> None:
    limit = int(batch["max_rounds"].max())
    fig, ax = _axes(f"Rounds used per scenario, limit {limit} (measured)")
    used = batch["rounds_used"].dropna().astype(int)
    # Runs past the limit stay on the chart, because hiding them would hide a coordinator bug.
    top = max(limit, int(used.max()) if len(used) else limit)
    counts = used.value_counts().reindex(range(1, top + 1), fill_value=0)
    ax.bar([str(i) for i in counts.index], counts.values, color=BLUE, width=0.55)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xlabel("Rounds", fontsize=9, color=MUTED)
    ax.set_ylabel("Scenarios", fontsize=9, color=MUTED)
    _save(fig, out, "rounds_histogram")


def chart_outcomes(batch, out) -> None:
    fig, ax = _axes("Outcome mix (measured)", (7.5, 3.2))
    shares = batch["outcome"].value_counts(normalize=True).reindex(OUTCOMES, fill_value=0) * 100
    _hbars(ax, OUTCOMES, shares.tolist(), "{:.1f}%")
    ax.set_xlabel("Share of scenarios (%)", fontsize=9, color=MUTED)
    _save(fig, out, "outcome_mix")


def chart_baselines(table, out) -> None:
    """Stacked per-agent cost, so a plan that just moves cost from one party to another is visible."""
    rows = table[table["plan"] != PLANS["sp"]].reset_index(drop=True)
    fig, ax = _axes(f"Mean cost per agent, {int(rows['common_scenarios'].iloc[0]):,} scenarios where every plan is "
                    "legal\n(modelled, cost constants mostly ASSUMED)", (7.5, 3.6))
    left = pd.Series(0.0, index=rows.index)
    for role, color in zip(ROLES, (BLUE, ORANGE, GREEN)):
        values = rows[f"mean_cost_{role}_usd"].fillna(0)
        ax.barh(rows.index, values, left=left, color=color, height=0.55, label=role)
        left += values
    for pos, total in zip(rows.index, left):
        ax.text(total * 1.01, pos, f"${total:,.0f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(rows.index), rows["plan"].tolist())
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlim(0, max(float(left.max()), 1) * 1.18)
    ax.set_xlabel("USD-equivalent", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=3)
    _save(fig, out, "baseline_comparison")


def chart_constraints(table, out) -> None:
    grouped = table.groupby("constraint_id")["failures"].sum().sort_values(ascending=False)
    fig, ax = _axes("Unsafe proposals caught, by constraint (measured)", (7.5, max(2.6, 0.4 * len(grouped) + 1.2)))
    _hbars(ax, grouped.index.tolist(), grouped.tolist())
    ax.set_xlabel("Failed checks", fontsize=9, color=MUTED)
    _save(fig, out, "constraint_failures")


def chart_tornado(table, out) -> None:
    run = table[(table["status"] == "OK") & (table["parameter"] != "BASELINE")].copy()
    if run.empty:
        return
    run["changed"] = (1 - run["decision_stability"]) * 100
    order = run.groupby("parameter")["changed"].max().sort_values(ascending=False).index.tolist()
    fig, ax = _axes("Decisions that change under a parameter shift (modelled)", (7.5, max(2.6, 0.5 * len(order) + 1.4)))
    for factor, color, sign, label in ((0.7, BLUE, -1, "-30%"), (1.3, ORANGE, 1, "+30%")):
        part = run[run["factor"] == factor].set_index("parameter").reindex(order)["changed"].fillna(0)
        ax.barh(range(len(order)), sign * part.values, color=color, height=0.55, label=label)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.set_major_formatter(lambda value, _: f"{abs(value):.0f}%")
    ax.set_xlabel("Share of decisions changed", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    _save(fig, out, "sensitivity_tornado")


def chart_cascade(points, validation, out) -> None:
    fig, ax = _axes("Next-leg departure delay: predicted vs recorded (measured)", (6.2, 5.2))
    limit = float(points["next_departure_delay"].quantile(0.995))
    ax.scatter(points["predicted_departure_delay"], points["next_departure_delay"],
               s=8, color=BLUE, alpha=0.18, linewidths=0)
    ax.plot([0, limit], [0, limit], color=MUTED, linewidth=1)
    ax.text(limit * 0.98, limit * 0.9, "perfect prediction", ha="right", fontsize=9, color=MUTED)
    best = validation[(validation["target"] == "next_departure_delay")
                      & validation["predictor"].str.startswith("model_fitted")].iloc[0]
    ax.text(0.02, 0.97, f"Held-out months: MAE {best['mae_min']:.1f} min, r = {best['pearson_r']:.2f}\n"
            "Only delayed legs are in the file", transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel("Predicted delay (min)", fontsize=9, color=MUTED)
    ax.set_ylabel("Recorded delay (min)", fontsize=9, color=MUTED)
    _save(fig, out, "cascade_scatter")


def build_e4(batch: pd.DataFrame, results: Path) -> str:
    notes = (SYNTHETIC_NOTE, COST_NOTE)
    baselines = summarise_baselines(batch)
    table, failures = headline(batch), do_nothing_failures(batch)
    write_results(per_scenario_costs(batch), "e4_per_scenario", notes, results)
    write_results(baselines, "e4_baselines", notes, results)
    write_results(table, "e4_headline", notes, results)
    write_results(failures, "e4_do_nothing_failures", (SYNTHETIC_NOTE,), results)
    text = slide_text(table, failures, int(batch["max_rounds"].max()))
    (results / "e4_slide.md").write_text(text, encoding="utf-8")
    if len(baselines) and baselines["common_scenarios"].iloc[0] > 0:
        chart_baselines(baselines, results)
    return text


def build_report(results_dir: Path | str = RESULTS_DIR, batches: tuple[str, ...] = ("batch_rule", "batch_llm")) -> list[str]:
    """Returns the experiments skipped because their CSV is missing."""
    results = Path(results_dir)
    skipped = []

    def have(name: str) -> bool:
        exists = (results / f"{name}.csv").exists()
        if not exists:
            skipped.append(name)
        return exists

    frames = {name: read_results(name, results) for name in batches if have(name)}
    if frames:
        batch = pd.concat(frames.values(), ignore_index=True)
        write_results(summarise_negotiation(batch), "e3_negotiation", (), results)
        build_e4(batch, results)
        chart_rounds(batch, results)
        chart_outcomes(batch, results)
    if have("e1_failures_by_constraint"):
        table = read_results("e1_failures_by_constraint", results)
        if len(table):
            chart_constraints(table, results)
    if have("e6_sensitivity"):
        chart_tornado(read_results("e6_sensitivity", results), results)
    if have("e5_points") and have("e5_validation"):
        chart_cascade(read_results("e5_points", results), read_results("e5_validation", results), results)
    return skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    skipped = build_report()
    print("Report built." + (f" Not run yet: {', '.join(skipped)}" if skipped else ""))
    slide = RESULTS_DIR / "e4_slide.md"
    if slide.exists():
        print(slide.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
