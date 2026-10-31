"""Turn eval/results/*.csv into the E3 and E4 summary tables and the PNG charts.

Any experiment that has not run yet is skipped and named. Stub results get STUB in the
chart title, so a placeholder chart cannot reach a slide unnoticed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eval.results import RESULTS_DIR, STUB_NOTE, read_notes, read_results, write_results

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
PLANS = {"neg": "Negotiated", "dn": "Do-nothing / FCFS", "fb": "Fallback rule",
         "cs": "Central solver", "sp": "Single-prompt LLM (banned design)"}
OUTCOMES = ["CONSENSUS", "COMPROMISE", "FALLBACK", "ESCALATED"]


def summarise_negotiation(batch: pd.DataFrame) -> pd.DataFrame:
    """E3: one row per agent kind."""
    rows = []
    for kind, group in batch.groupby("agent_kind"):
        ok = group[group["error"].fillna("") == ""]
        outcomes = ok["outcome"].value_counts(normalize=True)
        rows.append({
            "agent_kind": kind, "scenarios": len(group), "crashed": len(group) - len(ok),
            **{f"{o.lower()}_rate": float(outcomes.get(o, 0.0)) for o in OUTCOMES},
            "max_rounds": int(ok["max_rounds"].max()) if len(ok) else 0,
            "agreed_within_limit_rate": float((ok["outcome"].isin(["CONSENSUS", "COMPROMISE"])
                                               & (ok["rounds_used"] <= ok["max_rounds"])).mean()),
            "unsafe_accepted": int(ok["unsafe_accepted"].fillna(False).astype(bool).sum()),
            "guardrail_overrides": int(ok["guardrail_overrides"].sum()),
            "median_latency_s": float(ok["latency_s"].median()),
            "llm_calls_per_scenario": float(ok["llm_calls"].mean()), "label": "measured",
        })
    return pd.DataFrame(rows)


def summarise_baselines(batch: pd.DataFrame) -> pd.DataFrame:
    """E4. Costs are means over `common_scenarios`, where every plan is safe and within policy.

    A plan that breaks a rule can look cheap, so including it would flatter it. The central
    solver only ever picks policy-compliant plans, so this is also what keeps it the upper bound.
    """
    ok = batch[batch["error"].fillna("") == ""]
    present = [p for p in PLANS if f"{p}_safe" in ok and ok[f"{p}_safe"].notna().any()]
    legal = pd.Series(True, index=ok.index)
    for prefix in ("neg", "dn", "fb", "cs"):
        legal &= (ok[f"{prefix}_safe"] == True) & (ok[f"{prefix}_policy_ok"] == True)  # noqa: E712
    common = ok[legal]
    rows = []
    for prefix in present:
        has_plan = ok[f"{prefix}_has_plan"] == True  # noqa: E712
        unsafe = has_plan & (ok[f"{prefix}_safe"] == False)  # noqa: E712
        rows.append({
            "plan": PLANS[prefix], "scenarios": len(ok), "with_plan": int(has_plan.sum()),
            "kernel_violation_rate": float(unsafe.sum() / has_plan.sum()) if has_plan.any() else float("nan"),
            "common_scenarios": len(common),
            "mean_total_cost_usd": float(common[f"{prefix}_total_cost"].mean()),
            "mean_gate_delay_min": float(common[f"{prefix}_gate_delay_min"].mean()),
            "mean_missed_pax": float(common[f"{prefix}_missed_pax"].mean()),
            "mean_overtime_min": float(common[f"{prefix}_overtime_min"].mean()),
            "mean_hold_fuel_kg": float(common[f"{prefix}_hold_fuel_kg"].mean()),
            "label_violation": "measured", "label_cost": "modelled, ASSUMED constants",
        })
    return pd.DataFrame(rows)


def _axes(title: str, stub: bool, size=(7.5, 4.2)):
    fig, ax = plt.subplots(figsize=size, dpi=160)
    ax.set_title(("STUB, DO NOT CITE: " if stub else "") + title, loc="left", fontsize=11, color=INK)
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


def chart_rounds(batch, out, stub) -> None:
    limit = int(batch["max_rounds"].max())
    fig, ax = _axes(f"Rounds used per scenario, limit {limit} (measured)", stub)
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


def chart_outcomes(batch, out, stub) -> None:
    fig, ax = _axes("Outcome mix (measured)", stub, (7.5, 3.2))
    shares = batch["outcome"].value_counts(normalize=True).reindex(OUTCOMES, fill_value=0) * 100
    _hbars(ax, OUTCOMES, shares.tolist(), "{:.1f}%")
    ax.set_xlabel("Share of scenarios (%)", fontsize=9, color=MUTED)
    _save(fig, out, "outcome_mix")


def chart_baselines(table, out, stub) -> None:
    fig, ax = _axes(f"Mean total cost, {int(table['common_scenarios'].iloc[0]):,} common scenarios\n"
                    "(modelled, ASSUMED constants)", stub, (7.5, 3.6))
    _hbars(ax, table["plan"].tolist(), table["mean_total_cost_usd"].tolist(), "${:,.0f}")
    ax.set_xlabel("USD-equivalent", fontsize=9, color=MUTED)
    _save(fig, out, "baseline_comparison")


def chart_constraints(table, out, stub) -> None:
    grouped = table.groupby("constraint_id")["failures"].sum().sort_values(ascending=False)
    fig, ax = _axes("Unsafe proposals caught, by constraint (measured)", stub,
                    (7.5, max(2.6, 0.4 * len(grouped) + 1.2)))
    _hbars(ax, grouped.index.tolist(), grouped.tolist())
    ax.set_xlabel("Failed checks", fontsize=9, color=MUTED)
    _save(fig, out, "constraint_failures")


def chart_tornado(table, out, stub) -> None:
    run = table[(table["status"] == "OK") & (table["parameter"] != "BASELINE")].copy()
    if run.empty:
        return
    run["changed"] = (1 - run["decision_stability"]) * 100
    order = run.groupby("parameter")["changed"].max().sort_values(ascending=False).index.tolist()
    fig, ax = _axes("Decisions that change under a parameter shift (modelled)", stub,
                    (7.5, max(2.6, 0.5 * len(order) + 1.4)))
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
    fig, ax = _axes("Next-leg departure delay: predicted vs recorded (measured)", False, (6.2, 5.2))
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


def build_report(results_dir: Path | str = RESULTS_DIR, batches: tuple[str, ...] = ("batch_rule", "batch_llm")) -> list[str]:
    """Returns the experiments skipped because their CSV is missing."""
    results = Path(results_dir)
    skipped = []

    def have(name: str) -> bool:
        exists = (results / f"{name}.csv").exists()
        if not exists:
            skipped.append(name)
        return exists

    def stub(name: str) -> bool:
        return STUB_NOTE in read_notes(name, results)

    frames = {name: read_results(name, results) for name in batches if have(name)}
    if frames:
        batch = pd.concat(frames.values(), ignore_index=True)
        is_stub = any(stub(name) for name in frames)
        notes = (STUB_NOTE,) if is_stub else ()
        write_results(summarise_negotiation(batch), "e3_negotiation", notes, results)
        baselines = summarise_baselines(batch)
        write_results(baselines, "e4_baselines", notes, results)
        privacy = batch["price_of_privacy_usd"].dropna()
        if len(privacy):
            write_results(pd.DataFrame([{
                "scenarios": len(privacy), "central_strictly_better_rate": float((privacy > 1e-6).mean()),
                "mean_gap_usd": float(privacy.mean()), "median_gap_usd": float(privacy.median()),
                "bound_violations": int((privacy < -1e-6).sum()), "label": "modelled, ASSUMED constants",
            }]), "e4_price_of_privacy", notes, results)
        chart_rounds(batch, results, is_stub)
        chart_outcomes(batch, results, is_stub)
        chart_baselines(baselines, results, is_stub)
    if have("e1_failures_by_constraint"):
        table = read_results("e1_failures_by_constraint", results)
        if len(table):
            chart_constraints(table, results, stub("e1_failures_by_constraint"))
    if have("e6_sensitivity"):
        chart_tornado(read_results("e6_sensitivity", results), results, stub("e6_sensitivity"))
    if have("e5_points") and have("e5_validation"):
        chart_cascade(read_results("e5_points", results), read_results("e5_validation", results), results)
    return skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    skipped = build_report()
    print("Report built." + (f" Not run yet: {', '.join(skipped)}" if skipped else ""))


if __name__ == "__main__":
    main()
