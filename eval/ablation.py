"""E1 kernel ablation: random and adversarial proposers replace the LLMs."""
from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd

from eval import integration
from eval.results import write_results
from eval.run_batch import load_sample, run_batch


def summarise(batch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns the per-kind summary and the failure breakdown by constraint ID."""
    summary, breakdown = [], []
    for kind, group in batch.groupby("agent_kind"):
        ok = group[group["error"] == ""]
        failures: Counter = Counter()
        for cell in ok["agent_failures_by_constraint"]:
            failures.update(json.loads(cell))
        summary.append({
            "agent_kind": kind, "worlds": len(group), "crashed": len(group) - len(ok),
            "options_proposed": int(ok["agent_options"].sum()),
            "options_unsafe": int(ok["agent_options_unsafe"].sum()),
            "unsafe_accepted": int(ok["unsafe_accepted"].sum()),
            "kernel_entries_unparsed": int(ok["kernel_entries_unparsed"].sum()),
            "label": "measured",
        })
        breakdown += [{"agent_kind": kind, "constraint_id": cid, "failures": n, "label": "measured"}
                      for cid, n in sorted(failures.items())]
    columns = ["agent_kind", "constraint_id", "failures", "label"]
    return pd.DataFrame(summary), pd.DataFrame(breakdown, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    rows, cfg = load_sample(args.n), integration.load_cfg()
    batch = pd.concat([run_batch(rows, kind, cfg, range(args.seeds)) for kind in ("random", "adversarial")])
    summary, breakdown = summarise(batch)
    notes = ("unsafe_accepted = decisions failing a SAFETY or PHYSICAL constraint on re-check.",)
    write_results(summary, "e1_ablation", notes)
    write_results(breakdown, "e1_failures_by_constraint", notes)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
