"""`python -m eval.smoke`: main must always pass this.

Runs one real scenario end to end (A's world, B's kernel, C's coordinator, D's baselines), saves its
ledger, checks the SCENARIO entry carries the round cap from config/protocol.yaml, and runs verify.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from cdm import cascade
from eval import integration
from eval.run_batch import run_scenario
from eval.sample import sample_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # verify's summary uses "·", which cp1252 consoles can't print

    flights = integration.load_flights()
    print(f"[ok] data: {len(flights):,} rows")

    _, stats = cascade.chain_legs(cascade.load_legs())
    if not 0.94 <= stats["continuity_rate"] <= 0.96:
        print(f"[FAIL] cascade continuity {stats['continuity_rate']:.3f}, expected about 0.951")
        return 1
    print(f"[ok] cascade: {stats['continuity_rate']:.1%} continuity, {stats['chained_pairs']:,} chained pairs")

    sample = sample_rows(flights, 10, seed=2015)
    print(f"[ok] sample: {len(sample)} rows")

    cfg = integration.load_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        record = run_scenario(sample.iloc[0], args.seed, "rule", cfg, Path(tmp))
        if record["unsafe_accepted"]:
            print("[FAIL] the decision failed the kernel on re-check")
            return 1
        print(f"[ok] scenario {record['flight_key']}: {record['outcome']} in {record['rounds_used']} round(s)")
        if record["logged_max_rounds"] != cfg.max_rounds:
            print(f"[FAIL] ledger SCENARIO max_rounds {record['logged_max_rounds']}, "
                  f"config/protocol.yaml says {cfg.max_rounds}")
            return 1
        print(f"[ok] ledger SCENARIO max_rounds = {record['logged_max_rounds']} (config/protocol.yaml)")
        from cdm.verify import verify
        report = verify(record["ledger"])
        if not report.ok:
            print(f"[FAIL] verify: {report.message}")
            return 1
        print(f"[ok] verify: {report.message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
