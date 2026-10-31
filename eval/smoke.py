"""`python -m eval.smoke`: main must always pass this.

Cascade and sampling always run on the real file. The scenario uses the real pipeline when
A, B and C have landed, otherwise the stubs, and says which. --strict fails on stubs.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

from cdm import cascade
from eval import integration
from eval.run_batch import run_scenario
from eval.sample import sample_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail unless the real pipeline ran")
    args = parser.parse_args()

    dtype = {"SCHEDULED_DEPARTURE": str, "SCHEDULED_ARRIVAL": str}
    flights = pd.read_csv(integration.ROOT / "data" / "flights.csv", dtype=dtype)
    print(f"[ok] data: {len(flights):,} rows")

    _, stats = cascade.chain_legs(flights)
    if not 0.94 <= stats["continuity_rate"] <= 0.96:
        print(f"[FAIL] cascade continuity {stats['continuity_rate']:.3f}, expected about 0.951")
        return 1
    print(f"[ok] cascade: {stats['continuity_rate']:.1%} continuity, {stats['chained_pairs']:,} chained pairs")

    sample = sample_rows(flights, 10, seed=2015)
    print(f"[ok] sample: {len(sample)} rows")

    missing = integration.missing_modules()
    if args.strict and missing:
        print(f"[FAIL] --strict: not landed yet: {', '.join(missing)}")
        return 1
    stubbed = integration.use_pipeline(allow_stubs=True)
    if stubbed:
        print(f"[STUB] not landed, using stubs for: {', '.join(missing)}")
    with tempfile.TemporaryDirectory() as tmp:
        record = run_scenario(sample.iloc[0], 0, "rule", integration.load_cfg(), Path(tmp))
    if record["unsafe_accepted"]:
        print("[FAIL] the decision failed the kernel on re-check")
        return 1
    label = "STUB" if stubbed else "ok"
    print(f"[{label}] scenario {record['flight_key']}: {record['outcome']} in {record['rounds_used']} round(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
