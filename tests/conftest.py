"""Part D tests run on the real pipeline (A's world, B's kernel, C's coordinator)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so bare `pytest` finds eval/ and cdm/

from eval import integration  # noqa: E402


@pytest.fixture(scope="session")
def flights() -> pd.DataFrame:
    return integration.load_flights()


@pytest.fixture(scope="session")
def rows(flights) -> pd.DataFrame:
    from eval.sample import sample_rows
    return sample_rows(flights, 20, seed=2015)


@pytest.fixture(scope="session")
def cfg():
    return integration.load_cfg()


@pytest.fixture(scope="session")
def saved_run(rows, cfg, tmp_path_factory) -> dict:
    """One real negotiation with its ledger on disk, for the dashboard and verify tests."""
    from eval.run_batch import run_scenario
    return run_scenario(rows.iloc[0], 0, "rule", cfg, tmp_path_factory.mktemp("ledgers"))
