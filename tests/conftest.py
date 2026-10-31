"""Part D tests run on the stubs until the real pipeline lands, then on the real thing."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so bare `pytest` finds eval/ and cdm/

from eval import integration  # noqa: E402

integration.use_pipeline(allow_stubs=True)


@pytest.fixture(scope="session")
def flights() -> pd.DataFrame:
    dtype = {"SCHEDULED_DEPARTURE": str, "SCHEDULED_ARRIVAL": str}
    return pd.read_csv(integration.ROOT / "data" / "flights.csv", dtype=dtype)


@pytest.fixture(scope="session")
def rows(flights) -> pd.DataFrame:
    from eval.sample import sample_rows
    return sample_rows(flights, 20, seed=2015)
