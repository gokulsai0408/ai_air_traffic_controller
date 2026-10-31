"""Part D tests run on the stubs until the real pipeline lands, then on the real thing."""
import pandas as pd
import pytest

from eval import integration

integration.use_pipeline(allow_stubs=True)


@pytest.fixture(scope="session")
def flights() -> pd.DataFrame:
    dtype = {"SCHEDULED_DEPARTURE": str, "SCHEDULED_ARRIVAL": str}
    return pd.read_csv(integration.ROOT / "data" / "flights.csv", dtype=dtype)


@pytest.fixture(scope="session")
def rows(flights) -> pd.DataFrame:
    from eval.sample import sample_rows
    return sample_rows(flights, 20, seed=2015)
