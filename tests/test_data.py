import pandas as pd
import pytest

from cdm.data import load_flights, parse_hhmm
from helpers import FIXTURES, sample_cfg, sample_flights


@pytest.mark.parametrize("value,expected", [
    ("0510", 310), (659, 419), ("659.0", 419), ("2400", 1440), ("0000", 0), ("5", 5), (None, None), ("", None),
])
def test_parse_hhmm(value, expected):
    assert parse_hhmm(value) == expected


@pytest.mark.parametrize("value", ["2401", "2500", "1260"])
def test_parse_hhmm_rejects_bad_times(value):
    with pytest.raises(ValueError):
        parse_hhmm(value)


def test_leading_zero_kept_as_string():
    df = sample_flights()
    assert df.loc[df["FLIGHT_NUMBER"] == 4321, "SCHEDULED_DEPARTURE"].iloc[0] == "0510"


def test_overnight_arrival_is_next_day():
    row = sample_flights().query("FLIGHT_NUMBER == 88").iloc[0]
    assert row["SCHED_ARR_DT"] == pd.Timestamp("2015-07-04 04:50")


def test_2400_arrival_is_midnight():
    row = sample_flights().query("FLIGHT_NUMBER == 700").iloc[0]
    assert row["SCHED_ARR_DT"] == pd.Timestamp("2015-08-21 00:00")


def test_next_leg_same_tail_from_destination():
    df = sample_flights()
    first = df.query("FLIGHT_NUMBER == 1234").iloc[0]
    assert first["NEXT_SCHED_DEP_DT"] == pd.Timestamp("2015-06-12 21:20")
    assert pd.isna(df.query("FLIGHT_NUMBER == 1500").iloc[0]["NEXT_SCHED_DEP_DT"])


def _with_scheduled_times(tmp_path, changes):
    df = pd.read_csv(FIXTURES / "flights_sample.csv", dtype=str)
    df["SCHEDULED_TIME"] = "120"
    for number, values in changes.items():
        for column, value in values.items():
            df.loc[df["FLIGHT_NUMBER"] == number, column] = value
    path = tmp_path / "flights.csv"
    df.to_csv(path, index=False)
    return load_flights(path)


def test_short_westbound_flight_stays_on_same_day(tmp_path):
    df = _with_scheduled_times(tmp_path, {"1234": {"SCHEDULED_DEPARTURE": "2000", "SCHEDULED_ARRIVAL": "1950", "SCHEDULED_TIME": "50"}})
    assert df.query("FLIGHT_NUMBER == 1234").iloc[0]["SCHED_ARR_DT"] == pd.Timestamp("2015-06-12 19:50")


def test_scheduled_time_keeps_real_overnight_arrival(tmp_path):
    df = _with_scheduled_times(tmp_path, {"88": {"SCHEDULED_TIME": "365"}})
    assert df.query("FLIGHT_NUMBER == 88").iloc[0]["SCHED_ARR_DT"] == pd.Timestamp("2015-07-04 04:50")


def test_next_leg_before_arrival_is_not_a_next_leg(tmp_path):
    df = _with_scheduled_times(tmp_path, {"1500": {"SCHEDULED_DEPARTURE": "1940"}})
    assert pd.isna(df.query("FLIGHT_NUMBER == 1234").iloc[0]["NEXT_SCHED_DEP_DT"])


def test_flight_key_has_contract_name():
    df = sample_flights()
    assert df["flight_key"].equals(df["FLIGHT_KEY"])


def test_flight_keys_unique():
    assert sample_flights()["FLIGHT_KEY"].is_unique


def test_airports_without_coordinates_dropped():
    assert "ECP" not in set(sample_cfg().airports["IATA_CODE"])
