import pandas as pd
import pytest

from cdm import cascade


def leg(tail, day, origin, dest, dep, arr, block, arr_delay=30, dep_delay=20, late=25, flight=1):
    return {"TAIL_NUMBER": tail, "MONTH": 1, "DAY": day, "AIRLINE": "AA", "FLIGHT_NUMBER": flight,
            "ORIGIN_AIRPORT": origin, "DESTINATION_AIRPORT": dest, "SCHEDULED_DEPARTURE": dep,
            "SCHEDULED_ARRIVAL": arr, "SCHEDULED_TIME": block, "ARRIVAL_DELAY": arr_delay,
            "DEPARTURE_DELAY": dep_delay, "LATE_AIRCRAFT_DELAY": late}


def test_propagate():
    assert cascade.propagate(60, 20) == 40
    assert cascade.propagate(10, 20) == 0
    assert cascade.propagate(10, -5) == 15


def test_time_parsing_handles_2400_and_leading_zeros():
    assert cascade.hhmm_to_min(pd.Series(["0510", "2400", "0001"])).tolist() == [310, 1440, 1]


def test_arrival_crossing_midnight_gives_positive_gap():
    df = pd.DataFrame([leg("N1", 1, "AAA", "BBB", "2200", "0100", 180),
                       leg("N1", 1, "BBB", "CCC", "2359", "0200", 120)])
    pairs, stats = cascade.chain_legs(df)
    assert stats["dropped_negative_gap"] == 1 and pairs.empty  # lands 01:00 next day, next leg left before


def test_westbound_timezone_early_clock_is_not_overnight():
    # Departs 00:30 east, arrives 00:15 west the same day after 165 min. Clock arrival < departure.
    df = pd.DataFrame([leg("N1", 1, "AAA", "BBB", "0030", "0015", 165),
                       leg("N1", 1, "BBB", "CCC", "0100", "0300", 120)])
    pairs, _ = cascade.chain_legs(df)
    assert pairs["sched_gap_min"].tolist() == [45]


def test_chain_rejects_non_connecting_legs_and_other_days():
    df = pd.DataFrame([leg("N1", 1, "AAA", "BBB", "0800", "0900", 60),
                       leg("N1", 1, "ZZZ", "CCC", "1000", "1100", 60),
                       leg("N1", 2, "CCC", "DDD", "0800", "0900", 60)])
    pairs, stats = cascade.chain_legs(df)
    assert pairs.empty
    assert stats["consecutive_same_day_legs"] == 1 and stats["continuity_rate"] == 0


def test_next_leg_matches_chain():
    df = pd.DataFrame([leg("N1", 1, "AAA", "BBB", "0800", "0900", 60, flight=1),
                       leg("N1", 1, "BBB", "CCC", "1000", "1100", 60, flight=2),
                       leg("N1", 1, "ZZZ", "DDD", "1200", "1300", 60, flight=3)])
    assert cascade.next_leg(df, df.iloc[0])["FLIGHT_NUMBER"] == 2
    assert cascade.next_leg(df, df.iloc[1]) is None
    assert cascade.next_leg(df, df.iloc[2]) is None


def test_fit_recovers_known_min_turn():
    pairs = pd.DataFrame({"arrival_delay": [100.0, 60.0, 90.0], "sched_gap_min": [50.0, 40.0, 70.0]})
    pairs["target"] = (pairs["arrival_delay"] - (pairs["sched_gap_min"] - 30)).clip(lower=0)
    assert cascade.fit_min_turn(pairs, "target") == 30


def test_real_data_reproduces_plan_figures(flights):
    pairs, stats = cascade.chain_legs(flights)
    assert stats["consecutive_same_day_legs"] == 152241
    assert stats["continuity_rate"] == pytest.approx(0.951, abs=0.001)
    assert (pairs["sched_gap_min"] >= 0).all()
    validation = cascade.validate_model(pairs)
    model = validation[validation["predictor"].str.startswith("model_fitted")]
    naive = validation[validation["predictor"] == "naive_inherit_full_delay"]
    assert (model["mae_min"].values < naive["mae_min"].values).all()
    assert set(validation["label"]) == {"measured"}
