import pandas as pd

from mep.data.event_table import window_positions

DATES = pd.bdate_range("2024-01-01", "2024-01-31")  # weekdays only


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="America/New_York")


def test_premarket_filing_uses_prior_close_to_same_day_close():
    ref, end = window_positions(ts("2024-01-10 08:00"), DATES)
    assert DATES[ref] == pd.Timestamp("2024-01-09")
    assert DATES[end] == pd.Timestamp("2024-01-10")


def test_after_close_filing_extends_to_next_day():
    ref, end = window_positions(ts("2024-01-10 17:00"), DATES)
    assert DATES[ref] == pd.Timestamp("2024-01-09")
    assert DATES[end] == pd.Timestamp("2024-01-11")


def test_saturday_filing_rolls_to_monday():
    ref, end = window_positions(ts("2024-01-13 10:00"), DATES)
    assert DATES[ref] == pd.Timestamp("2024-01-12")
    assert DATES[end] == pd.Timestamp("2024-01-15")


def test_reference_date_is_always_before_filing_day():
    for stamp in pd.date_range("2024-01-02", "2024-01-30", freq="3h"):
        t = ts(str(stamp))
        out = window_positions(t, DATES)
        if out is None:
            continue
        ref, end = out
        assert DATES[ref] < pd.Timestamp(t.year, t.month, t.day)
        assert end > ref