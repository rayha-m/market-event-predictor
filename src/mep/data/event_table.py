from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROC = Path("data/processed")
EXCLUDE = {"ECHO"}  # stopped announcing earnings via Item 2.02 8-K after 2017
MIN_HISTORY = 252  # trading days of prior prices required (about 1 year)


def window_positions(ts_et: pd.Timestamp, dates: pd.DatetimeIndex):
    """Return (ref_pos, end_pos): row positions in the trading calendar.

    ref_pos is the last trading day strictly BEFORE the filing day, so all
    features may only use prices up to and including it.
    end_pos is where the reaction window ends:
      - filed before 4pm on a trading day -> that same day's close
      - filed at/after 4pm                -> next trading day's close (safe side)
      - filed on a weekend/holiday        -> the next trading day's close
    """
    day = pd.Timestamp(ts_et.year, ts_et.month, ts_et.day)
    d = dates.searchsorted(day, side="left")  # first trading day >= filing day
    if d == 0 or d >= len(dates):
        return None
    is_trading_day = dates[d] == day
    after_close = is_trading_day and ts_et.hour >= 16
    end = d + (1 if after_close else 0)
    if end >= len(dates):
        return None
    return d - 1, end


def build_event_table() -> pd.DataFrame:
    close = pd.read_parquet(RAW / "prices.parquet")["Close"]
    dates = close.index
    spy = close["SPY"].to_numpy()

    ev = pd.read_parquet(RAW / "events.parquet")
    ev = ev[~ev["ticker"].isin(EXCLUDE) & ev["ticker"].isin(close.columns)].copy()
    ev["ts_et"] = pd.to_datetime(ev["acceptanceDateTime"]).dt.tz_convert("America/New_York")
    ev["year"] = ev["ts_et"].dt.year

    # drop ticker-years with fewer than 3 events (keep each ticker's latest year)
    n = ev.groupby(["ticker", "year"])["year"].transform("size")
    latest = ev.groupby("ticker")["year"].transform("max")
    ev = ev[(n >= 3) | (ev["year"] == latest)].copy()

    wins = [window_positions(t, dates) for t in ev["ts_et"]]
    ok = np.array([w is not None for w in wins])
    ev = ev[ok].copy()
    ref_pos = np.array([w[0] for w in wins if w is not None])
    end_pos = np.array([w[1] for w in wins if w is not None])

    arr = close.to_numpy()
    col = close.columns.get_indexer(ev["ticker"])
    ev["ref_date"] = dates[ref_pos]
    ev["end_date"] = dates[end_pos]
    ev["window_days"] = end_pos - ref_pos
    ev["history_days"] = close.notna().cumsum().to_numpy()[ref_pos, col]
    ev["ret"] = arr[end_pos, col] / arr[ref_pos, col] - 1
    ev["mkt_ret"] = spy[end_pos] / spy[ref_pos] - 1

    ev = ev[ev["ret"].notna() & (ev["history_days"] >= MIN_HISTORY)].copy()
    ev["abn_ret"] = ev["ret"] - ev["mkt_ret"]
    ev["y_dir"] = (ev["abn_ret"] > 0).astype(int)  # target 1: direction
    ev["y_abs"] = ev["abn_ret"].abs()  # target 3: size of move, ignoring sign
    # target 2 (signed magnitude) is abn_ret itself

    PROC.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(PROC / "event_table.parquet")
    return ev


if __name__ == "__main__":
    t = build_event_table()
    print(len(t), "events,", t["ticker"].nunique(), "tickers")
    print("share of events with positive abnormal return:", round(t["y_dir"].mean(), 3))
    print("share with a 2-day window (filed after 4pm or on weekends):",
          round((t["window_days"] > 1).mean(), 3))
    print(t[["abn_ret", "y_abs"]].describe())