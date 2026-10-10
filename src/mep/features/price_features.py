from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROC = Path("data/processed")


def price_features(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each feature at row t uses data up to and including row t only."""
    daily = close / close.shift(1) - 1
    return {
        "ret_5d": close / close.shift(5) - 1,
        "ret_21d": close / close.shift(21) - 1,
        "ret_63d": close / close.shift(63) - 1,
        "vol_21d": daily.rolling(21, min_periods=15).std(),
        "vol_63d": daily.rolling(63, min_periods=45).std(),
        "dist_high_252": close / close.rolling(252, min_periods=200).max() - 1,
        "log_volume_21d": np.log1p(volume.rolling(21, min_periods=15).mean()),
    }


def build_feature_table() -> pd.DataFrame:
    px = pd.read_parquet(RAW / "prices.parquet")
    close, volume = px["Close"], px["Volume"]
    ev = (pd.read_parquet(PROC / "event_table.parquet")
          .sort_values(["ticker", "ts_et"]).reset_index(drop=True))

    # look up each feature at (day before filing, ticker)
    row = close.index.get_indexer(ev["ref_date"])
    col = close.columns.get_indexer(ev["ticker"])
    for name, mat in price_features(close, volume).items():
        ev[name] = mat.to_numpy()[row, col]

    # market-wide context from SPY on the same day
    spy = price_features(close[["SPY"]], volume[["SPY"]])
    ev["mkt_ret_21d"] = spy["ret_21d"].to_numpy()[row, 0]
    ev["mkt_vol_21d"] = spy["vol_21d"].to_numpy()[row, 0]
    ev["rel_ret_21d"] = ev["ret_21d"] - ev["mkt_ret_21d"]

    # company history from PRIOR events only (shift(1) excludes the current event)
    g = ev.groupby("ticker")
    ev["past_n"] = g.cumcount()
    ev["past_abs"] = g["y_abs"].transform(lambda s: s.expanding().mean().shift(1))
    ev["last_abn_ret"] = g["abn_ret"].shift(1)
    ev["usual_after_close"] = g["window_days"].transform(
        lambda s: (s > 1).astype(float).expanding().mean().shift(1)
    )

    year = ev["ts_et"].dt.year
    ev["split"] = np.where(year <= 2021, "train", np.where(year <= 2023, "val", "test"))
    ev.to_parquet(PROC / "features.parquet")
    return ev


if __name__ == "__main__":
    t = build_feature_table()
    print(t["split"].value_counts().to_dict())
    feats = ["ret_5d", "ret_21d", "ret_63d", "vol_21d", "vol_63d", "dist_high_252",
             "log_volume_21d", "mkt_ret_21d", "mkt_vol_21d", "rel_ret_21d",
             "past_abs", "last_abn_ret", "usual_after_close"]
    print("share of events missing each feature:")
    print(t[feats].isna().mean().round(3).to_string())