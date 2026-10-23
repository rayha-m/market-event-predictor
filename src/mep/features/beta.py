from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROC = Path("data/processed")


def rolling_beta(close: pd.DataFrame, window: int = 252, min_periods: int = 200) -> pd.DataFrame:
    """Beta of each column vs the 'SPY' column; the value at row t uses data up to t only."""
    daily = close / close.shift(1) - 1
    mkt = daily["SPY"]
    cov = daily.rolling(window, min_periods=min_periods).cov(mkt)
    var = mkt.rolling(window, min_periods=min_periods).var()
    return cov.div(var, axis=0)


def build_beta_table() -> pd.DataFrame:
    close = pd.read_parquet(RAW / "prices.parquet")["Close"]
    beta = rolling_beta(close)
    ev = pd.read_parquet(PROC / "features.parquet")
    row = close.index.get_indexer(ev["ref_date"])  # last close BEFORE the filing day
    col = close.columns.get_indexer(ev["ticker"])
    raw = beta.to_numpy()[row, col]
    ev["beta"] = 0.67 * raw + 0.33  # shrink toward 1: raw betas are noisy
    ev["abn_ret_beta"] = ev["ret"] - ev["beta"].fillna(1.0) * ev["mkt_ret"]
    ev["y_dir_beta"] = (ev["abn_ret_beta"] > 0).astype(int)
    ev.to_parquet(PROC / "features_beta.parquet")
    return ev


if __name__ == "__main__":
    t = build_beta_table()
    print(t[["beta"]].describe().round(3).to_string())
    print("share beta missing:", round(t["beta"].isna().mean(), 4))
    print("positive share, simple vs beta-adjusted:",
          round(t["y_dir"].mean(), 3), round(t["y_dir_beta"].mean(), 3))