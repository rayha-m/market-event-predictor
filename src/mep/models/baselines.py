from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, r2_score

PROC = Path("data/processed")


def split(df: pd.DataFrame):
    year = df["ts_et"].dt.year
    return df[year <= 2021], df[year.between(2022, 2023)], df[year >= 2024]


def main():
    df = pd.read_parquet(PROC / "event_table.parquet").sort_values(["ticker", "ts_et"])
    # each company's average absolute move over its own PRIOR events only
    df["past_abs"] = df.groupby("ticker")["y_abs"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    train, val, test = split(df)
    print(f"train {len(train)} | val {len(val)} | test {len(test)} (test not used here)\n")

    # 1) direction: always predict the training majority class
    p_up = train["y_dir"].mean()
    acc = (val["y_dir"] == round(p_up)).mean()
    ll = log_loss(val["y_dir"], np.full(len(val), p_up))
    print(f"DIRECTION  always-up accuracy {acc:.3f} | constant-p log-loss {ll:.4f}")

    # 2) signed size: always predict the training mean
    mu = train["abn_ret"].mean()
    pred = np.full(len(val), mu)
    print(f"SIGNED     MAE {mean_absolute_error(val['abn_ret'], pred):.4f} "
          f"| R2 {r2_score(val['abn_ret'], pred):.4f}")

    # 3) size of move: global mean vs the company's own history
    g = train["y_abs"].mean()
    global_pred = np.full(len(val), g)
    company_pred = val["past_abs"].fillna(g)
    print(f"SIZE global-mean   MAE {mean_absolute_error(val['y_abs'], global_pred):.4f} "
          f"| R2 {r2_score(val['y_abs'], global_pred):.4f}")
    print(f"SIZE company-hist  MAE {mean_absolute_error(val['y_abs'], company_pred):.4f} "
          f"| R2 {r2_score(val['y_abs'], company_pred):.4f}")
    print(f"(val events with no company history yet: {int(val['past_abs'].isna().sum())})")


if __name__ == "__main__":
    main()