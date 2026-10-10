import pandas as pd
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, log_loss, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROC = Path("data/processed")
FEATURES = ["ret_5d", "ret_21d", "ret_63d", "vol_21d", "vol_63d", "dist_high_252",
            "log_volume_21d", "mkt_ret_21d", "mkt_vol_21d", "rel_ret_21d",
            "past_abs", "last_abn_ret", "usual_after_close"]


def main():
    df = pd.read_parquet(PROC / "features.parquet")
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    Xtr, Xva = train[FEATURES], val[FEATURES]
    print(f"train {len(train)} | val {len(val)} (test not used)\n")

    # ---- direction ----
    p0 = train["y_dir"].mean()
    print(f"DIRECTION baseline always-up: acc "
          f"{(val['y_dir'] == round(p0)).mean():.3f} | log-loss "
          f"{log_loss(val['y_dir'], [p0] * len(val)):.4f}")
    models = {
        "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=0.1, max_iter=1000)),
        "boosted trees": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100,
            early_stopping=False, random_state=0),
    }
    for name, m in models.items():
        m.fit(Xtr, train["y_dir"])
        p = m.predict_proba(Xva)[:, 1]
        print(f"DIRECTION {name:14s}: acc {accuracy_score(val['y_dir'], p > 0.5):.3f} "
              f"| log-loss {log_loss(val['y_dir'], p):.4f} "
              f"| AUC {roc_auc_score(val['y_dir'], p):.3f}")

    # ---- size of move ----
    reg = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=100,
        early_stopping=False, random_state=0)
    reg.fit(Xtr, train["y_abs"])
    out = val.assign(model=reg.predict(Xva),
                     base=val["past_abs"].fillna(train["y_abs"].mean()))
    print(f"\nSIZE baseline company-history: MAE "
          f"{mean_absolute_error(out['y_abs'], out['base']):.4f} "
          f"| R2 {r2_score(out['y_abs'], out['base']):.4f}")
    print(f"SIZE boosted trees           : MAE "
          f"{mean_absolute_error(out['y_abs'], out['model']):.4f} "
          f"| R2 {r2_score(out['y_abs'], out['model']):.4f}")
    for w, g in out.groupby("window_days"):
        print(f"  window_days={w}: baseline MAE "
              f"{mean_absolute_error(g['y_abs'], g['base']):.4f} "
              f"| model MAE {mean_absolute_error(g['y_abs'], g['model']):.4f}")


if __name__ == "__main__":
    main()