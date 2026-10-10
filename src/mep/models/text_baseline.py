import gzip
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error, r2_score, roc_auc_score

PROC = Path("data/processed")
TEXTS = Path("data/raw/press_releases")


def clean(text: str) -> str:
    text = re.sub(r"^ex-99\S*\s+\d+\s+\S+\s*", "", text, flags=re.I)  # filing header
    text = text.lower()
    return re.sub(r"\d", "0", text)  # 4.2 -> 0.0, so exact figures and years can't be memorized


def load_texts(accessions) -> dict:
    out = {}
    for a in accessions:
        p = TEXTS / f"{a}.txt.gz"
        if p.exists():
            try:
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    out[a] = clean(f.read())
            except (EOFError, OSError):
                pass  # a file still being written by the downloader
    return out


def main():
    df = pd.read_parquet(PROC / "features.parquet")
    texts = load_texts(df["accessionNumber"].unique())
    df = df[df["accessionNumber"].isin(texts)].copy()
    df["text"] = df["accessionNumber"].map(texts)
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    print(f"events with text: {len(df)} | train {len(train)} | val {len(val)} (test not used)")
    print("sample after cleaning:", repr(train["text"].iloc[0][:200]), "\n")

    # vocabulary and word weights come from TRAIN ONLY
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000,
                          sublinear_tf=True, stop_words="english")
    Xtr = vec.fit_transform(train["text"])
    Xva = vec.transform(val["text"])
    one_day = (val["window_days"] == 1).to_numpy()

    # ---- direction ----
    p0 = train["y_dir"].mean()
    print(f"DIRECTION baseline always-up: acc {(val['y_dir'] == round(p0)).mean():.3f} "
          f"| log-loss {log_loss(val['y_dir'], [p0] * len(val)):.4f}")
    for C in (0.1, 1, 10):
        m = LogisticRegression(C=C, max_iter=2000).fit(Xtr, train["y_dir"])
        p = m.predict_proba(Xva)[:, 1]
        print(f"DIRECTION text C={C:<4}: acc {((p > 0.5) == val['y_dir']).mean():.3f} "
              f"| log-loss {log_loss(val['y_dir'], p):.4f} "
              f"| AUC {roc_auc_score(val['y_dir'], p):.3f} "
              f"| AUC 1-day only {roc_auc_score(val['y_dir'][one_day], p[one_day]):.3f}")

    # ---- size of move: can text improve on the company-history baseline? ----
    mean_tr = train["y_abs"].mean()
    base_tr = train["past_abs"].fillna(mean_tr)
    base_va = val["past_abs"].fillna(mean_tr)
    print(f"\nSIZE baseline company-history: MAE "
          f"{mean_absolute_error(val['y_abs'], base_va):.4f} "
          f"| R2 {r2_score(val['y_abs'], base_va):.4f}")
    for alpha in (0.1, 1, 10):
        r = Ridge(alpha=alpha).fit(Xtr, train["y_abs"] - base_tr)
        pred = base_va + r.predict(Xva)
        print(f"SIZE baseline + text residual a={alpha:<4}: MAE "
              f"{mean_absolute_error(val['y_abs'], pred):.4f} "
              f"| R2 {r2_score(val['y_abs'], pred):.4f}")


if __name__ == "__main__":
    main()