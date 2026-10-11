import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from mep.evaluation import block_bootstrap, week_blocks
from mep.models.text_baseline import PROC, load_texts

df = pd.read_parquet(PROC / "features.parquet")
df = df[df["split"] != "test"]  # 2024+ stays untouched
texts = load_texts(df["accessionNumber"].unique())
df = df[df["accessionNumber"].isin(texts)].copy()
df["text"] = df["accessionNumber"].map(texts)
df["year"] = df["ts_et"].dt.year

folds = []
print(f"{'year':6s}{'train':>7s}{'test':>7s}{'text AUC':>10s}{'ticker AUC':>12s}")
for year in range(2019, 2024):
    tr, te = df[df["year"] < year], df[df["year"] == year].copy()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000,
                          sublinear_tf=True, stop_words="english")
    Xtr, Xte = vec.fit_transform(tr["text"]), vec.transform(te["text"])
    te["p_text"] = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, tr["y_dir"]).predict_proba(Xte)[:, 1]
    enc = OneHotEncoder(handle_unknown="ignore")
    Ttr, Tte = enc.fit_transform(tr[["ticker"]]), enc.transform(te[["ticker"]])
    te["p_tick"] = LogisticRegression(C=0.1, max_iter=2000).fit(Ttr, tr["y_dir"]).predict_proba(Tte)[:, 1]
    print(f"{year:<6d}{len(tr):>7d}{len(te):>7d}"
          f"{roc_auc_score(te['y_dir'], te['p_text']):>10.3f}"
          f"{roc_auc_score(te['y_dir'], te['p_tick']):>12.3f}")
    folds.append(te)

pool = pd.concat(folds, ignore_index=True)
blocks = week_blocks(pool["ts_et"])


def yearly_mean(d, fn, lo, hi):
    d = d[d["year"].between(lo, hi)]
    return float(np.mean([fn(g) for _, g in d.groupby("year")]))


def text_auc(g):
    return roc_auc_score(g["y_dir"], g["p_text"]) - 0.5


def paired(g):
    return roc_auc_score(g["y_dir"], g["p_text"]) - roc_auc_score(g["y_dir"], g["p_tick"])


def show(name, res):
    pt, lo, hi = res
    print(f"{name:50s} {pt:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")


print("\naverage across years, week-block bootstrap")
for label, lo, hi in [("2019-2021 (fresh)", 2019, 2021),
                      ("2022-2023 (already seen)", 2022, 2023),
                      ("all five years", 2019, 2023)]:
    show(f"text AUC minus 0.5   {label}",
         block_bootstrap(pool, lambda d: yearly_mean(d, text_auc, lo, hi), blocks, n_boot=500))
    show(f"text minus ticker    {label}",
         block_bootstrap(pool, lambda d: yearly_mean(d, paired, lo, hi), blocks, n_boot=500))