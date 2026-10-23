import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from mep.evaluation import block_bootstrap, week_blocks
from mep.models.text_baseline import PROC, load_texts

TARGETS = ["y_dir", "y_dir_beta"]

df = pd.read_parquet(PROC / "features_beta.parquet")
df = df[df["split"] != "test"]  # 2024+ stays untouched
texts = load_texts(df["accessionNumber"].unique())
df = df[df["accessionNumber"].isin(texts)].copy()
df["text"] = df["accessionNumber"].map(texts)
df["year"] = df["ts_et"].dt.year

folds = []
print(f"{'year':6s}" + "".join(f"{t + ' text':>18s}{t + ' tick':>18s}" for t in TARGETS))
for year in range(2019, 2024):
    tr, te = df[df["year"] < year], df[df["year"] == year].copy()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000,
                          sublinear_tf=True, stop_words="english")
    Xtr, Xte = vec.fit_transform(tr["text"]), vec.transform(te["text"])
    enc = OneHotEncoder(handle_unknown="ignore")
    Ttr, Tte = enc.fit_transform(tr[["ticker"]]), enc.transform(te[["ticker"]])
    line = f"{year:<6d}"
    for tg in TARGETS:
        te[f"pt_{tg}"] = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, tr[tg]).predict_proba(Xte)[:, 1]
        te[f"pk_{tg}"] = LogisticRegression(C=0.1, max_iter=2000).fit(Ttr, tr[tg]).predict_proba(Tte)[:, 1]
        line += f"{roc_auc_score(te[tg], te[f'pt_{tg}']):>18.3f}{roc_auc_score(te[tg], te[f'pk_{tg}']):>18.3f}"
    print(line)
    folds.append(te)

pool = pd.concat(folds, ignore_index=True)
blocks = week_blocks(pool["ts_et"])


def yearly_mean(d, fn, lo, hi):
    d = d[d["year"].between(lo, hi)]
    return float(np.mean([fn(g) for _, g in d.groupby("year")]))


def show(name, res):
    pt, lo, hi = res
    print(f"{name:58s} {pt:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")


for tg in TARGETS:
    def text_auc(g, tg=tg): return roc_auc_score(g[tg], g[f"pt_{tg}"]) - 0.5
    def tick_auc(g, tg=tg): return roc_auc_score(g[tg], g[f"pk_{tg}"]) - 0.5
    def paired(g, tg=tg): return (roc_auc_score(g[tg], g[f"pt_{tg}"])
                                  - roc_auc_score(g[tg], g[f"pk_{tg}"]))
    print(f"\n=== target: {tg} ===")
    for label, lo, hi in [("2019-2021", 2019, 2021), ("2022-2023", 2022, 2023)]:
        for nm, fn in [("text AUC-0.5", text_auc), ("ticker AUC-0.5", tick_auc), ("text minus ticker", paired)]:
            show(f"{nm:18s} {label}",
                 block_bootstrap(pool, lambda d, fn=fn, lo=lo, hi=hi: yearly_mean(d, fn, lo, hi),
                                 blocks, n_boot=500))