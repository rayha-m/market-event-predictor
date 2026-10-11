import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from mep.evaluation import block_bootstrap, week_blocks
from mep.models.text_baseline import PROC, load_texts


def show(name, res):
    pt, lo, hi = res
    print(f"{name:52s} {pt:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")


df = pd.read_parquet(PROC / "features.parquet")
texts = load_texts(df["accessionNumber"].unique())
df = df[df["accessionNumber"].isin(texts)].copy()
df["text"] = df["accessionNumber"].map(texts)
train = df[df["split"] == "train"]
val = df[df["split"] == "val"].reset_index(drop=True)
blocks = week_blocks(val["ts_et"])
print(f"{len(val)} validation events in {blocks.nunique()} distinct weeks\n")

# text model, same settings as before (C=0.1)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000,
                      sublinear_tf=True, stop_words="english")
Xtr, Xva = vec.fit_transform(train["text"]), vec.transform(val["text"])
val["p_text"] = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, train["y_dir"]).predict_proba(Xva)[:, 1]

# control: ticker only, no text and no prices
enc = OneHotEncoder(handle_unknown="ignore")
Ttr, Tva = enc.fit_transform(train[["ticker"]]), enc.transform(val[["ticker"]])
val["p_tick"] = LogisticRegression(C=0.1, max_iter=2000).fit(Ttr, train["y_dir"]).predict_proba(Tva)[:, 1]

show("text model        AUC minus 0.5",
     block_bootstrap(val, lambda d: roc_auc_score(d["y_dir"], d["p_text"]) - 0.5, blocks))
show("ticker-only       AUC minus 0.5",
     block_bootstrap(val, lambda d: roc_auc_score(d["y_dir"], d["p_tick"]) - 0.5, blocks))
show("text minus ticker-only (paired AUC difference)",
     block_bootstrap(val, lambda d: roc_auc_score(d["y_dir"], d["p_text"])
                     - roc_auc_score(d["y_dir"], d["p_tick"]), blocks))