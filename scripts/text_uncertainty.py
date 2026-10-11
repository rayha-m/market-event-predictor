import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score

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

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000,
                      sublinear_tf=True, stop_words="english")
Xtr = vec.fit_transform(train["text"])  # vocabulary from train only
Xva = vec.transform(val["text"])

# ---- direction ----
val["p0"] = train["y_dir"].mean()
val["p"] = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, train["y_dir"]).predict_proba(Xva)[:, 1]
val["p10"] = LogisticRegression(C=10, max_iter=2000).fit(Xtr, train["y_dir"]).predict_proba(Xva)[:, 1]

print("PRE-CHOSEN SETTINGS")
show("DIRECTION C=0.1  AUC minus 0.5",
     block_bootstrap(val, lambda d: roc_auc_score(d["y_dir"], d["p"]) - 0.5, blocks))
show("DIRECTION C=0.1  log-loss: model minus baseline",
     block_bootstrap(val, lambda d: log_loss(d["y_dir"], d["p"]) - log_loss(d["y_dir"], d["p0"]), blocks))

# ---- size of move (text predicts the baseline's error) ----
mean_tr = train["y_abs"].mean()
base_tr = train["past_abs"].fillna(mean_tr)
val["base"] = val["past_abs"].fillna(mean_tr)
ridge = Ridge(alpha=10).fit(Xtr, train["y_abs"] - base_tr)
val["model"] = val["base"] + ridge.predict(Xva)
show("SIZE alpha=10  MAE: model minus baseline",
     block_bootstrap(val, lambda d: (d["y_abs"] - d["model"]).abs().mean()
                     - (d["y_abs"] - d["base"]).abs().mean(), blocks))

print("\nEXPLORATORY (chosen after seeing results, so optimistic)")
show("DIRECTION C=10   AUC minus 0.5",
     block_bootstrap(val, lambda d: roc_auc_score(d["y_dir"], d["p10"]) - 0.5, blocks))
print("\nFor log-loss and MAE, a NEGATIVE difference means the model beat the baseline.")