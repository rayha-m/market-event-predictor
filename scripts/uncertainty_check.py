import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mep.evaluation import block_bootstrap, week_blocks
from mep.models.first_models import FEATURES


def show(name, res):
    pt, lo, hi = res
    print(f"{name:46s} {pt:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")


df = pd.read_parquet("data/processed/features.parquet")
train = df[df["split"] == "train"]
val = df[df["split"] == "val"].reset_index(drop=True)
blocks = week_blocks(val["ts_et"])
print(f"{len(val)} validation events in {blocks.nunique()} distinct weeks\n")

# ---- direction: logistic regression on price features ----
clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                    LogisticRegression(C=0.1, max_iter=1000))
clf.fit(train[FEATURES], train["y_dir"])
val["p"] = clf.predict_proba(val[FEATURES])[:, 1]
val["p0"] = train["y_dir"].mean()

show("DIRECTION AUC minus 0.5 (0 = chance)",
     block_bootstrap(val, lambda d: roc_auc_score(d["y_dir"], d["p"]) - 0.5, blocks))
show("DIRECTION log-loss: model minus baseline",
     block_bootstrap(val, lambda d: log_loss(d["y_dir"], d["p"]) - log_loss(d["y_dir"], d["p0"]), blocks))

# ---- size of move: boosted trees vs company-history baseline ----
reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                    min_samples_leaf=100, early_stopping=False, random_state=0)
reg.fit(train[FEATURES], train["y_abs"])
val["model"] = reg.predict(val[FEATURES])
val["base"] = val["past_abs"].fillna(train["y_abs"].mean())

show("SIZE MAE: model minus baseline",
     block_bootstrap(val, lambda d: (d["y_abs"] - d["model"]).abs().mean()
                     - (d["y_abs"] - d["base"]).abs().mean(), blocks))
print("\nFor log-loss and MAE, a NEGATIVE difference means the model beat the baseline.")
print("A result only counts if the whole interval is on one side of zero.")