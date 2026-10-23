import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from mep.evaluation import block_bootstrap, week_blocks
from mep.models.text_baseline import PROC, load_texts

DRY = "dry" in sys.argv
RESULTS = Path("docs/test_results.txt")
N_BOOT = 1000

if not DRY and RESULTS.exists():
    sys.exit(f"{RESULTS} already exists: the test set has been scored. Do not rerun.")

df = pd.read_parquet(PROC / "features_beta.parquet")
texts = load_texts(df["accessionNumber"].unique())
n_all = len(df)
df = df[df["accessionNumber"].isin(texts)].copy()
df["text"] = df["accessionNumber"].map(texts)

if DRY:
    train = df[df["split"] == "train"]
    test = df[df["split"] == "val"].reset_index(drop=True)
else:
    train = df[df["split"].isin(["train", "val"])]
    test = df[df["split"] == "test"].reset_index(drop=True)
blocks = week_blocks(test["ts_et"])

lines = []


def out(s=""):
    lines.append(s)


def ci(name, res):
    pt, lo, hi = res
    out(f"{name:58s} {pt:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")


def boot(fn):
    return block_bootstrap(test, fn, blocks, n_boot=N_BOOT)


def auc(d, col, y="y_dir"):
    return roc_auc_score(d[y], d[col])


out("DRY RUN: validation data used as the exam, nothing frozen" if DRY
    else "FINAL TEST RUN: 2016-2023 train, 2024-2026 test")
out(f"train {len(train)} | test {len(test)} in {blocks.nunique()} weeks "
    f"| events without text dropped: {n_all - len(df)}\n")

# ---------- shared features ----------
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000,
                      sublinear_tf=True, stop_words="english")
Xtr, Xte = vec.fit_transform(train["text"]), vec.transform(test["text"])  # train only
enc = OneHotEncoder(handle_unknown="ignore")
Ttr, Tte = enc.fit_transform(train[["ticker"]]), enc.transform(test[["ticker"]])

# ---------- direction (primary question) ----------
p0 = train["y_dir"].mean()
test["p0"] = p0
test["p_text"] = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, train["y_dir"]).predict_proba(Xte)[:, 1]
test["p_tick"] = LogisticRegression(C=0.1, max_iter=2000).fit(Ttr, train["y_dir"]).predict_proba(Tte)[:, 1]

out("DIRECTION (target y_dir)")
out(f"always-up accuracy {(test['y_dir'] == round(p0)).mean():.3f} | "
    f"text model accuracy {((test['p_text'] > 0.5) == test['y_dir']).mean():.3f}")
text_res = boot(lambda d: auc(d, "p_text") - 0.5)
tick_res = boot(lambda d: auc(d, "p_tick") - 0.5)
pair_res = boot(lambda d: auc(d, "p_text") - auc(d, "p_tick"))
ci("text model AUC minus 0.5", text_res)
ci("ticker-only AUC minus 0.5", tick_res)
ci("text minus ticker-only (paired AUC)", pair_res)
ci("log-loss: text model minus always-up baseline",
   boot(lambda d: log_loss(d["y_dir"], d["p_text"]) - log_loss(d["y_dir"], d["p0"])))
verdict = "SUCCESS" if (text_res[1] > 0 and pair_res[1] > 0) else "NO DETECTABLE SIGNAL"
out(f"Pre-registered criterion (both intervals above zero): {verdict}\n")

# ---------- size of the move ----------
mean_tr = train["y_abs"].mean()
base_tr = train["past_abs"].fillna(mean_tr)
test["base"] = test["past_abs"].fillna(mean_tr)
ridge = Ridge(alpha=10).fit(Xtr, train["y_abs"] - base_tr)
test["model"] = test["base"] + ridge.predict(Xte)

out("SIZE OF MOVE (target y_abs)")
for nm, col in [("company-history baseline", "base"), ("baseline + text residual", "model")]:
    out(f"{nm:34s} MAE {mean_absolute_error(test['y_abs'], test[col]):.4f} "
        f"| R2 {r2_score(test['y_abs'], test[col]):.4f}")
ci("MAE: baseline+text minus baseline",
   boot(lambda d: (d["y_abs"] - d["model"]).abs().mean() - (d["y_abs"] - d["base"]).abs().mean()))
out()

# ---------- signed size ----------
mu = train["abn_ret"].mean()
out("SIGNED SIZE (target abn_ret), predict the training mean")
out(f"MAE {mean_absolute_error(test['abn_ret'], [mu] * len(test)):.4f} "
    f"| R2 {r2_score(test['abn_ret'], [mu] * len(test)):.4f}\n")

# ---------- by window length ----------
out("BY WINDOW LENGTH (point estimates only)")
for w, g in test.groupby("window_days"):
    out(f"window_days={w}: n={len(g)} | text AUC {auc(g, 'p_text'):.3f} "
        f"| ticker AUC {auc(g, 'p_tick'):.3f} "
        f"| size MAE baseline {mean_absolute_error(g['y_abs'], g['base']):.4f} "
        f"vs +text {mean_absolute_error(g['y_abs'], g['model']):.4f}")
out()

# ---------- robustness: beta-adjusted direction ----------
test["pb_text"] = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, train["y_dir_beta"]).predict_proba(Xte)[:, 1]
test["pb_tick"] = LogisticRegression(C=0.1, max_iter=2000).fit(Ttr, train["y_dir_beta"]).predict_proba(Tte)[:, 1]
out("ROBUSTNESS: beta-adjusted direction (target y_dir_beta), not primary")
ci("text model AUC minus 0.5", boot(lambda d: auc(d, "pb_text", "y_dir_beta") - 0.5))
ci("text minus ticker-only (paired AUC)",
   boot(lambda d: auc(d, "pb_text", "y_dir_beta") - auc(d, "pb_tick", "y_dir_beta")))

# ---------- print everything only now, then lock ----------
report = "\n".join(lines)
print(report)
if not DRY:
    RESULTS.write_text(report + "\n")
    print(f"\nSaved to {RESULTS}. The test set is now used; do not rerun.")