import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mep.backtest.pairs import net_return, pair_returns
from mep.evaluation import block_bootstrap

DRY = "dry" in sys.argv
RESULTS = Path("docs/backtest_results.txt")
HOLD = 5
COSTS = [0, 5, 10]  # basis points per side
N_BOOT = 1000

if not DRY and RESULTS.exists():
    sys.exit(f"{RESULTS} already exists: the backtest has been scored. Do not rerun.")

close = pd.read_parquet("data/raw/prices.parquet")["Close"]
ev = pd.read_parquet("data/processed/features_beta.parquet")[["ticker", "split", "end_date", "abn_ret"]].copy()
ev["end_date"] = pd.to_datetime(ev["end_date"])
ev["sign"] = np.where(ev["abn_ret"] > 0, 1.0, -1.0)  # zero counts as negative, per the plan
ev["gross"] = pair_returns(close, ev["ticker"].tolist(), pd.DatetimeIndex(ev["end_date"]),
                           ev["sign"].to_numpy(), HOLD)
ev["wk"] = ev["end_date"].dt.to_period("W")
ev["week"] = ev["wk"].astype(str)

periods = {"2016-2023 (development, code check)": ev[ev["split"].isin(["train", "val"])]}
if not DRY:
    periods["2024-2026 (second look at an already-scored period)"] = ev[ev["split"] == "test"]

lines = []


def out(s=""):
    lines.append(s)


def bps_ci(d, label):
    """Mean net return per trade in basis points, with a week-block bootstrap interval."""
    pt, lo, hi = block_bootstrap(d, lambda x: x["net"].mean() * 1e4, d["week"], n_boot=N_BOOT)
    out(f"{label:44s} {pt:+8.1f} bps   95% CI [{lo:+8.1f}, {hi:+8.1f}]")
    return lo


out("DRY RUN: 2016-2023 only, nothing frozen or saved" if DRY
    else "FINAL BACKTEST RUN: both periods, results saved and locked")
out(f"Rule: sign of reaction-window abnormal return; enter at window close, exit after {HOLD} trading days\n")

for name, d in periods.items():
    n_all = len(d)
    d = d[d["gross"].notna()].reset_index(drop=True)
    out(f"===== {name} =====")
    out(f"trades {len(d)} | dropped (no price / exit past data end): {n_all - len(d)}")

    for c in COSTS:
        d["net"] = net_return(d["gross"], c)
        lo = bps_ci(d, f"all trades, cost {c} bps per side")
        if c == 10:
            verdict = "SUCCESS" if lo > 0 else "NOT PROFITABLE AFTER COSTS"
            out(f"Pre-registered criterion (interval above zero at 10 bps): {verdict}")
            out(f"win rate {(d['net'] > 0).mean():.3f} | check: gross minus net = "
                f"{(d['gross'].mean() - d['net'].mean()) * 1e4:.1f} bps (should be 40.0)")

    # legs reported separately: long stock / short SPY vs short stock / long SPY
    for s, label in [(1.0, "long stock, short SPY"), (-1.0, "short stock, long SPY")]:
        sub = d[d["sign"] == s].reset_index(drop=True)
        out(f"-- {label}: {len(sub)} trades")
        for c in (0, 10):
            sub["net"] = net_return(sub["gross"], c)
            bps_ci(sub, f"   cost {c} bps per side")

    # rough weekly-cohort risk numbers at the primary cost level
    d["net"] = net_return(d["gross"], 10)
    weekly = d.groupby("wk")["net"].mean().reindex(
        pd.period_range(d["wk"].min(), d["wk"].max(), freq="W"), fill_value=0.0)
    curve = (1 + weekly).cumprod()
    years = (d["end_date"].max() - d["end_date"].min()).days / 365.25
    out(f"weekly cohorts (rough, ignores overlap): Sharpe {weekly.mean() / weekly.std() * np.sqrt(52):+.2f} "
        f"| max drawdown {(curve / curve.cummax() - 1).min():.1%} | trades per year {len(d) / years:.0f}\n")

report = "\n".join(lines)
print(report)
if not DRY:
    RESULTS.write_text(report + "\n")
    print(f"\nSaved to {RESULTS}. The backtest is now scored; do not rerun.")