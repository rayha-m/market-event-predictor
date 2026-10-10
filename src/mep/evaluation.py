import numpy as np
import pandas as pd


def week_blocks(ts: pd.Series) -> pd.Series:
    """Label each event with its calendar week (timezone dropped first)."""
    return ts.dt.tz_localize(None).dt.to_period("W").astype(str)


def block_bootstrap(df, stat_fn, block, n_boot=1000, seed=0):
    """Resample whole blocks (e.g. weeks) with replacement.

    stat_fn(resampled_df) -> float. Returns (point estimate, 2.5%, 97.5%).
    """
    df = df.reset_index(drop=True)
    groups = df.groupby(block.reset_index(drop=True)).indices  # block -> row positions
    keys = list(groups)
    rng = np.random.default_rng(seed)
    point = stat_fn(df)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[i]] for i in pick])
        try:
            stats.append(stat_fn(df.iloc[idx]))
        except ValueError:  # e.g. a resample containing only one class
            continue
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, lo, hi