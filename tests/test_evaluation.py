import numpy as np
import pandas as pd

from mep.evaluation import block_bootstrap


def make(seed=0):
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, 1, 40)
    block = np.repeat(np.arange(40), 25)
    x = shocks[block] + rng.normal(0, 0.3, 1000)
    return pd.DataFrame({"x": x}), pd.Series(block)


def test_interval_contains_point_estimate():
    df, block = make()
    pt, lo, hi = block_bootstrap(df, lambda d: d["x"].mean(), block, n_boot=300)
    assert lo < pt < hi


def test_clustering_widens_the_interval():
    df, block = make()
    _, lo_b, hi_b = block_bootstrap(df, lambda d: d["x"].mean(), block, n_boot=300)
    rows = pd.Series(np.arange(len(df)))  # every row its own block = naive bootstrap
    _, lo_n, hi_n = block_bootstrap(df, lambda d: d["x"].mean(), rows, n_boot=300)
    assert (hi_b - lo_b) > 2 * (hi_n - lo_n)