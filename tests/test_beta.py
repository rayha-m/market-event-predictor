import numpy as np
import pandas as pd

from mep.features.beta import rolling_beta


def make_close(n=500, seed=0, true_beta=1.5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    spy_r = rng.normal(0, 0.01, n)
    a_r = true_beta * spy_r + rng.normal(0, 0.002, n)
    b_r = rng.normal(0, 0.01, n)
    r = np.column_stack([a_r, b_r, spy_r])
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=["A", "B", "SPY"])


def test_beta_uses_no_future_data():
    close = make_close()
    full = rolling_beta(close)
    for t in [260, 350, 499]:
        part = rolling_beta(close.iloc[: t + 1])
        pd.testing.assert_series_equal(full.iloc[t], part.iloc[-1], check_names=False)


def test_beta_recovers_a_known_value():
    beta = rolling_beta(make_close(true_beta=1.5))
    assert abs(beta["A"].iloc[-1] - 1.5) < 0.1
    assert abs(beta["SPY"].iloc[-1] - 1.0) < 1e-9