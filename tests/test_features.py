import numpy as np
import pandas as pd

from mep.features.price_features import price_features


def make_prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = list("ABC")
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0)), index=idx, columns=cols
    )
    volume = pd.DataFrame(
        rng.integers(100_000, 1_000_000, (n, 3)).astype(float), index=idx, columns=cols
    )
    return close, volume


def test_features_use_no_future_data():
    close, volume = make_prices()
    full = price_features(close, volume)
    for t in [100, 250, 399]:
        part = price_features(close.iloc[: t + 1], volume.iloc[: t + 1])
        for name in full:
            pd.testing.assert_series_equal(
                full[name].iloc[t], part[name].iloc[-1], check_names=False
            )