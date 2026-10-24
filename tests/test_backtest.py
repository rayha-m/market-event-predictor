import numpy as np
import pandas as pd

from mep.backtest.pairs import net_return, pair_returns


def make_close():
    idx = pd.bdate_range("2024-01-01", periods=10)
    a = 100 * 1.1 ** np.arange(10)  # stock rises 10% a day
    spy = np.full(10, 100.0)        # market flat
    return pd.DataFrame({"A": a, "SPY": spy}, index=idx)


def test_long_and_short_pair_return():
    close = make_close()
    entry = close.index[[0]]
    up = pair_returns(close, ["A"], entry, np.array([1.0]), hold=2)
    dn = pair_returns(close, ["A"], entry, np.array([-1.0]), hold=2)
    assert abs(up[0] - 0.21) < 1e-9
    assert abs(dn[0] + 0.21) < 1e-9


def test_exit_past_end_of_data_is_nan():
    close = make_close()
    out = pair_returns(close, ["A"], close.index[[8]], np.array([1.0]), hold=5)
    assert np.isnan(out[0])


def test_result_ignores_prices_after_the_exit():
    close = make_close()
    changed = close.copy()
    changed.iloc[3:, :] = 1.0  # wreck every price after the exit day (index 2)
    a = pair_returns(close, ["A"], close.index[[0]], np.array([1.0]), hold=2)
    b = pair_returns(changed, ["A"], close.index[[0]], np.array([1.0]), hold=2)
    assert a[0] == b[0]


def test_cost_is_four_fills():
    assert abs(net_return(0.01, 10) - 0.006) < 1e-12