import numpy as np
import pandas as pd


def pair_returns(close: pd.DataFrame, tickers, entry_dates: pd.DatetimeIndex,
                 signs: np.ndarray, hold: int = 5) -> np.ndarray:
    """Gross return of a stock-vs-SPY pair trade, one value per trade.

    Enter at the close on entry_date, exit at the close `hold` trading days later.
    sign = +1: long stock, short SPY. sign = -1: short stock, long SPY.
    Returns NaN when the entry date is not a trading day, the ticker is unknown,
    or the exit would fall past the end of the data. Uses nothing after the exit day.
    """
    dates = close.index
    entry = dates.get_indexer(entry_dates)
    col = close.columns.get_indexer(list(tickers))
    exit_ = np.where(entry >= 0, entry + hold, -1)
    ok = (entry >= 0) & (col >= 0) & (exit_ < len(dates))
    arr = close.to_numpy()
    spy = close["SPY"].to_numpy()
    out = np.full(len(entry), np.nan)
    i = np.where(ok)[0]
    r_stock = arr[exit_[i], col[i]] / arr[entry[i], col[i]] - 1
    r_spy = spy[exit_[i]] / spy[entry[i]] - 1
    out[i] = signs[i] * (r_stock - r_spy)
    return out


def net_return(gross, cost_bps: float):
    """Subtract trading costs: 4 fills (stock in/out, SPY in/out) at cost_bps each."""
    return gross - 4 * cost_bps / 1e4