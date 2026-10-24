from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

RAW = Path("data/raw")


def get_sp500_tickers() -> list[str]:
    """Current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    table = pd.read_html(StringIO(html))[0]
    # Yahoo uses "BRK-B", Wikipedia uses "BRK.B"
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


def download_prices(tickers: list[str], start: str = "2015-01-01") -> Path:
    """Daily adjusted prices for all tickers plus SPY (our market benchmark)."""
    RAW.mkdir(parents=True, exist_ok=True)
    df = yf.download(tickers + ["SPY"], start=start, auto_adjust=True, progress=True)
    out = RAW / "prices.parquet"
    df.to_parquet(out)
    return out


if __name__ == "__main__":
    tickers = get_sp500_tickers()
    print(f"{len(tickers)} tickers")
    print(download_prices(tickers))