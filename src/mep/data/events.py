import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from mep.data.prices import get_sp500_tickers

load_dotenv()
RAW = Path("data/raw")
EARNINGS_ITEM = r"(?:^|,)2\.02(?:,|$)"  # 8-K Item 2.02 = earnings results


def _get_json(url: str) -> dict:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError("Set SEC_USER_AGENT in your .env file")
    time.sleep(0.15)  # stay under SEC's 10 requests/second limit
    r = requests.get(url, headers={"User-Agent": ua}, timeout=30)
    r.raise_for_status()
    return r.json()


def ticker_to_cik() -> dict[str, str]:
    """Map tickers to SEC's 10-digit company IDs (CIKs)."""
    data = _get_json("https://www.sec.gov/files/company_tickers.json")
    return {
        v["ticker"].replace(".", "-"): str(v["cik_str"]).zfill(10)
        for v in data.values()
    }


def earnings_filings(cik: str, since: str = "2015-01-01") -> pd.DataFrame:
    """One earnings 8-K per reported quarter, anchored to the 10-Q/10-K."""
    sub = _get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    blocks = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):  # older filings live in extra files
        blocks.append(_get_json(f"https://data.sec.gov/submissions/{f['name']}"))
    df = pd.concat([pd.DataFrame(b) for b in blocks], ignore_index=True)
    df = df.drop_duplicates("accessionNumber")

    eight_k = df[(df["form"] == "8-K") & df["items"].str.contains(EARNINGS_ITEM, na=False)].copy()
    eight_k["fd"] = pd.to_datetime(eight_k["filingDate"])
    eight_k = eight_k.sort_values("acceptanceDateTime")
    periodic = df[df["form"].isin(["10-Q", "10-K"]) & (df["filingDate"] >= since)]

    picked = []
    for _, p in periodic.iterrows():
        period_end = pd.to_datetime(p["reportDate"], errors="coerce")
        filed = pd.to_datetime(p["filingDate"])
        if pd.isna(period_end):
            continue
        cand = eight_k[(eight_k["fd"] >= period_end) & (eight_k["fd"] <= filed + pd.Timedelta(days=3))]
        if len(cand):
            row = cand.iloc[-1].copy()  # latest earnings 8-K before the 10-Q
            row["period_end"] = p["reportDate"]
            row["periodic_form"] = p["form"]
            picked.append(row)

    cols = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime",
            "items", "primaryDocument", "period_end", "periodic_form"]
    if not picked:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(picked).drop_duplicates("accessionNumber")
    return out[cols]


def build_events(limit: int | None = None) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    ciks = ticker_to_cik()
    tickers = [t for t in get_sp500_tickers() if t in ciks]
    if limit:
        tickers = tickers[:limit]
    frames = []
    for i, t in enumerate(tickers, 1):
        try:
            df = earnings_filings(ciks[t])
            df.insert(0, "ticker", t)
            df.insert(1, "cik", ciks[t])
            frames.append(df)
        except Exception as e:
            print(f"[{i}/{len(tickers)}] {t} failed: {e}")
            continue
        print(f"[{i}/{len(tickers)}] {t}: {len(df)} events")
    out = RAW / "events.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(out)
    return out


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(build_events(limit))