import gzip
import os
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
UA = os.environ["SEC_USER_AGENT"]
OUT = Path("data/raw/press_releases")


def get(url: str, tries: int = 4) -> str:
    for k in range(tries):
        time.sleep(0.12)  # stay under SEC's 10 requests/second limit
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        if r.status_code in (403, 429, 500, 502, 503):
            time.sleep(5 * (k + 1))  # back off, then retry
            continue
        r.raise_for_status()
        return r.text
    raise RuntimeError(f"gave up on {url} (last status {r.status_code})")


def find_exhibit_url(cik, acc):
    nodash = acc.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}"
    for tbl in pd.read_html(StringIO(get(f"{base}/{acc}-index.htm"))):
        if "Type" in tbl.columns and "Document" in tbl.columns:
            ex = tbl[tbl["Type"].astype(str).str.upper().str.startswith("EX-99")]
            if len(ex):
                return f"{base}/{str(ex.iloc[0]['Document']).split()[0]}"
    return None


def html_to_text(html: str) -> str:
    return BeautifulSoup(html, "lxml").get_text(" ", strip=True)


def main(limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/features.parquet")
    # GOOG/GOOGL etc. share filings, so fetch each filing once
    df = df.drop_duplicates("accessionNumber")
    # random order, so a partial run is still spread across all years
    df = df.sample(frac=1, random_state=0)
    if limit:
        df = df.head(limit)

    fetched = skipped = failed = 0
    for i, (_, r) in enumerate(df.iterrows(), 1):
        acc = r["accessionNumber"]
        if (OUT / f"{acc}.txt.gz").exists() or (OUT / f"{acc}.none").exists():
            skipped += 1
            continue
        try:
            url = find_exhibit_url(r["cik"], acc)
            if url is None:
                (OUT / f"{acc}.none").touch()  # no EX-99 exhibit; don't retry
            else:
                text = html_to_text(get(url))
                with gzip.open(OUT / f"{acc}.txt.gz", "wt", encoding="utf-8") as f:
                    f.write(text)
            fetched += 1
        except Exception as e:  # not marked, so a re-run retries it
            failed += 1
            print(f"[{i}] {r['ticker']} {acc}: {type(e).__name__}: {e}")
        if i % 200 == 0:
            print(f"[{i}/{len(df)}] fetched {fetched} | skipped {skipped} | failed {failed}")

    print(f"done: fetched {fetched} | skipped {skipped} | failed {failed}")
    print("saved:", len(list(OUT.glob("*.txt.gz"))), "texts |",
          len(list(OUT.glob("*.none"))), "with no exhibit")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)