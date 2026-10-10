import os
import time
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
UA = os.environ["SEC_USER_AGENT"]


def get(url: str) -> str:
    time.sleep(0.15)  # stay under SEC's 10 requests/second limit
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.text


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


df = pd.read_parquet("data/processed/features.parquet")
for _, r in df.sample(8, random_state=1).iterrows():
    tag = f"{r['ticker']} {r['ts_et']:%Y-%m-%d}"
    try:
        url = find_exhibit_url(r["cik"], r["accessionNumber"])
        if url is None:
            print(f"{tag}: no EX-99 exhibit found")
            continue
        text = html_to_text(get(url))
        print(f"{tag}: {len(text):,} chars | {text[:120]!r}")
    except Exception as e:
        print(f"{tag}: failed ({type(e).__name__}: {e})")