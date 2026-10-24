import pandas as pd

from mep.data.events import EARNINGS_ITEM, _get_json, ticker_to_cik

ciks = ticker_to_cik()
for t in ["ECHO", "BRK-B", "TROW"]:
    sub = _get_json(f"https://data.sec.gov/submissions/CIK{ciks[t]}.json")
    blocks = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):
        blocks.append(_get_json(f"https://data.sec.gov/submissions/{f['name']}"))
    df = pd.concat([pd.DataFrame(b) for b in blocks], ignore_index=True)
    df = df[df["filingDate"] >= "2015-01-01"]
    e = df[(df.form == "8-K") & df["items"].str.contains(EARNINGS_ITEM, na=False)]
    p = df[df.form.isin(["10-Q", "10-K"])]
    by_year = pd.DataFrame({
        "earnings_8k": e.groupby(e.filingDate.str[:4]).size(),
        "10Q_10K": p.groupby(p.filingDate.str[:4]).size(),
    }).fillna(0).astype(int)
    print(f"\n=== {t} ===")
    print(by_year.to_string())
    print("latest earnings 8-K dates:", sorted(e.filingDate)[-4:])
    print("latest 10-Q/10-K dates:   ", sorted(p.filingDate)[-4:])