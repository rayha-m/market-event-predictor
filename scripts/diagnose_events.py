import pandas as pd

from mep.data.events import EARNINGS_ITEM, _get_json, ticker_to_cik

events = pd.read_parquet("data/raw/events.parquet")
n = events.groupby("ticker").size()
short = n[n < 30].index.tolist()
ciks = ticker_to_cik()

rows = []
for t in short:
    sub = _get_json(f"https://data.sec.gov/submissions/CIK{ciks[t]}.json")
    blocks = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):
        blocks.append(_get_json(f"https://data.sec.gov/submissions/{f['name']}"))
    df = pd.concat([pd.DataFrame(b) for b in blocks], ignore_index=True)
    df = df[df["filingDate"] >= "2015-01-01"]
    e8k = df[(df.form == "8-K") & df["items"].str.contains(EARNINGS_ITEM, na=False)]
    periodic = df[df.form.isin(["10-Q", "10-K"])]
    rows.append({
        "ticker": t,
        "matched": int(n[t]),
        "earn_8k": len(e8k),
        "periodic": len(periodic),
        "first_filing": df.filingDate.min(),
        "name": sub.get("name", "")[:28],
        "former": ", ".join(x["name"] for x in sub.get("formerNames", [])[:1])[:28],
    })

pd.set_option("display.width", 220)
print(pd.DataFrame(rows).sort_values("matched").to_string(index=False))