# Market-Moving Event Predictor

Can information available around an S&P 500 earnings release predict the stock's reaction?
The project asks three questions on one event table: **direction** of the abnormal return,
its **signed size**, and the **size of the move** regardless of sign.

**Status: work in progress.** Price-feature results are in. Text features are being built.
The test period (2024-2026) has not been used.

## Data

| Source | What | How |
|---|---|---|
| Wikipedia | Current S&P 500 constituents | `mep.data.prices` |
| Yahoo Finance (`yfinance`) | Daily split/dividend-adjusted prices from 2015, plus SPY as the market benchmark | `mep.data.prices` |
| SEC EDGAR | 8-K filings with Item 2.02 (earnings results), with exact acceptance timestamps; press-release exhibits | `mep.data.events` |

Each earnings event is the latest Item 2.02 8-K filed between a fiscal period's end and 3 days
after that period's 10-Q/10-K. This gives one event per reported quarter and drops mid-quarter
guidance updates. Final table: 20,003 events across 498 tickers.

## Method

- **Abnormal return** = stock return minus SPY return over the same window.
- **Timing rule (no lookahead).** Features use prices through the close *before* the filing day.
  Filed before 4pm ET: window is prior close to same-day close. Filed at or after 4pm ET: prior
  close to next trading day's close. Weekend or holiday filings roll to the next trading day.
  This is conservative: the filing time is an upper bound on when news came out.
- **Targets:** `y_dir` (abnormal return > 0), `abn_ret` (signed), `y_abs` (absolute size).
- **Filters:** ECHO excluded (no earnings 8-Ks since 2017); ticker-years with fewer than 3 events
  dropped; events with under 252 trading days of prior prices dropped.
- **Split by date, never random:** train 2016-2021 (10,771), validation 2022-2023 (3,838),
  test 2024-2026 (5,394, untouched).
- **Baselines to beat:** always predict "up"; predict the training mean; predict a company's own
  past average absolute move (using only earlier events).
- **Uncertainty:** week-level block bootstrap, since many companies report in the same week.
- **Tests:** 7 tests cover the timing rule, a no-lookahead check on every rolling price feature,
  and the bootstrap.

## Results so far (validation, 2022-2023)

Baselines: always-up accuracy 0.534, log-loss 0.6913. Company-history size baseline: MAE 0.0276, R2 0.154.

Price features (momentum, volatility, volume, market context, company history) vs baselines:

| Comparison | Result | 95% interval (week-block bootstrap) |
|---|---|---|
| Direction AUC minus 0.5 | +0.0055 | [-0.014, +0.024] |
| Direction log-loss, model minus baseline | +0.0002 | [-0.0013, +0.0017] |
| Size MAE, model minus baseline | +0.0002 | [-0.0003, +0.0007] |

Price history before an earnings report adds no detectable information about direction, and no
meaningful improvement on a company's own volatility history for the size of the move.

## Known limitations

- **Survivorship bias:** the universe is today's S&P 500, so companies removed since 2015 are missing.
- **Yahoo Finance data is free but unofficial.**
- **Filing time is an upper bound on release time.** About 44% of events use a 2-day window, which
  adds a day of noise (2-day moves are about 30% larger on average).
- **Companies that changed legal entities** (e.g. XOM, BLK, BG) have short histories under their
  current SEC ID and are mostly excluded by the history filter.
- **Abnormal return is simple** (stock minus SPY), not beta-adjusted.
- **Reaction prediction, not a forecast:** press releases appear on the filing day, after the
  window's starting price. A trading strategy would need to enter after the release.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and a `.env` file containing
`SEC_USER_AGENT="Your Name your.email@example.com"` (SEC requires it).

```
uv sync
uv run python -m mep.data.prices
uv run python -m mep.data.events
uv run python -m mep.data.event_table
uv run python -m mep.features.price_features
uv run python -m mep.models.baselines
uv run python -m mep.models.first_models
uv run python scripts/uncertainty_check.py
uv run pytest
```