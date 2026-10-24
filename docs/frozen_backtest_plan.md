# Frozen backtest plan

Written on 2026-10-23, before any backtest code was run on any data.
Nothing below may be changed after the first backtest result is seen. If a bug is found
afterwards, it is documented in the README and not silently fixed and rerun.

## Context (stated honestly)

- The 2024-2026 test set was already scored once for the direction and size models
  (`docs/test_results.txt`). This backtest is therefore a second look at that period, not
  a clean untouched test. Results are labelled that way.
- The signal below has no fitted parameters. The holding period and cost level are
  fixed defaults, not tuned.

## Signal

- For each earnings event, take the sign of the abnormal return over the reaction window
  (`abn_ret`, already computed: stock return minus SPY return).
- Positive: go long the stock and short SPY. Negative or zero: short the stock, long SPY.
- This is the announcement-return version of post-earnings drift: does the reaction
  continue in the same direction afterwards? No model is used.

## Trades

- Entry: close of the last day of the reaction window (`end_date`), the earliest price
  that is fully known. Assumes a trade at that closing price.
- Exit: close 5 trading days after entry.
- Equal notional on each leg, one unit per leg (stock and SPY).
- Events without a price at entry or exit are dropped, and the count is reported.
- Each ticker-event is its own trade, including events sharing one filing (e.g. GOOG and GOOGL).

## Costs

- Primary: 10 basis points per side, applied to each of the four fills (stock in, stock out,
  SPY in, SPY out), so 40 basis points of round-trip cost per pair trade.
- Also reported: 0 (gross) and 5 basis points per side. No other levels.

## Metrics

- Primary: average net return per trade, with a 95% week-block bootstrap interval
  (blocks = calendar week of entry).
- Secondary, approximate: weekly cohort returns (mean net return of the trades entered
  that week; weeks with no trades count as 0). From these: annualized Sharpe
  (mean / std x sqrt(52)), maximum drawdown of the compounded cohort curve, trades per year.
- The cohort series ignores the overlap between consecutive weeks' holding periods,
  so Sharpe and drawdown are rough.

## Order of work

1. Build the backtest and check the code on 2016-2023 data (no parameters are fitted,
   so this checks the code only).
2. Score 2024-2026 once, and save the report to `docs/backtest_results.txt`.

## Counts as success

- Net average return per trade has a 95% interval entirely above zero at 10 basis points
  per side.
- Anything else is reported as no profitable strategy after costs.

## Reported regardless of outcome

- Gross and net results at all three cost levels, for both 2016-2023 and 2024-2026.
- Long-only and short-only legs reported separately.