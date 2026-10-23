# Frozen evaluation plan

Written on 2026-10-23, before any model was scored on the 2024-2026 test set.
Nothing in this file may be changed after the test set is scored. If a bug is
found afterwards, it is documented in the README and not silently fixed and rerun.

## Setup

- Train on 2016-2023 events. Score on 2024-2026 events, once.
- Primary direction target: `y_dir` (stock return minus SPY return, positive or not).
- Uncertainty: week-block bootstrap, 95% intervals.

## Direction (primary question)

- Model: TF-IDF text features (1-2 word pairs, min_df=5, 50,000 features, stop words
  removed, digits masked), logistic regression with C=0.1.
- Control: ticker-only logistic regression with C=0.1.
- Reported: AUC minus 0.5 for each, the paired AUC difference (text minus ticker-only),
  and log-loss against the always-up baseline.
- Counts as success only if BOTH: the text model's AUC interval is entirely above 0.5,
  and the paired difference interval (text minus ticker-only) is entirely above 0.
- Anything else is reported as no detectable signal.

## Size of the move

- Baseline: a company's own past average absolute move (earlier events only).
- Also reported for comparison: baseline plus ridge text residual, alpha=10.
- Reported: MAE and R2 for each, and the paired MAE difference with a 95% interval.

## Signed size

- Training average prediction. Reported only.

## Reported regardless of outcome

- Results split by 1-day and 2-day windows.
- The beta-adjusted direction target (`y_dir_beta`) as a robustness check, not primary.
- All of the above, whether or not the result is favourable.