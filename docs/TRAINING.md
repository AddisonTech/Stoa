# Training

How Stoa's model is built, tested and kept honest. Short version: a plain statistical model predicts the chance a stock beats SPY over the next two weeks. It is trained only on the past, tested on the future, checked for calibration, and penalized for every version ever tried.

The goal is calibrated edge after costs, not accuracy. Being right 55% of the time is elite. The money comes from knowing how sure the model is, sizing to that, and losing small when wrong.

## What the model predicts

For each stock on each trading day t:

- Buy at the open of t+1 (that is when Stoa can actually trade a decision made after the close of t).
- Sell at the close of t+10 (the horizon, `model.horizon`).
- The label is 1 if that return beats SPY's return over exactly the same span, else 0.

The label for day t is only known after the close of t+10. Training never uses a label that is not yet known.

The model outputs a probability: "62% chance this stock beats SPY over the next 10 sessions."

## What the model sees

Everything is known at the close of day t. Nothing later.

Price features (from split and dividend adjusted bars):

| Feature | Meaning |
| --- | --- |
| `f_mom_12_1` | 12 month return, skipping the last month (classic momentum) |
| `f_mom_6_1` | 6 month return, skipping the last month |
| `f_mom_3` | 3 month return |
| `f_rev_1m` | minus the last month's return (short-term reversal) |
| `f_trend_200`, `f_trend_50` | distance from the 200 and 50 day averages (trend) |
| `f_vol_60` | minus 60 day volatility (calmer is better) |
| `f_dist_52w` | distance from the 52 week high |
| `f_rs_spy_60` | 60 day return minus SPY's |
| `f_beta_120` | minus 120 day beta to SPY |

Quality features (from SEC filings, `stoa fundamentals fetch`):

| Feature | Meaning |
| --- | --- |
| `q_gross_profitability` | gross profit over total assets |
| `q_roe` | net income over equity |
| `q_accruals` | (net income minus operating cash flow) over assets |
| `q_leverage` | liabilities over assets |
| `q_earnings_stability` | minus the volatility of ROE over the last 8 quarters |

Text features (optional, from the Claude analyst, `model.use_text`): average sentiment, guidance direction, results surprise, risk flags, materiality and count of recent news items.

Every feature is scaled across all stocks on the same day: a z-score clipped at 3 (`model.normalize: zscore`), or the percentile rank scaled to -1 to 1 (`rank`), with missing values in the middle either way. The model learns which stocks look better than the others today, not what the market will do. `model.price_features` picks a subset of the price features (null means all ten).

`model.rebalance: monthly` lets the model open positions only when the orders execute on the first session of a month. Exits, stops and the kill switch still run every day. The current configuration is trial 6, pre-registered in DECISIONS.md.

## Point in time

Every record that is not a price bar carries `known_at`: the moment it became public, in UTC. A decision after the close of day t may use it only if `known_at` is at or before that close. That rule lives in one function, `stoa.pit.visible`, and everything goes through it.

- SEC fundamentals: the acceptance time of the filing that first reported the number. A later restatement has its own, later `known_at`, so it never leaks back.
- News: the article's last update time (`updated_at`), which can only delay an item.
- Analyst scores: the same timestamp as the item they score.
- A source with only a date (no time) is stamped at 11:59:59 pm New York time, so it is visible from the next close.

A test perturbs every bar after a date and checks that no feature on or before that date changes.

## No survivorship bias

The training universe on each day is the S&P 500 as it was on that day, from a vendored membership history (`stoa/reference/sp500_membership.csv`, MIT license, source in that folder's README). A company that went bankrupt or was bought out is in the training data up to the day it left the index.

If a stock stops trading inside a label's 10 day horizon, the label uses its last close. Without that, the final weeks of every failed company would vanish from training.

Remaining bias, said plainly:

- Alpaca's bar history covers 2016 onward. Alpaca does not document whether it keeps bars for symbols that were delisted, and its asset list has no history. `stoa universe build` fetches every symbol that was ever a member since 2016, then writes `coverage.json` with how many members have bars, by year, and how many of the names that left the index have bars. Read that file. Any name missing there is a hole, and the holes are mostly companies that failed or were acquired, which biases results upward.
- A symbol that was renamed shows up as two symbols, matching the bar provider.
- Sectors are today's GICS sectors applied to the past. Names that left before the snapshot count as `Unknown`.

## Walk-forward with a gap

Train on the past, test on the next slice, roll forward.

```
|------------- train -------------|-gap-|-- test --|
|------------------ train ------------------|-gap-|-- test --|
```

- Test slices are 63 sessions (`model.test_sessions`).
- Training uses all history before the gap (`expanding`) or the last 3 years (`rolling`).
- The gap (embargo) is at least the 10 session horizon. Training rows whose label would end inside or after the test slice are dropped (purged). So no training label ever overlaps a test label. Tests prove it for both window types and several horizons.
- Calibration is fit inside each training set only, on its last quarter, with its own gap.
- Until the holdout has been scored, no training row or label touches 2026-07-01 to 2026-10-23. Sessions after it may be used (see HOLDOUT.md).

Every out-of-sample probability comes from a model that never saw that test slice. The backtests in the scoreboard use only those probabilities.

## Two models

- **Logistic regression** (the baseline): one weight per feature, heavily regularized.
- **Gradient boosted trees** (the challenger): small, slow-learning trees that can catch simple interactions.

The winner is the one with the lowest out-of-sample log loss, with Brier score as the tiebreak. For the scoreboard, the choice uses only folds that end before the validation window. Accuracy is never used to choose.

## Calibration

When the model says 60%, it should be right about 60% of the time. The report shows:

- **Log loss**: how surprised the model is by what happened. Lower is better. Always predicting the base rate is the bar to beat.
- **Brier score**: average squared gap between the probability and what happened. Lower is better.
- **AUC**: how well it ranks winners above losers. 0.5 is a coin flip. Above 0.55 is good for stocks.
- **ECE** (expected calibration error): the average gap between what it said and how often it was right, across probability bins. Under 0.05 is required to pass.
- **Hit rate by bucket**: of the predictions between 0.55 and 0.60, how many beat SPY. These should line up with the bucket.

## Counting every experiment

Try 100 versions and the best one looks great by luck alone. So every training and backtest run is appended to `experiments/registry.jsonl`: time, code commit, config hash, data hash, parameters and metrics. Nothing is ever deleted.

The **Deflated Sharpe Ratio** (Bailey and Lopez de Prado, 2014) takes the best result and asks: given how many distinct versions were tried and how much their Sharpe ratios varied, how likely is it that this one's true Sharpe is above zero, allowing for skew and fat tails? The pass bar is 0.95. The code reproduces the paper's own worked example (DSR 0.9004) in a test.

What counts as a trial: a distinct configuration (effective settings plus run parameters). Retraining the same configuration on new data is not a new try. Changing anything that changes decisions is.

The **Probability of Backtest Overfitting** (PBO) is reported too: across many splits of the shared history, how often does the version that looked best in one half land in the bottom half of the other? Under 0.1 is good. Above 0.5 means the selection process is worse than random.

## Sizing

Weight for a new position = edge divided by volatility, then scaled so the whole book targets 15% annual volatility.

- Edge is a quarter of the Kelly bet on the calibrated probability, using the average win and loss from the training data.
- Volatility is 20 day realized volatility, floored at 10%.
- Halved on FOMC and CPI days and the day before.
- Then clipped by every hard cap: 20% per position, 5 positions, 2% cash, 40% per sector, and no position more than 0.8 correlated with one already held.

Sizing can only ask for less than the caps allow, never more.

## Exam order

1. **Walk-forward.** `stoa model train`. Must be calibrated, beat the base rate, and pass the deflated Sharpe bar.
2. **The sealed holdout.** `stoa holdout seal` refuses until walk-forward has passed. It records the promoted model, and `stoa holdout score` runs exactly that model, once.
3. **Paper.** At least 42 sessions (63 is the target) and 20 closed trades, graded on Stoa's cost-adjusted record: lower drawdown and higher Sharpe than SPY.
4. **Live.** Refused, on top of the triple lock, until 1 to 3 have passed. `stoa exam status` shows where things stand.

## Keeping it honest after launch

- **Monthly retrain** (`monthly.yml`): new data, same protocol. A new model replaces the current one only if it is no worse out of sample on the same recent rows, and calibrated. Otherwise the current model stays and the report says why. Every model is a versioned artifact with its metadata on the state branch.
- **Drift fallback**: if the rolling 63 session Sharpe falls below the walk-forward 5th percentile, or drawdown passes the walk-forward 95th percentile, Stoa stops opening positions and holds SPY.
- **Quit rule**: after six months of paper and live combined with no edge over SPY after costs, it holds SPY.
- Both need `stoa fallback reset` to undo.

## Reading the reports

`stoa model train` writes `results/<time>-model/calibration.md`:

1. The model table: compare log loss with the base rate's (shown in the verdict when it fails). ECE under 0.05. AUC above 0.5.
2. Hit rate by bucket: the hit rate should rise with the bucket. If 0.60 to 0.65 hits less often than 0.55 to 0.60, the top end is not trustworthy.
3. Per fold: look for a few folds carrying everything. A steady model has similar numbers in most folds.
4. The verdict: PASS or FAIL with the reasons, and the deflated Sharpe after all recorded trials.

`stoa model report --ablation` adds the text ablation: the same model with and without Claude's scores, split at Claude's knowledge cutoff (July 2026). Before the cutoff, Claude may remember what happened, so only the after-cutoff comparison counts as evidence. Keep `model.use_text: false` unless the clean period shows lower log loss with text.

`stoa score` writes `results/<time>/report.md`: every strategy on every window, the risk checks, the deflated Sharpe, and the verdicts against the 50/200 rule and MTUM.

## Example output (synthetic data)

`python scripts/demo.py` on seeded random data with a small planted momentum edge (AUC about 0.57). It shows the format only. The numbers mean nothing about real markets.

```text
## Walk-forward calibration (price+quality, 10 sessions)

Chosen model: **logistic** (lowest pooled out-of-sample log loss).

| Model | Rows | Base rate | Log loss | Brier | AUC | ECE | Hit rate when p > 0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic | 76117 | 0.530 | 0.6876 | 0.2463 | 0.568 | 0.015 | 0.567 (47811) |
| hgb | 76117 | 0.530 | 0.6895 | 0.2475 | 0.553 | 0.010 | 0.552 (53947) |

Hit rate by probability bucket (chosen model):

| Bucket | Rows | Mean p | Hit rate |
| --- | --- | --- | --- |
| 0.00 to 0.45 | 11993 | 0.410 | 0.444 |
| 0.45 to 0.50 | 16247 | 0.473 | 0.485 |
| 0.50 to 0.55 | 17356 | 0.527 | 0.522 |
| 0.55 to 0.60 | 13783 | 0.565 | 0.573 |
| 0.60 to 0.65 | 10247 | 0.616 | 0.583 |
| 0.65 to 1.00 | 6491 | 0.679 | 0.648 |

## Walk-forward verdict: PASS
```

The same run with no planted edge (`demo.signal: 0`) gives an AUC near 0.51, a log loss above the base rate, and a FAIL. That is the point: the protocol finds an edge that exists and does not invent one that does not.
