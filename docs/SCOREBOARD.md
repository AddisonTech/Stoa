# Scoreboard

Stoa is graded on risk, not raw return. It passes only if it has less drawdown and more return per unit of risk than holding SPY, after costs, with the score penalized for every version ever tried.

Every strategy runs on the same windows, the same costs and the same NYSE calendar:

- **model**: the v0.2 strategy. Its probabilities in every backtest come only from the walk-forward: each session's numbers were produced by a model trained on data that ended before that session's test slice. This is the strategy the pass bar grades.
- **agents**: the v1 pipeline (rule or Claude scout, quant gates, risk engine), kept as a comparison.
- **spy**: buy SPY with all cash at the first open and hold.
- **sma**: fully in SPY when its 50 day average is above its 200 day average, otherwise cash.
- **mtum**: buy the iShares MSCI USA Momentum Factor ETF (MTUM) and hold. MTUM history starts in 2013.

The baselines run through the same backtest engine, simulated broker, slippage and fees. They are not risk managed, because a baseline with a cash buffer and a 20% cap would no longer be the thing it claims to measure.

## Costs

- Slippage: 5 bps against you on every fill, both sides.
- Regulatory sell fees: 0.3 bps of the sale, rounded up to the cent, at least $0.01.
- Buys fill at the next open plus slippage. Positions are marked at the close.
- A stock that stops trading while held is sold at its last close less slippage.

## Windows

| Window | Dates | Use |
| --- | --- | --- |
| tune_a | 2019-01-01 to 2020-12-31 | Tuning |
| tune_b | 2021-01-01 to 2022-12-31 | Tuning |
| tune_c | 2023-01-01 to 2024-12-31 | Tuning |
| validation | 2025-01-01 to 2026-06-30 | Checked, not tuned on. The model choice ignores it. |
| holdout | 2026-07-01 to 2026-10-23 | Sealed, scored once. See HOLDOUT.md |

The model's walk-forward needs about two years of training before its first test slice, so it starts making predictions in early 2019. Data begins in 2016.

## The pass bar

From `config/scoreboard.yaml`. The graded strategy passes only if every line holds.

| Check | Rule |
| --- | --- |
| drawdown_vs_spy | Max drawdown strictly lower than SPY's on the validation window. |
| sharpe_vs_spy | Sharpe strictly higher than SPY's on the validation window. |
| sortino_vs_spy | Sortino strictly higher than SPY's on the validation window. |
| deflated_sharpe | Deflated Sharpe ratio at least 0.95, counting every distinct configuration in the experiment registry. |
| drawdown_cap | Max drawdown never past 25% in any window. |
| tune_windows | Sharpe above SPY in at least 2 of the 3 tune windows. |
| trade_count | 30 or more closed trades across all windows. |
| api_cost | Claude spend under $25 in every month. |

Also reported, every run:

- **Versus the 50/200 rule**: better or not on risk.
- **Versus MTUM**: if Stoa does not beat MTUM on drawdown, Sharpe and Sortino together, the report says so and recommends buying the fund. Set `beat_mtum_for_pass: true` to make that a hard check.

Total return is reported but is not a pass criterion. A strategy that makes a little less than SPY with half the drawdown passes. One that makes more with a deeper drawdown does not.

The bar is locked by `stoa holdout seal` before the holdout is scored.

## The trial penalty

Every `stoa score`, `stoa model train` and `stoa model report` run is appended to the experiment registry. The scoreboard logs the graded run, then deflates its Sharpe by every distinct configuration recorded so far. Rerunning the same configuration does not add a trial. Changing a setting that changes decisions does. See TRAINING.md.

## Running it

```bash
stoa data fetch          # watchlist and MTUM bars, needs Alpaca paper keys (or --provider yahoo)
stoa universe build      # the point-in-time training universe
stoa fundamentals fetch  # SEC quality features and earnings times
stoa score               # walk-forward the model, backtest everything, grade
stoa score --no-model    # grade the v1 pipeline instead
```

Each run writes `results/<run-id>/report.md` and `report.json` with every metric: total return, CAGR, volatility, Sharpe, Sortino, max drawdown and its length, Calmar, profit factor, win rate, average win and loss, expectancy, exposure, turnover, trade count, monthly returns, a seeded block bootstrap of the max drawdown, the deflated Sharpe and the PBO.

## Current status

Not passed. The bar has not been run on real market data in this repository yet, because that needs a data key.

## Sample report (synthetic data)

This is `python scripts/demo.py`: seeded random data with a small planted momentum edge. It shows the format only. The numbers mean nothing about real markets.

It also shows the bar doing its job. The model beats SPY on drawdown, Sharpe and Sortino and beats MTUM, and still fails: three recorded trials are enough to pull its deflated Sharpe under 0.95. One more lucky-looking variant would pull it lower.

| Check | Result | Detail |
| --- | --- | --- |
| drawdown_vs_spy | pass | model 4.42% vs SPY 7.40% (validation) |
| sharpe_vs_spy | pass | model 1.69 vs SPY 1.18 (validation) |
| sortino_vs_spy | pass | model 2.53 vs SPY 1.74 (validation) |
| deflated_sharpe | FAIL | DSR 0.914 after 3 recorded trials (need 0.95) |
| drawdown_cap | pass | worst model drawdown 7.02% (cap 25%) |
| tune_windows | pass | Sharpe above SPY in 3 of 3 (tune_a, tune_b, tune_c) |
| trade_count | pass | 234 trades across all windows (need 30) |
| api_cost | pass | max $0.00 in any month (cap $25.00) |

| Window | Strategy | Return | CAGR | Max DD | Sharpe | Sortino | PF | Trades | Exp. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tune_a | agents | -2.03% | -1.02% | 6.93% | -0.14 | -0.18 | 0.21 | 13 | 12% |
| tune_a | model | +0.27% | +0.13% | 4.47% | 0.05 | 0.07 | 0.53 | 13 | 8% |
| tune_a | spy | -31.78% | -17.38% | 38.87% | -1.15 | -1.56 | n/a | 0 | 100% |
| tune_a | sma | -7.34% | -3.73% | 12.30% | -0.88 | -1.13 | 0.00 | 1 | 7% |
| tune_a | mtum | -48.37% | -28.10% | 63.33% | -1.33 | -1.76 | n/a | 0 | 100% |
| tune_b | agents | +84.03% | +35.74% | 11.47% | 1.96 | 3.21 | 2.58 | 102 | 70% |
| tune_b | model | +63.56% | +27.95% | 5.70% | 2.53 | 4.48 | 3.64 | 116 | 44% |
| tune_b | spy | +31.93% | +14.89% | 21.06% | 0.91 | 1.39 | n/a | 0 | 100% |
| tune_b | sma | +33.44% | +15.55% | 14.67% | 1.00 | 1.56 | inf | 1 | 85% |
| tune_b | mtum | +111.90% | +45.67% | 36.36% | 1.71 | 2.74 | n/a | 0 | 100% |
| tune_c | agents | -1.64% | -0.82% | 9.83% | -0.11 | -0.16 | 0.85 | 15 | 9% |
| tune_c | model | -4.31% | -2.19% | 7.02% | -0.79 | -1.03 | 0.32 | 16 | 4% |
| tune_c | spy | -32.77% | -18.07% | 38.98% | -1.17 | -1.60 | n/a | 0 | 100% |
| tune_c | sma | -11.47% | -5.93% | 11.65% | -1.47 | -1.73 | 0.00 | 1 | 8% |
| tune_c | mtum | +22.22% | +10.60% | 23.98% | 0.56 | 0.84 | n/a | 0 | 100% |
| validation | agents | +35.69% | +22.90% | 11.46% | 1.36 | 1.97 | 1.96 | 80 | 77% |
| validation | model | +20.41% | +13.37% | 4.42% | 1.69 | 2.53 | 2.52 | 89 | 36% |
| validation | spy | +28.68% | +18.57% | 7.40% | 1.18 | 1.74 | n/a | 0 | 100% |
| validation | sma | +21.32% | +13.95% | 7.21% | 1.01 | 1.48 | inf | 1 | 80% |
| validation | mtum | +18.12% | +11.91% | 21.36% | 0.62 | 0.93 | n/a | 0 | 100% |

- Versus the 50/200 rule: better on risk (drawdown 4.4% vs 7.2%, Sharpe 1.69 vs 1.01).
- Stoa beats MTUM on the risk-adjusted bar.

Deflated Sharpe: 0.914 with 3 trials in the registry (daily Sharpe 0.1062, luck threshold SR0 0.0341, PBO 0.09).

Bootstrap max drawdown (model, 5th / 50th / 95th percentile):

- tune_a: 2.0% / 4.6% / 9.4%
- tune_b: 4.0% / 6.2% / 10.2%
- tune_c: 1.6% / 5.5% / 12.2%
- validation: 3.1% / 4.8% / 7.9%
