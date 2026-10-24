# Scoreboard

Stoa is graded against two baselines on the same windows, the same costs and the same NYSE calendar:

- **SPY**: buy SPY with all cash at the first open and hold.
- **50/200**: fully in SPY when its 50 day average is above its 200 day average, otherwise in cash.

Both baselines run through the same backtest engine, simulated broker, slippage and fees as the agents. They are not risk managed, because a baseline with a cash buffer and a 20% cap would no longer be the thing it claims to measure.

## Costs

- Slippage: 5 bps against you on every fill, both sides.
- Regulatory sell fees: 0.3 bps of the sale, rounded up to the cent, at least $0.01.
- Buys fill at the next open plus slippage. Positions are marked at the close.

## Windows

| Window | Dates | Use |
| --- | --- | --- |
| tune_a | 2019-01-01 to 2020-12-31 | Tuning |
| tune_b | 2021-01-01 to 2022-12-31 | Tuning |
| tune_c | 2023-01-01 to 2024-12-31 | Tuning |
| validation | 2025-01-01 to 2026-06-30 | Checked, not tuned on |
| holdout | 2026-07-01 onward | Sealed, scored once. See HOLDOUT.md |

Every window needs 260 sessions of history before it starts, so data begins in 2016. The first decision is made on the close before the window, and the first fill is on the window's first open, for every strategy.

## The pass bar

From `config/scoreboard.yaml`. The agents pass only if every line holds.

| Check | Rule |
| --- | --- |
| beats_spy | Total return above SPY buy-and-hold after costs, on the validation window. |
| beats_sma | Total return above the 50/200 rule after costs, on the validation window. |
| drawdown_vs_spy | Max drawdown no worse than SPY's on the validation window. |
| drawdown_cap | Max drawdown never past 25% in any window. |
| profit_factor | Gross profit over gross loss at least 1.2 on the validation window. |
| tune_windows | Beats SPY in at least 2 of the 3 tune windows. |
| trade_count | 30 or more closed trades across all windows. |
| api_cost | Claude API spend under $5 in every month. |

The bar is locked by `stoa holdout seal` before the holdout is scored.

## Running it

```bash
stoa data fetch          # once, needs Alpaca paper keys (or --provider yahoo)
stoa score               # rule scout
stoa score --llm         # Claude scout; replays cached answers, pays only for new ones
```

Each run writes `results/<run-id>/report.md` and `report.json` with every metric: total return, CAGR, volatility, Sharpe, Sortino, max drawdown and its length, Calmar, profit factor, win rate, average win and loss, expectancy, exposure, turnover, trade count, monthly returns, and a seeded block bootstrap of the max drawdown.

## Current status

Not passed. The bar has not been run on real market data in this repository yet, because that needs a data key.

## Sample report (synthetic data)

This is `python scripts/demo.py` on seeded random data. It shows the format only. The numbers mean nothing about real markets. It also shows the bar doing its job: the rule scout wins the tune windows on this data and still fails on validation.

| Check | Result | Detail |
| --- | --- | --- |
| beats_spy | FAIL | agents +9.53% vs SPY +28.68% (validation) |
| beats_sma | FAIL | agents +9.53% vs 50/200 +21.32% (validation) |
| drawdown_vs_spy | FAIL | agents 14.25% vs SPY 7.40% (validation) |
| drawdown_cap | pass | worst agents drawdown 16.55% (cap 25%) |
| profit_factor | FAIL | 1.13 (need 1.2) (validation) |
| tune_windows | pass | beat SPY in 3 of 3 (tune_a, tune_b, tune_c) |
| trade_count | pass | 241 trades across all windows (need 30) |
| api_cost | pass | max $0.00 in any month (cap $5.00) |

| Window | Strategy | Return | CAGR | Max DD | Sharpe | Sortino | PF | Trades | Exp. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tune_a | agents | -0.72% | -0.36% | 7.65% | -0.03 | -0.04 | 0.16 | 14 | 10% |
| tune_a | spy | -31.78% | -17.38% | 38.87% | -1.15 | -1.56 | n/a | 0 | 100% |
| tune_a | sma | -7.34% | -3.73% | 12.30% | -0.88 | -1.13 | 0.00 | 1 | 7% |
| tune_b | agents | +37.17% | +17.16% | 16.55% | 1.05 | 1.59 | 1.67 | 114 | 69% |
| tune_b | spy | +31.93% | +14.89% | 21.06% | 0.91 | 1.39 | n/a | 0 | 100% |
| tune_b | sma | +33.44% | +15.55% | 14.67% | 1.00 | 1.56 | inf | 1 | 85% |
| tune_c | agents | -5.14% | -2.62% | 11.19% | -0.44 | -0.60 | 0.54 | 14 | 8% |
| tune_c | spy | -32.77% | -18.07% | 38.98% | -1.17 | -1.60 | n/a | 0 | 100% |
| tune_c | sma | -11.47% | -5.93% | 11.65% | -1.47 | -1.73 | 0.00 | 1 | 8% |
| validation | agents | +9.53% | +6.34% | 14.25% | 0.45 | 0.62 | 1.13 | 99 | 75% |
| validation | spy | +28.68% | +18.57% | 7.40% | 1.18 | 1.74 | n/a | 0 | 100% |
| validation | sma | +21.32% | +13.95% | 7.21% | 1.01 | 1.48 | inf | 1 | 80% |

Bootstrap max drawdown (agents, 5th / 50th / 95th percentile):

- tune_a: 4.3% / 7.3% / 13.2%
- tune_b: 9.7% / 16.1% / 28.6%
- tune_c: 3.2% / 8.6% / 19.0%
- validation: 9.8% / 17.1% / 30.6%
