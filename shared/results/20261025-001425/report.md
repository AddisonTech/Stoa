# Stoa scoreboard 20261025-001425

Data: cache:data\bars. Strategy hash: `85456507c9a9`.

Graded strategy: `event`.

**Result: FAIL**

| Check | Result | Detail |
| --- | --- | --- |
| drawdown_vs_spy | FAIL | event 34.21% vs SPY 18.75% (validation) |
| sharpe_vs_spy | FAIL | event -0.04 vs SPY 1.04 (validation) |
| sortino_vs_spy | FAIL | event -0.06 vs SPY 1.57 (validation) |
| deflated_sharpe | FAIL | DSR 0.070 after 8 recorded trials (need 0.95) |
| drawdown_cap | FAIL | worst event drawdown 51.80% (cap 25%) |
| tune_windows | FAIL | Sharpe above SPY in 0 of 3 (none) |
| trade_count | pass | 152 trades across all windows (need 30) |
| api_cost | pass | max $0.00 in any month (cap $25.00) |
| sharpe_vs_mtum | FAIL | event -0.04 vs MTUM 1.50 (validation) |

| Window | Strategy | Return | CAGR | Max DD | Sharpe | Sortino | PF | Trades | Exp. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tune_a | agents | +24.83% | +11.70% | 19.73% | 0.68 | 0.92 | 1.50 | 112 | 74% |
| tune_a | event | +1.60% | +0.80% | 51.80% | 0.18 | 0.25 | 0.32 | 27 | 77% |
| tune_a | spy | +57.85% | +25.58% | 33.79% | 1.06 | 1.45 | n/a | 0 | 100% |
| tune_a | sma | +7.53% | +3.69% | 33.79% | 0.28 | 0.36 | 0.00 | 1 | 75% |
| tune_a | mtum | +68.07% | +29.57% | 34.08% | 1.13 | 1.60 | n/a | 0 | 100% |
| tune_b | agents | +16.91% | +8.14% | 13.75% | 0.63 | 0.90 | 1.38 | 90 | 60% |
| tune_b | event | -13.24% | -6.87% | 35.23% | -0.21 | -0.31 | 0.85 | 67 | 68% |
| tune_b | spy | +4.88% | +2.41% | 24.50% | 0.22 | 0.31 | n/a | 0 | 100% |
| tune_b | sma | +16.94% | +8.16% | 12.86% | 0.73 | 1.05 | inf | 1 | 60% |
| tune_b | mtum | -7.80% | -3.99% | 32.29% | -0.06 | -0.08 | n/a | 0 | 100% |
| tune_c | agents | +39.55% | +18.21% | 17.78% | 1.03 | 1.50 | 1.69 | 128 | 84% |
| tune_c | event | +26.31% | +12.44% | 26.34% | 0.60 | 0.87 | 0.45 | 43 | 92% |
| tune_c | spy | +56.70% | +25.29% | 9.97% | 1.82 | 2.73 | n/a | 0 | 100% |
| tune_c | sma | +49.22% | +22.25% | 9.97% | 1.68 | 2.48 | n/a | 0 | 97% |
| tune_c | mtum | +45.27% | +20.62% | 12.96% | 1.22 | 1.80 | n/a | 0 | 100% |
| validation | agents | +28.54% | +18.48% | 16.98% | 1.15 | 1.68 | 1.41 | 93 | 74% |
| validation | event | -10.55% | -7.26% | 34.21% | -0.04 | -0.06 | 0.14 | 15 | 43% |
| validation | spy | +28.81% | +18.66% | 18.75% | 1.04 | 1.57 | n/a | 0 | 100% |
| validation | sma | +9.64% | +6.42% | 18.75% | 0.45 | 0.66 | 0.00 | 1 | 87% |
| validation | mtum | +66.50% | +41.12% | 20.99% | 1.50 | 2.20 | n/a | 0 | 100% |

- Versus the 50/200 rule: not better on risk (drawdown 34.2% vs 18.8%, Sharpe -0.04 vs 0.45).
- Stoa does not beat MTUM on the risk-adjusted bar (drawdown 34.2% vs 21.0%, Sharpe -0.04 vs 1.50, Sortino -0.06 vs 2.20). Recommendation: buy the momentum fund (MTUM) instead.

Deflated Sharpe: 0.070 with 8 trials in the registry (daily Sharpe -0.0026, luck threshold SR0 0.0737, PBO nan).

Bootstrap max drawdown (event, 5th / 50th / 95th percentile):

- tune_a: 16.1% / 40.2% / 69.1%
- tune_b: 17.4% / 30.9% / 49.4%
- tune_c: 14.8% / 26.7% / 46.1%
- validation: 21.6% / 42.0% / 66.7%

Per-signal diagnostics (never graded or picked between):

| Window | Signal | Trades | Beat SPY | Avg excess per trade |
| --- | --- | --- | --- | --- |
| tune_a | insider_cluster | 15 | 13.3% | -9.07% |
| tune_a | earnings_drift | 12 | 33.3% | -0.87% |
| tune_b | insider_cluster | 34 | 32.4% | -4.54% |
| tune_b | earnings_drift | 33 | 51.5% | +1.11% |
| tune_c | insider_cluster | 26 | 34.6% | -3.50% |
| tune_c | earnings_drift | 17 | 23.5% | -3.87% |
| validation | insider_cluster | 9 | 22.2% | -6.14% |
| validation | earnings_drift | 6 | 33.3% | -0.07% |
