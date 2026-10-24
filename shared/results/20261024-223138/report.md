# Stoa scoreboard 20261024-223138

Data: cache:data\bars. Strategy hash: `85d4eec04d72`.

Graded strategy: `model`.

**Result: FAIL**

| Check | Result | Detail |
| --- | --- | --- |
| drawdown_vs_spy | pass | model 1.27% vs SPY 18.75% (validation) |
| sharpe_vs_spy | FAIL | model -1.41 vs SPY 1.04 (validation) |
| sortino_vs_spy | FAIL | model -1.43 vs SPY 1.57 (validation) |
| deflated_sharpe | FAIL | DSR 0.000 after 7 recorded trials (need 0.95) |
| drawdown_cap | pass | worst model drawdown 2.73% (cap 25%) |
| tune_windows | FAIL | Sharpe above SPY in 0 of 3 (none) |
| trade_count | FAIL | 16 trades across all windows (need 30) |
| api_cost | pass | max $0.00 in any month (cap $25.00) |

| Window | Strategy | Return | CAGR | Max DD | Sharpe | Sortino | PF | Trades | Exp. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tune_a | agents | +24.83% | +11.70% | 19.73% | 0.68 | 0.92 | 1.50 | 112 | 74% |
| tune_a | model | +0.77% | +0.38% | 2.73% | 0.20 | 0.27 | 1.23 | 15 | 7% |
| tune_a | spy | +57.85% | +25.58% | 33.79% | 1.06 | 1.45 | n/a | 0 | 100% |
| tune_a | sma | +7.53% | +3.69% | 33.79% | 0.28 | 0.36 | 0.00 | 1 | 75% |
| tune_a | mtum | +68.07% | +29.58% | 34.08% | 1.13 | 1.60 | n/a | 0 | 100% |
| tune_b | agents | +16.91% | +8.14% | 13.75% | 0.63 | 0.90 | 1.38 | 90 | 60% |
| tune_b | model | +0.00% | +0.00% | -0.00% | 0.00 | 0.00 | n/a | 0 | 0% |
| tune_b | spy | +4.88% | +2.41% | 24.50% | 0.22 | 0.31 | n/a | 0 | 100% |
| tune_b | sma | +16.94% | +8.16% | 12.86% | 0.73 | 1.05 | inf | 1 | 60% |
| tune_b | mtum | -7.78% | -3.97% | 32.29% | -0.06 | -0.08 | n/a | 0 | 100% |
| tune_c | agents | +39.55% | +18.21% | 17.78% | 1.03 | 1.50 | 1.69 | 128 | 84% |
| tune_c | model | +1.10% | +0.55% | 0.71% | 0.79 | 1.23 | n/a | 0 | 1% |
| tune_c | spy | +56.70% | +25.29% | 9.97% | 1.82 | 2.73 | n/a | 0 | 100% |
| tune_c | sma | +49.22% | +22.25% | 9.97% | 1.68 | 2.48 | n/a | 0 | 97% |
| tune_c | mtum | +45.35% | +20.65% | 12.96% | 1.22 | 1.81 | n/a | 0 | 100% |
| validation | agents | +28.54% | +18.48% | 16.98% | 1.15 | 1.68 | 1.41 | 93 | 74% |
| validation | model | -1.27% | -0.86% | 1.27% | -1.41 | -1.43 | 0.00 | 1 | 0% |
| validation | spy | +28.81% | +18.66% | 18.75% | 1.04 | 1.57 | n/a | 0 | 100% |
| validation | sma | +9.64% | +6.42% | 18.75% | 0.45 | 0.66 | 0.00 | 1 | 87% |
| validation | mtum | +66.51% | +41.13% | 20.99% | 1.50 | 2.20 | n/a | 0 | 100% |

- Versus the 50/200 rule: not better on risk (drawdown 1.3% vs 18.8%, Sharpe -1.41 vs 0.45).
- Stoa does not beat MTUM on the risk-adjusted bar (drawdown 1.3% vs 21.0%, Sharpe -1.41 vs 1.50, Sortino -1.43 vs 2.20). Recommendation: buy the momentum fund (MTUM) instead.

Deflated Sharpe: 0.000 with 7 trials in the registry (daily Sharpe -0.0888, luck threshold SR0 0.0871, PBO nan).

Bootstrap max drawdown (model, 5th / 50th / 95th percentile):

- tune_a: 1.0% / 2.4% / 5.1%
- tune_b: 0.0% / 0.0% / 0.0%
- tune_c: 0.0% / 0.7% / 1.1%
- validation: 0.0% / 1.3% / 3.4%
