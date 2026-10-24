# Risk

Every limit lives in `config/default.yaml`, is validated when the config loads, and is enforced in `stoa/risk.py` (with sizing in `stoa/sizing.py`). Both are plain code: no network, no model. The same functions run in backtests, signal, paper and live.

Risk is checked twice. Once after the close when the plan is made, and again after the open right before each order, against the account as it is at that moment.

## The rules

| Rule | Default | When it is checked | Reason code |
| --- | --- | --- | --- |
| Max position size | 20% of equity | plan and execution | sized down, never above |
| Max open positions | 5 | plan and execution | `max_positions` |
| Max gross exposure | 100% of equity | plan and execution | sized down |
| Cash buffer | 2% of equity | plan and execution | `insufficient_cash` |
| Sector cap | 40% of equity per GICS sector | plan and execution | `sector_cap` |
| Correlation check | no new position whose 60 day return correlation with a held or planned one is above 0.8 | plan | `correlation` |
| Min order | $1.00 | plan and execution | `min_order` |
| Max orders per day | 10 | plan and execution | `max_orders` |
| Daily loss limit | no new buys after a 3% down day (deposits and withdrawals excluded) | plan and execution | `daily_loss_limit` |
| Regime filter | no new buys while SPY is below its 200 day average | plan | `risk_off` |
| Earnings, no entry | no new position with a confirmed release inside the next 10 sessions, or with no confirmed date when the last report is over 80 days old | plan | `earnings` |
| Event days | new positions sized at 0.5x on FOMC and CPI days and the day before | plan | (size note in the alert) |
| Kill switch | flatten and halt at 25% below the high-water mark | plan and execution | `kill_switch` |
| Fallback halt | no new buys except SPY while the drift fallback or quit rule is active | plan and execution | `halted` |
| Same-day round trip | never | plan and execution | `same_day_round_trip` |
| Watchlist only | 29 symbols | plan and execution | `not_in_watchlist` |
| Fractionable only | from the broker's asset flags | plan and execution | `not_fractionable` |
| Price sanity | skip a buy if the price moved more than 8% from the plan | execution | `price_sanity` |
| Price cross-check | skip a buy if a second source differs by more than 1%, or cannot be reached | execution | `price_check` |
| Hand approval | skip any order not approved with the plan's nonce before the window closes | execution | `not_approved` |
| Shorting, options, leverage | never | always | not possible in code |

Sells always run first and are never blocked by the daily loss limit, the regime filter, price sanity or the price cross-check. Skipping a stop after a gap down would keep the account in the position it is trying to leave. During the hand-approval month, sells need a tap too (they are trades); the one exception is the kill switch flatten, which never waits.

Buys are sized again at execution from the real cash after sells have filled. A plan never assumes money it does not have.

## Sizing

The model gives a calibrated probability p of beating SPY. For each new idea:

1. Edge = a quarter of the Kelly bet: 0.25 x (p - (1 - p) / b), where b is the average win over the average loss (in excess return) from the training data. Never negative.
2. Raw weight = edge / annualized 20 day volatility (floored at 10%).
3. All new weights are scaled by one factor so the whole book, existing positions included, targets 15% annualized volatility.
4. Times 0.5 on event days.
5. Clipped by every cap above, in order: position cap, sector room, cash after the buffer, gross exposure.

So a 60% probability on a calm stock gets more than a 60% probability on a jumpy one, and both get less than a 70% probability on the calm one, and none of them can ever exceed 20%.

## Exits

All exits are checked on the close and sell the whole position at the next open.

| Exit | Rule | Reason code |
| --- | --- | --- |
| Stop | close at or below entry minus 2.5 x ATR(14) at entry | `stop_loss` |
| Trailing stop | close at or below the highest high since entry minus 3 x ATR at entry (never below the first stop) | `trailing_stop` |
| Target | optional, off by default | `take_profit` |
| Time | held 30 sessions | `time_exit` |
| Trend break | two closes in a row below the 50 day average | `trend_break` |
| Earnings | a confirmed release before the open after next: sell at the next open | `earnings_exit` |
| Earnings unknown | no confirmed date and the last report is over 80 days old | `earnings_unknown` |
| Horizon | held 10 sessions and the model no longer gives it over 55% | `horizon_exit` |

Entry price and ATR are rescaled whenever a split or dividend re-adjusts history, so a split never fires a stop.

### Stops at the broker

The rules above look at the close. During the day, a real stop order at Alpaca covers every position held from a prior day.

- **Price.** DECIDE stores each position's stop: the stop or the trailing stop above, whichever is higher, rounded up to the cent. It only ever goes up, so it is never looser than the rule, and it is rescaled with the entry on a split.
- **Placed.** EXECUTE runs the plan first, then places a DAY stop sell for the whole position. Positions are fractional at this size and Alpaca only takes fractional orders as DAY orders, so the stops expire at the close and are placed again every morning. A position bought that morning gets no stop, so nothing is ever bought and sold the same day. If the price is already at or below the stop at the open, no order is placed (it would fill at once) and you get an alert; the close rule sells it if it stays there.
- **Before a sell.** Any planned sell, including the kill switch flatten, first cancels that position's open Stoa stop and waits for the broker to confirm, so the sell is never rejected for shares held by the stop.
- **Fills.** DECIDE checks every stop it placed. A filled one is recorded as a `stop_out` trade with its fill price and time, the position is closed in state, and an alert goes out.
- **Only its own.** Every stop's client order id starts with `stoa-stop-`. Stoa never lists, cancels or replaces any other order.
- **Failures.** If placing, cancelling or checking a stop fails, an urgent alert goes out and the rest of the run carries on.
- **Kill switch.** No new buys, as always. Protective stops stay: any position still held after the flatten gets its stop.

Paper and live use the same code. The live lock is unchanged.

## The kill switch

The high-water mark is the highest equity at any close. If equity falls 25% or more below it, the kill switch trips:

1. An urgent alert goes to your phone.
2. Every Stoa position is sold at the next open (its broker stop is cancelled first).
3. No new buys happen until a person resets it.

The check runs after each close and again at the open, so an overnight gap can trip it too. Positions outside the watchlist are left alone.

Withdrawals do not trip it. Stoa reads cash transfers from Alpaca at DECIDE and again at EXECUTE, before the check, and moves the high-water mark by the same amount.

## The drift fallback and the quit rule

Walk-forward gives an expected rolling Sharpe band and a drawdown band. If paper or live drifts worse than that (rolling 63 session Sharpe below the 5th percentile, or drawdown past the 95th percentile), or six months of paper and live together show no edge over SPY after costs:

1. An urgent alert goes to your phone.
2. No new positions open. Existing ones exit by their normal rules.
3. Stoa buys SPY with the freed cash, up to 98% of equity, still under the kill switch and the cash buffer.
4. Only `stoa fallback reset` undoes it.

## Honest paper

Alpaca's paper fills have no slippage. Stoa charges itself the backtest's 5 bps per side plus regulatory sell fees on every paper and live fill, and grades the paper exam, the fallback and the quit rule on that cost-adjusted record. The raw broker number is shown too.

## Live money

Live trading needs all of these. Any one missing and DECIDE and EXECUTE both refuse and send an urgent alert.

1. `mode: live` in config (or the `STOA_MODE` repository variable set to `live`).
2. The secret `STOA_LIVE_CONFIRM` set to exactly `I_UNDERSTAND`.
3. The secrets `ALPACA_LIVE_KEY_ID` and `ALPACA_LIVE_SECRET`.
4. A passed walk-forward and a passed paper exam (`stoa exam status`).

Paper keys are never used for live orders. They are separate secrets with separate names.

## Things the broker would allow that Stoa does not

Alpaca has no cash accounts. An account under $2,000 is a limited margin account with 1x buying power. Stoa does not rely on that. It caps gross exposure at 100% of equity itself, keeps a cash buffer and never submits an order larger than its cash.
