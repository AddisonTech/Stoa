# Risk

Every limit lives in `config/default.yaml`, is validated when the config loads, and is enforced in `stoa/risk.py`. That file is plain code: no network, no model. The same functions run in backtests, signal, paper and live.

Risk is checked twice. Once after the close when the plan is made, and again after the open right before each order, against the account as it is at that moment.

## The rules

| Rule | Default | When it is checked | Reason code |
| --- | --- | --- | --- |
| Max position size | 20% of equity | plan and execution | sized down, never above |
| Max open positions | 5 | plan and execution | `max_positions` |
| Max gross exposure | 100% of equity | plan and execution | sized down |
| Cash buffer | 2% of equity | plan and execution | `insufficient_cash` |
| Min order | $1.00 | plan and execution | `min_order` |
| Max orders per day | 10 | plan and execution | `max_orders` |
| Daily loss limit | no new buys after a 3% down day | plan and execution | `daily_loss_limit` |
| Risk off | no new buys while SPY is below its 200 day average | plan | `risk_off` |
| Kill switch | flatten and halt at 25% below the high-water mark | plan and execution | `kill_switch` |
| Same-day round trip | never | plan and execution | `same_day_round_trip` |
| Watchlist only | 29 symbols | plan and execution | `not_in_watchlist` |
| Fractionable only | from the broker's asset flags | plan and execution | `not_fractionable` |
| Price sanity | skip a buy if the price moved more than 8% from the plan | execution | `price_sanity` |
| Shorting, options, leverage | never | always | not possible in code |

Sells always run first and are never blocked by the daily loss limit, risk off, or price sanity. Skipping a stop after a gap down would keep the account in the position it is trying to leave.

Buys are sized again at execution from the real cash after sells have filled. A plan never assumes money it does not have.

## Exits

All exits are checked on the close and sell the whole position at the next open.

| Exit | Rule | Reason code |
| --- | --- | --- |
| Stop | close at or below entry minus 2.5 x ATR(14) at entry | `stop_loss` |
| Trailing stop | close at or below the highest high since entry minus 3 x ATR at entry (never below the first stop) | `trailing_stop` |
| Target | optional, off by default | `take_profit` |
| Time | held 30 sessions | `time_exit` |
| Trend break | two closes in a row below the 50 day average | `trend_break` |

## The kill switch

The high-water mark is the highest equity at any close. If equity falls 25% or more below it, the kill switch trips:

1. An urgent alert goes to your phone.
2. Every Stoa position is sold at the next open.
3. No new buys happen until a person resets it.

The check runs after each close and again at the open, so an overnight gap can trip it too. Positions outside the watchlist are left alone.

To reset, see the runbook. A reset sets the high-water mark to current equity, so the switch does not trip again the next day.

Withdrawals do not trip it. When Stoa sees cash leave the account in Alpaca's activity feed, it lowers the high-water mark by the same amount.

## Live money

Live trading needs all three of these. Any one missing and DECIDE and EXECUTE both refuse and send an urgent alert.

1. `mode: live` in config (or the `STOA_MODE` repository variable set to `live`).
2. The secret `STOA_LIVE_CONFIRM` set to exactly `I_UNDERSTAND`.
3. The secrets `ALPACA_LIVE_KEY_ID` and `ALPACA_LIVE_SECRET`.

Paper keys are never used for live orders. They are separate secrets with separate names.

## Things the broker would allow that Stoa does not

Alpaca has no cash accounts. An account under $2,000 is a limited margin account with 1x buying power. Stoa does not rely on that. It caps gross exposure at 100% of equity itself, keeps a cash buffer and never submits an order larger than its cash.
