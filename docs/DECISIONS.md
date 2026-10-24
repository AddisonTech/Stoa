# Decisions

Choices made while building Stoa, and why. Newest last.

## Modules beyond the architecture list

Every module in ARCHITECTURE.md exists. A few more were added to keep boundaries clean:

- `models.py`: shared value types (orders, fills, trades) and Decimal helpers.
- `strategy.py`: the agent pipeline (exits, scout, quant, risk) and the strategy factory.
- `trader.py`: the one execution path shared by backtests, signal, paper and live.
- `net.py`: the HTTP retry rule shared by the trading and market data clients.
- `data/base.py`, `data/alpaca.py`, `data/yahoo.py`, `data/csv.py`, `data/cache.py`, `data/synthetic.py`, `data/factory.py`: the providers behind one interface.
- `scripts/state-branch.sh`, `scripts/check_text.py`, `scripts/demo.py`: workflow and CI helpers and the offline demo.

## Money is Decimal, indicators are float

Cash, quantities, fill prices and fees are `Decimal` end to end, so the accounting tests can assert exact equality. Indicators stay float in pandas. A float crosses into money in one place, `models.dec`, which uses the float's shortest repr (and unwraps numpy floats first).

## Fixed windows for every indicator

ATR and RSI use simple rolling averages instead of Wilder's exponential smoothing. An exponential average depends on how far back the history starts, so a live run that fetches 450 days would get slightly different numbers than a 10 year backtest. Fixed windows make the two identical, and a test checks that.

## One execution path

`trader.execute_plan` runs in backtests, signal mode, paper and live. It sells first, then resizes buys to the cash actually available, rechecks every rule, and looks up each client order id before submitting. Because the backtest depends on it, the trader and the execution-time risk checks were committed with v0.1 rather than v0.3. The risk tests themselves came in v0.3.

## Sells always close the whole position

Every exit sells everything held in that symbol, at the quantity the broker reports at execution. A split overnight changes the share count; selling the planned count would leave a stub.

## Baselines skip the risk engine

SPY buy-and-hold and the 50/200 rule are executed without the 20% cap, the cash buffer or the kill switch. Otherwise they would not measure what their names say. They use the same fills, slippage, fees and calendar.

## The first decision is the close before the window

So every strategy holds cash at the window's first open and gets its first fill there. Equity curves start at the starting equity on the day before the window.

## Backtests never reset the kill switch

Once tripped in a backtest, the agents stay flat for the rest of the window. That is the pessimistic reading of "manual reset only".

## Kill switch checked at the close and at the open

The architecture says the switch trips at a 25% drawdown. Stoa checks after each close (and plans the flatten) and again at the open before any order (so an overnight gap trips it the same morning). The high-water mark only moves on closes.

## Equity at DECIDE is marked at the official close

Alpaca marks positions at the latest trade, which after 4 pm can be an after-hours print. DECIDE re-marks watchlist positions at the session's closing bar so sizing, the daily loss limit and the kill switch use the same prices as the backtest.

## Daily loss baseline

The daily loss limit compares today's close to Stoa's own record of the previous close. If Stoa has no record for that session, it falls back to Alpaca's `last_equity`.

## Signal mode keeps a shadow portfolio

Signal mode never touches a broker, but it still needs equity, positions and stops to make sensible plans. It keeps a simulated portfolio in state and settles the previous plan at today's open, using today's completed bar, through the same execution code. It starts from `money.deposit`.

## Empty plans are kept

A plan with no orders still goes to EXECUTE, so the kill switch is checked at every open.

## Orders still working after 90 seconds

EXECUTE waits up to `alpaca.fill_timeout_seconds` for each order to finish. Anything still open is recorded in state, alerted, and settled at the next DECIDE (entry metadata for buys, realized P&L for sells).

## Data freshness

DECIDE fetches about 450 calendar days of bars for every run and requires every watchlist symbol's latest bar to be the session that just closed. Anything else is a data failure: urgent alert, non-zero exit, nothing traded.

The bar request ends 16 minutes before now, because the free plan does not serve the latest 15 minutes of SIP data. If SIP is refused, the request is retried on IEX. DECIDE runs 30 minutes or more after the close, so the daily bar is complete either way. IEX volume is a slice of consolidated volume, so the liquidity gate is conservative on IEX.

## Live prices at execution

Price sanity uses the latest trade from `/v2/stocks/trades/latest` on the IEX feed, which the free plan allows in real time. The feed is set by `data.quote_feed`.

## Market clock cross-check

EXECUTE trusts the NYSE calendar for its window, and also asks Alpaca's `/v2/clock` whether the market is open before sending anything.

## Withdrawals and the high-water mark

`stoa money confirm-withdrawal` rebases the high-water mark to current equity. Separately, DECIDE reads cash transfers (activity types TRANS, CSD, CSW) after the inception date and moves the high-water mark by each one, so a withdrawal never looks like a loss. When a confirmed withdrawal later shows up in that feed, it is matched and not subtracted twice. A TRANS entry that mirrors a CSD or CSW entry on the same date and amount is counted once.

## The money plan needs a matching account size

Alpaca paper accounts start at $100,000 unless reset. If the account's first equity is not within half to one and a half times `money.deposit`, the plan stays off and the first summary says why. Reset the paper balance to $100 in the Alpaca dashboard, or change `money.deposit`.

## Withdraw alert title

The architecture asks for "Withdraw $100 now". The title prints whole dollars that way and falls back to cents only when the deposit has cents.

## Realized P&L and the tax reserve

Realized gains are tracked per calendar year from fills (sell price minus the broker's average entry price, minus fees). The reserve is the configured rate on net realized gains, rounded up to the cent, and zero when the year is net negative. Every gain is labeled short-term because no position is held for a year. It is an estimate, not tax advice.

## LLM cost accounting

Token counts come from `response.usage` and are priced from `llm.prices` in config. Spend is recorded per calendar month in state, so the monthly cap survives between workflow runs. In backtests, a replay cache hit costs nothing. The scoreboard's `api_cost` check uses list price per month of simulated time, because that is what live operation would cost. One real-money budget is shared across a whole `stoa score --llm` run.

## Anonymized prompts

With `llm.anonymize: true`, prompts show only scale-free features (returns, distances, volatility, RSI, relative strength) and no prices, dates or tickers. Aliases are reshuffled every session, so "Asset C" is not the same stock two days in a row. The model's answer is mapped back and any unknown id, duplicate or out-of-range conviction is dropped.

## Schema limits

The Pydantic schema sent to Claude has no numeric or length constraints, per the structured output limits. Ranges are checked after parsing, and theses are trimmed to 280 characters.

## Rule scout conviction

The rule scout gives conviction 5 to its top idea, then 4, 3, 2, 1. Conviction does not change position size; every entry is sized to the 20% cap.

## State on a branch

The workflow keeps `state/` on a branch named `stoa-state`, checked out as a git worktree. If the branch does not exist yet, the first run creates it as an orphan branch. Pushes retry with a rebase. The LLM replay cache is stored there too, so live runs never pay twice for the same prompt.

## Workflow permissions

`daily.yml` needs `contents: write` to push the state branch. CI needs only `contents: read`.

## Running before setup

If `STOA_NTFY_TOPIC` or the Alpaca paper keys are missing, the daily workflow logs which ones and exits 0, so the repository stays green until it is configured.

## Commit message check

CI fails on co-author or sign-off trailers and on em dashes in commit messages, over the commits in a push or pull request.

## Repository visibility

The task description calls the repository private, but GitHub reports it as public. GitHub disables scheduled workflows in a public repository after 60 days without activity. It is not documented whether bot commits to the state branch count, so the runbook says not to rely on them.

# v0.2

## v1 review: defects found and fixed

A skeptical pass over every v1 module before adding anything. Each item below was a real defect and has a regression test in `tests/test_v1_fixes.py`.

1. **Splits and dividends broke stored prices.** Live bars use `adjustment=all`, so every split or dividend rewrites all earlier prices. Stoa stored entry price and entry ATR from an earlier run and compared them with re-adjusted closes. A 4 for 1 split looked like a 75% crash: the stop fired, and in signal mode the shadow book's share count no longer matched and the kill switch could trip. The inception SPY close drifted the same way on every SPY dividend. Fix: `stoa/adjust.py`. Every stored price carries an anchor (a session and the adjusted close seen then). Each DECIDE compares the anchor bar today with the stored close and rescales entry price, ATR and the signal-mode share count and average cost by the same factor, then moves the anchor to today.
2. **Entry ATR on the wrong scale after an overnight split.** The plan's ATR came from the DECIDE bars, the entry price from the next morning's fill. A split effective at that open put the stop on the old scale. Fix: the ATR is carried as a fraction of the decision close and applied to the fill price (`trader.scaled_atr`), for immediate fills and for orders settled later.
3. **A withdrawal after the close could trip the kill switch at the open.** Cash transfers were only read at DECIDE. A withdrawal made in the evening lowered equity at the next open, and EXECUTE compared it with the old high-water mark. Fix: EXECUTE reads transfers too, before the kill switch check (`jobs.sync_flows`).
4. **Withdrawals looked like a down day.** The daily loss limit compared equity with the last close, without the money that moved since. A $50 withdrawal on a $100 account read as a 50% loss and blocked buying. Fix: flows applied since the last close are added to the baseline (`state.flows_since_close`).
5. **One delisted or halted symbol stopped all trading.** DECIDE required every watchlist symbol to have today's bar, and the Alpaca provider raised if any symbol came back empty. A single halt or delisting meant no trades and no stop checks for anyone. Fix: the benchmark and every held symbol must be fresh (a stale held symbol still fails loudly, because its stop cannot be checked). Other stale symbols are skipped for the day and named in the summary, as long as at least `data.min_fresh_fraction` (90%) are fresh. The provider now reports missing symbols instead of raising; `stoa data fetch` still refuses an incomplete watchlist.
6. **The simulated broker failed the whole execution when one symbol had no price.** Alpaca leaves a missing symbol out of its latest trades answer; the simulator raised. Backtests could crash on a symbol that stopped trading. Fix: the simulator matches Alpaca, and the risk check skips that one buy for having no live price.
7. **Secrets could reach logs and alerts.** Exception text went straight into the urgent failure alert, and an HTTP error can carry a URL. v0.2 adds secret-bearing URLs (the healthcheck ping URL, a quote API token in a query string). Fix: `stoa/redact.py`. Every secret read from the environment is registered; a logging filter masks them in every record, `httpcore` debug logging is capped at WARNING, and every alert title and message is scrubbed before it is sent or printed.
8. **Signal mode's daily loss baseline ignored the session check.** It used the last recorded close even when that close was not the previous session. Fix: it goes through the same `prev_close_equity` rule as paper and live.

## v1 review: checked and correct

- **Lookahead.** Features at t use bars up to t only; decisions at the close of t fill at the open of t+1; exits use the close of t. The perturbation test still holds and is extended to the model features in v0.2.
- **Session alignment.** Cron slots, DST and early closes behave as documented. The second slot of each pair exits as already done.
- **Idempotent EXECUTE.** A crash after submitting and before saving state re-runs safely: every order is looked up by its deterministic client order id and adopted, never resubmitted.
- **Workflow.** DECIDE and EXECUTE share one concurrency group, state is saved even on failure, and a failure outside Stoa still sends its own urgent alert.
- **The demo's $20 of cash.** On a $100 account the plan bought four names at $20 each. The scout's fifth idea (PG) was killed by the overbought gate (RSI 92.5 over 85). Had it passed, it would have been sized to $18, not $20: the 2% cash buffer ($2) comes out of the last slot. So the most a full book can hold is 98% invested with five positions, which is the intended reading of 5 positions, 20% each, 2% cash. A position can drift above 20% by rising after entry; Stoa never buys above the cap and does not trim winners.
