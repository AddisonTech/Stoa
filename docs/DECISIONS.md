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

## Data sources chosen

| Need | Source | License and limits | Why |
| --- | --- | --- | --- |
| Index membership history | `S&P 500 Historical Components & Changes (Updated).csv` from github.com/fja05680/sp500, vendored at commit a2430f2, changes through 2026-08-18 | MIT. A community dataset, reconciled by hand against Wikipedia and news; may be off by a day on some changes. 503 to 505 members on every date checked since 2016. | The only free, reproducible, point-in-time list found. `scripts/build_reference.py` rebuilds Stoa's interval file from it. |
| Sectors | `sp500.csv` in the same repository (GICS sector from Wikipedia) | MIT. Today's sectors applied to the past; names that left before the snapshot are `Unknown`. | Free. The sector cap treats `Unknown` as its own bucket. |
| Bars | Alpaca market data (as in v1), into a second cache namespace `universe/` | Free plan, SIP history from 2016 except the last 15 minutes. Alpaca does not document whether delisted symbols keep their bars and has no historical asset list. | Already the data source. Coverage is measured, not assumed: `stoa universe build` writes `coverage.json` by year and for names that left. |
| Fundamentals and earnings history | SEC EDGAR `data.sec.gov` (`submissions`, `api/xbrl/companyfacts`, `files/company_tickers.json`) | US government, public domain. Needs a User-Agent with a contact email; 10 requests a second (Stoa uses 8) and caches everything to disk. The ticker map only covers current filers, so some delisted names have no CIK. | Acceptance times give true point-in-time stamps. 8-K item 2.02 is the earnings release. |
| News | Alpaca News API, `/v1beta1/news` (Benzinga) | Free with the market data keys, from 2015, 50 items a page. | Already authorized; timestamps on every item. |
| Earnings call transcripts | none | No free, legitimate, bulk source was found. | Documented gap; the analyst reads news and 8-K text instead. |
| Upcoming earnings dates | Finnhub `/calendar/earnings` with a free key, else Yahoo through `yfinance` (optional install) | Finnhub free tier: 60 calls a minute. Yahoo has no key and rate-limits cloud IP ranges at times. | Finnhub works from GitHub Actions with a key. Without either, the conservative unknown-date rule applies. |
| Second price | Finnhub `/quote` (key in the `X-Finnhub-Token` header, never the URL), else Yahoo | Same limits as above. A quote older than 30 minutes counts as unavailable. | Independent of Alpaca. Stooq was considered and dropped: since March 2026 its CSV download needs a key. |
| FOMC dates | federalreserve.gov: the 2016 to 2020 tentative schedules from the press releases of 2015-05-11, 2016-06-28, 2017-05-11, 2018-05-25 and 2019-05-17; 2021 to 2027 from the FOMC calendar page; the two March 2020 emergency actions | Public domain. 2027 is tentative until confirmed. | Vendored with a source URL on every row. |
| CPI dates | bls.gov: the CPI release archive (2016 to 2025, from archive file names) and the 2026 and 2027 release schedules | Public domain. October 2025 CPI was never published (funding lapse); September 2025 moved to 10-24. | Vendored with a source URL on every row. |

## The model, not the scout, makes the prediction

The training plan says Claude remembers 2019 to mid 2026, so it cannot honestly predict that period. In v0.2 the prediction is a calibrated logistic regression or gradient boosted tree trained by walk-forward. Claude is demoted to an optional analyst that reads text into scores with no prices in view. The v1 scout pipeline stays as the `agents` baseline and as DECIDE's fallback while no model is promoted.

## Label and execution price

The label measures from the next open, Stoa's actual execution price, to the close H sessions later, against SPY over exactly the same span. A stock that stops trading inside the horizon is valued at its last close; otherwise every company's final weeks before a failure would be dropped from training.

## Features are ranked, not raw

Every feature is a cross-sectional z-score per day, clipped at 3, with missing values set to the day's mean. The model learns which names look better today, so it does not depend on price levels or the market's overall direction (the regime filter handles that).

## Calibration inside each training fold

Each training set is split by time: the earlier 75% fits the classifier, then an embargo of H sessions, then the last 25% fits the isotonic (or Platt) calibrator. The test slice never touches either. Probabilities are clipped to [0.01, 0.99].

## Choosing the model

By pooled out-of-sample log loss, then Brier. For backtests in the scoreboard, only folds that end before the validation window count, so the validation window never influences which model is graded on it.

## What counts as a trial

A trial is a distinct configuration: the effective settings hash plus the run's parameters. Rerunning the same configuration on more data (the monthly retrain, or the same `stoa score` twice) does not add a trial. Changing a setting that changes decisions does. The variance of trial Sharpe ratios comes from trials measured the same way; with fewer than two of those, all trials are used, so the penalty never silently drops to zero.

## The ranking's Sharpe uses non-overlapping returns

The registry's Sharpe for a walk-forward run is computed on the top-ranked basket's return minus the average scored name's, every H-th day only. Averaging overlapping 10 day returns every day would shrink volatility by about the square root of 10 and inflate the Sharpe by the same factor. Subtracting the average scored name removes the market: if every stock beat SPY that period, a random basket would too, and that is not skill. Stored Sharpe ratios are per session so trials with different horizons are comparable; the deflated Sharpe rescales the variance to the series' period.

## The deflated Sharpe on the scoreboard

The scoreboard logs the graded validation run as a trial, then deflates it by every distinct configuration in the registry. A strategy can beat SPY on every risk measure and still fail on the trial penalty; the synthetic sample in SCOREBOARD.md shows that happening with three trials.

## Sizing defaults

Quarter Kelly on the calibrated probability (with the payoff ratio from the training fold), divided by 20 day realized volatility floored at 10%, scaled so the book targets 15% annual volatility, then every hard cap. Quarter Kelly because full Kelly on an estimated edge is famously too aggressive; the floor because a quiet stock would otherwise ask for an enormous weight. `sizing.method: equal` restores v1's every-entry-at-the-cap rule.

## Sector and correlation caps

Sector room is counted after this plan's sells and including this plan's buys. With no sector map at all (unit tests, the v1 baselines), the cap is skipped; with a map, an unknown sector is its own bucket. The correlation check uses 60 sessions of daily returns and also compares buys planned together. When return history is supplied but a symbol has none, that symbol is refused, because the check cannot be done.

## Earnings timing

Stoa trades a few minutes after the open. The last safe session before a release is the last session whose execution time is before the release: a release before the open of day D makes D - 1 the last safe session; during the day or after the close, D itself. Exits sell at the open of the session before the last safe one would be passed, so no position is held through a release. A position whose date only became known late is sold at the next open and the alert says so.

Historical release dates come from SEC acceptance times, but EDGAR does not record when the *date* was announced. Stoa assumes 10 sessions ahead (`earnings.confirm_lead_sessions`). Real announcements are usually 2 to 4 weeks ahead, so this is on the cautious side for the no-entry rule. Symbols without SEC history use a volume-spike proxy, marked `proxy`.

The unknown-date rule also exits a held position once its last report is over 80 days old with no confirmed next date, rather than risk holding through an unannounced report.

## Event days

"Go light" applies on the decision day before an FOMC statement or CPI release and on the event day itself, because a buy decided the evening before executes on the event day. Only new positions are sized down. A weekend event (the Sunday 15 March 2020 cut) maps to the next session.

## Hand approval

The phone posts `approve <plan id> <symbol> <nonce>` to a second, secret ntfy topic. EXECUTE polls it with `poll=1&since=` the plan's creation time, keeps the newest message per symbol with the right plan id and nonce published between plan creation and the end of the execute window, and sends only approved orders. No token or key is ever in a notification. Polling is bounded both by the clock and by a count, so a stalled clock cannot hang the job. Kill switch flattening never waits for a tap. Sells otherwise need one during the approval month; they are trades.

`approval.mode: auto` turns it on in live mode before the first live fill and for 30 calendar days after it. `always` turns it on in any mode (the way to try it on paper).

## Honest paper

Each paper or live fill adds the backtest's slippage (and the regulatory fee the broker did not charge) to a running cost drag. Each close records broker equity and broker equity minus that drag. The paper exam, the drift fallback and the quit rule use the adjusted number; the raw number is reported beside it. Signal mode already pays backtest costs in its shadow book. Returns between records exclude deposits and withdrawals.

## The exam grades only the strategy that would go live

Each daily record carries the strategy whose plan produced it. The paper exam counts only the latest unbroken run of the configured strategy, so days the rule pipeline traded while no model was promoted do not count toward the model's exam.

## Drift fallback and quit rule

The expectation bands come from the walk-forward's out-of-sample book: the 5th percentile of rolling 63 session Sharpe and the 95th percentile of a block-bootstrapped max drawdown. They are written next to the promoted model and only change when a model is promoted. The quit rule joins paper and live returns segment by segment (never across the seam between two accounts of different sizes). In fallback the benchmark may be held up to `risk.max_benchmark_pct` (98%, leaving the cash buffer), exempt from the 20% and sector caps; every other cap, the kill switch and the cash buffer still apply, and the regime filter does not block the SPY buy.

## Live state is kept apart from the paper record

In live mode the workflow uses `live-state/` on the state branch, and points `exam.paper_state_dir` at `state/`, so the live account starts fresh while the exam check and the quit rule still read the paper record. Models, the registry, data and exam records live in `shared/`. v1's advice to rename the state branch before going live is replaced by this.

## Monthly retrain and model artifacts

Models are pickled `TrainedModel` objects with a JSON metadata file (code SHA, config hash, data hash, features, training window, out-of-sample scores, experiment id) under `shared/models/<version>/`, with `current.json` naming the live one and `promotions.jsonl` recording every change. They are on the state branch, so a scheduled run always loads the model it was promoted with. The promotion gate requires calibration error under 0.05 and log loss and Brier no worse than the current model's on the same rows of the last four folds; the first model must beat the base rate. `--force-promote` and `stoa model promote <version>` bypass the gate by hand and say so in the record.

The monthly workflow restores the bar and SEC cache from the Actions cache, refetches, retrains and commits the result. The default cron (11:17 UTC on the 2nd) is before the open and outside the daily slots; both workflows share one concurrency group so they never touch state at the same time.

## Training cutoff before the holdout

Until the holdout is scored, `stoa model train` and `stoa model report` stop at the session before the holdout start, whatever the date. Walk-forward test labels that would finish after the cutoff are blanked, so no holdout price ever enters a score. After the holdout is scored, training uses everything.

## The analyst budget

One ledger (`state.llm_spend_by_month`) and one cap (`budget.monthly_cap_usd`, $25) for every Claude call: the v1 scout's own cap is never allowed above it. A backfill prints its estimate first and refuses if it exceeds what is left this month, unless `--allow-spend`; even then it stops submitting batches once the cap is reached. Batch ids are written before polling, so an interrupted backfill collects the same batch on the next run instead of paying again. The replay cache is keyed by a hash of the model, the system prompt, the item text and the schema.

The system prompt is about 930 tokens, above Haiku 5.5's 512 token caching minimum (a shorter prompt is silently not cached). Daily calls use the 5 minute cache; batches use the 1 hour cache because batches can take longer than 5 minutes.

## Cost estimates

At list price for Claude Haiku 5.5 ($0.10 per million input tokens, $0.50 output, cache reads at 0.1x, 1 hour cache writes at 2x, batch at half), with a 930 token cached system prompt, about 320 tokens of item text and about 120 output tokens per item:

Figures below come from `analyst.estimate_cost`, the same estimator the backfill uses before it spends anything: about **$0.061 per 1,000 items in batch** and **$0.115 per 1,000 one by one** (output tokens are most of it).

- Historical backfill, watchlist only (28 names, 2016 to 2026, at most 3 items per name per day): roughly 60,000 to 100,000 items, so about **$4 to $6** once. `stoa analyst backfill --estimate` prints the exact figure from the stored news.
- Backfill for the whole index (`analyst.symbols: universe`): roughly 1 to 2 million items, about **$60 to $120**, several months of the cap. It would have to be spread out, or run with `--allow-spend` after raising the Console limit. Not recommended unless the watchlist ablation shows text helps.
- A normal month with text features on: about 30 to 90 new watchlist items a trading day, one by one with the cached prompt: about **$0.10 to $0.25 a month**. The v1 Claude scout, if used, adds about $0.50 to $3 a month. The owner's $3 to $10 estimate is comfortably above both, and the $25 cap is a hard ceiling either way.

## Things that could not be verified without real keys

- Whether Alpaca's bar history includes symbols that were delisted (the docs are silent; `coverage.json` will show it on the first real `stoa universe build`).
- The exact JSON shape of Alpaca's news API under the free plan beyond what the official SDK models (`news`, `next_page_token`, `created_at`, `updated_at`, `symbols`).
- Live behavior of the Finnhub and Yahoo endpoints from GitHub Actions.
- The SEC endpoints were not reachable from the build machine; the parser was built from the documented format and tested on fixtures.
- ntfy action buttons on a real phone, and healthchecks.io's ntfy integration.
- Anthropic batch and caching behavior against the live API (built against the installed SDK's types and the official docs; tested with fakes).

## Version number

The owner calls this release v0.2, so the package version is 0.2.0. v1's internal 1.0.0 was the label of the last build stage, not a published release; nothing depends on the number.

## Strict typing

mypy strict covers every module that handles money, sizing, approvals, the model or secrets: the v1 list plus `adjust`, `analyst`, `approvals`, `calibration`, `crosscheck`, `events`, `exam`, `experiments`, `healthcheck`, `membership`, `model`, `pit`, `redact`, `retrain`, `sizing`, `tracking` and `walkforward`. scikit-learn ships no type information, so its imports are untyped and the classifier is reached through a small `Classifier` protocol.

## v0.3: broker stops and the dashboard

- **Which stop goes to the broker.** The one the exit rules already use: the ATR stop or, once higher, the trailing stop. DECIDE stores it on the position and only ever raises it, so the broker stop is never looser than the rule. Rounded up to the tick for the same reason.
- **DAY stops, re-placed each morning.** Positions are fractional at this account size and Alpaca takes fractional orders only as DAY orders. The gap between the open and EXECUTE is the same window the close-based rules always had.
- **No stop on a position bought that morning.** A stop that fills the same day would be a round trip, which the risk engine never allows.
- **Price already through the stop at the open.** No order is placed, because a stop above the market fills at once and that would be a new market sell nobody planned. An alert goes out and the close rule decides. A split that lands overnight shows up the same way and is handled the same way.
- **Cancel before any sell, confirmed.** A working stop holds the shares, so the planned sell would be rejected. The kill switch flatten goes through the same hook.
- **MTUM on the dashboard.** DECIDE now fetches the reference funds beside the watchlist (never traded, never fatal) and records each close in the daily record, rescaled for dividends like the SPY close. The chart starts on the first day every line has a value.
- **The dashboard is read only by construction.** It is built in a workflow that gets no secrets, from committed files, with no network calls and no scripts in the page. The output is also scrubbed for every secret name and any secret value in the build environment.
- **Whether the dead man's switch ping went out.** Not recorded before; DECIDE now writes `healthcheck.json` (signal, time, sent or not, never the URL).
- **Version 0.3.0.** `stops`, `dashboard` and `files` join the strict mypy list.

## Trial 6: monthly horizon

Pre-registered on 2026-10-24, before any code for it was written and before it was trained. Trial 5 and earlier (the 10 and 5 session price and quality models) failed walk-forward on real data: out-of-sample AUC 0.496, log loss worse than the base rate. This is the one follow-up design. It is trained once. If it fails, the result is reported and nothing is tuned or retried.

- **Target:** the probability a stock beats SPY over the next 21 sessions, from the next open to the close 21 sessions later (`model.horizon: 21`).
- **Rebalance:** new positions only when the execute session is the first session of a month (`model.rebalance: monthly`). Stops, the broker stops, earnings exits, the other exit rules and the kill switch still run every day.
- **Features:** 12-1 month momentum (`f_mom_12_1`), 6-1 month momentum (`f_mom_6_1`), distance from the 200 day average (`f_trend_200`), 60 day volatility (`f_vol_60`), and the five existing quality features (`q_gross_profitability`, `q_roe`, `q_accruals`, `q_leverage`, `q_earnings_stability`). All rank-normalized across stocks each day: the percentile rank, scaled to -1 to 1, missing values at 0 (`model.price_features`, `model.normalize: rank`).
- **Model:** logistic regression only, one model, one horizon (`model.candidates: [logistic]`, `model.test_horizon: null`).
- **Unchanged:** the walk-forward (63 session test slices, expanding window, 504 sessions minimum training), the embargo (at least the horizon, so 21), isotonic calibration inside each training fold, the 0.55 threshold, up to 10 candidates, costs, the scoreboard windows and the pass bar (calibration error under 0.05, log loss better than the base rate, deflated Sharpe at least 0.95 counting every recorded trial, and every scoreboard check).
- **Data:** the universe and fundamentals as of 2026-10-24, with the renamed tickers (BK, FI, JEC, WLTW) fetched under their current symbols. Training stops on 2026-06-30, the day before the holdout. The holdout is not sealed or scored.
- **What it adds to the registry:** `stoa model train` logs exactly one new trial (logistic, 21 sessions). `stoa model report` reruns the same configuration, which is not a new trial. `stoa score` appends its scoreboard backtest record, as every scoreboard run does.
- **If it passes every walk-forward check:** promote it and set `model.rebalance: monthly` in `config/default.yaml` so paper DECIDE uses it. **If it fails:** report it and stop.

## The holdout gets a fixed end

Decided 2026-10-24, before the holdout was sealed. The holdout was every session from 2026-07-01 onward, which left no clean data for a forward test of anything. It is now 2026-07-01 to 2026-10-23 (`backtest.holdout_end`), about 80 sessions after the analyst model's knowledge cutoff. Until it is scored, no training row or label touches it; later sessions are free for forward tests. Features after the end look back over the window as live decisions always do. `stoa holdout score` is the only reader. A perturbation test guards it.

## Trial 7: analyst forward test

Pre-registered on 2026-10-24, before any code for it was written. It asks one question: do the Claude analyst's news scores add anything to the price and quality model, measured only on news the model could not have known about?

- **Why forward only.** The analyst model (`claude-haiku-5-5`) knows events through mid-2026, so scoring older news can leak outcomes. Only news published on or after **2026-10-26 00:00 New York time** is scored for this trial. The holdout ends 2026-10-23 (see "The holdout gets a fixed end"), so the forward window does not overlap it.
- **Shadow mode.** Scores are stored with their `known_at` in `shared/shadow/` on the state branch and never reach DECIDE, EXECUTE or any order. `model.use_text` stays false.
- **Scope.** Every stock in the point-in-time S&P 500 on each session (the vendored membership file), plus nothing else. News from the existing Alpaca (Benzinga) feed, full text: headline plus body, up to 1200 characters, at most 3 items per symbol per day (longest first), as `analyst.text_chars` and `analyst.max_items_per_symbol_per_day` already set. Sized on the week of 2026-10-19: about 620 items a session, about $0.033 a session through the Batch API with prompt caching at the configured prices. Target under $0.60, so no trimming.
- **Sessions.** A session's news is everything published from the previous session's close to this session's close (the first session starts at 2026-10-26 00:00 New York time). A session counts as scored once all of its news is scored and stored. A session the job misses, or skips because the spend cap paused the analyst, does not count.
- **Job.** A separate weekday workflow after the daily DECIDE: fetch the session's news from Alpaca, score it through the Batch API, poll to completion, commit the scores to the state branch. It never delays or blocks DECIDE or EXECUTE; a failure sends one alert. The analyst spend cap (80% heads-up, pause and ask at the cap, $75 ceiling) applies, counting every Claude call in the month.
- **Evaluation.** Once, after **210 scored sessions** (2027-08-26 at the earliest), with `stoa analyst evaluate`. Two walk-forward runs, identical except for the text features: logistic regression, 10 session horizon, trial 1's features (all ten price features and the five quality features, z-scored), with and without the six text features (`t_sentiment`, `t_guidance`, `t_surprise`, `t_risk`, `t_materiality`, `t_count`). Same walk-forward (63 session test slices, expanding window, 504 sessions minimum training, embargo 10, isotonic calibration inside each fold), same costs. The holdout stays excluded. 210 sessions is what the unchanged walk-forward needs for a full test fold to train on a meaningful stretch of scored news; 60 was not enough (see the exchange on 2026-10-24).
- **Pass bar.** Measured on the out-of-sample rows dated 2026-10-26 or later, for the model with text: calibration error under 0.05, log loss better than the base rate, log loss lower than the same model without text, and deflated Sharpe at least 0.95 counting every recorded trial. All four, or it fails.
- **Registry.** `stoa analyst evaluate` logs exactly one trial and refuses to run a second time. It refuses to run before 210 scored sessions.

## Trial 8: event-driven

Pre-registered on 2026-10-24, before any code for it was written. A fixed rule with no fitted parameters: two well-known event effects (post-earnings drift after a strong surprise, and clusters of insider buying), traded on a universe widened into mid and small caps. It is run once with `stoa score --strategy event`. If it fails, the result is reported and nothing is tuned or retried.

**Universe.** Point-in-time S&P 400 and S&P 600 membership with dated removals is not freely available (only commercial sources), so the universe is a liquidity universe, measured each day with no lookahead:

- Candidates: every Alpaca US equity listed on NYSE, Nasdaq, NYSE American, NYSE Arca or Cboe BZX, active or inactive (delisted). OTC is excluded. On 2026-10-24 that is 13,187 active and 2,864 inactive symbols.
- On each session: close at least $5 and 20 session average dollar volume (close times volume, the 20 sessions ending that day) at least $5 million. On 2026-10-23, 4,287 names passed. ETFs and funds can pass but never produce an event.
- Account fit, applied at each decision in backtests exactly as live: the stock is fractionable at Alpaca, or its close is below the position size (20% of current equity). Fractionability is Alpaca's flag as of 2026-10-24 (delisted names keep their last flag). On 2026-10-23, 4,205 of the 4,287 passed.
- Known bias: Alpaca's inactive list is the only record of delisted names, and it may miss names delisted before its coverage. Any such gap favors survivors.

**Data, all point in time.** Daily bars from Alpaca (split and dividend adjusted), 2016 on. Insider purchases from the SEC Insider Transactions Data Sets, quarterly, 2015 Q3 to 2026 Q3. Earnings release times from SEC 8-K item 2.02 acceptance times by CIK; the CIK comes from the SEC ticker map, or for delisted names from the issuer ticker and CIK in the insider data sets. Sector for the 40% cap: the SIC major group (first two digits) from SEC submissions, `Unknown` when there is none.

**Earnings drift signal.** For a release with a known time: the close before the release is the last close before the release time; the first full session after it is the first session that opens after the release (a release before the open uses that day; after the close, the next day; during a session, the next day). The excess return over SPY from that close to the close of the first full session must be at least +5%, and that session's volume at least 2 times the average volume of the 20 sessions before it. Known at the close of that session. Enter at the next open. Hold 20 sessions. Releases without a time (the volume-spike proxy) never produce this signal.

**Insider cluster signal.** Open-market purchases (transaction code P, acquired) by officers or directors (relationship includes Officer or Director). Purchases flagged 10b5-1 (`AFF10B5ONE` true, where the flag exists, 2023 on) are excluded. An amendment that repeats a purchase (same insider, date, shares and price) counts once. A cluster is 2 or more distinct insiders whose purchase dates fall within 10 sessions, at least $100,000 combined (shares times price). **Known at the close of the first session after the filing date of the purchase that completes the cluster.** The brief said the close of the filing date, but the SEC data sets give the filing date without a time and Form 4s are filed until 10 pm, so that would use filings before they existed; this is Stoa's rule for date-only data. Enter at the next open. Hold 40 sessions. An issuer signals at most once in any 40 sessions.

**Slots.** When signals exceed open slots: insider clusters first (largest combined purchase first), then earnings drift by size of the excess move. A signal not entered at its next open is dropped, not queued. A stock already held is not bought again.

**Exits.** The hold period (decided at the close of the 20th or 40th session held, sold at the next open), the ATR stop and trailing stop (2.5 and 3 ATR, `exits`), the earnings rules (never hold through the next report), and the kill switch. Not the 30 session time exit, the trend break or the model's horizon exit.

**Risk, unchanged.** 5 positions, 20% each, 40% per sector, 2% cash, the correlation check, the regime filter (no new entries while SPY is below its 200 day average), the daily loss limit, the 25% kill switch, and broker stops live.

**Costs.** One-way slippage in basis points = 25 times the square root of ($10 million divided by the stock's 20 session average dollar volume), at least 5 and at most 75, per fill, using the average up to the decision day. $100 million a day costs about 8 bps, $10 million 25 bps, $5 million 35 bps. Sell fees unchanged. Applied to every strategy in this scoreboard run, so SPY and MTUM sit at the 5 bps floor as before.

**Grading.** The combined strategy only. Per-signal results (trades, hit rate against SPY over the same holding period, average excess return per trade) are reported as diagnostics, never picked between.

**Pass bar.** Every scoreboard check (Sharpe and Sortino above SPY and max drawdown below SPY on the validation window, Sharpe above SPY in at least 2 of the 3 tune windows, the 25% drawdown cap, at least 30 trades, the API cost cap), deflated Sharpe at least 0.95 counting every recorded trial, and Sharpe above MTUM's on the validation window. All, or it fails.

**Holdout.** 2026-07-01 to 2026-10-23 stays excluded and unscored. The scoreboard windows end 2026-06-30.

**Registry.** The one `stoa score --strategy event` run logs exactly one trial.

## The new day: summary at 5:15 pm, picks at 9:00 am, orders at 9:45 am

Decided 2026-10-25.

- **DECIDE at 5:15 pm is quiet.** It records the day and sends the day summary (each position's move today, today's buys against their fill price, sells, the account against SPY today and since the start), then plans the next open without announcing it. Safety alerts (kill switch, fallback, stop-outs, the withdraw milestone) still go out at once.
- **The sweep at 9:00 am.** It reads every article published after the plan's close and visible by 9:00 (created after the close, last updated by 9:00) on the planned buys and the reserves (only articles that tag at most 3 tickers, since roundups are rarely about the name), with ProsusAI/finbert pinned at revision `0bc2ac10`, on CPU, headline plus summary, 128 tokens. A buy is dropped when one article is at least 0.80 negative and the night's mean tone (positive minus negative) is at or below -0.30. A reserve is added when one article is at least 0.90 positive and the mean tone is at or above +0.50, only into a free slot, at most one a day. Reserves are candidates the rules and quant gates passed and the risk engine turned down only for capacity, so an add can never bypass the regime filter, the kill switch or the daily loss limit; EXECUTE resizes it and re-runs every check. **These thresholds are a judgment call, not a tested edge.** Trials 9 and 10 measure whether this kind of news signal has any value. Sells are never changed by the sweep.
- **The sweep starts at 8:57** and Stoa waits until 9:00 so the night's news is complete. Task Scheduler on the Dell starts it (and DECIDE and EXECUTE) by `workflow_dispatch`; GitHub cron is the backup at 9:05, because on 2026-10-26 the 8:55 cron never fired (RUNBOOK, section 13). Window: 8:50 am until EXECUTE starts. A sweep that runs after 9:40 still sends the picks but says they will be skipped.
- **Every buy needs a tap** (`approval.buys: always`), in paper too. The approval code is drawn at the sweep, so a tap can only answer the final picks. Taps count until 9:40 am. Sells follow `approval.mode` as before (live: the first 30 days). The kill switch never waits.
- **Taps close out loud** (decided 2026-10-26). At 9:40 the close job sends "Approvals closed at 9:40: approved X, Y; skipped the rest", and replies "Too late, approvals closed at 9:40." to every tap until 10:15. On the first morning the picks reached the phone at 9:16, taps came at 9:49 and nothing said they were too late. The cutoff itself is unchanged.
- **A missed evening run.** If no plan exists for today and the last session was never decided, the sweep runs the evening work first (without the evening summary). This also covers the very first day.
- **No sweep, no buys.** If the sweep never runs, EXECUTE finds no approval code and skips every buy with a "Skipped" alert; sells still run.
- **FinBERT on CPU in Actions.** CPU-only torch and transformers are installed only for the sweep (`.[reader]`), and the model files are cached by revision. A night of watchlist news is a few dozen articles, a few seconds on the runner's CPU.

## Trial 9: the news brain, daily

Pre-registered on 2026-10-25, before any feature, label or training code for it was written and before any of its data was looked at beyond pull smoke tests. It is trained once and tested once. If it fails, the result is reported and nothing is tuned or retried.

**Question.** Does news, read by a free open-source model, predict which stocks beat SPY after realistic costs, in a book run through Stoa's real risk rails?

**Data, all point in time.** Alpaca (Benzinga) news from 2015-01-01 for the 14,858 symbols of the wide universe (id, created and updated times, symbols, headline, summary); an article is known at its `updated_at`. SEC EDGAR text: 8-K item 2.02 main documents and EX-99 exhibits, 10-K paragraphs that tie a customer or supplier to a share of sales, other 8-Ks; each known at its EDGAR acceptance time. Daily bars from the wide cache. Every article and filing is read by ProsusAI/finbert at revision `0bc2ac10` (positive, negative, neutral): news from headline plus summary (128 tokens), filings from the first section (the first substantial paragraphs, about 1,500 characters, 256 tokens). Only articles that tag at most 3 tickers are read, as in the live sweep. No Claude, no paid service.

**Timing.** The decision for session t is made at 9:00 am New York time on t. It sees daily bars through the close of t-1, and articles and filings known by 9:00 am on t; news after 4:00 pm counts for the next session by construction. A session's fresh news is everything known after 9:00 am on t-1 and by 9:00 am on t. Entry at 9:45 am on t.

**Universe.** Trial 8's liquidity universe, measured each day with no lookahead: close at least $5, 20 session average dollar volume at least $5 million, fractionable or priced below the position size. A row is a (symbol, session) with at least one fresh article, a filing in the last 5 sessions, or a ripple signal; other rows are not scored or traded.

**Features (each per symbol and session, from information known by 9:00 am).**

- `sent`: weighted mean FinBERT tone (positive minus negative) of the fresh articles; first breaks weigh 1, rewrites 0.25.
- `surprise`: `sent` minus the symbol's mean daily `sent` over its previous 60 sessions with news, over their standard deviation (at least 5 such sessions, else 0).
- `first_break`: share of fresh articles that are first on their story. An article is a rewrite when the same symbol had an article in the previous 3 days whose headline shares at least half its word set (Jaccard of lower-case words of 3 or more letters, at least 0.5).
- `quiet_spike`: log((1 + fresh articles) / (1 + the symbol's average daily articles over the previous 60 sessions)), and `coverage`: log(1 + that average). A rarely covered name with news scores high on the first and low on the second.
- `ripple`: the mean `sent` times `first_break` of the company's linked customers and suppliers over sessions t-1 to t-3 (never the same day). Links come from 10-K paragraphs that tie a named customer or supplier to a share of sales; names are matched to universe companies by their SEC names (legal suffixes dropped, at least 5 characters, among the 1,500 most traded names that month). A link is known from the 10-K's acceptance and lasts until the next 10-K or 15 months. Signals pass from the larger to the smaller company only (by 20 session dollar volume).
- `filing_tone`, `guidance`, `beat_miss`: for an 8-K item 2.02 accepted in the last 5 sessions, FinBERT's tone of the press release's first section (the EX-99, else the main document); guidance +1 when the text says guidance or outlook was raised, -1 when lowered, cut or reduced, else 0; beat_miss +1 for exceeded, above or beat expectations or consensus, -1 for below expectations or consensus or missed, else 0.
- Controls from daily bars: the return of t-1, the 5 session return to t-1, 20 session volatility, log 20 session dollar volume.

**Labels.** For h in 1, 5 and 20 sessions: the return from the open of t to the open of t+h, minus SPY's over the same span, minus costs. Costs per fill in basis points: 25 times the square root of ($10 million over the 20 session dollar volume), at least 5 and at most 75 (trial 8's rule, wider for small caps), on both fills, plus the sell fee. The label is 1 when the net excess return is above 0. Training uses the open as the entry; the replay uses the real 9:45 price.

**Model.** Gradient boosted trees (scikit-learn HistGradientBoostingClassifier: max_depth 3, learning_rate 0.05, max_iter 300, l2_regularization 1.0, min_samples_leaf 200), isotonic calibration inside each training set on its last quarter with its own gap. Fixed now; nothing is tuned.

**Walk-forward on 2016-01-01 to 2022-12-31.** 63 session test slices, expanding window, 504 sessions minimum training, embargo at least the horizon, purged labels: the existing protocol. Reported: log loss against the base rate, Brier, AUC, calibration error, hit rate by bucket. Then the model is frozen: one fit on every row whose label ends by 2022-12-30, calibrated the same way. Nothing from 2023-01-01 on is read by any training, tuning or feature-selection step.

**Fast-forward test, 2023-01-01 to 2026-06-30.** Replayed session by session through the backtest engine, the simulated broker and the risk engine, as if live. At 9:00 am the frozen model scores that morning's rows; names with a probability of at least 0.55 are candidates, best first, up to 10. Risk unchanged: 5 positions, 20% each, 40% per sector (SIC major group), 2% cash, the correlation check, the regime filter, the daily loss limit, the 25% kill switch, the ATR stop and trailing stop (2.5 and 3 ATR), and the earnings rules (never hold through a confirmed next report). Hold h sessions, decided at the close of the h-th session and sold at 9:45 am the next session, like every exit. Buys and sells fill at the 9:45 am price from Alpaca minute bars plus the slippage above. $100 start, fractional shares. SPY and MTUM run through the same engine.

**Grading.** The 5 session horizon is the graded strategy. The 1 and 20 session runs are reported beside it as diagnostics and logged as trials, so they count in the deflated Sharpe; they are never picked between. Reported for each: total return, Sharpe, max drawdown, hit rate (closed trades that beat SPY over their holding span after costs) and trade count, against SPY and MTUM, and by size: small (market value under $2 billion), mid ($2 to $10 billion), large (over $10 billion), from SEC shares outstanding times price at entry, else by dollar volume (under $20 million small, under $100 million mid).

**Pass bar (SCOREBOARD.md, applied to the replay).** All of: max drawdown below SPY's, and Sharpe and Sortino above SPY's, on the validation window (2025-01-01 to 2026-06-30); Sharpe above SPY's on tune_c (2023-01-01 to 2024-12-31), the only tune window outside training, so it must pass; max drawdown never past 25%; at least 30 closed trades; Claude spend under $25 in every month (it is $0); deflated Sharpe at least 0.95 counting every distinct configuration in the registry; and, as for trial 8, Sharpe above MTUM's on the validation window.

**Known limits, said now.** The test window overlaps tune_c and validation, which earlier trials were scored on: it is out of sample for this model but not unseen by the research. FinBERT's training text predates 2019, so it cannot know how any test-period story ended. Alpaca's inactive list is the only record of delisted names. 10-K name matching misses customers named loosely ("a large retailer").

**Holdout.** 2026-07-01 to 2026-10-23 stays sealed and unscored.

**Registry.** One run logs three trials (1, 5 and 20 sessions).

**The holdout wall** (added 2026-10-26, before any trial 9 run). `stoa/newsbrain/sealed.py` stops trials 9 and 10 from reading 2026-07-01 to 2026-10-23 in code, not only by date choice: the panel calendar, the replay windows and the test window refuse an end date inside it; 9:45 am prices and minute bars refuse a holdout day before any request or file read; holdout news months are never opened; SEC filings accepted from midnight New York time on 2026-07-01 are dropped at load; panel rows from the holdout are dropped before training. Tests hold the dates to `backtest.holdout_start` and `holdout_end`. The raw minute bars for July to October are on disk (the pull covers 2016 to now), but nothing in the trials opens them. `stoa holdout seal` was tried the same day and refused, as designed, because no model has passed walk-forward; no HOLDOUT.lock exists. The owner decided the same day not to force it: the holdout stays unsealed, with the code wall in place, and is sealed once a model passes walk-forward.

## Trial 10: the news brain, with minute timing

Pre-registered on 2026-10-25 with trial 9. Identical to trial 9 (data, features, model, frozen fit, replay, grading on the 5 session horizon, pass bar) except the entry time, which uses Alpaca minute bars.

**Entry.** For a buy on session t, follow the minute bars from 9:30 am. Let r5 be the absolute 5 minute return (the close of the minute over the close 5 minutes earlier, less 1) and P the largest r5 seen since 9:30. Enter at the open of the first minute at or after 9:45 am whose r5 is at most a third of P: the first spike has cooled. If none by 11:00 am, enter at 11:00 am. Sells stay at 9:45 am. The model, its features (known by 9:00 am) and its labels do not change.

**When.** Run once, after the minute bar pull is complete. Trial 9 does not wait for it.

**Registry.** One run logs three trials (1, 5 and 20 sessions).
