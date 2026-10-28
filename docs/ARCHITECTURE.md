# Stoa Architecture

Stoa is a daily-cadence equity trading system. In v0.2 a calibrated statistical model ranks stocks by their chance of beating SPY, a rule-based risk engine sizes and approves, and every result is graded on risk against three baselines: buy-and-hold SPY, a 50/200 day moving average rule, and the momentum ETF MTUM. It runs in four modes from one engine: backtest, signal, paper and live.

The design goal is a system that is boring in the right places. The model estimates probabilities. Claude, when used at all, only reads text into scores. Neither is ever allowed to size a position, skip a limit, or touch money.

## v0.2 data flow

```
   point-in-time inputs                        known_at <= close of t, enforced in stoa/pit.py
   ---------------------------------------------------------------------------------------
   bars (Alpaca, adjusted)        S&P 500 membership history     SEC EDGAR
   universe incl. dead names      (vendored, MIT)                quality facts, 8-K 2.02 times
          |                              |                              |
          |        Alpaca news  ->  Claude analyst (optional)  ->  text scores
          |                         prompt cache, batch API, $ cap      |
          v                              v                              v
   +---------------------------------------------------------------------------+
   | dataset.py   one row per (day, stock): standardized features + label      |
   |              label = beats SPY from the next open to the close 10 later    |
   +---------------------------------------------------------------------------+
          |
          v
   walkforward.py  purged, embargoed folds     model.py  logistic | boosted trees
   training.py     out-of-sample predictions   calibration inside each fold only
   calibration.py  log loss, Brier, AUC, ECE   experiments.py  registry, DSR, PBO
          |
          v
   retrain.py  promotion gate -> versioned model artifact (state branch)
          |
          v   every DECIDE
   live.py  today's features -> calibrated probabilities
   model_strategy.py  exits (stops, earnings, horizon) -> ranked ideas -> quant gates
                      -> earnings and event rules -> sizing.py -> risk.py caps
          |
          v   plan, one FYI alert (Approve / Skip only when a tap is required)
   EXECUTE  approvals.py (tap if required) -> crosscheck.py (second price) -> trader.py
          |
          v
   tracking.py  cost-adjusted daily record -> drift fallback, quit rule, paper exam
   exam.py      walk-forward -> holdout -> paper -> live
```

The v1 pipeline (scout, quant gates, risk) is still there. `strategy.kind: rule` runs it, the scoreboard always reports it as `agents`, and DECIDE falls back to it while no model has been promoted.

## Hard facts the design is built on

These were checked against the broker and API docs before any code was written.

| Fact | Consequence in Stoa |
| --- | --- |
| Alpaca has no cash accounts. Accounts under $2,000 equity are limited margin with 1x buying power. | Stoa never relies on the broker to stop leverage. The risk engine caps gross exposure at 100% of equity itself. |
| Fractional and notional orders support `day` time in force only. `opg` (market on open) is not allowed for fractional. | Orders are submitted as notional `day` market orders after the open, not queued overnight. |
| Day orders submitted after 4:00 pm ET are queued for the next session, and fractional handling outside hours is not documented. | Stoa splits the day in two: decide after the close, execute after the open. Nothing is queued blind overnight. |
| The SEC approved removing the pattern day trader rule in April 2026, but brokers have until October 2027 to implement. | Stoa never opens and closes the same symbol on the same day, so day trade counting can never matter. |
| Claude Haiku 5.5 has a reliable knowledge cutoff of June 2026. | A model may remember how stocks moved before July 2026. The sealed holdout starts 2026-07-01, and prompts anonymize tickers and dates by default. |
| GitHub Actions cron runs in UTC, can start late under load, and does not follow daylight saving time. | Jobs are scheduled in both EDT and EST slots and every job checks New York time and the NYSE calendar itself before acting. |
| ntfy topics have no login. The topic name is the password. | The topic is a long random string stored as a secret, never committed. |
| ntfy notifications can carry up to three `http` action buttons that POST a body to a URL, and a topic can be polled with `GET /<topic>/json?poll=1&since=<unix time>`. Messages are cached about 12 hours by default. | Approve and Skip buttons post `approve <plan> <symbol> <nonce>` to a second secret topic; EXECUTE polls it once after the open. |
| healthchecks.io takes success, `/start` and `/fail` pings and alerts (including through ntfy) when an expected ping is late, on a cron schedule with a time zone and grace period. | The dead man's switch catches a job that never ran at all. |
| SEC EDGAR's JSON APIs are free, need a User-Agent with a contact email, and allow 10 requests a second. Each filing has an acceptance time; 8-K item 2.02 is the earnings release. | Fundamentals and historical earnings times are point in time from acceptance times. Stoa throttles to 8 a second and caches every response. |
| Alpaca's News API (Benzinga, from 2015) is on the free plan at `/v1beta1/news`, 50 items a page, with `created_at` and `updated_at`. | News is stored with `known_at = updated_at`. |
| Claude Haiku 5.5 caches prompts of 512 tokens or more (writes 1.25x, reads 0.1x the input price) and the Batch API is half price; the two discounts stack. | The analyst system prompt is about 930 tokens and marked for caching; backfills go through batches. |

## Daily cycle

```
            4:00 pm ET close
                  |
   DECIDE  (about 5:15 pm ET)
   1. Refresh daily bars, adjusted for splits and dividends
   2. Build features using data up to and including today only
   3. Mark open positions: stops, targets, time exits, trend breaks
   4. Scout proposes candidates from the watchlist
   5. Quant gates kill ideas the numbers do not back
   6. Risk engine sizes and approves, or rejects with a reason
   7. Write the plan, journal every step, push alerts
                  |
            9:30 am ET open
                  |
   EXECUTE (about 9:45 am ET)
   1. Load the plan from last night
   2. Re-check the account, positions and the kill switch
   3. Price sanity: skip any order whose price gapped too far from plan
   4. Submit notional day market orders with idempotent client order ids
   5. Confirm fills, journal, alert
```

Signal mode runs DECIDE and sends the plan to the phone. It never touches a broker. Paper and live modes run both halves.

## Modules

v0.2 additions:

```
stoa/
  pit.py             The one place the known_at rule is enforced
  membership.py      Point-in-time index membership and sectors
  reference/         Vendored membership, sectors, FOMC and CPI dates, with sources
  events.py          FOMC and CPI days, the go-light rule
  adjust.py          Keeps stored prices on the scale of freshly adjusted bars
  data/sec.py        EDGAR fundamentals and 8-K earnings times, throttled and cached
  data/news.py       Alpaca news, stored point in time
  data/external.py   Finnhub and Yahoo: upcoming earnings, the second price
  earnings.py        Exit before a report, no entry into one, the unknown-date rule
  dataset.py         Features, labels and the survivorship-free universe
  walkforward.py     Purged, embargoed folds
  calibration.py     Brier, log loss, AUC, reliability, ECE
  model.py           Classifiers, the calibrator, versioned artifacts
  training.py        Walk-forward, reports, the ablation, the promotion gate
  experiments.py     Append-only registry, deflated Sharpe, PBO
  retrain.py         `stoa model train` and the monthly retrain
  pipeline.py        Assembles inputs the same way for every caller
  model_strategy.py  The model strategy and its predictors
  live.py            Builds today's model strategy for DECIDE
  sizing.py          Confidence and volatility sizing, pure functions
  analyst.py         Claude reads text into scores; caching, batches, cost cap
  approvals.py       ntfy Approve and Skip buttons, nonce checks
  crosscheck.py      Second-source price check before a buy
  healthcheck.py     Dead man's switch pings
  tracking.py        Honest paper costs, drift fallback, quit rule, paper exam
  exam.py            Exam order and `stoa exam status`
  redact.py          Keeps every secret out of logs and alerts
  cli_v2.py          The v0.2 commands
```

From v1:

```
stoa/
  config.py          Typed settings (YAML plus environment), validated at load
  clock.py           New York time, NYSE sessions, early closes, job windows
  data/              Bar providers (Alpaca, Yahoo, CSV cache) behind one interface
  features.py        Indicators computed strictly from past data
  universe.py        Watchlist and asset metadata
  roles/
    scout.py         LLM or rule-based idea generation
    quant.py         Deterministic gates on every idea
    llm.py           Claude client with schema output, cost cap, replay cache
    prompts.py       Prompt templates with anonymization
  risk.py            The final word. Pure functions, no network.
  portfolio.py       Positions, cash, equity, high-water mark
  exits.py           Stop, target, time and trend exits for open positions
  planner.py         Turns approved decisions into an executable plan
  broker/
    base.py          Broker interface
    sim.py           Simulated fills for backtests
    alpaca.py        REST client for paper and live
  backtest.py        Event loop that runs the same decide and execute code
  baselines.py       SPY buy-and-hold and the 50/200 rule
  metrics.py         Return, drawdown, Sharpe, profit factor, consistency
  scoreboard.py      Scores runs against the pass bar
  holdout.py         Seal and one-time scoring of the holdout window
  money.py           The $100 to $200 plan, withdraw alerts, tax reserve
  alerts.py          ntfy notifier with priorities and formatting
  journal.py         Append-only JSONL decision log
  state.py           Durable state: kill switch, high-water mark, executions
  jobs.py            decide and execute entry points with all guards
  cli.py             Command line
```

## The roles

Scout. Reads a compact, anonymized feature table for the watchlist plus market regime and current holdings. Returns up to N candidates, each with a conviction score and a written thesis, through a strict JSON schema. With no API key, a rule-based scout runs instead (momentum plus trend), which also serves as the deterministic stand-in for tests.

Quant. Plain code. Every idea must pass: uptrend (close above the 50 day average), positive 60 day momentum, volatility under a ceiling, no extreme short-term spike, enough average dollar volume, and not already held. Failing ideas are logged with the gate that killed them.

Risk. Plain code, no network, no model. The rules in the next section. Every rejection carries a reason code.

Trader. Plain code. Converts approved decisions into notional orders with deterministic client order ids, then reconciles fills.

## Risk rules

All limits live in config, are validated at load, and are enforced in `risk.py`.

| Rule | Default |
| --- | --- |
| Max position size | 20% of equity |
| Max open positions | 5 |
| Max gross exposure | 100% of equity, never margin |
| Cash buffer kept | 2% of equity |
| Min order | $1.00 notional |
| Daily loss limit | No new entries after a 3% down day |
| Kill switch | Flatten and halt at 25% drawdown from the high-water mark. Manual reset only. |
| Same-day round trips | Never |
| Shorting, options, leverage | Never |
| Symbols | Watchlist only, fractionable only |
| Price sanity at execution | Skip if price moved more than 8% from the plan reference |
| Max orders per day | 10 |
| Stop loss | 2.5 x ATR(14) below entry, checked on the close |
| Max hold | 30 sessions |

## Data rules

Bars are daily, split and dividend adjusted. Features at day t use bars up to t only. A test perturbs every bar after t and asserts that nothing at t changes. Missing or stale data fails the run loudly and sends an alert. It never trades on partial data.

## Grading

Three baselines run on the same windows, the same costs and the same calendar: SPY, the 50/200 rule and MTUM. The pass bar is on risk (drawdown, Sharpe, Sortino, deflated Sharpe). See SCOREBOARD.md. Costs are 5 bps slippage per side plus regulatory sell fees. Windows:

| Window | Dates | Use |
| --- | --- | --- |
| Tune A | 2019-01-01 to 2020-12-31 | Tuning |
| Tune B | 2021-01-01 to 2022-12-31 | Tuning |
| Tune C | 2023-01-01 to 2024-12-31 | Tuning |
| Validation | 2025-01-01 to 2026-06-30 | Checked, not tuned on |
| Holdout | 2026-07-01 onward | Sealed. Scored once. |

The pass bar lives in `docs/SCOREBOARD.md` and in `config/scoreboard.yaml`. It is locked before the holdout is scored.

## Alerts

Every alert goes through ntfy with a priority:

| Priority | Event |
| --- | --- |
| 5 urgent | Kill switch tripped, job failure, data failure, live order rejected |
| 4 high | New trade signal, order filled, money plan milestone |
| 3 default | Daily summary, even on days with no trades, so silence always means something broke |
| 2 low | Informational |

A failed workflow sends its own urgent alert from a separate step, so a crash still reaches the phone.

## The money plan

Start with $100. When equity reaches 2x the deposit, Stoa sends a withdraw alert. The withdrawal happens on Alpaca's site, because API keys cannot move money. Once marked done, the high-water mark and kill switch rebase to the house money. A tax reserve estimate on realized short-term gains rides along on every summary.

## Safety around live money

Live mode requires all three: `mode: live` in config, the `STOA_LIVE_CONFIRM=I_UNDERSTAND` environment variable, and live keys. Any one missing and Stoa refuses to start. Paper and live keys are separate secrets so one can never be used as the other.
