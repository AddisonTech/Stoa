# Runbook

How to set Stoa up, run it every day, and what to do when it talks to you. No coding needed for the daily part. Setup takes about 45 minutes on a laptop, with your phone nearby.

Stoa starts in **paper** mode: fake money in an Alpaca paper account. Leave it there for months before thinking about live.

## 0. Lock it down first

Do these before anything else.

1. **Make the repository private.** On GitHub: the Stoa repository, **Settings**, **General**, scroll to **Danger Zone**, **Change repository visibility**, **Make private**. Stoa cannot do this for you. GitHub currently reports this repository as public. A private repository keeps the state branch (your positions, plans and journal) out of view, and its scheduled workflows are not paused after 60 days of no activity.
2. **Trade-only keys.** Alpaca API keys can trade and read the account. They cannot move money to or from a bank; that is only possible on Alpaca's site. Generate a separate pair for paper and, later, for live. Never reuse the live pair anywhere else. You chose not to turn on two-factor at Alpaca; that makes the keys and your Alpaca password the only locks, so keep both in a password manager.
3. **A separate Anthropic workspace, only if you want the analyst.** At console.anthropic.com: **Settings**, **Workspaces**, **Create workspace** (name it Stoa). Open it, **Limits**, set a monthly spend limit of **$25**. Create the API key inside that workspace (**API keys**, **Create key**, pick the Stoa workspace). The key then can only spend from that workspace, and Anthropic stops it at $25 even if Stoa's own cap failed. Stoa also stops itself at `budget.monthly_cap_usd` ($25).
4. Secrets live only in GitHub secrets. Never in a file, a commit, an issue or a chat.

## 1. Create an Alpaca paper account and keys

1. On a laptop, go to alpaca.markets and sign up. You only need the free account.
2. Log in. At the top left, make sure the account switcher says **Paper**.
3. Paper accounts start with $100,000. Stoa's money plan assumes $100. In the paper account menu, choose **Reset account** (or "Paper account settings") and set the balance to **100**. If you skip this, Stoa still trades, but the withdraw milestone stays off and the first summary tells you why.
4. On the paper dashboard, find **API Keys** and click **Generate New Keys**.
5. Copy the **Key ID** and the **Secret Key** into a note. The secret is shown only once.

## 2. Set up phone alerts with ntfy

1. Install the **ntfy** app on your phone (App Store or Google Play). It is free.
2. Make up a long random topic name. This is the password, because anyone who knows it can read your alerts. Something like `stoa-` followed by 24 random letters and numbers. A password manager can generate it.
3. In the app, tap **+**, choose **Subscribe to topic**, type your topic name, and keep the server as `ntfy.sh`.
4. Optional: in the app's notification settings, let priority 5 alerts break through Do Not Disturb.

Do not put the topic name in any file in the repository. It goes only into GitHub secrets.

## 3. Add GitHub secrets

1. On GitHub, open the Stoa repository, then **Settings**, **Secrets and variables**, **Actions**.
2. Under **Secrets**, click **New repository secret** for each of these:

| Name | Value | Needed for |
| --- | --- | --- |
| `ALPACA_PAPER_KEY_ID` | the paper Key ID | paper mode, and market data in every mode |
| `ALPACA_PAPER_SECRET` | the paper Secret Key | same |
| `STOA_NTFY_TOPIC` | your topic name | alerts |
| `STOA_NTFY_APPROVAL_TOPIC` | a second long random topic name, different from the first | hand approval of trades (required live, first 30 days) |
| `STOA_HEALTHCHECK_URL` | the ping URL from healthchecks.io (section 5) | the dead man's switch |
| `FINNHUB_API_KEY` | a free key from finnhub.io | recommended: the second price source and upcoming earnings dates |
| `ANTHROPIC_API_KEY` | the key from the Stoa workspace (section 0) | optional; without it nothing uses Claude |
| `STOA_NTFY_TOKEN` | an ntfy access token | optional; only for a protected topic or server |

Leave `ALPACA_LIVE_KEY_ID`, `ALPACA_LIVE_SECRET` and `STOA_LIVE_CONFIRM` empty until the go-live checklist.

3. Under **Variables** (same page, other tab), you can add `STOA_MODE` with `paper` or `signal`. If you leave it out, Stoa uses paper. Add `STOA_NTFY_SERVER` only if you run your own ntfy server.

Without `FINNHUB_API_KEY`, Stoa tries Yahoo (no key) for the second price and upcoming earnings. Yahoo sometimes blocks cloud servers like GitHub's. When the second price cannot be fetched, buys are skipped (on purpose: `crosscheck.on_unavailable: skip`), so the Finnhub key is worth the two minutes: sign up at finnhub.io, copy the API key from the dashboard.

## 4. Turn on the workflow

1. Open the **Actions** tab. If GitHub asks, click **I understand my workflows, go ahead and enable them**.
2. Click the **daily** workflow on the left. If it says it is disabled, click **Enable workflow**.
3. Click **Run workflow**, pick **alert-test**, and click the green **Run workflow** button.
4. Within a couple of minutes your phone should show "Stoa test alert". If not, see "No alerts at all" below.

From now on it runs by itself on weekdays. You can watch it from the GitHub app on your phone too (Actions tab).

## 5. The dead man's switch (healthchecks.io)

Stoa already alerts you when it fails. This catches the case where it never runs at all (GitHub down, the workflow disabled, a schedule that silently stopped).

1. Sign up at healthchecks.io (free: 20 checks).
2. **Add Check**. Name it `Stoa daily`.
3. Schedule: pick **Cron**. Cron expression `15 16 * * 1-5`. Time zone `America/New_York`. Grace time **4 hours**. That means: after 4:15 pm New York time on a weekday, Stoa must report success by 8:15 pm. (DECIDE runs at about 5:15 pm, so the success ping always lands after 4:15 pm and inside the window, with room for GitHub's delays and the second slot an hour later.)
4. Copy the check's **Ping URL** (looks like `https://hc-ping.com/<long id>`). Add it as the GitHub secret `STOA_HEALTHCHECK_URL`. It is a secret: anyone with it can mark your job as fine.
5. Connect your phone: **Integrations**, **ntfy**. Server `https://ntfy.sh`, topic: your `STOA_NTFY_TOPIC` value (or a third topic just for this). Priority: urgent. Then on the check, make sure the ntfy integration is ticked.
6. Market holidays: the check will alert on NYSE holidays because Stoa does not run. Either add the year's holidays as pauses, or accept one alert on each holiday and ignore it.

Stoa pings `/start` when DECIDE begins, the plain URL after the daily summary is sent, and `/fail` when DECIDE fails or refuses (and the workflow pings `/fail` if it crashes before Stoa starts). The URL never appears in a log or alert.

## 6. Hand approval of trades

Every buy, in paper and live, comes as a 9:00 am pick alert with **Approve** and **Skip** buttons. Nothing is bought without Approve. Sells (stops, time exits and the other exit rules) go through on their own, except during the first 30 days after the first live trade, when they need a tap too. The kill switch flatten never waits.

1. Make a second long random topic name, different from the alerts topic. Add it as `STOA_NTFY_APPROVAL_TOPIC`. You do not need to subscribe to it in the app; the buttons post to it.
2. When the pick alert arrives at about 9:00 am New York time, tap **Approve** or **Skip** by **9:40 am**. A tap after that is ignored.
3. EXECUTE reads your taps at 9:45 am. Approved orders go through the usual checks. Anything skipped, or not answered, is not sent, and you get a "Skipped" alert saying why.
4. To test it now (on paper): Actions, daily, Run workflow, pick **approve-test**. Tap a button within three minutes. The log says what it saw. To let paper buys go through without a tap, add the repository variable `STOA__APPROVAL__BUYS` with `mode`.

Each plan has its own random code. An old tap, a tap from a different plan, or a forged message without the code does nothing.

## 7. What happens each day

All times are New York time.

- **About 5:15 pm**: DECIDE. Stoa reads the day's bars, records the day and sends the **day summary** (priority 3), every trading day: how each position and each of today's buys did today, and the whole account against SPY, today and since the start. Then it quietly plans tomorrow (checks stops and earnings dates, runs the v1 rules and the risk engine). The daily workflow sets `STOA__SCOUT__KIND=rule`, so paper trading never calls Claude; Claude is used only by trial 7's shadow scoring. No pick alert goes out in the evening.
- **About 9:00 am**: the SWEEP. Stoa fetches every article published since the close on the planned buys and on the reserves (names the rules passed but had no slot for), and reads each headline and summary with FinBERT, a free open-source model, on GitHub's CPU. A planned buy with clearly bad news overnight is dropped (you get a "Dropped BUY" alert with the headline). A reserve with clearly good news can take a free slot (at most one a day). Then you get the **final picks**, one alert per order (priority 4), buys with Approve and Skip buttons. Taps count until **9:40 am**. If the evening run was missed, the sweep makes the plan first from the last close. If the reader cannot run, the rules' picks go out unchanged and an alert says so. To check the reader on GitHub by hand: Actions, daily, Run workflow, pick **reader-check** (it reads two test headlines and keeps the model cached for the morning run).
- **9:40 am**: taps close. You get **Approvals closed at 9:40** (priority 4): "Approved X, Y; skipped the rest (...)". Any tap after 9:40 gets the reply "Too late, approvals closed at 9:40." until 10:15 am, and is not acted on. This is the close workflow: it only reads state and has its own concurrency group, so it never holds up EXECUTE. If the taps came late and you still want the buys on paper, see "Taps came too late" in section 9.
- **The 2nd of each month, early morning**: the monthly retrain. New data, the same walk-forward, and a new model only if it is no worse than the current one. You get a short summary.
  **Currently off** (disabled on 2026-10-24 with `gh workflow disable monthly`). It stays off until a design passes walk-forward: none has (trials 1 to 6 failed, see DECISIONS.md), and retraining a failed design only adds trials to the registry. Turn it back on with `gh workflow enable monthly` or the Actions tab once a model is promoted.
- **About 9:45 am**: EXECUTE. Stoa places the approved buys and the planned sells and sends one alert per fill with its price, then places a stop order at Alpaca for every position held from a prior day (see docs/RISK.md, Stops at the broker). If a stop fills during the day, DECIDE records it that evening and alerts you.
- **About 7:45 pm**: the analyst workflow (trial 7, the analyst forward test in DECISIONS.md). It scores the session's news for the point-in-time S&P 500 through the Batch API in shadow mode and commits the scores to `shared/shadow/` on the state branch. Shadow scores never reach a trade. It has its own concurrency group, so it never holds up DECIDE or EXECUTE. The daily summary and the dashboard show how many of the 210 sessions are scored. After 210, run `stoa analyst evaluate` once.
- **After each daily run**: the pages workflow rebuilds the read-only dashboard at https://addisontech.github.io/Stoa/ from the committed state. It gets no secrets and has no buttons; approvals stay on your phone. Its **trade log** is public: every morning pick, drop, fill, skip and closed trade with its date, symbol, side, size, reason and profit or loss (paper money). To rebuild by hand: Actions, pages, Run workflow. One-time setup: Settings, Pages, Source: GitHub Actions.

Each job is started on time by Windows Task Scheduler on the Dell (section 13), with GitHub cron as the backup a few minutes later, in two slots an hour apart (because it ignores daylight saving time). Any later start sees the first already did the work and does nothing. Weekends and market holidays are skipped. On early close days (like the day after Thanksgiving) DECIDE runs earlier on its own.

## 8. Reading alerts

**Pick** (priority 4, about 9:00 am)
> BUY NVDA $19.40 (19% of account) at 9:45 am. Stop $91.25. Why: ...
> Overnight news: 2 articles, tone +0.31.
> Tap Approve or Skip by 9:40 am. Nothing is sent without Approve.

Tap Approve to buy it at 9:45, Skip (or nothing) to pass. In signal mode, place it yourself if you want to follow along. The stop is where Stoa would sell on a close. The tone is FinBERT's average read of the night's articles, from -1 (all negative) to +1 (all positive).

> SELL KO all 0.12 sh at 9:45 am. Why: stop loss ...

Stoa sells the whole position. Reasons: stop loss, trailing stop, time exit (30 sessions), trend break (two closes under the 50 day average), kill switch.

**Filled** (priority 4): the order went through, with shares and price.

**Dropped BUY** (priority 3, about 9:00 am): bad overnight news on a planned buy; the headline is in the alert. Nothing to do.

**Morning sweep** (priority 2): a note from the sweep, such as the reader not running or the plan being made in the morning. **No trades today** (priority 2): nothing planned for the open.

**Day summary** (priority 3, about 5:15 pm): equity and today's change against SPY's, each position's move today (and today's buys against their fill price), anything sold today, return since start against SPY, cash, Claude API spend this month, realized short-term gains this year and how much to set aside for tax. Tomorrow's picks are not in it; they come at 9:00 am.

**Skipped** (priority 2 for a buy, 4 for a sell): a check stopped an order. Usually the price jumped more than 8% overnight, or the order was not approved. For a buy, nothing to do. For a sell (during the approval month, an exit you did not approve), the position is still open: approve the next one, or sell it yourself in the Alpaca app.

**Skipped buy ...: price check** (priority 4): the second price source disagreed with Alpaca by more than 1%, or could not be reached. Stoa did not buy. If it happens every day, check the `FINNHUB_API_KEY` secret.

## 9. When an alert says...

**"Stoa decide failed" or "Stoa execute failed" (priority 5)**
Something crashed. Nothing else happens until it is fixed. Tap the alert or open the Actions tab and read the red run's log. If it was a hiccup (GitHub or Alpaca down), the next slot usually fixes itself. To retry now: Actions, daily, Run workflow, pick the job, tick **force**.

**"Stoa workflow failed" (priority 5)**
Same as above, but the failure was outside Stoa (install, checkout, saving state). Open the log.

**"Stoa decide: bad data" (priority 5)**
The day's bars were missing or late. Stoa did not trade. It tries again at the next slot an hour later. If it keeps happening, check status.alpaca.markets.

**"Stoa decide refused to run" (priority 5)**
A guard said no. The message says which: missing keys, live mode locked, or the account is blocked. Fix what it names.

**"Stop place failed", "Stop replace failed", "Stop cancel failed" or "Stop check failed" (priority 5)**
Stoa could not place, cancel or check a protective stop at Alpaca. The rest of the run carried on. Open the Alpaca app and look at the position and its open orders. A missing stop is placed again at the next EXECUTE; the close rules still apply in the meantime.

**"Stoa analyst shadow failed" (priority 4)**
The trial 7 shadow scoring did not finish. Trading is not affected. That session is not counted; the forward test simply needs one more session. If it keeps failing, open the log. To retry the same evening: Actions, analyst, Run workflow.

**"Analyst spend at 80% of the cap" (priority 3)**
The Claude analyst has used 80% of this month's cap. Once a month at most. Nothing to do.

**"Analyst paused at its cap" (priority 4, Approve and Skip buttons)**
The analyst reached this month's cap and stopped spending. Approve raises this month's cap by $10 (`budget.raise_step_usd`); Skip, no tap within the approval wait, a stale button or a late tap leaves it paused until the 1st. Raises end with the month, and no month goes past `budget.hard_ceiling_usd` ($75). Trading carries on either way: the model just runs without new news scores, as it does with the analyst off. Every raise is recorded in state and the dashboard shows spend against the cap.

**"Analyst stays paused" or "Analyst cap raised" (priority 3)**
The outcome of the ask above. Nothing to do.

**"Stopped out: SYMBOL" (priority 4)**
A protective stop filled during the day. Stoa recorded it as a `stop_out` trade at DECIDE. Nothing to do.

**"KILL SWITCH TRIPPED" (priority 5)**
Equity fell 25% below its best close. Stoa is selling everything at the next open and will not buy again until you reset it. Do not rush. See "Kill switch" below.

**"Order rejected" (priority 5 live, 4 paper)**
Alpaca refused an order. Open the Alpaca app and look at Orders. Common causes: account restricted, asset halted.

**"Order still working" (priority 4)**
An order did not finish in 90 seconds. Stoa checks it again at DECIDE. Nothing to do.

**"Scout fell back to rules" (priority 4)**
Claude was not used today (no key, refused, too long, or the monthly cost cap hit). The rule scout ran instead. If it says cost cap, it resets next month, or raise `llm.monthly_cost_cap_usd`.

**"FALLBACK: switching to SPY" (priority 5)**
Live or paper results drifted worse than walk-forward said they should, or six months passed with no edge over SPY. Stoa stops opening positions, lets the current ones exit normally, and buys SPY with the cash. Read `stoa fallback status` (or the summaries) before doing anything. To go back to the model: Actions, daily, Run workflow, **fallback-reset**. Only do that after you understand what changed.

**"Stoa monthly retrain" (priority 3)**
The monthly summary. "not promoted" is normal and fine: the current model stays.

**"Withdraw $100 now" (priority 4)**
The money plan milestone. See "Withdrawing" below.

**"Detected withdrawal" or "Detected deposit" (priority 4)**
Stoa saw money move in Alpaca and adjusted its high-water mark so it is not counted as a gain or loss.

**No pick alert by 9:20 am on a trading day**
Either nothing was planned (you get "No trades today"), or the sweep did not run. Check the Actions tab. Without a sweep no buy is approved, so nothing is bought; sells still run at 9:45.

**No summary by 7 pm on a trading day**
Silence means something broke. Check the Actions tab. If there is no run at all, see "Scheduled runs stopped".

### Taps came too late (paper only)

Actions, daily, Run workflow, pick **override**. It runs once a session, only in paper, only while the market is open and only after EXECUTE: it sends today's planned buys you tapped Approve on (at any time), or the ones named in **symbols**, or every planned buy if neither names one. The sector cap is lifted for that run only; every other check still runs. The journal and the trade log mark it as a manual override. It never changes the 9:40 cutoff or lets EXECUTE run twice.

## 10. Withdrawing (the money plan)

The plan: grow $100 to $200, then take the $100 out and let Stoa trade only the profit.

1. When equity reaches $200, you get "Withdraw $100 now". It is sent once.
2. Open the Alpaca app (live account, once you are live), go to **Banking** or **Transfers**, and withdraw $100 to your bank. API keys cannot move money, so this is always done by you.
3. If cash is short because money is in positions, wait for the next sell.
4. Tell Stoa it is done: Actions, daily, Run workflow, pick **money-confirm**. Leave equity blank to read it from Alpaca.
5. Stoa rebases its high-water mark to what is left, so the withdrawal does not count as a drawdown.

In signal mode, money-confirm takes the $100 out of Stoa's own paper portfolio instead.

## 11. Kill switch

When it trips, Stoa cancels every open order in the account and sells every position at the next open, watchlist or not, and stops buying.

Before resetting:

1. Read the last few daily summaries and the Actions logs. Was it the market, or did Stoa do something wrong?
2. If you are not sure, leave it off. Cash is a fine place to be.

To reset: Actions, daily, Run workflow, pick **kill-switch-reset**. Leave equity blank to read it from Alpaca. The high-water mark becomes today's equity, and buying resumes at the next DECIDE.

On a laptop: `stoa kill-switch status` and `stoa kill-switch reset`.

## 12. Go-live checklist

Do not go live until all of these are true. Stoa enforces the first three itself and refuses live mode until they hold (`stoa exam status`).

- [ ] `stoa model train` on real data passed walk-forward, and you have read the calibration report.
- [ ] The holdout is sealed and scored once, and it passed (see HOLDOUT.md).
- [ ] The paper exam passed: at least 42 sessions (63 is better) and 20 closed trades, with lower drawdown and higher Sharpe than SPY after simulated costs.
- [ ] `stoa score` passes on real data (see SCOREBOARD.md), and it beats MTUM. If it does not beat MTUM, buy MTUM instead.
- [ ] The healthchecks.io check has been green for a month.
- [ ] `STOA_NTFY_APPROVAL_TOPIC` is set and approve-test works.
- [ ] You understand every alert type above.
- [ ] You have money you can afford to lose entirely. Start with $100.

Then:

1. Open and fund a live Alpaca account with $100. Generate **live** API keys (switch the account selector to Live first).
2. Add secrets `ALPACA_LIVE_KEY_ID` and `ALPACA_LIVE_SECRET`.
3. Add secret `STOA_LIVE_CONFIRM` with exactly `I_UNDERSTAND`.
4. Set the repository variable `STOA_MODE` to `live`.
5. Nothing to rename. In live mode the workflow keeps state in `live-state/` on the `stoa-state` branch, so the live account starts fresh (its own high-water mark and money plan), while the paper record in `state/` stays where the exam check and the quit rule read it. The model, the experiment registry and the exam records are shared in `shared/`.
6. For the first 30 days after the first live fill, every trade needs your tap.

If any of the three locks is missing, Stoa refuses and sends an urgent alert. To stop live trading at any time, delete `STOA_LIVE_CONFIRM` or set `STOA_MODE` back to `paper`.

## 13. Scheduled runs stopped

GitHub disables scheduled workflows in a public repository after 60 days with no activity, and it is not clear that Stoa's own state commits count. Make the repository private (section 0) and this does not apply. Until then, make any small commit to the main branch once a month, or re-enable the **daily** workflow if the Actions tab shows a banner. The healthchecks.io check (section 5) tells you if the daily run stops.

GitHub cron can start late under load, or not at all: on 2026-10-26, the first scheduled morning, no scheduled run of any workflow had ever started in this repository, and the 8:55 sweep never came. So the Dell starts each job on time:

- `scripts\dispatch.ps1 -Job <decide|sweep|execute>` starts the daily workflow by `workflow_dispatch` with the GitHub CLI (`gh`, logged in as AddisonTech) and logs to `data\dispatch.log`.
- `scripts\install-dispatch.ps1` registers five Task Scheduler tasks, weekdays, New York time: **Stoa sweep** at 8:57 am (the run holds until 9:00, so the picks go out at 9:00 with the night's news complete), **Stoa close** at 9:39 am (it holds until 9:40), **Stoa execute** at 9:45 am, **Stoa decide** at 5:15 pm and **Stoa analyst** (trial 7's shadow scoring, the analyst workflow) at 7:45 pm. They wake the Dell from sleep and catch up if it was off at the time. Run it again after changing a time.
- GitHub cron stays as the backup: DECIDE at 5:20 pm, SWEEP at 9:05 am, CLOSE at 9:41 am, EXECUTE at 9:50 am, the analyst at 7:50 pm (6:50 pm in winter).
- They can never double-run. Every run is in the `stoa-state` concurrency group, so runs go one at a time, and each job records its session on the state branch: a second start for the same job logs "already swept" (or decided, or executed) and sends nothing. Taps still close at 9:40 am whichever start sent the picks.

If the Dell is off or asleep through a whole window, the backup cron covers it. If both miss the morning, start the sweep by hand before 9:40 (Actions, daily, Run workflow, **sweep**).

## 13a. The live positions panel (Vercel)

The dashboard's **Live** card shows the paper account's equity, today's change and every open position (quantity, price, market value, today's change and total profit or loss), refreshed every 30 seconds while the tab is open and the market is open, with "Market closed" outside regular hours. GitHub Pages is static and public, so the keys cannot live there. A small read-only function on Vercel's free tier holds them instead:

- **Code:** `live/api/positions.py`, standard library only. It makes exactly two kinds of request, both GET, to the paper API: `/v2/account` and `/v2/positions`. There is no order, cancel or other write call anywhere in it; a test checks the source.
- **Paper only:** it refuses (HTTP 503) if `ALPACA_LIVE_KEY_ID`, `ALPACA_LIVE_SECRET` or `STOA_MODE=live` is set in Vercel, or if the key is not a paper key (paper keys start with `PK`).
- **Rate limit:** answers are cached 25 seconds, in memory and at Vercel's edge (`s-maxage=25`), so however many tabs poll, Alpaca sees at most two requests every 25 seconds, far from the 200 a minute the pulls and the daily jobs share.
- **CORS:** only `https://addisontech.github.io` may read it from a browser.
- **If it is down:** the card says "Live data unavailable right now" and the rest of the dashboard, built from committed state, loads as usual.
- **Where:** the Vercel project is `stoa-live` and the address is `https://stoa-live.vercel.app/api/positions`, set as `dashboard.live_url` in `config/default.yaml`; blank hides the card.

Deploying or changing it (from the repository root, with Node installed):

```
npx vercel@65.2.0 login                       # once, in a browser
npx vercel@65.2.0 link --yes --project stoa-live --cwd live   # once per machine
npx vercel@65.2.0 deploy --prod --cwd live    # after a change to live/
```

`live/vercel.json` sets `"framework": null`. Without it Vercel treats the folder as a Python web app and refuses the deploy with "No python entrypoint found".

**Rotating the paper keys means updating them in Vercel too**, or the card goes blank: Vercel, the stoa-live project, Settings, Environment Variables, edit `ALPACA_PAPER_KEY_ID` and `ALPACA_PAPER_SECRET` (Production), then redeploy (`npx vercel@65.2.0 deploy --prod --cwd live`). Or from the command line: `npx vercel@65.2.0 env rm ALPACA_PAPER_KEY_ID production --cwd live`, then `npx vercel@65.2.0 env add ALPACA_PAPER_KEY_ID production --cwd live` (it asks for the value), the same for the secret, then redeploy. Never add the live keys there.

## 14. Changing settings

Every limit is in `config/default.yaml`. Change it in a commit, or override one value with a repository variable passed as an environment variable named `STOA__SECTION__KEY` (for example `STOA__RISK__MAX_POSITIONS=4`). Changing settings after the holdout is sealed makes `stoa holdout score` refuse, on purpose.

## 15. Running on a laptop

```bash
pip install -e ".[yahoo]"
export ALPACA_PAPER_KEY_ID=... ALPACA_PAPER_SECRET=... STOA_NTFY_TOPIC=...
stoa status
stoa alert test
stoa data fetch && stoa universe build && stoa fundamentals fetch
stoa model train         # walk-forward, report in results/, promote past the gate
stoa model report        # both horizons; add --ablation once analyst scores exist
stoa score               # every strategy, graded on risk
stoa exam status
stoa decide --force      # outside the window, uses the last closed session
```

The analyst (optional, costs money):

```bash
export ANTHROPIC_API_KEY=...         # the Stoa workspace key
stoa news fetch                      # watchlist news since 2016
stoa analyst backfill --estimate     # prints the cost; spends nothing
stoa analyst backfill                # batch API at half price; refuses if over budget
stoa model report --ablation         # does text help after Claude's knowledge cutoff?
```

Only set `STOA__MODEL__USE_TEXT=true` if the ablation shows lower log loss with text in the clean period (after July 2026).

Local runs keep state in `state/`. The workflow keeps its own copy on the `stoa-state` branch. Do not run both against the same Alpaca account on the same day.

## No alerts at all

1. In the ntfy app, check the topic name matches the `STOA_NTFY_TOPIC` secret exactly.
2. Run the **alert-test** job again and open its log. "Sent." means ntfy accepted it; the problem is on the phone side (notifications blocked for the app).
3. If the log says "Stoa is not set up yet. Missing secrets", add the secrets it lists.

## Sample run

This is part 3 of `python scripts/demo.py`: DECIDE at 5:15 pm, the 9:00 am sweep and EXECUTE at 9:45 am against the simulated broker on synthetic data, with every alert printed instead of sent. The night is quiet (no news), so the sweep leaves the rules' picks alone. The phone is simulated: it approves the first pick and leaves the rest unanswered. The prices are made up.

```text
DECIDE 2021-10-12 21:15 UTC -> decided

[p3] Stoa 2021-10-12 (paper)
    Equity $100.00
    Since start +0.0% vs SPY +0.0% (+0.0%)
    Today: no positions
    Cash $100.00
    Tomorrow: the picks come at 9:00 am, after the overnight news sweep.
    API spend $0.00 of $25.00 this month
    Realized short-term gains this year $0.00, set aside $0.00 for tax
    Analyst shadow: 0 of 210 sessions scored; Claude spend $0.00 of $25.00 in 2021-10

SWEEP 2021-10-13 13:00 UTC -> swept

[p4] BUY NVDA
    BUY NVDA $20.00 (20% of account) at 9:45 am. Stop $120.27. Why: Momentum score 0.432: 60 day +45.7%, 120 day +39.5%, above its 50 and 200 day averages.
    No overnight news.
    Tap Approve or Skip by 9:40 am. Nothing is sent without Approve.
    [button] Approve
    [button] Skip
[p4] BUY HD
    BUY HD $20.00 (20% of account) at 9:45 am. Stop $1,152.07. Why: Momentum score 0.271: 60 day +10.3%, 120 day +52.2%, above its 50 and 200 day averages.
    No overnight news.
    Tap Approve or Skip by 9:40 am. Nothing is sent without Approve.
    [button] Approve
    [button] Skip
[p4] BUY UNH
    BUY UNH $20.00 (20% of account) at 9:45 am. Stop $1,050.99. Why: Momentum score 0.253: 60 day +19.6%, 120 day +34.0%, above its 50 and 200 day averages.
    No overnight news.
    Tap Approve or Skip by 9:40 am. Nothing is sent without Approve.
    [button] Approve
    [button] Skip
[p4] BUY AMZN
    BUY AMZN $20.00 (20% of account) at 9:45 am. Stop $375.18. Why: Momentum score 0.250: 60 day +23.1%, 120 day +27.9%, above its 50 and 200 day averages.
    No overnight news.
    Tap Approve or Skip by 9:40 am. Nothing is sent without Approve.
    [button] Approve
    [button] Skip

EXECUTE 2021-10-13 13:45 UTC (first order approved, rest unanswered) -> executed

[p4] Filled buy NVDA
    BUY NVDA: 0.154398 sh at $129.54 = $20.00 (paper).
[p2] Skipped buy HD
    Not approved (no answer). Nothing was sent for HD.
[p2] Skipped buy UNH
    Not approved (no answer). Nothing was sent for UNH.
[p2] Skipped buy AMZN
    Not approved (no answer). Nothing was sent for AMZN.

Sim broker cash 80.00, positions:
    NVDA 0.154398287 sh at 129.5351
```
