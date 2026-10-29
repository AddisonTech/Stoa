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
| `STOA_NTFY_APPROVAL_TOPIC` | a second long random topic name, different from the first | hand approval of trades: live mode's first 30 days, or if taps are turned back on (section 6) |
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

## 6. Hand approval of trades (off on paper)

Since 2026-10-28 the $100 account's picks go through on their own, the same way the big account's do: the owner does not want to approve trades he cannot judge (docs/DECISIONS.md). At about 9:00 am one FYI alert lists what will be bought and sold at 9:45 am, with no buttons. Every risk cap, the price checks, the broker stops and the kill switch are exactly as before. The setting is `approval.buys: mode`: buys follow `approval.mode`, which asks for no tap on paper.

The Approve and Skip buttons are still in the code, for two cases:

- **Live mode.** For the first 30 days after the first live trade (`approval.mode: auto`), every trade, buy or sell, needs a tap again. Changing that is a separate decision, to be made before going live.
- **Turning them back on for paper:** set `approval.buys: always` in `config/default.yaml` (or the repository variable `STOA__APPROVAL__BUYS` with `always`).

When taps are on:

1. Make a second long random topic name, different from the alerts topic. Add it as `STOA_NTFY_APPROVAL_TOPIC`. You do not need to subscribe to it in the app; the buttons post to it.
2. Each pick comes as its own alert with **Approve** and **Skip**. Tap by **9:40 am**; a tap after that is ignored.
3. EXECUTE reads your taps at 9:45 am. Approved orders go through the usual checks. Anything skipped, or not answered, is not sent, and you get a "Skipped" alert saying why. The kill switch flatten never waits.
4. To test it: Actions, daily, Run workflow, pick **approve-test**. Tap a button within three minutes. The log says what it saw.

Each plan has its own random code. An old tap, a tap from a different plan, or a forged message without the code does nothing.

## 7. What happens each day

All times are New York time.

- **About 5:15 pm**: DECIDE. Stoa reads the day's bars, records the day and sends the **day summary** (priority 3), every trading day: how each position and each of today's buys did today, and the whole account against SPY, today and since the start. Then it quietly plans tomorrow (checks stops and earnings dates, runs the v1 rules and the risk engine). The daily workflow sets `STOA__SCOUT__KIND=rule`, so paper trading never calls Claude; Claude is used only by trial 7's shadow scoring. No pick alert goes out in the evening.
- **About 9:00 am**: the SWEEP. Stoa fetches every article published since the close on the planned buys and on the reserves (names the rules passed but had no slot for), and reads each headline and summary with FinBERT, a free open-source model, on GitHub's CPU. A planned buy with clearly bad news overnight is dropped (you get a "Dropped BUY" alert with the headline). A reserve with clearly good news can take a free slot (at most one a day). Then you get the **final picks** in one alert (priority 3) listing every buy and sell; they go in at 9:45 am on their own, with no buttons. If the evening run was missed, the sweep makes the plan first from the last close. If the reader cannot run, the rules' picks go out unchanged and an alert says so. To check the reader on GitHub by hand: Actions, daily, Run workflow, pick **reader-check** (it reads two test headlines and keeps the model cached for the morning run).
- **9:40 am**: the close workflow. It matters only when a tap is needed (section 6): then taps close and you get **Approvals closed at 9:40** (priority 4), and any later tap gets "Too late" until 10:15 am. On paper nothing waits for a tap, so it finds nothing to close and sends nothing. It only reads state and has its own concurrency group, so it never holds up EXECUTE.
- **The 2nd of each month, early morning**: the monthly retrain. New data, the same walk-forward, and a new model only if it is no worse than the current one. You get a short summary.
  **Currently off** (disabled on 2026-10-24 with `gh workflow disable monthly`). It stays off until a design passes walk-forward: none has (trials 1 to 6 failed, see DECISIONS.md), and retraining a failed design only adds trials to the registry. Turn it back on with `gh workflow enable monthly` or the Actions tab once a model is promoted.
- **About 9:45 am**: EXECUTE. Stoa places the planned buys and sells and sends one alert per fill with its price, then places a stop order at Alpaca for every position held from a prior day (see docs/RISK.md, Stops at the broker). If a stop fills during the day, DECIDE records it that evening and alerts you.
- **About 7:45 pm**: the analyst workflow (trial 7, the analyst forward test in DECISIONS.md). It scores the session's news for the point-in-time S&P 500 through the Batch API in shadow mode and commits the scores to `shared/shadow/` on the state branch. Shadow scores never reach a trade. It has its own concurrency group, so it never holds up DECIDE or EXECUTE. The daily summary and the dashboard show how many of the 210 sessions are scored. After 210, run `stoa analyst evaluate` once.
- **After each daily run**: the pages workflow rebuilds the read-only dashboard at https://addisontech.github.io/Stoa/ from the committed state. It gets no secrets and has no buttons; nothing is ever traded from it. Its **trade log** is public: every morning pick, drop, fill, skip and closed trade with its date, symbol, side, size, reason and profit or loss (paper money). To rebuild by hand: Actions, pages, Run workflow. One-time setup: Settings, Pages, Source: GitHub Actions.

Each job is started on time by cron-job.org (section 13), with Windows Task Scheduler on the Dell as a backup at the same minute and GitHub cron as the last backup a few minutes later, in two slots an hour apart (because it ignores daylight saving time). If cron-job.org misses a slot, the next job sends a "Trigger missed" alert. Any later start sees the first already did the work and does nothing. Weekends and market holidays are skipped. On early close days (like the day after Thanksgiving) DECIDE runs earlier on its own.

## 8. Reading alerts

**Picks** (priority 3, about 9:00 am)
> Picks for 2026-10-29. They go in on their own at 9:45 am; nothing to tap.
> BUY NVDA $19.40 (19% of account) at 9:45 am. Stop $91.25. Why: ...
> Overnight news: 2 articles, tone +0.31.
> SELL KO all 0.12 sh at 9:45 am. Why: stop loss ...

For your information only: nothing to do. In signal mode it says "You place them yourself". The stop is where Stoa would sell on a close. The tone is FinBERT's average read of the night's articles, from -1 (all negative) to +1 (all positive). When taps are turned on (section 6), each pick comes as its own alert with Approve and Skip instead.

For a sell, Stoa sells the whole position. Reasons: stop loss, trailing stop, time exit (30 sessions), trend break (two closes under the 50 day average), kill switch.

**Filled** (priority 4): the order went through, with shares and price.

**Dropped BUY** (priority 3, about 9:00 am): bad overnight news on a planned buy; the headline is in the alert. Nothing to do.

**Morning sweep** (priority 2): a note from the sweep, such as the reader not running or the plan being made in the morning. **No trades today** (priority 2): nothing planned for the open.

**Day summary** (priority 3, about 5:15 pm): equity and today's change against SPY's, each position's move today (and today's buys against their fill price), anything sold today, return since start against SPY, cash, Claude API spend this month, realized short-term gains this year and how much to set aside for tax. Tomorrow's picks are not in it; they come at 9:00 am.

**Skipped** (priority 2 for a buy, 4 for a sell): a check stopped an order. Usually the price jumped more than 8% overnight, or (only when taps are on) the order was not approved. For a buy, nothing to do. For a sell (during the approval month, an exit you did not approve), the position is still open: approve the next one, or sell it yourself in the Alpaca app.

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
Either nothing was planned (you get "No trades today"), or the sweep did not run. Check the Actions tab. Without a sweep no buy goes in; sells still run at 9:45.

**No summary by 7 pm on a trading day**
Silence means something broke. Check the Actions tab. If there is no run at all, see "Scheduled runs stopped".

### Taps came too late (paper only)

Only when taps are turned on (section 6). Actions, daily, Run workflow, pick **override**. It runs once a session, only in paper, only while the market is open and only after EXECUTE: it sends today's planned buys you tapped Approve on (at any time), or the ones named in **symbols**, or every planned buy if neither names one. The sector cap is lifted for that run only; every other check still runs. The journal and the trade log mark it as a manual override. It never changes the 9:40 cutoff or lets EXECUTE run twice.

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

## 11a. The owner's stop

Since 2026-10-29 the $100 account (`stop.usd100`) and the shadow scoreboard (`stop.shadow`) are stopped (`stop:` in `config/default.yaml`). The big paper account (`stop.big_paper`) was stopped the same day and turned back on that evening as trial 15 (section 13c); its first trial 15 orders go in at 9:45 am on 2026-10-30.

- **A stopped account** sells everything it holds at its next 9:45 am run, SPY and anything bought by hand included, and buys nothing. After that it trades nothing. It cancels its open orders first, the same way the kill switch does, but it is not the kill switch: no drawdown is involved, the kill switch stays off and nothing needs a reset.
- **Alerts.** The morning alert says what will be sold, the 9:45 am alert starts with "STOPPED by the owner", and the 5:15 pm summary says the account is stopped. Once an account holds nothing, its mornings are quiet. The daily summary still records equity against SPY.
- **The shadow scoreboard** scores, locks and grades nothing, and the summary has no shadow lines.
- **Still running.** The workflows and the big account's 8:15 am feed, which keeps the news brain's data current for the next trial.
- **To start again.** Set the account's `stop` value back to `false`, as its own recorded decision, after the next trial is registered.

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

GitHub cron can start late under load, or not at all: on 2026-10-26, the first scheduled morning, no scheduled run of any workflow had ever started in this repository, and the 8:55 sweep never came. So three clocks start each job, and the Dell can be closed:

- **cron-job.org is the primary clock** (free account, set up 2026-10-26). Five jobs, weekdays, America/New_York: **stoa sweep** 8:57 am (the run holds until 9:00, so the picks go out at 9:00 with the night's news complete), **stoa close** 9:39 am (it holds until 9:40), **stoa execute** 9:45 am, **stoa decide** 5:15 pm and **stoa analyst** 7:45 pm (trial 7's shadow scoring). Each is a POST to `https://api.github.com/repos/AddisonTech/Stoa/actions/workflows/<daily.yml|close.yml|analyst.yml>/dispatches` with the body `{"ref":"main","inputs":{"job":"sweep","source":"cron-job.org"}}` (no `job` for close and analyst) and the headers `Authorization: Bearer <token>`, `Accept: application/vnd.github+json` and `X-GitHub-Api-Version: 2022-11-28`. GitHub answers 204. A failed call emails the account's address. Proven on 2026-10-26: planned 2:21:00 pm, called 2:21:01, run created 2:21:04.
- **The Dell is the first backup**, at the same minute. `scripts\install-dispatch.ps1` registers five Task Scheduler tasks with those times (they wake the Dell from sleep and catch up if it was off); each runs `scripts\dispatch.ps1 -Job <job>`, which starts the workflow with the GitHub CLI (`gh`, logged in as AddisonTech), names the run "via dell" and logs to `data\dispatch.log`. Run the installer again after changing a time.
- **GitHub cron is the last backup**: SWEEP at 9:05 am, CLOSE at 9:41 am, EXECUTE at 9:50 am, DECIDE at 5:20 pm, the analyst at 7:50 pm (6:50 pm in winter).
- **They can never double-run.** Every daily run is in the `stoa-state` concurrency group, so runs go one at a time, and each job records its session on the state branch: a second start for the same job logs "already swept" (or decided, or executed) and sends nothing. Close and analyst have their own groups and the same once-a-session rule. Taps still close at 9:40 am whichever start sent the picks.
- **A missed trigger is reported.** Each run's name says what started it ("sweep via cron-job.org", "execute via dell", "schedule 5 13 * * 1-5"). At the end of each scheduled job, `stoa triggers check` looks at the slot before it (sweep checks the previous evening's analyst, close checks the sweep, and so on). If no run started by cron-job.org was created within 5 minutes of that slot, it sends one alert, "Trigger missed: 8:57 am sweep Mon Nov 2", saying which trigger did start it. Priority 4 when a backup covered it; priority 5 when nothing started it within 5 minutes. It reads only, never fails the job, and checks the alert topic first so the same miss is reported once.

If every clock misses the morning, start the sweep by hand before 9:40 (Actions, daily, Run workflow, **sweep**).

**Pausing.** cron-job.org has no single switch for all jobs, and pausing it alone would leave the Dell and GitHub cron running. The one place that stops every clock is GitHub: disable the workflows, and every start from any clock is ignored until they are enabled again. From the Dell:

```
gh workflow disable daily.yml -R AddisonTech/Stoa; gh workflow disable close.yml -R AddisonTech/Stoa; gh workflow disable analyst.yml -R AddisonTech/Stoa
gh workflow enable daily.yml -R AddisonTech/Stoa; gh workflow enable close.yml -R AddisonTech/Stoa; gh workflow enable analyst.yml -R AddisonTech/Stoa
```

Or in the browser: Actions, each of daily, close and analyst, the three dots, **Disable workflow**. To stop only cron-job.org (say, to test the backups), switch each of the five jobs off in its console; the others keep running.

**The GitHub token expires on 2027-10-26.** It is a fine-grained token named `stoa-cron-job-org`, for AddisonTech/Stoa only, with Actions read and write and nothing else (GitHub adds Metadata read-only). It lives only in cron-job.org, in each job's Authorization header. When it expires, every cron-job.org call fails: cron-job.org emails you, the next job's trigger check alerts, and the Dell and GitHub cron keep the jobs running meanwhile. To rotate, before it expires:

1. GitHub, Settings, Developer settings, Fine-grained tokens: open `stoa-cron-job-org` and click **Regenerate token** (or make a new one with the same settings), one year.
2. In cron-job.org, open each of the five jobs, Advanced, Headers, and replace the value of `Authorization` with `Bearer ` and the new token. Save.
3. Run one job now (the job's menu, **Test run**) on **close**, which does nothing outside its window, and check that it answered 204 and that a "close via cron-job.org" run appeared in Actions.

The cron-job.org API key (Settings, API) was used only to create the jobs. Delete it there if it is not needed; the jobs keep running without it.

## 13a. The live positions panel (Vercel)

The dashboard's **Live** card shows the paper account's equity, today's change and every open position (quantity, price, market value, today's change and total profit or loss), refreshed every 30 seconds while the tab is open and the market is open, with "Market closed" outside regular hours. GitHub Pages is static and public, so the keys cannot live there. A small read-only function on Vercel's free tier holds them instead:

- **Code:** `live/api/positions.py`, standard library only. It makes exactly two kinds of request, both GET, to the paper API: `/v2/account` and `/v2/positions`. There is no order, cancel or other write call anywhere in it; a test checks the source.
- **Paper only:** it refuses (HTTP 503) if `ALPACA_LIVE_KEY_ID`, `ALPACA_LIVE_SECRET` or `STOA_MODE=live` is set in Vercel, or if the key is not a paper key (paper keys start with `PK`).
- **Rate limit:** answers are cached 25 seconds, in memory and at Vercel's edge (`s-maxage=25`), so however many tabs poll, Alpaca sees at most two requests every 25 seconds, far from the 200 a minute the pulls and the daily jobs share.
- **CORS:** only `https://addisontech.github.io` may read it from a browser.
- **If it is down:** the card says "Live data unavailable right now" and the rest of the dashboard, built from committed state, loads as usual.
- **Where:** the Vercel project is `stoa-live` and the address is `https://stoa-live.vercel.app/api/positions`, set as `dashboard.live_url` in `config/default.yaml`; blank hides the card.

**Deploying** happens on its own: the stoa-live project is connected to this repository, with its Root Directory set to `live` (Vercel, stoa-live, Settings, Build and Deployment). A push to main that changes anything under `live/` builds and deploys to production. Every other push is skipped by the ignored build step (`live/ignore-build.sh`, named in `live/vercel.json`) and shows in Vercel as Canceled, not as an error. A redeploy of the commit already live always builds. The step compares live/ with the last deployed commit. Vercel clones only the last few commits, so from 2026-10-27, once main was more than about ten commits past the last deployment, that commit was missing from the clone, the step stopped with `fatal: bad object`, and every push failed with an error email. Since 2026-10-28 the step fetches the missing commit first, and builds if it still cannot tell (checked by a test in a real shallow clone). Only main deploys: preview deployments are turned off for the project (`previewDeploymentsDisabled`, set 2026-10-26; in the dashboard, Settings, Git), so commits to stoa-state and any other branch create no deployment and send no failure email. Before that, every stoa-state commit started a preview that failed with 'Root Directory "live" does not exist', since that branch has no live/ folder; `vercel.json` cannot fix it because Vercel reads it from the missing folder. To turn previews back on: `npx vercel@65.2.0 api /v9/projects/stoa-live --method PATCH` with the body `{"previewDeploymentsDisabled": false}`.

`live/vercel.json` also sets `"framework": null`. Without it Vercel treats the folder as a Python web app and fails with "No python entrypoint found". A test checks both settings.

To redeploy by hand (from the repository root, with Node installed), for example after changing a key:

```
npx vercel@65.2.0 login                       # once, in a browser
npx vercel@65.2.0 link --yes --project stoa-live --cwd live   # once per machine
npx vercel@65.2.0 redeploy stoa-live.vercel.app --target production --cwd live
```

Do not use `vercel deploy --cwd live`: with the Root Directory set it fails with 'Root Directory "live" does not exist'. Deploying from the repository root would upload the whole repository, data folder included.

**Rotating the paper keys means updating them in Vercel too**, or the card goes blank. Environment variables only reach new builds, so redeploy afterwards. In the dashboard: stoa-live, Settings, Environment Variables, edit `ALPACA_PAPER_KEY_ID` and `ALPACA_PAPER_SECRET` (Production), then Deployments, the top production deployment, Redeploy. Or from the command line: `npx vercel@65.2.0 env rm ALPACA_PAPER_KEY_ID production --cwd live`, then `npx vercel@65.2.0 env add ALPACA_PAPER_KEY_ID production --cwd live` (it asks for the value), the same for the secret, then the redeploy command above. Never add the live keys there.

## 13b. The big paper account (the plumbing; trial 14 as it ran)

Since 2026-10-29 the account runs trial 15 (section 13c). The keys, the feed, the dataset, the Windows runners, the dry runs, the kill switch and the dashboard below are unchanged and serve trial 15 too. What follows is how trial 14 ran.

A second, separate Alpaca paper account of about $100,000 runs trial 14 (spec `f385bc9d5040ef7b`, `big_paper.strategy: trial14`, `big_paper.idle_money: cash`): trial 12 exactly as registered (spec `e75fa8775469de92`), the news brain's picks, its sizing and its rails, with no overlay, but with idle money held as cash instead of SPY, so the account's result shows the picks alone. SPY stays the line it is measured against. From 2026-10-28 to 2026-10-29 it ran trial 12 with its SPY core; the first 9:45 am run after the switch sells that SPY alongside the day's picks, and SPY is never bought again while `idle_money` is `cash`. Each trial keeps its own idle money: the config refuses `trial12` or `trial13` with `cash`, and `trial14` with `spy`. The $100 account above keeps the v1 rules, its keys and its state, and nothing here changes it (since 2026-10-28 its picks also go through without a tap, section 6). Trial 12 (`trial12` with `spy`) and trial 13's overlay (`trial13` with `spy`, `stoa/overlay.py`) are still there and run only if the settings are changed.

**Sizing on $100,000.** Up to 5 picks at a time, each 20% of the account ($20,000 at the start; the fifth gets $18,000 because 2% always stays in cash). Every other dollar stays in cash: with no pick held, all of it; with one pick, $20,000 in it and about $80,000 cash. Cash earns nothing here, so on a day with no pick the account does not move while SPY does. Each pick is sold at 9:45 am the next session. In trial 12's test window (875 sessions) 85% of sessions held no pick at all, 92 sessions held 1, 19 held 2, 11 held 3, 5 held 4 and 2 held 5. Trial 12's backtest ran on $100, but all 458 of its picks were stocks Alpaca sells in fractions, so the bigger account buys the same picks at the same weights.

**Kept apart.** Its own keys, `ALPACA_BIG_PAPER_KEY_ID` and `ALPACA_BIG_PAPER_SECRET`, are read only by `stoa/bigpaper.py` and passed only to `.github/workflows/bigpaper.yml`; the daily workflow never sees them, and the big account refuses to run if they are the $100 account's keys. The feed pulls its news and bars on these keys too. Its state is `big-paper/` on the stoa-state branch, its workflow has its own concurrency group, and every alert it sends starts with **BIG PAPER**.

**The day,** on the same clock as the $100 account:

- **8:15 am feed** (`stoa bigpaper feed`). News since the last pull from Alpaca, scored by FinBERT on the CPU at the pinned revision (`ProsusAI/finbert` at `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`, the one that scored the history); earnings 8-Ks and 10-Ks of the universe's companies from EDGAR; the daily bars of the close before for every stock, with the 25 sessions before it. Everything waits in the feed's `pending/` folder.
- **9:00 am picks.** First the same pulls again for whatever arrived since 8:15, then the news brain's rolling state moves to the session with exactly what was known by 9:00 am, as the model was trained. Up to 10 picks at p of at least 0.55 go through trial 12's rails (5 positions at 20%, the regime filter, the daily loss stop, sector and correlation limits, the earnings rule). Paper with hundreds of trades to come, so picks are auto-approved: the alert lists what will be bought and sold, with no buttons. If anything is stale or a step failed, there are no buys that day, yesterday's picks are still sold, the account sits in cash, and the alert says why (priority high). Any SPY still held is in the plan as a sale (reason `idle_cash`).
- **9:45 am execute.** Yesterday's picks are sold (and any SPY left), today's bought, and idle money stays in cash. The alert lists the fills.
- **5:15 pm summary.** The trial and where idle money sits, equity, the day against SPY, the drawdown from the high, and the kill switch. The 9:00 am picks alert starts with the same label ("Trial 14, idle money in cash").

**Kill switch.** At 25% off the high, the next 9:45 am run cancels every working order and sells every position in the account: the picks, SPY, and anything bought by hand. It stays on until reset by hand: Actions, bigpaper, Run workflow, job `kill-switch-reset`.

**The news brain's rolling state** (`stoa/newsbrain/rolling.py`, `stoa/newsbrain/feed.py`). Trial 9's 13 features read history: a stock's last 60 days with news (years back for a quiet stock), headlines of the 3 days before each article, article counts over 60 sessions, earnings 8-Ks of 5 sessions, 10-K supplier links good for 315 sessions, and 20 sessions of daily bars, with pandas' running sums for the volatility and dollar volume carried from 2015. The state holds exactly that: every scored headline (trial 9's own news code runs on it each morning, about a minute), recent filings, the live links, and per stock the last 21 closes and pandas' own running sums. A split or dividend changes a stock's past prices; the feed sees it in the 25 sessions it re-reads and rebuilds that stock's sums from its full history. Checked against trial 9's full rebuild from 2015: every row and all 13 features bit for bit (results below). The state refuses to run on a pandas other than the one it was built with. The feed's pinned versions are in the workflow.

After the seed, nothing in the big account's day needs the Dell.

**Turning it on.** It trades only when all of these are done. Steps 1 to 4 were done on 2026-10-28, after the dry runs below matched the Dell exactly.

1. In the Alpaca dashboard, create a new paper account with about $100,000 and generate its API keys. It must be a different paper account from the $100 one.
2. Add two repository secrets (Settings, Secrets and variables, Actions): `ALPACA_BIG_PAPER_KEY_ID` and `ALPACA_BIG_PAPER_SECRET`.
3. Push the seed once (below), and add the repository secret `HF_TOKEN`: a Hugging Face token with write access to `ASmith91/stoa-feed`.
4. Set `big_paper.enabled: true` in `config/default.yaml` and push. To stop it, set it back to `false` and push.
5. Add four cron-job.org jobs that start `bigpaper.yml`, weekdays, America/New_York, each a POST to `https://api.github.com/repos/AddisonTech/Stoa/actions/workflows/bigpaper.yml/dispatches` with the same three headers as the $100 account's jobs (section 13, the same `stoa-cron-job-org` token) and the body `{"ref":"main","inputs":{"job":"feed","source":"cron-job.org"}}`: `feed` at 8:15 am, `picks` at 9:00 am, `execute` at 9:45 am, `summary` at 5:15 pm (change `job` in the body). GitHub answers 204. The GitHub crons in the workflow are only the late backup.

**The dry runs, 2026-10-28.** A replay of the mornings of 2026-10-26, 10-27 and 10-28 from the seed, on Windows, rebuilt on the Dell from its own history plus exactly what the feed pulled: 1,140, 1,288 and 1,391 stocks, every one of the 49,647 feature values and every score equal, no picks on either side (the best score was 0.50; a buy needs 0.55). The same replay on Linux had given the same picks but 261 to 410 values a morning off in the last digit, which is why the feed and the picks run on Windows. Times on Windows: the 8:15 am feed 564 to 638 s for a morning's pulls (about 2,700 articles) plus about 3 minutes of setup, so done by about 8:30; the 9:00 am picks 85 to 183 s plus setup, done by about 9:05. Then the live state was reset to the seed (job `feed-reset`) and caught up by one dry feed on Windows, so it holds no number computed on Linux.

**Where the feed's state lives.** In the private Hugging Face dataset `ASmith91/stoa-feed` (`big_paper.feed_repo`), never on the stoa-state branch or in this repository: the seed is 85 MB (the headline archive is most of it) and the bar sums, about 6 MB, change every morning, which is too big for the stoa-state branch that every Stoa workflow checks out. The 8:15 am feed downloads the folder, updates it and uploads it (only changed files travel); the 9:00 am picks job downloads it and moves its own copy to the session. The runner keeps it in a temporary folder that is never committed. `HF_TOKEN` reaches only the workflow step that runs Stoa, which passes it to each Hugging Face call and never saves it or logs in; both calls refuse a dataset that is not private. If the token is missing or the download fails, the picks make no buys and the alert says why.

**The seed, once, from the Dell.** `python -m stoa.newsbrain feedseed 2026-10-26` builds it into `state/big-paper-feed` (done 2026-10-28). Then `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\push-feed-seed.ps1` asks for a write token without showing it and uploads the folder; the token lives only in that one process. The first feed run after that catches up from where the Dell's data ends (news at 2026-10-25 17:44 UTC, SEC filings from 2026-10-24, bars after 2026-10-23), skipping anything the seed already holds.

**Why the feed and the picks run on Windows.** Windows and Linux round `log` and `log1p` differently in the last of about 16 digits. Two features (`coverage`, `quiet_spike`) are logs of article counts, and many of the model's cut points sit exactly on the Dell's values, so on Linux a stock can land on the other side of a cut and its score moves by up to 0.015. A Linux dry run on 2026-10-28 showed it: same picks, but 5 to 23 stocks a session scored differently. So the feed and the picks (and replays) run on `windows-latest` with Python 3.12.10, like the Dell, and with numpy's AVX-512 paths switched off, because the Dell's processor has none. EXECUTE and SUMMARY do no feature maths and stay on Linux.

**Dry runs and replays.** Actions, bigpaper, Run workflow:

- job `feed` or `picks` with **dry** ticked: the real morning's pulls and picks, with no orders, nothing saved to the account and alerts only logged. A dry feed still moves the live state on, as a real feed would. Each dry run keeps what it pulled under `raw/<time>-<job>/` in the dataset, and a dry picks run its rows and picks under `checks/<time>-picks/`.
- job `execute` with **dry** ticked (on Windows, like the picks): makes the morning's plan as the picks would, then runs it against a copy of the account in the simulator at the latest prices, through the same broker wrapper and rails as the real 9:45 am run. No order is sent and nothing is saved. The log shows every order it would place and how much SPY it would sell and buy.
- **replay** with steps such as `picks@2026-10-26T09:00 feed@2026-10-27T08:15 picks@2026-10-27T09:00` (New York times): each step runs dry on a fresh folder, as if at that time. The replay starts from the dataset as of **replay_from** (by default the seed's commit, `seed 2026-10-26`) and keeps its own copy between steps, so the live state never moves. Only the raw pulls and checks are uploaded.

To check them on the Dell, `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\pull-feed.ps1` (a read token, never saved) downloads the dataset into `data\newsbrain\feedcheck\hf`, and the Dell rebuilds the same sessions from its own history plus exactly those pulls and compares every number.

**Dashboard.** The public page has a "Big paper account (trial 14)" card with its equity against SPY and its trade log, rebuilt after every bigpaper run, and a "Shadow scoreboard" card (below).

**The shadow scoreboard** (`stoa/shadowboard.py`). It grades every stock the brain scores each morning, buys or not, so the model is judged on a thousand stocks a day instead of a handful of trades.

- **Lock, before the open.** The 9:00 am picks job writes the full list it scored (symbol, p, whether p cleared 0.55, the model version, the dollar volume the cost model needs) to the private dataset under `scoreboard/locked/<session>/`, with the UTC time it was written and the file's sha256 in a `meta.json` beside it. The same hash is appended to `big-paper/scoreboard/locks.jsonl`, which the workflow commits to the stoa-state branch. Nothing is written at or after 9:30 am, and a session already locked is never replaced. A failure here never stops the picks: it is logged and that morning goes ungraded.
- **Grade, the session after.** The 5:15 pm summary grades every locked morning whose exit price is in: in at the 9:45 am price of the session, out at the 9:45 am price of the next (the open of the first minute bar from 9:45 to 10:15, adjusted, SIP feed: the trades' and the trials' own rule), against SPY over the same span. A hit is the model's own label: the stock beat SPY after a round trip of slippage and the fee. So Thursday's lock is graded Friday evening. The grader refuses a file whose bytes do not match its hash or the hash on the state branch, or that was written or uploaded (the dataset server's own commit time) at or after 9:30 am of its session; a refused morning is listed, never counted.
- **What it records.** Per morning and as a running total since day 1: stocks scored, stocks above the line, the hit rate above the line, calibration by bucket (under 0.50, 0.50-0.55, 0.55-0.60, 0.60-0.65, 0.65+: average p against the share that actually beat SPY), the average return of the above-line group against SPY, and the top 10 by p against SPY. The totals are in `big-paper/scoreboard/scoreboard.json` (aggregates only; the per-stock files stay in the private dataset).
- **The summary line.** The 5:15 pm alert ends with "Shadow: X scored, Y above line, above-line avg vs SPY +/-Z% (date)" for the latest locked morning graded, then one line per shadow model: "Shadow <version>: X scored, Y above line, above-line avg vs SPY +/-Z%, top 10 vs SPY +/-W% (date)".
- **Shadow models, side by side** (`big_paper.shadow_models`, version: sha256). A model that never trades is scored on the same stocks the same morning and graded the same way, in its own column. Today: `t12-all-20261027`, the trial 12 retrain-all model, beside the frozen picks model. The picks step is unchanged. The next step of the same run ("Shadow models") reads the rows the picks just scored from the feed folder they left. It checks that the picks' model gives those rows the picks' locked p to the last bit (or locks nothing). It then downloads `models/<version>/` from the private dataset, refuses it unless its sha256 is the pinned one, scores the rows and locks the list under `scoreboard/shadow/<version>/locked/<session>/`, with the same 9:30 am deadline, hash, meta.json and line in `locks.jsonl`. That step may fail without failing the run, and it runs before "Save state", so its hash reaches the state branch. The grader checks a shadow file exactly as it checks the picks' file, and also that its meta.json names the folder's model. The card has one column per model, each with its own totals, calibration and backfill. To add or drop a shadow model, edit `shadow_models` and push; nothing else changes.
- **Backfill.** Mornings scored after the fact (`scoreboard/backfill/`) carry no lock. They are graded the same way and kept in their own totals, never mixed with locked mornings, in the column of the model that scored them. The first backfill is the retrained model's last five sessions, which are inside its own training window, so they flatter it.
- **By hand.** Actions, bigpaper, Run workflow, job `scoreboard`: grades what is ready, with no alert.

**The model and rolling it back.** `big_paper.model` names the model the picks use. Blank is trial 9's frozen model, kept in the feed state (`static/model_h1.pkl`); it is blank today. The retrained `t12-all-20261027` was not swapped in: its calibrated p never goes above 0.520, so it could never buy (docs/DECISIONS.md, "Trial 12 retrain-all"). It is a shadow model instead (above). A version (say `t12-all-20261027`) reads `models/<version>/model_h1.pkl` from the private dataset, and the picks refuse it unless its sha256 equals `big_paper.model_sha256`, so a changed file cannot slip in. To roll back, set `model` back to blank (and `model_sha256` to blank) and push: the next 9:00 am picks use the frozen model again. A new version goes in only after three checks: a dry picks run on the Windows runner with that version (Run workflow, job `picks`, dry ticked, `model` set), `model-check` (it downloads the version and its backfilled mornings, scores the Dell's rows again on the Windows runner and requires the Dell's p to the last bit and picks without errors), and the feature match of section "The dry runs" above.

**Retraining** (`stoa/newsbrain/retrain.py`, docs/DECISIONS.md "Trial 12 retrain-all"). On the Dell: `python -m stoa.newsbrain retrain <last session>` refits trial 9's model, same 13 features, same 1 session label, same fit and calibration split, on every session labelled by the open of `<last session>`, and saves `data\newsbrain\models\<version>\` with a `meta.json` (train window, rows, features, git sha, the pickle's sha256). `python -m stoa.newsbrain backfill <version> <sessions...>` scores past sessions for the scoreboard. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\push-model.ps1 <version>` uploads both (a write token, never saved).

## 13c. Trial 15: the owner's brain, 5 picks a day

Registered 2026-10-29 (spec `557c95176a137379`, docs/DECISIONS.md "Trial 15"), `big_paper.strategy: trial15`, `big_paper.idle_money: cash`.

**The day.**

- **8:15 am feed.** As in 13b, plus the local event reader (`big_paper.reader: true`): Qwen2.5-1.5B-Instruct on the runner's CPU labels each new article (event type, good or bad for the company, how big), for up to 20 minutes; whatever it does not reach waits in a backlog for the next run. Labels go to `events/` in the private dataset, beside the rolling state, never inside it. A reader failure is logged and never stops the feed. On the dry feed of 2026-10-29 it labelled 448 items, nothing left over, and the whole feed took 9 minutes.
- **9:00 am picks.** The brain ranks every stock it scores; the top 5 operating companies by conviction (funds and SPY never) are the picks. Yesterday's 5 are sold and today's 5 bought at (equity x 0.98) / 5 each; a stock picked again is resized, not sold and bought back. The reader gets 4 more minutes on what arrived since 8:15. Claude then reads the day's news behind the top 20 (the existing analyst, inside the one monthly cap); its reading is listed in the alert and never changes the picks. The plan also keeps the morning's rows for the nightly learner and locks the learner's own top 5 before 9:30 am.
- **9:45 am execute.** Sells, then buys, with the kill switch and the price sanity check (an order is refused if its price moved more than 8% from the 9:00 am price). No SPY, no sector cap, no daily loss stop, no regime filter.
- **5:15 pm summary.** Equity against SPY, today's 5 from 9:45 am to the close (not final), the session before's 5 graded 9:45 am to 9:45 am after costs with that day's hit rate, why each loser lost, the learner's shadow record and the running hit rate against the 75% target. Then the nightly learner refits (the "Nightly learning" step, never fails the run).

**Files on the state branch** (`big-paper/trial15/`): `picks/<session>.json` (what each pick saw), `days.jsonl` (grades), `reviews/<session>.json`, `lessons.json` (reasons counted, Claude's live record), `rule_checks.json` (the rule menu's decade checks), `claude/<session>.json`, `learner.json`, `shadow/<session>.json` (locked with a sha256), `shadow_days.jsonl`, and `promoted.json` once a learner takes over. `shared/trial15/claude_ledger.json` is the Claude reader's spend, counted in the one cap.

**Owner's one-time step for the nightly learner.** It needs the decade's rows in the private dataset. On the Dell: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\push-learn-base.ps1` (asks for a write token without showing it; uploads `data\newsbrain\trial15\learn_base.parquet`, 1,633,249 rows, as `learn/base.parquet`). Until then each evening's learner says "waiting_for_base" and nothing changes.

**Rolling back.** A learner that took over is in `big-paper/trial15/promoted.json`; delete that file on the state branch and the picks go back to `big_paper.model` (blank: trial 9's frozen brain). To turn off the reader set `big_paper.reader: false`. To stop the account, `stop.big_paper: true` (section 11a).

**History tools (Dell).** `python -m stoa.newsbrain events` (the reader over every month, GPU, resumable, newest first; progress in `data\newsbrain\events\progress.jsonl`), `python -c "from stoa.newsbrain import ripplemap; ripplemap.main()"` (the 10-K ripple map), `python -c "from stoa.newsbrain import catchup; catchup.main()"` (the pre-market catch-up table), `python -c "from stoa.newsbrain import compare15; compare15.main()"` (the registered comparison, run once).

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
4. To watch a locked phone at a known time, run the **ntfy-test** workflow with a New York time (HH:MM, within 30 minutes). At that time it sends the standard test alert (priority 3), then a plain priority 4 alert 60 seconds later. Its log shows both send times.
5. Research jobs on the Dell (trial 9 and 10) alert through the **notify** workflow: `gh workflow run notify.yml -R AddisonTech/Stoa -f title=... -f message=... -f priority=4`. It sends one alert to the same topic and nothing else.

## Sample run

This is part 3 of `python scripts/demo.py`: DECIDE at 5:15 pm, the 9:00 am sweep and EXECUTE at 9:45 am against the simulated broker on synthetic data, with every alert printed instead of sent. The night is quiet (no news), so the sweep leaves the rules' picks alone, and on paper nothing waits for a tap. The prices are made up.

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

[p3] Picks 2021-10-13
    Picks for 2021-10-13. They go in on their own at 9:45 am; nothing to tap.
    BUY NVDA $20.00 (20% of account) at 9:45 am. Stop $120.27. Why: Momentum score 0.432: 60 day +45.7%, 120 day +39.5%, above its 50 and 200 day averages.
    No overnight news.
    BUY HD $20.00 (20% of account) at 9:45 am. Stop $1,152.07. Why: Momentum score 0.271: 60 day +10.3%, 120 day +52.2%, above its 50 and 200 day averages.
    No overnight news.
    BUY UNH $20.00 (20% of account) at 9:45 am. Stop $1,050.99. Why: Momentum score 0.253: 60 day +19.6%, 120 day +34.0%, above its 50 and 200 day averages.
    No overnight news.
    BUY AMZN $20.00 (20% of account) at 9:45 am. Stop $375.18. Why: Momentum score 0.250: 60 day +23.1%, 120 day +27.9%, above its 50 and 200 day averages.
    No overnight news.

EXECUTE 2021-10-13 13:45 UTC -> executed

[p4] Filled buy NVDA
    BUY NVDA: 0.154398 sh at $129.54 = $20.00 (paper).
[p4] Filled buy HD
    BUY HD: 0.016104 sh at $1,241.91 = $20.00 (paper).
[p4] Filled buy UNH
    BUY UNH: 0.017574 sh at $1,138.04 = $20.00 (paper).
[p4] Filled buy AMZN
    BUY AMZN: 0.050041 sh at $399.67 = $20.00 (paper).

Sim broker cash 20.00, positions:
    AMZN 0.050041081 sh at 399.6716
    HD 0.016104260 sh at 1241.9074
    NVDA 0.154398287 sh at 129.5351
    UNH 0.017573999 sh at 1138.0448
```
