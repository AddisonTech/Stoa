# Runbook

How to set Stoa up, run it every day, and what to do when it talks to you. No coding needed for the daily part. Setup takes about 30 minutes on a laptop, with your phone nearby.

Stoa starts in **paper** mode: fake money in an Alpaca paper account. Leave it there for months before thinking about live.

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
| `ANTHROPIC_API_KEY` | a Claude API key | optional; without it the rule scout runs |
| `STOA_NTFY_TOKEN` | an ntfy access token | optional; only for a protected topic or server |

Leave `ALPACA_LIVE_KEY_ID`, `ALPACA_LIVE_SECRET` and `STOA_LIVE_CONFIRM` empty until the go-live checklist.

3. Under **Variables** (same page, other tab), you can add `STOA_MODE` with `paper` or `signal`. If you leave it out, Stoa uses paper. Add `STOA_NTFY_SERVER` only if you run your own ntfy server.

## 4. Turn on the workflow

1. Open the **Actions** tab. If GitHub asks, click **I understand my workflows, go ahead and enable them**.
2. Click the **daily** workflow on the left. If it says it is disabled, click **Enable workflow**.
3. Click **Run workflow**, pick **alert-test**, and click the green **Run workflow** button.
4. Within a couple of minutes your phone should show "Stoa test alert". If not, see "No alerts at all" below.

From now on it runs by itself on weekdays. You can watch it from the GitHub app on your phone too (Actions tab).

## 5. What happens each day

All times are New York time.

- **About 5:20 pm**: DECIDE. Stoa reads the day's bars, checks stops, asks the scout for ideas, runs the risk engine, and sends you:
  - one alert per planned trade (priority 4), and
  - the daily summary (priority 3), every trading day, even with no trades.
- **About 9:50 am**: EXECUTE. Stoa places last night's orders and sends one alert per fill.

GitHub runs each job twice an hour apart (because it ignores daylight saving time). The second run sees the first already did the work and does nothing. Weekends and market holidays are skipped. On early close days (like the day after Thanksgiving) DECIDE runs earlier on its own.

## 6. Reading alerts

**Signal** (priority 4)
> BUY NVDA $19.40 (19% of account) at the open. Stop $91.25. Why: ...

In paper and live mode you do nothing; Stoa places it at the open. In signal mode, place it yourself if you want to follow along. The stop is where Stoa would sell on a close.

> SELL KO all 0.12 sh at the open. Why: stop loss ...

Stoa sells the whole position. Reasons: stop loss, trailing stop, time exit (30 sessions), trend break (two closes under the 50 day average), kill switch.

**Filled** (priority 4): the order went through, with shares and price.

**Daily summary** (priority 3): equity and today's change, return since start against SPY, what is held, cash, what happens at the next open, Claude API spend this month, realized short-term gains this year and how much to set aside for tax.

**Skipped** (priority 2): a risk check stopped an order. Usually the price jumped more than 8% overnight. Nothing to do.

## 7. When an alert says...

**"Stoa decide failed" or "Stoa execute failed" (priority 5)**
Something crashed. Nothing else happens until it is fixed. Tap the alert or open the Actions tab and read the red run's log. If it was a hiccup (GitHub or Alpaca down), the next slot usually fixes itself. To retry now: Actions, daily, Run workflow, pick the job, tick **force**.

**"Stoa workflow failed" (priority 5)**
Same as above, but the failure was outside Stoa (install, checkout, saving state). Open the log.

**"Stoa decide: bad data" (priority 5)**
The day's bars were missing or late. Stoa did not trade. It tries again at the next slot an hour later. If it keeps happening, check status.alpaca.markets.

**"Stoa decide refused to run" (priority 5)**
A guard said no. The message says which: missing keys, live mode locked, or the account is blocked. Fix what it names.

**"KILL SWITCH TRIPPED" (priority 5)**
Equity fell 25% below its best close. Stoa is selling everything at the next open and will not buy again until you reset it. Do not rush. See "Kill switch" below.

**"Order rejected" (priority 5 live, 4 paper)**
Alpaca refused an order. Open the Alpaca app and look at Orders. Common causes: account restricted, asset halted.

**"Order still working" (priority 4)**
An order did not finish in 90 seconds. Stoa checks it again at DECIDE. Nothing to do.

**"Scout fell back to rules" (priority 4)**
Claude was not used today (no key, refused, too long, or the monthly cost cap hit). The rule scout ran instead. If it says cost cap, it resets next month, or raise `llm.monthly_cost_cap_usd`.

**"Withdraw $100 now" (priority 4)**
The money plan milestone. See "Withdrawing" below.

**"Detected withdrawal" or "Detected deposit" (priority 4)**
Stoa saw money move in Alpaca and adjusted its high-water mark so it is not counted as a gain or loss.

**No summary by 7 pm on a trading day**
Silence means something broke. Check the Actions tab. If there is no run at all, see "Scheduled runs stopped".

## 8. Withdrawing (the money plan)

The plan: grow $100 to $200, then take the $100 out and let Stoa trade only the profit.

1. When equity reaches $200, you get "Withdraw $100 now". It is sent once.
2. Open the Alpaca app (live account, once you are live), go to **Banking** or **Transfers**, and withdraw $100 to your bank. API keys cannot move money, so this is always done by you.
3. If cash is short because money is in positions, wait for the next sell.
4. Tell Stoa it is done: Actions, daily, Run workflow, pick **money-confirm**. Leave equity blank to read it from Alpaca.
5. Stoa rebases its high-water mark to what is left, so the withdrawal does not count as a drawdown.

In signal mode, money-confirm takes the $100 out of Stoa's own paper portfolio instead.

## 9. Kill switch

When it trips, Stoa sells every watchlist position at the next open and stops buying.

Before resetting:

1. Read the last few daily summaries and the Actions logs. Was it the market, or did Stoa do something wrong?
2. If you are not sure, leave it off. Cash is a fine place to be.

To reset: Actions, daily, Run workflow, pick **kill-switch-reset**. Leave equity blank to read it from Alpaca. The high-water mark becomes today's equity, and buying resumes at the next DECIDE.

On a laptop: `stoa kill-switch status` and `stoa kill-switch reset`.

## 10. Go-live checklist

Do not go live until all of these are true.

- [ ] At least 3 months of paper trading with daily summaries arriving every trading day.
- [ ] `stoa score` passes on real data (see SCOREBOARD.md), and you have read the report.
- [ ] The holdout is sealed and scored once, and it passed (see HOLDOUT.md).
- [ ] You understand every alert type above.
- [ ] You have money you can afford to lose entirely. Start with $100.

Then:

1. Open and fund a live Alpaca account with $100. Generate **live** API keys (switch the account selector to Live first).
2. Add secrets `ALPACA_LIVE_KEY_ID` and `ALPACA_LIVE_SECRET`.
3. Add secret `STOA_LIVE_CONFIRM` with exactly `I_UNDERSTAND`.
4. Set the repository variable `STOA_MODE` to `live`.
5. Before the first live run, rename the `stoa-state` branch on GitHub to `stoa-state-paper` (Code, Branches, the pencil icon). The live account then starts with fresh state (its own high-water mark and money plan), and the paper history is kept.

If any of the three locks is missing, Stoa refuses and sends an urgent alert. To stop live trading at any time, delete `STOA_LIVE_CONFIRM` or set `STOA_MODE` back to `paper`.

## 11. Scheduled runs stopped

GitHub disables scheduled workflows in a public repository after 60 days with no activity, and it is not clear that Stoa's own state commits count. Once a month, make any small commit to the main branch, or open the Actions tab and re-enable the **daily** workflow if it shows a banner. If the repository is private this does not apply.

GitHub cron can also start late under load. Stoa's windows (DECIDE until midnight, EXECUTE for two hours after 9:45 am) leave plenty of slack.

## 12. Changing settings

Every limit is in `config/default.yaml`. Change it in a commit, or override one value with a repository variable passed as an environment variable named `STOA__SECTION__KEY` (for example `STOA__RISK__MAX_POSITIONS=4`). Changing settings after the holdout is sealed makes `stoa holdout score` refuse, on purpose.

## 13. Running on a laptop

```bash
pip install -e .
export ALPACA_PAPER_KEY_ID=... ALPACA_PAPER_SECRET=... STOA_NTFY_TOPIC=...
stoa status
stoa alert test
stoa decide --force      # outside the window, uses the last closed session
stoa money status
```

Local runs keep state in `state/`. The workflow keeps its own copy on the `stoa-state` branch. Do not run both against the same Alpaca account on the same day.

## No alerts at all

1. In the ntfy app, check the topic name matches the `STOA_NTFY_TOPIC` secret exactly.
2. Run the **alert-test** job again and open its log. "Sent." means ntfy accepted it; the problem is on the phone side (notifications blocked for the app).
3. If the log says "Stoa is not set up yet. Missing secrets", add the secrets it lists.

## Sample run

This is `python scripts/demo.py`: DECIDE and EXECUTE against the simulated broker on synthetic data, with every alert printed instead of sent. The prices are made up.

```text
DECIDE 2021-10-12 22:20 UTC -> decided

[p4] BUY NVDA
    BUY NVDA $20.00 (20% of account) at the open. Stop $120.27. Why: Momentum score 0.432: 60 day +45.7%, 120 day +39.5%, above its 50 and 200 day averages.
[p4] BUY HD
    BUY HD $20.00 (20% of account) at the open. Stop $1,152.07. Why: Momentum score 0.271: 60 day +10.3%, 120 day +52.2%, above its 50 and 200 day averages.
[p4] BUY UNH
    BUY UNH $20.00 (20% of account) at the open. Stop $1,050.99. Why: Momentum score 0.253: 60 day +19.6%, 120 day +34.0%, above its 50 and 200 day averages.
[p4] BUY AMZN
    BUY AMZN $20.00 (20% of account) at the open. Stop $375.18. Why: Momentum score 0.250: 60 day +23.1%, 120 day +27.9%, above its 50 and 200 day averages.
[p3] Stoa 2021-10-12 (paper)
    Equity $100.00
    Since start +0.0% vs SPY +0.0% (+0.0%)
    Holding: nothing
    Cash $100.00
    Next: BUY NVDA $20.00; BUY HD $20.00; BUY UNH $20.00; BUY AMZN $20.00
    API spend $0.00 of $5.00 this month
    Realized short-term gains this year $0.00, set aside $0.00 for tax

EXECUTE 2021-10-13 13:50 UTC -> executed

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
