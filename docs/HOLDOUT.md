# Holdout

The holdout is every session from 2026-07-01 to 2026-10-23 (`backtest.holdout_start` and `backtest.holdout_end`), about 80 sessions. It is scored exactly once, after everything else is frozen.

## A fixed end

The end was fixed on 2026-10-24, before the holdout was sealed, so the sessions after it can be used for forward tests (trial 7, see DECISIONS.md). Until the holdout is scored:

- Nothing inside it trains, tests or scores a model. Every training dataset drops the rows dated 2026-07-01 to 2026-10-23 and every earlier row whose label would reach into them (`holdout.blocked_window`, applied in `training.dataset_for`; `pipeline.load_inputs` blocks it by default).
- Before 2026-10-23 training still stops the session before the start. After it, later sessions are usable again.
- Features of sessions after the end look back over the window (a year of momentum, for example), as every live decision does. No label, test row or score comes from it.
- `stoa holdout score` is the one command that reads it, and it stops at the end.

A test wrecks every price inside a stand-in window and checks that no training label and no earlier row changes.

## Why July 2026

Claude Haiku 5.5 has a reliable knowledge cutoff of June 2026. A model may remember how a stock moved before then, even with the ticker hidden. From July 2026 on, it cannot. Tickers and dates are anonymized in prompts too, but the holdout is the real guard.

## Exam order

The holdout is step 2 of 4. `stoa holdout seal` refuses until `stoa model train` has run and its walk-forward passed: there is no point spending the one-shot holdout on a model that already failed. Until the holdout is scored, training never uses a session on or after its start date.

## Seal

```bash
stoa model train          # walk-forward must pass, and the model is promoted
git status                # must be clean
stoa holdout seal
git add HOLDOUT.lock && git commit -m "Seal the holdout"
```

`HOLDOUT.lock` records:

- the git commit
- a hash of every tracked file under `stoa/`
- a hash of `config/default.yaml`
- a hash of `config/scoreboard.yaml`
- a hash of the effective strategy settings (so an environment override cannot sneak a change in)
- the promoted model's version (scoring uses exactly that model, and refuses if a different one is live)
- the holdout start date and the time of sealing

Sealing refuses a dirty tree and refuses if a lock already exists.

## Score

The holdout data is complete (it ended 2026-10-23). Fetch fresh data, then:

```bash
stoa data fetch
stoa holdout score            # or --llm for the Claude scout
```

Scoring refuses if:

- the holdout is not sealed
- it was already scored (the score is written into the lock)
- the working tree has any change other than the lock file
- the code under `stoa/`, either config file, or the effective strategy settings differ from the seal
- the promoted model is not the one recorded at the seal

It runs the sealed model (each holdout session predicted on its own data, no retraining), the v1 agents, SPY, the 50/200 rule and MTUM over the holdout, applies the risk bar (lower drawdown, higher Sharpe and Sortino than SPY, the drawdown cap, the API cost cap), and writes the result into `HOLDOUT.lock`, with the MTUM verdict. Commit the lock afterwards so the result is public and permanent.

## The rule

One shot. If the holdout fails, the right move is to write down what was learned, not to tweak and score again. A new holdout would need a new, later window and a new seal. The code enforces the one shot; honesty covers the rest.
