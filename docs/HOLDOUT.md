# Holdout

The holdout is every session from 2026-07-01 onward. It is scored exactly once, after everything else is frozen.

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

Wait until there is enough holdout data (at least a month; more is better). Fetch fresh data, then:

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

One shot. If the holdout fails, the right move is to write down what was learned, not to tweak and score again. A new holdout would need a new, later start date and a new seal. The code enforces the one shot; honesty covers the rest.
