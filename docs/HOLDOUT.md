# Holdout

The holdout is every session from 2026-07-01 onward. It is scored exactly once, after everything else is frozen.

## Why July 2026

Claude Haiku 5.5 has a reliable knowledge cutoff of June 2026. A model may remember how a stock moved before then, even with the ticker hidden. From July 2026 on, it cannot. Tickers and dates are anonymized in prompts too, but the holdout is the real guard.

## Seal

```bash
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

It runs the agents, SPY and the 50/200 rule over the holdout, applies the pass bar, and writes the result into `HOLDOUT.lock`. Commit the lock afterwards so the result is public and permanent.

## The rule

One shot. If the holdout fails, the right move is to write down what was learned, not to tweak and score again. A new holdout would need a new, later start date and a new seal. The code enforces the one shot; honesty covers the rest.
