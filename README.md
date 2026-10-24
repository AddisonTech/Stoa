# Stoa

A daily equity trading system that has to prove it is better than just holding SPY, on risk, before it touches real money.

**Dashboard:** [addisontech.github.io/Stoa](https://addisontech.github.io/Stoa/), read only, rebuilt after every daily run.

Stoa runs once a day. A plain statistical model, trained the honest way, estimates the chance each stock beats SPY over the next two weeks. A rule-based risk engine sizes by confidence and volatility and has the final word. Claude is optional and only reads news into scores. Every decision is logged, and every version ever tried counts against the result.

## How it works

1. After the close, Stoa pulls split and dividend adjusted bars and checks open positions against their stops, the time limit and the next earnings date.
2. The model ranks the watchlist by its calibrated probability of beating SPY over 10 sessions.
3. Quant gates, the earnings rule, the regime filter, the 40% sector cap and a correlation check cut the list.
4. Sizing: bigger when the model is sure and the stock is calm, halved on FOMC and CPI days, always under the caps.
5. The plan goes to your phone with Approve and Skip buttons (required for the first live month). After the open, Stoa checks the price against a second source and places the approved orders.
6. Then it places a stop order at the broker for every position held from a prior day, at the stop the risk engine already uses, so a crash during the day is not left for the next close. A position bought that morning gets none: nothing is ever bought and sold the same day.
7. A dead man's switch on healthchecks.io alerts your phone if the daily summary never arrives.

## Quickstart

```bash
git clone https://github.com/AddisonTech/Stoa.git
cd Stoa
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                   # offline, about two minutes
python scripts/demo.py                   # synthetic end to end demo, no keys needed
stoa model train --synthetic             # walk-forward and the calibration report
```

With an Alpaca paper account (free):

```bash
export ALPACA_PAPER_KEY_ID=...  ALPACA_PAPER_SECRET=...
stoa data fetch && stoa universe build   # bars, including companies that left the index
stoa fundamentals fetch                  # SEC filings and earnings times
stoa model train                         # walk-forward, report, promote
stoa score                               # every strategy, graded on risk
stoa exam status
```

To run it every day from GitHub Actions, follow [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Exam order

1. Walk-forward results must pass (calibrated, better than the base rate, deflated Sharpe above 0.95).
2. Then the sealed holdout, scored once.
3. Then at least two months of paper trading with 20 trades, graded after simulated costs.
4. Only then does live unlock, on top of the three live switches.

If it cannot beat SPY on risk, or the momentum fund MTUM, the report says so. If live drifts below what walk-forward promised, or six months pass with no edge, it holds SPY instead.

## Safety

- Gross exposure never above 100% of equity. No borrowing, shorting or options.
- At most 5 positions, 20% each, 40% per sector, 2% always in cash.
- Never holds through an earnings report.
- Every position held overnight has a stop order at the broker during the day. Stops only move up.
- A 25% drawdown trips the kill switch. A person resets it.
- Live needs `mode: live`, `STOA_LIVE_CONFIRM=I_UNDERSTAND`, separate live keys and a passed exam.

Details in [docs/RISK.md](docs/RISK.md).

## Scoreboard status

Not passed. The bar has not been run on real market data in this repository. Sample reports in the docs are synthetic and only show the format. The holdout is not sealed.

## Docs

- [docs/TRAINING.md](docs/TRAINING.md) the model, walk-forward, calibration, the trial penalty
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) the design
- [docs/RUNBOOK.md](docs/RUNBOOK.md) setup and daily operation
- [docs/RISK.md](docs/RISK.md) every limit and how it is enforced
- [docs/SCOREBOARD.md](docs/SCOREBOARD.md) how runs are graded
- [docs/HOLDOUT.md](docs/HOLDOUT.md) the one-shot holdout
- [docs/DECISIONS.md](docs/DECISIONS.md) choices made while building

## License

MIT. See [LICENSE](LICENSE). Not investment advice.
