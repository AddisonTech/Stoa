# Stoa

AI trading agents that tell you what to buy and when, graded against just holding SPY.

Stoa runs once a day. A team of roles proposes trades, a rule-based risk engine has the final word, and every decision is logged with its reasoning and scored against two baselines. It has to prove itself in backtest, on a sealed holdout and in paper trading before it ever touches real money.

## How it works

1. After the close, Stoa pulls split and dividend adjusted daily bars for a 29 symbol watchlist.
2. It checks open positions against their stops, trailing stops, time limit and trend.
3. A scout (Claude, or a plain momentum rule) suggests a few names to buy, with a short reason.
4. Quant gates kill ideas the numbers do not back, and the risk engine sizes what is left.
5. The plan goes to your phone. After the next open, Stoa places notional market orders.
6. Every step is journaled, and every run is graded against SPY and a 50/200 day rule.

The model is allowed to have ideas. It never sizes a position, skips a limit or touches money.

## Quickstart

```bash
git clone https://github.com/AddisonTech/Stoa.git
cd Stoa
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                   # offline, about a minute
python scripts/demo.py                   # synthetic end to end demo, no keys needed
stoa backtest --strategy agents --window tune_a --synthetic
```

With an Alpaca paper account (free):

```bash
export ALPACA_PAPER_KEY_ID=...  ALPACA_PAPER_SECRET=...
stoa data fetch                          # daily bars from 2016 into data/bars
stoa score                               # all strategies, all windows, graded
stoa status
```

To run it every day from GitHub Actions and get alerts on your phone, follow [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Modes

| Mode | What it does |
| --- | --- |
| backtest | Replays history with the same decision, risk and execution code as live. |
| signal | Sends the plan to your phone. Never touches a broker. Tracks a paper portfolio of its own. |
| paper | Trades an Alpaca paper account. This is the default. |
| live | Trades real money. Locked unless three separate switches are all on. |

## Safety

- Gross exposure never goes above 100% of equity. Stoa never borrows, shorts or trades options.
- At most 5 positions, 20% each, with 2% always kept in cash.
- No new buys after a 3% down day, or while SPY is below its 200 day average.
- A 25% drawdown from the high-water mark trips the kill switch: sell everything, buy nothing until a person resets it.
- A symbol is never bought and sold on the same day.
- Live trading needs `mode: live`, `STOA_LIVE_CONFIRM=I_UNDERSTAND` and separate live keys. Any one missing and Stoa refuses.
- Silence means something broke: a summary is sent after every close, even with no trades.

Details in [docs/RISK.md](docs/RISK.md).

## Scoreboard status

Not passed yet. The pass bar is in [docs/SCOREBOARD.md](docs/SCOREBOARD.md) and has not been run on real data in this repository. The sample report in that file is on synthetic data and only shows the format. The holdout (July 2026 onward) is not sealed. See [docs/HOLDOUT.md](docs/HOLDOUT.md).

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) the design
- [docs/RUNBOOK.md](docs/RUNBOOK.md) setup and daily operation
- [docs/RISK.md](docs/RISK.md) every limit and how it is enforced
- [docs/SCOREBOARD.md](docs/SCOREBOARD.md) how runs are graded
- [docs/HOLDOUT.md](docs/HOLDOUT.md) the one-shot holdout
- [docs/DECISIONS.md](docs/DECISIONS.md) choices made while building

## License

MIT. See [LICENSE](LICENSE). Not investment advice.
