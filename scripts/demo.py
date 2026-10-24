"""Offline end-to-end demo on seeded synthetic data. No network, no keys.

1. Backtest agents, SPY buy-and-hold and the 50/200 rule on every window.
2. Grade them against config/scoreboard.yaml and write the report.
3. Run DECIDE then EXECUTE against the simulated broker and print every alert.

    python scripts/demo.py [output-dir]
"""

from __future__ import annotations

import sys
import tempfile
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd

from stoa import alerts, jobs, scoreboard
from stoa.broker.sim import SimBroker
from stoa.config import Secrets, load_settings
from stoa.data.base import Bars
from stoa.data.synthetic import synthetic_bars
from stoa.features import cell
from stoa.models import dec
from stoa.portfolio import Portfolio
from stoa.universe import Universe

DECIDE_AT = datetime(2021, 10, 12, 22, 20, tzinfo=UTC)
EXECUTE_AT = datetime(2021, 10, 13, 13, 50, tzinfo=UTC)


class FixedProvider:
    name = "synthetic"

    def __init__(self, bars: Bars) -> None:
        self.bars = bars

    def fetch(self, symbols: list[str], start: date, end: date) -> Bars:
        return {s: self.bars[s].loc[pd.Timestamp(start) : pd.Timestamp(end)] for s in symbols}


def prices(bars: Bars, day: date, column: str) -> dict[str, Decimal]:
    return {s: dec(cell(df, pd.Timestamp(day), column)) for s, df in bars.items()}


def print_alerts(title: str, rec: alerts.RecordingNotifier) -> None:
    print(f"\n### {title}\n")
    for a in rec.alerts:
        print(f"[p{a.priority}] {a.title}")
        for line in a.message.splitlines():
            print(f"    {line}")


def main(out: Path) -> int:
    settings = load_settings(Path("config/default.yaml"), env={})
    secrets = Secrets.from_env({})

    print("## Scoreboard on synthetic data\n")
    report = scoreboard.score_all(settings, secrets, synthetic=True)
    path = scoreboard.write_report(report, out)
    print(scoreboard.render_markdown(report))
    print(f"(written to {path})")

    with tempfile.TemporaryDirectory() as tmp:
        s = load_settings(
            Path("config/default.yaml"),
            env={
                "STOA_STATE_DIR": tmp,
                "STOA__SCOUT__KIND": "rule",
                "STOA__STRATEGY__KIND": "rule",
                "STOA__CROSSCHECK__ENABLED": "false",
                "STOA__DATA__PROVIDER": "csv",
            },
        )
        bars = synthetic_bars(list(s.watchlist), date(2016, 1, 4), date(2021, 12, 31), seed=11)
        universe = Universe.assume_all_fractionable(s.watchlist, s.benchmark)
        broker = SimBroker(Portfolio(cash=Decimal(100)), s.costs, universe)
        broker.begin_session(DECIDE_AT.date(), prices(bars, DECIDE_AT.date(), "close"))
        env = Secrets.from_env({"ALPACA_PAPER_KEY_ID": "demo", "ALPACA_PAPER_SECRET": "demo"})

        rec = alerts.RecordingNotifier()
        rt = jobs.Runtime(
            s, env, rec, now=lambda: DECIDE_AT, provider=FixedProvider(bars), broker=broker
        )
        decided = jobs.decide(rt)
        print_alerts(f"DECIDE {DECIDE_AT:%Y-%m-%d %H:%M} UTC -> {decided.status}", rec)

        broker.begin_session(EXECUTE_AT.date(), prices(bars, EXECUTE_AT.date(), "open"))
        rec2 = alerts.RecordingNotifier()
        rt2 = jobs.Runtime(
            s, env, rec2, now=lambda: EXECUTE_AT, provider=FixedProvider(bars), broker=broker
        )
        executed = jobs.execute(rt2)
        print_alerts(f"EXECUTE {EXECUTE_AT:%Y-%m-%d %H:%M} UTC -> {executed.status}", rec2)

        print(f"\nSim broker cash {broker.portfolio.cash:.2f}, positions:")
        for sym, lot in sorted(broker.portfolio.positions.items()):
            print(f"    {sym} {lot.qty:.9f} sh at {lot.avg_cost:.4f}")
        return max(decided.code, executed.code)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    sys.exit(main(target))
