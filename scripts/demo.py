"""Offline end-to-end demo on seeded synthetic data. No network, no keys.

1. Walk-forward train the model on a synthetic universe (with dead names and a
   small planted momentum edge), print the calibration report, log the trials.
2. Backtest the model, the v1 rule pipeline, SPY, the 50/200 rule and MTUM on
   every window and grade on risk, with the deflated Sharpe.
3. Run DECIDE, the 9:00 am sweep, then EXECUTE against the simulated broker,
   and print every alert (the morning picks are one FYI; on paper nothing waits for a tap).

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
from stoa.experiments import Registry
from stoa.features import cell
from stoa.models import dec
from stoa.pipeline import load_inputs, walkforward_model_source
from stoa.portfolio import Portfolio
from stoa.retrain import train
from stoa.universe import Universe

DECIDE_AT = datetime(2021, 10, 12, 21, 15, tzinfo=UTC)  # 5:15 pm New York
SWEEP_AT = datetime(2021, 10, 13, 13, 0, tzinfo=UTC)  # 9:00 am
EXECUTE_AT = datetime(2021, 10, 13, 13, 45, tzinfo=UTC)  # 9:45 am
SIGNAL = "1.0"  # a small planted edge: AUC about 0.57, like a good real factor


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
        for act in a.actions:
            print(f"    [button] {act['label']}")


class QuietNight:
    """No news overnight, so the sweep leaves the rules' picks as they are."""

    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        from stoa.data.news import NEWS_COLUMNS

        return pd.DataFrame(columns=list(NEWS_COLUMNS))


def main(out: Path) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        env = {"STOA_STATE_DIR": f"{tmp}/state", "STOA__DEMO__SIGNAL": SIGNAL}
        settings = load_settings(Path("config/default.yaml"), env=env)
        secrets = Secrets.from_env({})

        print("## 1. Walk-forward training (synthetic universe)\n")
        cutoff = date(2026, 6, 30)
        inputs = load_inputs(settings, cutoff, synthetic=True)
        outcome = train(settings, inputs, cutoff=cutoff, results_dir=out, promote=True)
        print((outcome.report_dir / "calibration.md").read_text())
        print(
            f"Model {outcome.version}: {'promoted' if outcome.promoted else 'not promoted'}"
            f" ({outcome.gate_reason})."
        )

        print("\n## 2. Scoreboard on synthetic data, graded on risk\n")
        end = max(w.end for w in settings.backtest.windows.values())
        source, _ = walkforward_model_source(settings, inputs, end)
        report = scoreboard.score_all(
            settings,
            secrets,
            synthetic=True,
            model=source,
            registry=Registry(settings.registry_path),
        )
        path = scoreboard.write_report(report, out)
        print(scoreboard.render_markdown(report))
        print(f"(written to {path})")

        print("\n## 3. A day of operation: summary, morning picks, orders at the open\n")
        s = load_settings(
            Path("config/default.yaml"),
            env={
                "STOA_STATE_DIR": f"{tmp}/live",
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
        keys = Secrets.from_env({"ALPACA_PAPER_KEY_ID": "demo", "ALPACA_PAPER_SECRET": "demo"})

        rec = alerts.RecordingNotifier()
        rt = jobs.Runtime(
            s,
            keys,
            rec,
            now=lambda: DECIDE_AT,
            provider=FixedProvider(bars),
            broker=broker,
            use_external=False,
            sleep=lambda _s: None,
        )
        decided = jobs.decide(rt)
        print_alerts(f"DECIDE {DECIDE_AT:%Y-%m-%d %H:%M} UTC -> {decided.status}", rec)

        rec_s = alerts.RecordingNotifier()
        rt_s = jobs.Runtime(
            s,
            keys,
            rec_s,
            now=lambda: SWEEP_AT,
            provider=FixedProvider(bars),
            broker=broker,
            use_external=False,
            sleep=lambda _s: None,
            news_client=QuietNight(),
        )
        swept = jobs.sweep_job(rt_s)
        print_alerts(f"SWEEP {SWEEP_AT:%Y-%m-%d %H:%M} UTC -> {swept.status}", rec_s)

        broker.begin_session(EXECUTE_AT.date(), prices(bars, EXECUTE_AT.date(), "open"))
        rec2 = alerts.RecordingNotifier()
        rt2 = jobs.Runtime(
            s,
            keys,
            rec2,
            now=lambda: EXECUTE_AT,
            provider=FixedProvider(bars),
            broker=broker,
            use_external=False,
            sleep=lambda _s: None,
        )
        executed = jobs.execute(rt2)
        print_alerts(f"EXECUTE {EXECUTE_AT:%Y-%m-%d %H:%M} UTC -> {executed.status}", rec2)

        print(f"\nSim broker cash {broker.portfolio.cash:.2f}, positions:")
        for sym, lot in sorted(broker.portfolio.positions.items()):
            print(f"    {sym} {lot.qty:.9f} sh at {lot.avg_cost:.4f}")
        return max(decided.code, swept.code, executed.code)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    sys.exit(main(target))
