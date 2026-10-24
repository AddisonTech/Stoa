from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stoa import scoreboard
from stoa.config import Secrets, Settings
from stoa.metrics import Metrics
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def report(settings: Settings) -> scoreboard.Report:
    return scoreboard.score_all(
        settings,
        Secrets.from_env({}),
        synthetic=True,
        now=datetime(2026, 10, 1, tzinfo=UTC),
        board=scoreboard.load_scoreboard(ROOT / "config" / "scoreboard.yaml"),
    )


def test_score_all_runs_every_window_and_strategy(report: scoreboard.Report) -> None:
    pairs = {(r.window, r.strategy) for r in report.runs}
    assert pairs == {
        (w, s)
        for w in ("tune_a", "tune_b", "tune_c", "validation")
        for s in ("agents", "spy", "sma")
    }
    names = {c.name for c in report.checks}
    assert names == {
        "beats_spy",
        "beats_sma",
        "drawdown_vs_spy",
        "drawdown_cap",
        "profit_factor",
        "tune_windows",
        "trade_count",
        "api_cost",
    }
    assert report.run_id.endswith("-synthetic")


def test_report_files(report: scoreboard.Report, tmp_path: Path) -> None:
    out = scoreboard.write_report(report, tmp_path)
    md = (out / "report.md").read_text()
    data = json.loads((out / "report.json").read_text())
    assert "| Check | Result | Detail |" in md and ("PASS" in md or "FAIL" in md)
    assert data["passed"] == report.passed
    assert len(data["runs"]) == 12


def _metrics(total: float, mdd: float = 0.1, pf: float = 1.5, trades: int = 20) -> Metrics:
    return Metrics(
        start="2025-01-01",
        end="2025-12-31",
        sessions=250,
        starting_equity=100,
        final_equity=100 * (1 + total),
        total_return=total,
        cagr=total,
        ann_vol=0.1,
        sharpe=1,
        sortino=1,
        max_drawdown=mdd,
        max_drawdown_sessions=10,
        calmar=1,
        profit_factor=pf,
        win_rate=0.5,
        avg_win=1,
        avg_loss=-1,
        expectancy=0.1,
        exposure=0.5,
        turnover=1,
        trade_count=trades,
    )


def _run(strategy: str, window: str, m: Metrics, cost: float = 0.0) -> scoreboard.Run:
    return scoreboard.Run(strategy, window, m, {}, {}, cost, 0, 0, {}, {})


def synthetic_report(
    agents: float,
    spy: float,
    sma: float,
    *,
    mdd: float = 0.1,
    pf: float = 1.5,
    trades: int = 20,
    cost: float = 0.5,
    tune_beats: int = 3,
) -> scoreboard.Report:
    r = scoreboard.Report("x", datetime(2026, 1, 1, tzinfo=UTC), "test", "h")
    for i, w in enumerate(("tune_a", "tune_b", "tune_c")):
        a = 0.2 if i < tune_beats else 0.0
        r.runs += [
            _run("agents", w, _metrics(a, mdd, pf, trades), cost),
            _run("spy", w, _metrics(0.1, 0.2)),
            _run("sma", w, _metrics(0.05, 0.1)),
        ]
    r.runs += [
        _run("agents", "validation", _metrics(agents, mdd, pf, trades), cost),
        _run("spy", "validation", _metrics(spy, 0.2)),
        _run("sma", "validation", _metrics(sma, 0.1)),
    ]
    r.checks = scoreboard.evaluate(r, scoreboard.ScoreboardConfig())
    return r


def test_pass_bar_all_good() -> None:
    assert synthetic_report(0.15, 0.10, 0.05).passed


@pytest.mark.parametrize(
    ("kwargs", "failing"),
    [
        ({"agents": 0.05, "spy": 0.10, "sma": 0.0}, "beats_spy"),
        ({"agents": 0.15, "spy": 0.10, "sma": 0.20}, "beats_sma"),
        ({"agents": 0.15, "spy": 0.10, "sma": 0.05, "mdd": 0.21}, "drawdown_vs_spy"),
        ({"agents": 0.15, "spy": 0.10, "sma": 0.05, "pf": 1.1}, "profit_factor"),
        ({"agents": 0.15, "spy": 0.10, "sma": 0.05, "trades": 5}, "trade_count"),
        ({"agents": 0.15, "spy": 0.10, "sma": 0.05, "cost": 5.0}, "api_cost"),
        ({"agents": 0.15, "spy": 0.10, "sma": 0.05, "tune_beats": 1}, "tune_windows"),
    ],
)
def test_each_criterion_can_fail(kwargs: dict[str, float], failing: str) -> None:
    r = synthetic_report(**kwargs)  # type: ignore[arg-type]
    assert not r.passed
    assert {c.name for c in r.checks if not c.passed} == {failing}


def test_drawdown_cap_independent_of_spy() -> None:
    r = synthetic_report(0.15, 0.10, 0.05, mdd=0.26)
    failed = {c.name for c in r.checks if not c.passed}
    assert "drawdown_cap" in failed
