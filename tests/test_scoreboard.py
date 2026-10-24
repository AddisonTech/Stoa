from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stoa import scoreboard
from stoa.config import Secrets, Settings
from stoa.experiments import Registry
from stoa.metrics import Metrics
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def report(settings: Settings, tmp_path_factory: pytest.TempPathFactory) -> scoreboard.Report:
    reg = Registry(tmp_path_factory.mktemp("reg") / "registry.jsonl")
    return scoreboard.score_all(
        settings,
        Secrets.from_env({}),
        synthetic=True,
        now=datetime(2026, 10, 1, tzinfo=UTC),
        board=scoreboard.load_scoreboard(ROOT / "config" / "scoreboard.yaml"),
        registry=reg,
    )


def test_score_all_runs_every_window_and_strategy(report: scoreboard.Report) -> None:
    pairs = {(r.window, r.strategy) for r in report.runs}
    assert pairs == {
        (w, s)
        for w in ("tune_a", "tune_b", "tune_c", "validation")
        for s in ("agents", "spy", "sma", "mtum")
    }
    # No model source given: the bar falls back to grading the v1 pipeline.
    assert report.graded == "agents"
    names = {c.name for c in report.checks}
    assert names == {
        "drawdown_vs_spy",
        "sharpe_vs_spy",
        "sortino_vs_spy",
        "deflated_sharpe",
        "drawdown_cap",
        "tune_windows",
        "trade_count",
        "api_cost",
    }
    assert report.run_id.endswith("-synthetic")
    assert report.deflated["trials"] >= 1 and 0 <= report.deflated["dsr"] <= 1
    assert any("MTUM" in v for v in report.verdicts)


def test_report_files(report: scoreboard.Report, tmp_path: Path) -> None:
    out = scoreboard.write_report(report, tmp_path)
    md = (out / "report.md").read_text()
    data = json.loads((out / "report.json").read_text())
    assert "| Check | Result | Detail |" in md and ("PASS" in md or "FAIL" in md)
    assert "Deflated Sharpe" in md and "MTUM" in md
    assert data["passed"] == report.passed
    assert len(data["runs"]) == 16


def _metrics(
    total: float,
    mdd: float = 0.1,
    sharpe: float = 1.5,
    sortino: float = 2.0,
    trades: int = 20,
) -> Metrics:
    return Metrics(
        start="2025-01-01",
        end="2025-12-31",
        sessions=250,
        starting_equity=100,
        final_equity=100 * (1 + total),
        total_return=total,
        cagr=total,
        ann_vol=0.1,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        max_drawdown_sessions=10,
        calmar=1,
        profit_factor=1.5,
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


SPY = _metrics(0.10, 0.20, sharpe=1.0, sortino=1.3)
SMA = _metrics(0.05, 0.10, sharpe=0.8, sortino=1.0)
MTUM = _metrics(0.12, 0.22, sharpe=1.1, sortino=1.4)


def synthetic_report(
    *,
    mdd: float = 0.1,
    sharpe: float = 1.5,
    sortino: float = 2.0,
    trades: int = 20,
    cost: float = 0.5,
    tune_beats: int = 3,
    dsr: float = 0.99,
    mtum: Metrics = MTUM,
    board: scoreboard.ScoreboardConfig | None = None,
) -> scoreboard.Report:
    r = scoreboard.Report("x", datetime(2026, 1, 1, tzinfo=UTC), "test", "h", graded="model")
    for i, w in enumerate(("tune_a", "tune_b", "tune_c")):
        sh = 1.5 if i < tune_beats else 0.5
        r.runs += [
            _run("model", w, _metrics(0.1, mdd, sh, sortino, trades), cost),
            _run("spy", w, SPY),
            _run("sma", w, SMA),
            _run("mtum", w, mtum),
        ]
    r.runs += [
        _run("model", "validation", _metrics(0.1, mdd, sharpe, sortino, trades), cost),
        _run("spy", "validation", SPY),
        _run("sma", "validation", SMA),
        _run("mtum", "validation", mtum),
    ]
    r.deflated = {"dsr": dsr, "trials": 12}
    b = board or scoreboard.ScoreboardConfig()
    r.checks = scoreboard.evaluate(r, b)
    r.verdicts = scoreboard.verdicts(r, b)
    return r


def test_pass_bar_all_good() -> None:
    r = synthetic_report()
    assert r.passed
    assert any("beats MTUM" in v for v in r.verdicts)


@pytest.mark.parametrize(
    ("kwargs", "failing"),
    [
        ({"mdd": 0.20}, "drawdown_vs_spy"),  # equal to SPY is not lower
        ({"sharpe": 1.0}, "sharpe_vs_spy"),
        ({"sortino": 1.3}, "sortino_vs_spy"),
        ({"dsr": 0.80}, "deflated_sharpe"),
        ({"trades": 5}, "trade_count"),
        ({"cost": 25.0}, "api_cost"),
        ({"tune_beats": 1}, "tune_windows"),
    ],
)
def test_each_criterion_can_fail(kwargs: dict[str, float], failing: str) -> None:
    r = synthetic_report(**kwargs)  # type: ignore[arg-type]
    assert not r.passed
    assert {c.name for c in r.checks if not c.passed} == {failing}


def test_drawdown_cap_independent_of_spy() -> None:
    r = synthetic_report(mdd=0.26)
    failed = {c.name for c in r.checks if not c.passed}
    assert "drawdown_cap" in failed


def test_risk_not_return_decides() -> None:
    """Lower total return than SPY still passes when risk-adjusted results are better."""
    r = synthetic_report()
    assert r.get("model", "validation").metrics.total_return <= SPY.total_return
    assert r.passed


def test_mtum_verdict_recommends_the_fund_when_not_beaten() -> None:
    strong = _metrics(0.3, 0.05, sharpe=2.5, sortino=3.5)
    r = synthetic_report(mtum=strong)
    assert r.passed  # MTUM is reported, not a hard check by default
    assert any("buy the momentum fund (MTUM)" in v for v in r.verdicts)
    hard = scoreboard.ScoreboardConfig(criteria=scoreboard.Criteria(beat_mtum_for_pass=True))
    r2 = synthetic_report(mtum=strong, board=hard)
    assert not r2.passed and {c.name for c in r2.checks if not c.passed} == {"beats_mtum"}


def test_versus_50_200_reported() -> None:
    r = synthetic_report()
    assert any("50/200" in v for v in r.verdicts)
