"""Scores runs against the pass bar in config/scoreboard.yaml.

Every strategy runs on the same bars, windows, costs and calendar. The agents
pass only if every criterion holds. The report is written as markdown and JSON
to results/<run-id>/.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from stoa import metrics
from stoa.backtest import BacktestResult, run_backtest
from stoa.config import DEFAULT_SCOREBOARD_PATH, Secrets, Settings
from stoa.data.base import Bars
from stoa.data.cache import load_cached_bars
from stoa.data.synthetic import synthetic_bars
from stoa.strategy import build_strategy

STRATEGIES = ("agents", "spy", "sma")
SYNTHETIC_START = date(2016, 1, 4)


class Criteria(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    beat_spy: bool = True
    beat_sma: bool = True
    max_drawdown_vs_spy: bool = True
    max_drawdown_cap: float = 0.25
    min_profit_factor: float = 1.2
    min_tune_windows_beating_spy: int = 2
    min_trades: int = 30
    max_monthly_api_cost_usd: float = 5.0


class ScoreboardConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_window: str = "validation"
    tune_windows: tuple[str, ...] = ("tune_a", "tune_b", "tune_c")
    criteria: Criteria = Criteria()


def load_scoreboard(path: Path = DEFAULT_SCOREBOARD_PATH) -> ScoreboardConfig:
    return ScoreboardConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


@dataclass(frozen=True)
class Run:
    strategy: str
    window: str
    metrics: metrics.Metrics
    monthly: dict[str, dict[str, float]]
    bootstrap_mdd: dict[str, float]
    max_monthly_api_cost: float
    llm_fallbacks: int
    kill_switch_trips: int
    kills_by_gate: dict[str, int]
    rejections_by_reason: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "window": self.window,
            "metrics": self.metrics.to_dict(),
            "monthly_returns": self.monthly,
            "bootstrap_max_drawdown": self.bootstrap_mdd,
            "max_monthly_api_cost": self.max_monthly_api_cost,
            "llm_fallbacks": self.llm_fallbacks,
            "kill_switch_trips": self.kill_switch_trips,
            "kills_by_gate": self.kills_by_gate,
            "rejections_by_reason": self.rejections_by_reason,
        }


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Report:
    run_id: str
    created: datetime
    data_source: str
    strategy_hash: str
    runs: list[Run] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def get(self, strategy: str, window: str) -> Run:
        for r in self.runs:
            if r.strategy == strategy and r.window == window:
                return r
        raise KeyError((strategy, window))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created": self.created.isoformat(),
            "data_source": self.data_source,
            "strategy_hash": self.strategy_hash,
            "passed": self.passed,
            "checks": [c.__dict__ for c in self.checks],
            "runs": [r.to_dict() for r in self.runs],
        }


def load_bars(settings: Settings, end: date, *, synthetic: bool) -> Bars:
    """Cached real bars, or seeded synthetic bars for offline demos and tests."""
    symbols = list(settings.watchlist)
    if synthetic:
        return synthetic_bars(symbols, SYNTHETIC_START, end, seed=settings.metrics.seed)
    start = settings.data.history_start
    return load_cached_bars(settings.data.cache_dir, symbols, start, end)


def summarize(result: BacktestResult, settings: Settings, window: str) -> Run:
    m = metrics.compute(
        result.equity, result.trades, result.fills, result.exposure, settings.metrics.risk_free_rate
    )
    table = metrics.monthly_returns(result.equity.iloc[1:]) if len(result.equity) > 2 else None
    monthly = (
        {
            str(y): {str(k): float(v) for k, v in row.items() if not math.isnan(v)}
            for y, row in table.iterrows()
        }
        if table is not None
        else {}
    )
    boot = metrics.bootstrap_max_drawdown(
        metrics.daily_returns(result.equity),
        settings.metrics.bootstrap_samples,
        settings.metrics.bootstrap_block,
        settings.metrics.seed,
    )
    return Run(
        strategy=result.strategy,
        window=window,
        metrics=m,
        monthly=monthly,
        bootstrap_mdd=boot,
        max_monthly_api_cost=max(result.llm_cost_by_month.values(), default=0.0),
        llm_fallbacks=result.llm_fallbacks,
        kill_switch_trips=result.kill_switch_trips,
        kills_by_gate=result.kills_by_gate,
        rejections_by_reason=result.rejections_by_reason,
    )


def run_one(
    settings: Settings,
    secrets: Secrets,
    bars: Bars,
    strategy: str,
    window: str,
    start: date,
    end: date,
    *,
    llm: bool = False,
) -> Run:
    strat = build_strategy(settings, secrets, strategy, llm=llm)
    return summarize(run_backtest(settings, bars, strat, start, end), settings, window)


def evaluate(report: Report, board: ScoreboardConfig) -> list[Check]:
    c = board.criteria
    ev = board.evaluation_window
    agents, spy, sma = (report.get(s, ev).metrics for s in STRATEGIES)
    checks: list[Check] = []
    if c.beat_spy:
        checks.append(
            Check(
                "beats_spy",
                agents.total_return > spy.total_return,
                f"agents {agents.total_return:+.2%} vs SPY {spy.total_return:+.2%} ({ev})",
            )
        )
    if c.beat_sma:
        checks.append(
            Check(
                "beats_sma",
                agents.total_return > sma.total_return,
                f"agents {agents.total_return:+.2%} vs 50/200 {sma.total_return:+.2%} ({ev})",
            )
        )
    if c.max_drawdown_vs_spy:
        checks.append(
            Check(
                "drawdown_vs_spy",
                agents.max_drawdown <= spy.max_drawdown,
                f"agents {agents.max_drawdown:.2%} vs SPY {spy.max_drawdown:.2%} ({ev})",
            )
        )
    worst = max(r.metrics.max_drawdown for r in report.runs if r.strategy == "agents")
    checks.append(
        Check(
            "drawdown_cap",
            worst <= c.max_drawdown_cap,
            f"worst agents drawdown {worst:.2%} (cap {c.max_drawdown_cap:.0%})",
        )
    )
    checks.append(
        Check(
            "profit_factor",
            agents.profit_factor >= c.min_profit_factor,
            f"{agents.profit_factor:.2f} (need {c.min_profit_factor}) ({ev})",
        )
    )
    beats = [
        w
        for w in board.tune_windows
        if report.get("agents", w).metrics.total_return > report.get("spy", w).metrics.total_return
    ]
    checks.append(
        Check(
            "tune_windows",
            len(beats) >= c.min_tune_windows_beating_spy,
            f"beat SPY in {len(beats)} of {len(board.tune_windows)} ({', '.join(beats) or 'none'})",
        )
    )
    trades = sum(r.metrics.trade_count for r in report.runs if r.strategy == "agents")
    checks.append(
        Check(
            "trade_count",
            trades >= c.min_trades,
            f"{trades} trades across all windows (need {c.min_trades})",
        )
    )
    cost = max((r.max_monthly_api_cost for r in report.runs if r.strategy == "agents"), default=0.0)
    checks.append(
        Check(
            "api_cost",
            cost < c.max_monthly_api_cost_usd,
            f"max ${cost:.2f} in any month (cap ${c.max_monthly_api_cost_usd:.2f})",
        )
    )
    return checks


def score_all(
    settings: Settings,
    secrets: Secrets,
    *,
    llm: bool = False,
    synthetic: bool = False,
    board: ScoreboardConfig | None = None,
    now: datetime | None = None,
) -> Report:
    board = board or load_scoreboard()
    windows = [*board.tune_windows, board.evaluation_window]
    missing = [w for w in windows if w not in settings.backtest.windows]
    if missing:
        raise ValueError(f"scoreboard names unknown windows {missing}")
    last_end = max(settings.backtest.windows[w].end for w in windows)
    bars = load_bars(settings, last_end, synthetic=synthetic)
    created = now or datetime.now(UTC)
    report = Report(
        run_id=created.strftime("%Y%m%d-%H%M%S")
        + ("-synthetic" if synthetic else "")
        + ("-llm" if llm else ""),
        created=created,
        data_source="synthetic" if synthetic else f"cache:{settings.data.cache_dir}",
        strategy_hash=settings.strategy_hash(),
    )
    for w in windows:
        win = settings.backtest.windows[w]
        for strat in STRATEGIES:
            report.runs.append(
                run_one(
                    settings,
                    secrets,
                    bars,
                    strat,
                    w,
                    win.start,
                    win.end,
                    llm=llm and strat == "agents",
                )
            )
    report.checks = evaluate(report, board)
    return report


def _pct(x: float) -> str:
    return f"{x:+.2%}"


def render_markdown(report: Report) -> str:
    lines = [
        f"# Stoa scoreboard {report.run_id}",
        "",
        f"Data: {report.data_source}. Strategy hash: `{report.strategy_hash[:12]}`.",
        "",
        f"**Result: {'PASS' if report.passed else 'FAIL'}**",
        "",
        "| Check | Result | Detail |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {c.name} | {'pass' if c.passed else 'FAIL'} | {c.detail} |" for c in report.checks
    ]
    lines += [
        "",
        "| Window | Strategy | Return | CAGR | Max DD | Sharpe | Sortino | PF | Trades | Exp. |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.runs:
        m = r.metrics
        pf = "inf" if math.isinf(m.profit_factor) else f"{m.profit_factor:.2f}"
        lines.append(
            f"| {r.window} | {r.strategy} | {_pct(m.total_return)} | {_pct(m.cagr)} "
            f"| {m.max_drawdown:.2%} | {m.sharpe:.2f} "
            f"| {m.sortino:.2f} | {pf} | {m.trade_count} | {m.exposure:.0%} |"
        )
    lines += ["", "Bootstrap max drawdown (agents, 5th / 50th / 95th percentile):", ""]
    for r in report.runs:
        if r.strategy == "agents":
            b = r.bootstrap_mdd
            lines.append(f"- {r.window}: {b['p05']:.1%} / {b['p50']:.1%} / {b['p95']:.1%}")
    return "\n".join(lines) + "\n"


def write_report(report: Report, root: Path) -> Path:
    out = root / report.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (out / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    return out


def holdout_window(settings: Settings, end: date) -> tuple[date, date]:
    start = settings.backtest.holdout_start
    if end <= start + timedelta(days=30):
        raise ValueError("the holdout needs more than a month of data after its start")
    return start, end
