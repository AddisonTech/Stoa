"""Scores runs against the pass bar in config/scoreboard.yaml.

Every strategy runs on the same bars, windows, costs and calendar. The agents
pass only if every criterion holds. The report is written as markdown and JSON
to results/<run-id>/.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from stoa import metrics
from stoa.backtest import BacktestResult, run_backtest
from stoa.baselines import BuyAndHold
from stoa.broker.sim import Slippage
from stoa.config import DEFAULT_SCOREBOARD_PATH, Secrets, Settings
from stoa.data.base import Bars, DataError
from stoa.data.cache import load_cached_bars
from stoa.data.synthetic import synthetic_bars
from stoa.experiments import (
    Registry,
    deflate,
    deflated_sharpe,
    hash_frame_like,
    return_moments,
)
from stoa.models import ClosedTrade
from stoa.planner import Strategy
from stoa.roles.llm import Budget
from stoa.strategy import build_strategy
from stoa.universe import Universe


@dataclass
class ModelSource:
    """Everything the scoreboard needs to backtest the graded strategy (model or event).

    ``bars`` is what every strategy can trade. The event strategy trades
    thousands of names, so it brings ``window_bars`` (only the names with a
    signal near a window) and ``slippage`` (the liquidity-scaled cost model,
    applied to every strategy in the run).
    """

    strategy: Strategy
    bars: Bars
    universe: Universe | None = None
    name: str = "model"
    slippage: Slippage | None = None
    window_bars: Callable[[date, date], Bars] | None = None


STRATEGIES = ("agents", "spy", "sma")  # v1 set, still what the holdout scores
BASELINES = ("spy", "sma", "mtum")
SYNTHETIC_START = date(2016, 1, 4)


class Criteria(BaseModel):
    """The pass bar. Graded on risk: less drawdown and more return per unit of risk than SPY."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_drawdown_vs_spy: bool = True  # strictly lower max drawdown than SPY
    sharpe_vs_spy: bool = True  # strictly higher Sharpe than SPY
    sortino_vs_spy: bool = True  # strictly higher Sortino than SPY
    min_deflated_sharpe: float = 0.95  # DSR given every trial in the registry
    max_drawdown_cap: float = 0.25
    min_tune_windows_beating_spy: int = 2  # on Sharpe, in the tune windows
    min_trades: int = 30
    max_monthly_api_cost_usd: float = 25.0
    beat_mtum_for_pass: bool = False  # MTUM is always reported; this makes it a hard check
    sharpe_vs_mtum: bool = False  # trial 8: Sharpe above MTUM's on the evaluation window


class ScoreboardConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_window: str = "validation"
    tune_windows: tuple[str, ...] = ("tune_a", "tune_b", "tune_c")
    graded: str = "model"  # the strategy the pass bar applies to: model or agents
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
    daily_returns: tuple[float, ...] = ()
    trades: tuple[ClosedTrade, ...] = ()  # for per-signal diagnostics; not in the report

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
    graded: str = "agents"
    deflated: dict[str, float] = field(default_factory=dict)
    verdicts: list[str] = field(default_factory=list)
    calibration: dict[str, Any] = field(default_factory=dict)

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
            "graded": self.graded,
            "checks": [c.__dict__ for c in self.checks],
            "deflated": self.deflated,
            "verdicts": self.verdicts,
            "calibration": self.calibration,
            "runs": [r.to_dict() for r in self.runs],
        }

    def strategies(self) -> list[str]:
        seen: list[str] = []
        for r in self.runs:
            if r.strategy not in seen:
                seen.append(r.strategy)
        return seen

    def has(self, strategy: str, window: str) -> bool:
        return any(r.strategy == strategy and r.window == window for r in self.runs)


def load_bars(settings: Settings, end: date, *, synthetic: bool) -> Bars:
    """Cached real bars, or seeded synthetic bars for offline demos and tests.

    The watchlist is required. Reference symbols (MTUM) are loaded when cached
    and skipped (with that baseline missing from the report) when not.
    """
    symbols = list(settings.watchlist)
    refs = [r for r in settings.sources.reference_symbols if r not in symbols]
    if synthetic:
        return synthetic_bars([*symbols, *refs], SYNTHETIC_START, end, seed=settings.metrics.seed)
    start = settings.data.history_start
    bars = load_cached_bars(settings.data.cache_dir, symbols, start, end)
    for r in refs:
        try:
            bars.update(load_cached_bars(settings.data.cache_dir, [r], start, end))
        except DataError:
            continue
    return bars


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
        daily_returns=tuple(float(x) for x in metrics.daily_returns(result.equity)),
        trades=tuple(result.trades),
    )


def run_one(
    settings: Settings,
    secrets: Secrets,
    bars: Bars,
    strategy: str | Strategy,
    window: str,
    start: date,
    end: date,
    *,
    llm: bool = False,
    budget: Budget | None = None,
    universe: Universe | None = None,
    slippage: Slippage | None = None,
) -> Run:
    if isinstance(strategy, str):
        strat = build_strategy(settings, secrets, strategy, llm=llm, budget=budget)
    else:
        strat = strategy
    result = run_backtest(settings, bars, strat, start, end, universe=universe, slippage=slippage)
    return summarize(result, settings, window)


def evaluate(report: Report, board: ScoreboardConfig) -> list[Check]:
    """The pass bar for ``report.graded`` (risk first), plus the MTUM verdict."""
    c = board.criteria
    ev = board.evaluation_window
    g = report.graded
    me, spy = report.get(g, ev).metrics, report.get("spy", ev).metrics
    checks: list[Check] = []
    if c.max_drawdown_vs_spy:
        checks.append(
            Check(
                "drawdown_vs_spy",
                me.max_drawdown < spy.max_drawdown,
                f"{g} {me.max_drawdown:.2%} vs SPY {spy.max_drawdown:.2%} ({ev})",
            )
        )
    if c.sharpe_vs_spy:
        checks.append(
            Check(
                "sharpe_vs_spy",
                me.sharpe > spy.sharpe,
                f"{g} {me.sharpe:.2f} vs SPY {spy.sharpe:.2f} ({ev})",
            )
        )
    if c.sortino_vs_spy:
        checks.append(
            Check(
                "sortino_vs_spy",
                me.sortino > spy.sortino,
                f"{g} {me.sortino:.2f} vs SPY {spy.sortino:.2f} ({ev})",
            )
        )
    dsr = report.deflated.get("dsr", math.nan)
    trials = int(report.deflated.get("trials", 0))
    checks.append(
        Check(
            "deflated_sharpe",
            math.isfinite(dsr) and dsr >= c.min_deflated_sharpe,
            f"DSR {dsr:.3f} after {trials} recorded trials (need {c.min_deflated_sharpe})"
            if math.isfinite(dsr)
            else "no deflated Sharpe (no return series)",
        )
    )
    worst = max(r.metrics.max_drawdown for r in report.runs if r.strategy == g)
    checks.append(
        Check(
            "drawdown_cap",
            worst <= c.max_drawdown_cap,
            f"worst {g} drawdown {worst:.2%} (cap {c.max_drawdown_cap:.0%})",
        )
    )
    beats = [
        w
        for w in board.tune_windows
        if report.get(g, w).metrics.sharpe > report.get("spy", w).metrics.sharpe
    ]
    checks.append(
        Check(
            "tune_windows",
            len(beats) >= c.min_tune_windows_beating_spy,
            f"Sharpe above SPY in {len(beats)} of {len(board.tune_windows)} "
            f"({', '.join(beats) or 'none'})",
        )
    )
    trades = sum(r.metrics.trade_count for r in report.runs if r.strategy == g)
    checks.append(
        Check(
            "trade_count",
            trades >= c.min_trades,
            f"{trades} trades across all windows (need {c.min_trades})",
        )
    )
    cost = max((r.max_monthly_api_cost for r in report.runs if r.strategy == g), default=0.0)
    checks.append(
        Check(
            "api_cost",
            cost < c.max_monthly_api_cost_usd,
            f"max ${cost:.2f} in any month (cap ${c.max_monthly_api_cost_usd:.2f})",
        )
    )
    if c.beat_mtum_for_pass and report.has("mtum", ev):
        checks.append(mtum_check(report, g, ev))
    if c.sharpe_vs_mtum:
        if report.has("mtum", ev):
            mt = report.get("mtum", ev).metrics
            checks.append(
                Check(
                    "sharpe_vs_mtum",
                    me.sharpe > mt.sharpe,
                    f"{g} {me.sharpe:.2f} vs MTUM {mt.sharpe:.2f} ({ev})",
                )
            )
        else:
            checks.append(Check("sharpe_vs_mtum", False, "no MTUM bars to compare with"))
    return checks


def beats_on_risk(a: metrics.Metrics, b: metrics.Metrics) -> bool:
    return a.max_drawdown < b.max_drawdown and a.sharpe > b.sharpe and a.sortino > b.sortino


def mtum_check(report: Report, g: str, ev: str) -> Check:
    me, mt = report.get(g, ev).metrics, report.get("mtum", ev).metrics
    return Check(
        "beats_mtum",
        beats_on_risk(me, mt),
        f"{g} DD {me.max_drawdown:.2%} / Sharpe {me.sharpe:.2f} / Sortino {me.sortino:.2f} vs "
        f"MTUM {mt.max_drawdown:.2%} / {mt.sharpe:.2f} / {mt.sortino:.2f} ({ev})",
    )


def verdicts(report: Report, board: ScoreboardConfig) -> list[str]:
    """Plain-English verdicts: the 50/200 rule and MTUM comparisons, and what to do."""
    ev = board.evaluation_window
    g = report.graded
    out = []
    me = report.get(g, ev).metrics
    if report.has("sma", ev):
        sma = report.get("sma", ev).metrics
        out.append(
            f"Versus the 50/200 rule: {'better' if beats_on_risk(me, sma) else 'not better'} "
            f"on risk (drawdown {me.max_drawdown:.1%} vs {sma.max_drawdown:.1%}, "
            f"Sharpe {me.sharpe:.2f} vs {sma.sharpe:.2f})."
        )
    if report.has("mtum", ev):
        mt = report.get("mtum", ev).metrics
        if beats_on_risk(me, mt):
            out.append("Stoa beats MTUM on the risk-adjusted bar.")
        else:
            out.append(
                "Stoa does not beat MTUM on the risk-adjusted bar "
                f"(drawdown {me.max_drawdown:.1%} vs {mt.max_drawdown:.1%}, "
                f"Sharpe {me.sharpe:.2f} vs {mt.sharpe:.2f}, "
                f"Sortino {me.sortino:.2f} vs {mt.sortino:.2f}). "
                "Recommendation: buy the momentum fund (MTUM) instead."
            )
    else:
        out.append("MTUM baseline not run: no MTUM bars in the cache (`stoa data fetch`).")
    return out


def score_all(
    settings: Settings,
    secrets: Secrets,
    *,
    llm: bool = False,
    synthetic: bool = False,
    board: ScoreboardConfig | None = None,
    now: datetime | None = None,
    model: ModelSource | None = None,
    registry: Registry | None = None,
) -> Report:
    """Every strategy on every window, then the pass bar.

    Strategies: the v1 rule pipeline (``agents``), the model (when a model
    source is given), SPY buy-and-hold, the 50/200 rule, and MTUM buy-and-hold.
    The pass bar applies to ``board.graded`` (``model`` when present).
    """
    board = board or load_scoreboard()
    windows = [*board.tune_windows, board.evaluation_window]
    missing = [w for w in windows if w not in settings.backtest.windows]
    if missing:
        raise ValueError(f"scoreboard names unknown windows {missing}")
    last_end = max(settings.backtest.windows[w].end for w in windows)
    bars = model.bars if model is not None else load_bars(settings, last_end, synthetic=synthetic)
    created = now or datetime.now(UTC)
    graded = (
        model.name if model is not None else (board.graded if board.graded != "model" else "agents")
    )
    report = Report(
        run_id=created.strftime("%Y%m%d-%H%M%S")
        + ("-synthetic" if synthetic else "")
        + ("-llm" if llm else ""),
        created=created,
        data_source="synthetic" if synthetic else f"cache:{settings.data.cache_dir}",
        strategy_hash=settings.strategy_hash(),
        graded=graded,
    )
    budget = Budget(settings.llm.monthly_cost_cap_usd)  # one real-money cap for the whole run
    mtum = "MTUM" if "MTUM" in bars else None
    for w in windows:
        win = settings.backtest.windows[w]
        strategies: list[tuple[str, str | Strategy, Universe | None]] = [("agents", "agents", None)]
        if model is not None:
            strategies.append((model.name, model.strategy, model.universe))
        strategies += [("spy", "spy", None), ("sma", "sma", None)]
        if mtum is not None:
            ref_universe = Universe.assume_all_fractionable(
                (*settings.watchlist, mtum), settings.benchmark
            )
            strategies.append(("mtum", BuyAndHold(mtum, name="mtum"), ref_universe))
        for name, strat, universe in strategies:
            use_llm = llm and name == "agents"
            graded_bars = (
                model.window_bars(win.start, win.end)
                if model is not None and name == model.name and model.window_bars is not None
                else bars
            )
            run = run_one(
                settings,
                secrets,
                graded_bars,
                strat,
                w,
                win.start,
                win.end,
                llm=use_llm,
                budget=budget,
                universe=universe,
                slippage=model.slippage if model is not None else None,
            )
            report.runs.append(run)
            if name == graded and w == board.evaluation_window:
                report.deflated = graded_deflation(registry, run, settings, created)
    report.checks = evaluate(report, board)
    report.verdicts = verdicts(report, board)
    return report


def graded_deflation(
    registry: Registry | None, run: Run, settings: Settings, now: datetime
) -> dict[str, float]:
    """Log the graded run as a trial, then deflate its Sharpe by every trial so far."""
    rets = list(run.daily_returns)
    if registry is None or len(rets) < 3:
        sr, n, sk, ku = return_moments(rets)
        dsr, _ = deflated_sharpe(sr, 1, 0.0, n, sk, ku)
        return {"dsr": dsr, "trials": 1.0, "sharpe_daily": sr}
    import pandas as pd

    registry.append(
        kind="backtest",
        name=f"scoreboard:{run.strategy}:{run.window}",
        config_hash=settings.strategy_hash(),
        data_hash=hash_frame_like(rets),
        params={"strategy": run.strategy, "window": run.window},
        metrics={"sharpe_daily": return_moments(rets)[0], **run.metrics.to_dict()},
        returns=pd.Series(rets),
        now=now,
    )
    return deflate(registry, rets, variance_kind="backtest")


def _pct(x: float) -> str:
    return f"{x:+.2%}"


def render_markdown(report: Report) -> str:
    lines = [
        f"# Stoa scoreboard {report.run_id}",
        "",
        f"Data: {report.data_source}. Strategy hash: `{report.strategy_hash[:12]}`.",
        "",
        f"Graded strategy: `{report.graded}`.",
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
        if m.trade_count == 0:
            pf = "n/a"
        else:
            pf = "inf" if math.isinf(m.profit_factor) else f"{m.profit_factor:.2f}"
        lines.append(
            f"| {r.window} | {r.strategy} | {_pct(m.total_return)} | {_pct(m.cagr)} "
            f"| {m.max_drawdown:.2%} | {m.sharpe:.2f} "
            f"| {m.sortino:.2f} | {pf} | {m.trade_count} | {m.exposure:.0%} |"
        )
    if report.verdicts:
        lines += ["", *[f"- {v}" for v in report.verdicts]]
    if report.deflated:
        d = report.deflated
        lines += [
            "",
            f"Deflated Sharpe: {d.get('dsr', math.nan):.3f} with {int(d.get('trials', 0))} trials "
            f"in the registry (daily Sharpe {d.get('sharpe_daily', math.nan):.4f}, "
            f"luck threshold SR0 {d.get('sr0_daily', 0.0):.4f}, "
            f"PBO {d.get('pbo', math.nan):.2f}).",
        ]
    lines += ["", f"Bootstrap max drawdown ({report.graded}, 5th / 50th / 95th percentile):", ""]
    for r in report.runs:
        if r.strategy == report.graded:
            b = r.bootstrap_mdd
            lines.append(f"- {r.window}: {b['p05']:.1%} / {b['p50']:.1%} / {b['p95']:.1%}")
    fallbacks = sum(
        r.llm_fallbacks for r in report.runs if r.strategy == "agents" and r.llm_fallbacks
    )
    if fallbacks:
        lines += ["", f"The LLM scout fell back to the rule scout on {fallbacks} decisions."]
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
    if settings.backtest.holdout_end is not None:
        end = min(end, settings.backtest.holdout_end)
    if end <= start + timedelta(days=30):
        raise ValueError("the holdout needs more than a month of data after its start")
    return start, end
