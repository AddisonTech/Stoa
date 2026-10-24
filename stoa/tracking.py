"""Live and paper tracking: honest costs, the drift fallback, the quit rule and the exam.

Honest paper. Alpaca's paper fills are generous: no slippage at all on a
market order. So Stoa keeps its own record of every broker fill with the same
slippage and fee model as the backtest applied on top (a buy costs 5 bps
more, a sell gets 5 bps less and pays the regulatory fee). Each close it
records two equity numbers: the broker's raw equity, and the cost-adjusted
equity. The paper exam and the fallback are graded on the cost-adjusted one.

Drift fallback. Walk-forward results give an expected Sharpe and a drawdown
band. If live (or paper) results drift worse than that, Stoa stops opening
positions, lets the existing ones exit by their normal rules, and holds SPY
instead. Triggers:
- rolling Sharpe over ``fallback.sharpe_window`` sessions below the lower
  band (the 5th percentile of out-of-sample Sharpe for windows that long),
- drawdown beyond the 95th percentile of the out-of-sample bootstrap,
- the quit rule: after ``fallback.quit_after_sessions`` (about six months) of
  paper plus live, no edge over SPY (total return not above SPY's, after costs).
Only ``stoa fallback reset`` undoes it.

The exam. Going live is blocked until the paper exam has run at least
``exam.paper_min_sessions`` sessions with at least ``exam.paper_min_trades``
closed trades and passed: lower drawdown than SPY, and a higher Sharpe than
SPY, over the paper period, on the cost-adjusted record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from stoa.broker.sim import BPS, sell_fees
from stoa.config import CostConfig, ExamConfig, FallbackConfig
from stoa.models import ZERO, Side, dec

TRADING_DAYS = 252
SESSION_KEEP = 800


class DailyRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    session: date
    mode: str
    broker_equity: Decimal
    adjusted_equity: Decimal
    benchmark_close: Decimal
    trades_closed: int = 0
    flow: Decimal = ZERO  # deposits minus withdrawals since the previous record


class Expectation(BaseModel):
    """What walk-forward said live results should look like. Written by `stoa model train`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sharpe_mean: float
    sharpe_p05: float  # 5th percentile of rolling out-of-sample Sharpe at the fallback window
    drawdown_p95: float
    window: int
    source: str = ""


class Track(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    days: list[DailyRecord] = Field(default_factory=list)
    cost_drag: Decimal = ZERO  # cumulative simulated costs charged on top of broker fills
    trades_closed: int = 0
    pending_trades: int = 0  # closed since the last daily record
    expectation: Expectation | None = None
    fallback_active: bool = False
    fallback_reason: str = ""
    fallback_since: date | None = None
    first_live_session: date | None = None
    exam_passed_at: date | None = None

    def record(self, rec: DailyRecord) -> None:
        days = [d for d in self.days if d.session != rec.session] + [rec]
        self.days = sorted(days, key=lambda d: d.session)[-SESSION_KEEP:]


def fill_cost(
    side: Side, qty: Decimal, price: Decimal, broker_fees: Decimal, costs: CostConfig
) -> Decimal:
    """Extra cost Stoa charges itself on a broker fill so paper is as hard as the backtest.

    Slippage at ``costs.slippage_bps`` against the fill, and on sells the
    regulatory fee minus whatever the broker already charged.
    """
    gross = qty * price
    slip = gross * dec(costs.slippage_bps) / BPS
    fee = ZERO
    if side is Side.SELL:
        fee = max(ZERO, sell_fees(gross - slip, costs) - broker_fees)
    return slip + fee


def returns(values: list[float], flows: list[float] | None = None) -> np.ndarray:
    """Daily returns. Money moved in or out between closes is not counted as return."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return np.array([], dtype=float)
    f = np.zeros(len(v)) if flows is None else np.asarray(flows, dtype=float)
    out: np.ndarray = (v[1:] - f[1:]) / v[:-1] - 1.0
    return out


def curve(r: np.ndarray) -> list[float]:
    return [1.0, *np.cumprod(1.0 + r).tolist()]


def sharpe(r: np.ndarray) -> float:
    if len(r) < 2 or float(np.std(r, ddof=1)) == 0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(TRADING_DAYS))


def max_drawdown(values: list[float]) -> float:
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return 0.0
    peak = np.maximum.accumulate(v)
    return float(np.max(1 - v / peak))


@dataclass(frozen=True)
class Grade:
    sessions: int
    trades: int
    total_return: float
    spy_return: float
    sharpe: float
    spy_sharpe: float
    max_drawdown: float
    spy_drawdown: float
    raw_total_return: float

    @property
    def beats_spy_risk(self) -> bool:
        return self.max_drawdown < self.spy_drawdown and self.sharpe > self.spy_sharpe

    @property
    def edge(self) -> bool:
        return self.total_return > self.spy_return


def grade(days: list[DailyRecord]) -> Grade:
    if len(days) < 2:
        return Grade(len(days), 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    flows = [float(d.flow) for d in days]
    r_adj = returns([float(d.adjusted_equity) for d in days], flows)
    r_raw = returns([float(d.broker_equity) for d in days], flows)
    r_spy = returns([float(d.benchmark_close) for d in days])
    adj, raw, spy = curve(r_adj), curve(r_raw), curve(r_spy)
    return Grade(
        sessions=len(days) - 1,
        trades=sum(d.trades_closed for d in days[1:]),
        total_return=adj[-1] - 1,
        spy_return=spy[-1] - 1,
        sharpe=sharpe(r_adj),
        spy_sharpe=sharpe(r_spy),
        max_drawdown=max_drawdown(adj),
        spy_drawdown=max_drawdown(spy),
        raw_total_return=raw[-1] - 1,
    )


def segment_returns(segments: list[list[DailyRecord]]) -> tuple[np.ndarray, np.ndarray, int]:
    """Concatenated (strategy, SPY) daily returns over several records, and the trade count.

    Paper and live are separate accounts of different sizes, so returns are
    computed inside each segment and then joined, never across the seam.
    """
    rs, spy, trades = [], [], 0
    for seg in segments:
        if len(seg) < 2:
            continue
        rs.append(returns([float(d.adjusted_equity) for d in seg], [float(d.flow) for d in seg]))
        spy.append(returns([float(d.benchmark_close) for d in seg]))
        trades += sum(d.trades_closed for d in seg[1:])
    if not rs:
        return np.array([]), np.array([]), 0
    return np.concatenate(rs), np.concatenate(spy), trades


def check_fallback(
    track: Track,
    cfg: FallbackConfig,
    expectation: Expectation | None = None,
    prior: list[DailyRecord] | None = None,
) -> str | None:
    """The reason to switch to SPY-hold, or None. Pure: does not change ``track``.

    ``prior`` is the paper record when running live, so the quit rule counts
    paper and live together.
    """
    if not cfg.enabled or track.fallback_active:
        return None
    days = track.days
    r = returns([float(d.adjusted_equity) for d in days], [float(d.flow) for d in days])
    combined, combined_spy, _ = segment_returns([prior or [], days])
    if len(combined) >= cfg.quit_after_sessions:
        total = float(np.prod(1 + combined) - 1)
        spy_total = float(np.prod(1 + combined_spy) - 1)
        if total <= spy_total:
            return (
                f"quit rule: no edge over SPY after {len(combined)} sessions of paper and live "
                f"({total:+.1%} vs SPY {spy_total:+.1%}, after costs)"
            )
    if len(r) < cfg.min_history_sessions:
        return None
    adj = curve(r)
    exp = expectation or track.expectation
    if exp is not None:
        if len(r) >= cfg.sharpe_window:
            rs = sharpe(r[-cfg.sharpe_window :])
            if rs < exp.sharpe_p05:
                return (
                    f"rolling {cfg.sharpe_window} session Sharpe {rs:.2f} is below the "
                    f"walk-forward lower band {exp.sharpe_p05:.2f}"
                )
        dd = max_drawdown(adj)
        if dd > exp.drawdown_p95:
            return (
                f"drawdown {dd:.1%} is beyond the walk-forward 95th percentile "
                f"{exp.drawdown_p95:.1%}"
            )
    return None


def activate(track: Track, reason: str, session: date) -> None:
    track.fallback_active = True
    track.fallback_reason = reason
    track.fallback_since = session


def reset(track: Track) -> None:
    track.fallback_active = False
    track.fallback_reason = ""
    track.fallback_since = None


@dataclass(frozen=True)
class ExamStatus:
    walkforward_done: bool
    walkforward_passed: bool
    holdout_sealed: bool
    holdout_scored: bool
    holdout_passed: bool
    paper: Grade
    paper_sessions_needed: int
    paper_trades_needed: int
    paper_passed: bool

    @property
    def live_allowed(self) -> bool:
        return self.walkforward_passed and self.paper_passed

    def problems(self) -> list[str]:
        out = []
        if not self.walkforward_done:
            out.append("walk-forward has not been run (`stoa model train`)")
        elif not self.walkforward_passed:
            out.append("walk-forward results did not pass")
        if not self.paper_passed:
            g = self.paper
            out.append(
                f"paper exam not passed: {g.sessions} of {self.paper_sessions_needed} sessions, "
                f"{g.trades} of {self.paper_trades_needed} trades, "
                f"drawdown {g.max_drawdown:.1%} vs SPY {g.spy_drawdown:.1%}, "
                f"Sharpe {g.sharpe:.2f} vs SPY {g.spy_sharpe:.2f}"
            )
        return out


def paper_exam(track: Track, cfg: ExamConfig) -> tuple[Grade, bool]:
    paper_days = [d for d in track.days if d.mode in ("paper", "signal")]
    g = grade(paper_days)
    passed = (
        g.sessions >= cfg.paper_min_sessions
        and g.trades >= cfg.paper_min_trades
        and g.beats_spy_risk
    )
    return g, passed


def expectation_from_returns(
    daily: list[float], window: int, samples: int = 2000, seed: int = 7, source: str = ""
) -> Expectation:
    """Bands from out-of-sample daily returns: rolling Sharpe 5th pct and bootstrap MDD 95th."""
    r = np.asarray([x for x in daily if math.isfinite(x)], dtype=float)
    if len(r) < window + 2:
        raise ValueError("not enough out-of-sample returns to set the expectation")
    rolling = [sharpe(r[i : i + window]) for i in range(0, len(r) - window + 1)]
    rng = np.random.default_rng(seed)
    mdds = []
    n = len(r)
    block = 20
    for _ in range(samples):
        starts = rng.integers(0, n, size=math.ceil(n / block))
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        curve = np.cumprod(1 + r[idx])
        peak = np.maximum.accumulate(np.concatenate(([1.0], curve)))[1:]
        mdds.append(float(np.max(1 - curve / peak)))
    return Expectation(
        sharpe_mean=sharpe(r),
        sharpe_p05=float(np.percentile(rolling, 5)),
        drawdown_p95=float(np.percentile(mdds, 95)),
        window=window,
        source=source,
    )


def expectation_path(shared_dir: Path) -> Path:
    return shared_dir / "exam" / "expectation.json"


def save_expectation(shared_dir: Path, exp: Expectation) -> Path:
    p = expectation_path(shared_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(exp.model_dump_json(indent=2), encoding="utf-8")
    return p


def load_expectation(shared_dir: Path) -> Expectation | None:
    p = expectation_path(shared_dir)
    if not p.exists():
        return None
    return Expectation.model_validate_json(p.read_text(encoding="utf-8"))
