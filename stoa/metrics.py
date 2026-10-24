"""Performance metrics computed from a close-to-close equity curve and closed trades."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from stoa.models import ClosedTrade, Fill

TRADING_DAYS = 252


@dataclass(frozen=True)
class Metrics:
    start: str
    end: str
    sessions: int
    starting_equity: float
    final_equity: float
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_sessions: int
    calmar: float
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float
    exposure: float
    turnover: float
    trade_count: int

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1]) / float(equity.iloc[0]) - 1.0


def cagr(equity: pd.Series) -> float:
    n = len(equity) - 1
    if n <= 0 or equity.iloc[0] <= 0:
        return 0.0
    ratio = float(equity.iloc[-1]) / float(equity.iloc[0])
    if ratio <= 0:
        return -1.0
    return float(ratio ** (TRADING_DAYS / n) - 1.0)


def ann_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(returns) > 1 else 0.0


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / TRADING_DAYS
    sd = float(excess.std(ddof=1))
    return float(excess.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / TRADING_DAYS
    downside = np.minimum(excess.to_numpy(), 0.0)
    dd = math.sqrt(float(np.mean(downside**2)))
    return float(excess.mean() / dd * math.sqrt(TRADING_DAYS)) if dd > 0 else 0.0


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Largest peak-to-trough loss (positive fraction) and longest run under water in sessions."""
    if len(equity) < 2:
        return 0.0, 0
    dd = drawdown_series(equity)
    worst = float(-dd.min())
    longest = run = 0
    for under in (dd < 0).to_numpy():
        run = run + 1 if under else 0
        longest = max(longest, run)
    return worst, longest


def trade_stats(trades: Sequence[ClosedTrade]) -> tuple[float, float, float, float, float]:
    """Profit factor, win rate, average win, average loss (negative), expectancy, all in dollars."""
    pnl = [float(t.pnl) for t in trades]
    if not pnl:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    if gross_loss > 0:
        pf = gross_win / gross_loss
    else:
        pf = math.inf if gross_win > 0 else 0.0
    win_rate = len(wins) / len(pnl)
    avg_win = gross_win / len(wins) if wins else 0.0
    avg_loss = -gross_loss / len(losses) if losses else 0.0
    expectancy = sum(pnl) / len(pnl)
    return pf, win_rate, avg_win, avg_loss, expectancy


def turnover(fills: Sequence[Fill], equity: pd.Series) -> float:
    """Annualized one-way turnover: traded value divided by two, over average equity."""
    if len(equity) < 2:
        return 0.0
    traded = sum(float(f.gross) for f in fills) / 2.0
    years = (len(equity) - 1) / TRADING_DAYS
    avg = float(equity.mean())
    return traded / avg / years if avg > 0 and years > 0 else 0.0


def compute(
    equity: pd.Series,
    trades: Sequence[ClosedTrade],
    fills: Sequence[Fill],
    exposure: pd.Series,
    rf: float = 0.0,
) -> Metrics:
    rets = daily_returns(equity)
    mdd, mdd_len = max_drawdown(equity)
    cg = cagr(equity)
    pf, wr, aw, al, ex = trade_stats(trades)
    return Metrics(
        start=str(pd.Timestamp(equity.index[0]).date()),
        end=str(pd.Timestamp(equity.index[-1]).date()),
        sessions=len(equity) - 1,
        starting_equity=float(equity.iloc[0]),
        final_equity=float(equity.iloc[-1]),
        total_return=total_return(equity),
        cagr=cg,
        ann_vol=ann_vol(rets),
        sharpe=sharpe(rets, rf),
        sortino=sortino(rets, rf),
        max_drawdown=mdd,
        max_drawdown_sessions=mdd_len,
        calmar=cg / mdd if mdd > 0 else 0.0,
        profit_factor=pf,
        win_rate=wr,
        avg_win=aw,
        avg_loss=al,
        expectancy=ex,
        exposure=float(exposure.iloc[1:].mean()) if len(exposure) > 1 else 0.0,
        turnover=turnover(fills, equity),
        trade_count=len(trades),
    )


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Year by month table of returns, plus a full-year column."""
    idx = pd.DatetimeIndex(equity.index)
    month_end = equity.groupby([idx.year, idx.month]).last()
    prev = month_end.shift(1)
    prev.iloc[0] = equity.iloc[0]
    rets = month_end / prev - 1.0
    table = rets.unstack(level=1)
    table.index.name = "year"
    table.columns = [f"{m:02d}" for m in table.columns]
    year_end = equity.groupby(idx.year).last()
    year_prev = year_end.shift(1)
    year_prev.iloc[0] = equity.iloc[0]
    table["year"] = (year_end / year_prev - 1.0).to_numpy()
    return table


def bootstrap_max_drawdown(
    returns: pd.Series, samples: int, block: int, seed: int
) -> dict[str, float]:
    """Stationary-block bootstrap of daily returns into a max drawdown distribution."""
    r = returns.to_numpy(dtype=float)
    n = len(r)
    if n < 2:
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0, "mean": 0.0}
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n))
    out = np.empty(samples)
    for k in range(samples):
        starts = rng.integers(0, n, size=math.ceil(n / block))
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        curve = np.cumprod(1.0 + r[idx])
        peak = np.maximum.accumulate(np.concatenate(([1.0], curve)))[1:]
        out[k] = float(np.max(1.0 - curve / peak))
    return {
        "p05": float(np.percentile(out, 5)),
        "p50": float(np.percentile(out, 50)),
        "p95": float(np.percentile(out, 95)),
        "mean": float(out.mean()),
    }
