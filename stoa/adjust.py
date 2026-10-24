"""Keeps Stoa's stored prices on the same scale as freshly adjusted bars.

Live bars are requested with ``adjustment=all``, so every split and dividend
rewrites all earlier prices. Anything Stoa stored from an earlier run (entry
price, entry ATR, the signal-mode paper quantities, the benchmark close at
inception) must be rescaled by the same factor, or a 4 for 1 split looks like
a 75% crash: the stop fires and, in signal mode, the kill switch trips.

Each stored value carries an anchor: a session and the adjusted close seen for
it at the time. The next run compares that close with the same bar today. The
ratio is the adjustment factor. Factors within ``TOLERANCE`` of 1 are ignored
(float noise from the provider).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd

from stoa.data.base import Bars
from stoa.features import cell
from stoa.models import dec
from stoa.state import State

TOLERANCE = Decimal("0.000001")


def _close(bars: Bars, symbol: str, day: date) -> Decimal | None:
    df = bars.get(symbol)
    ts = pd.Timestamp(day)
    if df is None or ts not in df.index:
        return None
    return dec(cell(df, ts, "close"))


def factor(bars: Bars, symbol: str, ref_session: date | None, ref_close: Decimal | None) -> Decimal:
    """Today's adjusted close of the anchor bar over the close recorded then. 1 if unknown."""
    if ref_session is None or ref_close is None or ref_close <= 0:
        return Decimal(1)
    now = _close(bars, symbol, ref_session)
    if now is None or now <= 0:
        return Decimal(1)
    f = now / ref_close
    return Decimal(1) if abs(f - 1) <= TOLERANCE else f


def apply_adjustments(state: State, bars: Bars, *, shadow: bool) -> list[str]:
    """Rescale entry metadata (and the signal-mode paper book) for splits and dividends.

    Returns notes for the daily summary, one per symbol that was rescaled.
    """
    notes: list[str] = []
    meta = dict(state.positions_meta)
    sim_qty = dict(state.sim_positions)
    sim_cost = dict(state.sim_avg_cost)
    for sym, m in meta.items():
        f = factor(bars, sym, m.ref_session, m.ref_close)
        if f == 1:
            continue
        meta[sym] = m.model_copy(
            update={"entry_price": m.entry_price * f, "entry_atr": m.entry_atr * f}
        )
        if shadow and sym in sim_qty:
            sim_qty[sym] = sim_qty[sym] / f
            sim_cost[sym] = sim_cost[sym] * f
        notes.append(f"{sym} history was re-adjusted (factor {f:.6f}); stops rescaled to match")
    state.positions_meta = meta
    if shadow:
        state.sim_positions = sim_qty
        state.sim_avg_cost = sim_cost
    return notes


def adjust_benchmark(state: State, bars: Bars, benchmark: str) -> None:
    """Keep stored benchmark closes on today's scale, so every 'vs SPY' is total return."""
    if state.inception is None and not state.track.days:
        return
    f = factor(bars, benchmark, state.last_close_session, state.last_benchmark_close)
    if f != 1 and state.inception is not None:
        state.inception = state.inception.model_copy(
            update={"benchmark_close": state.inception.benchmark_close * f}
        )
    if f != 1:
        state.track.days = [
            d.model_copy(update={"benchmark_close": d.benchmark_close * f})
            for d in state.track.days
        ]


def anchor(state: State, bars: Bars, session: date, benchmark: str) -> None:
    """Move every anchor to ``session`` after this run's adjustments are applied."""
    meta = {}
    for sym, m in state.positions_meta.items():
        close = _close(bars, sym, session)
        meta[sym] = (
            m
            if close is None
            else m.model_copy(update={"ref_session": session, "ref_close": close})
        )
    state.positions_meta = meta
    bench = _close(bars, benchmark, session)
    if bench is not None:
        state.last_benchmark_close = bench
