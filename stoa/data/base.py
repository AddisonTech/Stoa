"""The bar provider interface and bar validation.

A ``Bars`` value maps a symbol to a DataFrame indexed by session date (a naive
``DatetimeIndex`` at midnight, sorted, unique) with float columns
open, high, low, close, volume. Prices are split and dividend adjusted.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

BAR_COLUMNS = ("open", "high", "low", "close", "volume")

Bars = dict[str, pd.DataFrame]


class DataError(RuntimeError):
    """Missing, stale or malformed market data. Never trade on it."""


class BarProvider(Protocol):
    name: str

    def fetch(self, symbols: list[str], start: date, end: date) -> Bars:
        """Daily adjusted bars for each symbol with start <= session <= end."""
        ...


def validate_bars(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy or raise DataError. Never fills or invents data."""
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{symbol}: missing columns {missing}")
    out = df.loc[:, list(BAR_COLUMNS)].astype(float).copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize().as_unit("ns")
    if out.index.tz is not None:
        raise DataError(f"{symbol}: index must be naive session dates")
    out = out.sort_index()
    if out.index.has_duplicates:
        raise DataError(f"{symbol}: duplicate session dates")
    if out.empty:
        raise DataError(f"{symbol}: no bars")
    prices = out[["open", "high", "low", "close"]].to_numpy()
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise DataError(f"{symbol}: non-positive or non-finite prices")
    if (out["high"] < out[["open", "close", "low"]].max(axis=1) - 1e-9).any() or (
        out["low"] > out[["open", "close", "high"]].min(axis=1) + 1e-9
    ).any():
        raise DataError(f"{symbol}: high/low inconsistent with open/close")
    if (out["volume"] < 0).any():
        raise DataError(f"{symbol}: negative volume")
    out.index.name = "date"
    return out


def latest_session(bars: Bars, symbol: str) -> date:
    return pd.Timestamp(bars[symbol].index[-1]).date()


def require_fresh(bars: Bars, symbols: list[str], session: date) -> None:
    """Every symbol must have a bar for exactly ``session`` as its latest bar."""
    stale = []
    for sym in symbols:
        if sym not in bars or bars[sym].empty:
            stale.append(f"{sym}=missing")
            continue
        last = latest_session(bars, sym)
        if last != session:
            stale.append(f"{sym}={last}")
    if stale:
        raise DataError(f"stale data for session {session}: {', '.join(stale)}")


def fresh_subset(
    bars: Bars,
    symbols: list[str],
    session: date,
    required: set[str],
    min_fraction: float,
) -> tuple[Bars, list[str]]:
    """Keep the symbols whose latest bar is ``session``; fail loudly when that is not enough.

    Every symbol in ``required`` (the benchmark and anything held) must be fresh.
    The rest may be dropped for the day (a halt, a pending delisting, a late
    print), but only while at least ``min_fraction`` of ``symbols`` are fresh.
    Returns the fresh bars and the dropped symbols.
    """
    fresh: Bars = {}
    dropped: list[str] = []
    for sym in symbols:
        df = bars.get(sym)
        if df is not None and not df.empty and latest_session(bars, sym) == session:
            fresh[sym] = df
        else:
            dropped.append(sym)
    bad = sorted(required & set(dropped))
    if bad:
        require_fresh(bars, bad, session)
    if symbols and len(fresh) / len(symbols) < min_fraction:
        raise DataError(
            f"only {len(fresh)} of {len(symbols)} symbols have a bar for {session}; "
            f"need {min_fraction:.0%}. Dropped: {', '.join(dropped[:10])}"
        )
    return fresh, dropped
