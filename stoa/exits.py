"""Exit rules for open positions. All checked on the close of the decision session.

Reason codes:
- stop_loss      close at or below entry minus stop_atr_multiple x ATR at entry
- trailing_stop  close at or below the trailing stop, which rises with the highest
                 high since entry and never falls below the initial stop
- take_profit    close at or above the optional profit target
- time_exit      held for max_hold_sessions sessions
- trend_break    closed below the 50 day average for trend_break_sessions in a row
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date

import pandas as pd

from stoa.clock import sessions_after
from stoa.config import ExitConfig
from stoa.models import ExitSignal, Holding


def initial_stop(entry_price: float, entry_atr: float, cfg: ExitConfig) -> float:
    return entry_price - cfg.stop_atr_multiple * entry_atr


def trailing_stop(
    highest_high: float, entry_price: float, entry_atr: float, cfg: ExitConfig
) -> float:
    return max(
        initial_stop(entry_price, entry_atr, cfg),
        highest_high - cfg.trailing_atr_multiple * entry_atr,
    )


def evaluate_position(
    holding: Holding, history: pd.DataFrame, session: date, cfg: ExitConfig
) -> ExitSignal | None:
    """First exit rule that fires for one position, or None. ``history`` ends at ``session``."""
    ts = pd.Timestamp(session)
    if history.empty or history.index[-1] != ts:
        return None
    meta = holding.meta
    close = float(history["close"].iloc[-1])
    entry = float(meta.entry_price)
    atr = float(meta.entry_atr)
    sym = holding.symbol

    stop = initial_stop(entry, atr, cfg)
    if close <= stop:
        return ExitSignal(
            symbol=sym, reason="stop_loss", detail=f"close {close:.2f} <= stop {stop:.2f}"
        )

    if cfg.trailing_enabled:
        since_entry = history.loc[pd.Timestamp(meta.entry_session) :]
        highest = float(since_entry["high"].max()) if not since_entry.empty else entry
        trail = trailing_stop(highest, entry, atr, cfg)
        if trail > stop and close <= trail:
            return ExitSignal(
                symbol=sym,
                reason="trailing_stop",
                detail=f"close {close:.2f} <= trailing {trail:.2f}",
            )

    if cfg.take_profit_atr_multiple is not None:
        target = entry + cfg.take_profit_atr_multiple * atr
        if close >= target:
            return ExitSignal(
                symbol=sym, reason="take_profit", detail=f"close {close:.2f} >= target {target:.2f}"
            )

    held = sessions_after(meta.entry_session, session)
    if held >= cfg.max_hold_sessions:
        return ExitSignal(symbol=sym, reason="time_exit", detail=f"held {held} sessions")

    n = cfg.trend_break_sessions
    tail = history.iloc[-n:]
    if len(tail) == n:
        below = [
            not math.isnan(s) and c < s
            for c, s in zip(tail["close"].astype(float), tail["sma_50"].astype(float), strict=True)
        ]
        if all(below):
            return ExitSignal(
                symbol=sym, reason="trend_break", detail=f"{n} closes below the 50 day average"
            )
    return None


def evaluate_exits(
    holdings: Mapping[str, Holding],
    features: Mapping[str, pd.DataFrame],
    session: date,
    cfg: ExitConfig,
) -> list[ExitSignal]:
    out = []
    for sym in sorted(holdings):
        hist = features.get(sym)
        if hist is None:
            continue
        signal = evaluate_position(holdings[sym], hist, session, cfg)
        if signal is not None:
            out.append(signal)
    return out
