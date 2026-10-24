"""Seeded synthetic daily bars on the real NYSE calendar.

Used by the test suite and the offline end-to-end demo. Never used for grading.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stoa.clock import sessions_between
from stoa.data.base import Bars, validate_bars


def synthetic_bars(
    symbols: list[str],
    start: date,
    end: date,
    *,
    seed: int = 1,
    benchmark: str = "SPY",
) -> Bars:
    """Correlated geometric random walks with a shared market factor and regimes."""
    days = sessions_between(start, end)
    n = len(days)
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days], name="date")
    regime = np.where(np.sin(np.arange(n) / 180.0) > -0.35, 0.0006, -0.0012)
    market = regime + rng.normal(0.0, 0.010, n)
    out: Bars = {}
    for k, sym in enumerate(symbols):
        sub = np.random.default_rng(seed * 1000 + k)
        beta = 1.0 if sym == benchmark else float(sub.uniform(0.6, 1.5))
        drift = 0.0 if sym == benchmark else float(sub.normal(0.0002, 0.0004))
        idio = 0.0 if sym == benchmark else float(sub.uniform(0.006, 0.018))
        rets = beta * market + drift + sub.normal(0.0, max(idio, 1e-9), n)
        close = float(sub.uniform(40, 400)) * np.exp(np.cumsum(rets))
        gap = sub.normal(0.0, 0.004, n)
        open_ = np.empty(n)
        open_[0] = close[0]
        open_[1:] = close[:-1] * np.exp(gap[1:])
        spread = np.abs(sub.normal(0.0, 0.008, n))
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        volume = sub.uniform(5e6, 4e7, n)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
        )
        out[sym] = validate_bars(sym, df)
    return out


class SyntheticProvider:
    name = "synthetic"

    def __init__(self, seed: int = 1, benchmark: str = "SPY", start: date = date(2016, 1, 4)) -> None:
        self.seed = seed
        self.benchmark = benchmark
        self.start = start

    def fetch(self, symbols: list[str], start: date, end: date) -> Bars:
        full = synthetic_bars(symbols, self.start, end, seed=self.seed, benchmark=self.benchmark)
        return {s: df.loc[pd.Timestamp(start) :] for s, df in full.items()}
