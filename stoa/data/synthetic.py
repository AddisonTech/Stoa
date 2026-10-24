"""Seeded synthetic daily bars on the real NYSE calendar.

Used by the test suite and the offline end-to-end demo. Never used for grading.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stoa.clock import sessions_between
from stoa.data.base import Bars, validate_bars
from stoa.membership import Interval, Membership


def synthetic_bars(
    symbols: list[str],
    start: date,
    end: date,
    *,
    seed: int = 1,
    benchmark: str = "SPY",
    signal: float = 0.0,
) -> Bars:
    """Correlated geometric random walks with a shared market factor and regimes.

    ``signal`` plants a persistent, slowly changing drift in each stock (an
    AR(1) process), so recent winners keep winning for a while: a momentum
    edge the model can learn. 0 (the default) means no edge at all, pure noise.
    Tests use both: the pipeline must find a planted edge and must not invent
    one in noise.
    """
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
        if signal > 0 and sym != benchmark:
            shocks = sub.normal(0.0, signal * 0.0001, n)
            persistent = np.zeros(n)
            for i in range(1, n):
                persistent[i] = 0.995 * persistent[i - 1] + shocks[i]
            rets = rets + persistent
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

    def __init__(
        self, seed: int = 1, benchmark: str = "SPY", start: date = date(2016, 1, 4)
    ) -> None:
        self.seed = seed
        self.benchmark = benchmark
        self.start = start

    def fetch(self, symbols: list[str], start: date, end: date) -> Bars:
        full = synthetic_bars(symbols, self.start, end, seed=self.seed, benchmark=self.benchmark)
        return {s: df.loc[pd.Timestamp(start) :] for s, df in full.items()}


def synthetic_universe(
    watchlist: list[str],
    start: date,
    end: date,
    *,
    seed: int = 1,
    benchmark: str = "SPY",
    extra: int = 16,
    references: tuple[str, ...] = ("MTUM",),
    signal: float = 0.0,
) -> tuple[Bars, Membership]:
    """Watchlist plus ``extra`` synthetic index names, some of which leave the index and die.

    A quarter of the extra names stop trading partway through (their bars end
    the day after they leave the index, like a delisting). A few join the index
    late. Used to exercise the survivorship machinery offline.
    """
    names = [f"SYN{i:02d}" for i in range(extra)]
    bars = synthetic_bars(
        [*watchlist, *references, *names],
        start,
        end,
        seed=seed,
        benchmark=benchmark,
        signal=signal,
    )
    days = sessions_between(start, end)
    rng = np.random.default_rng(seed + 99)
    intervals = [Interval(s, date(2000, 1, 3), None) for s in watchlist if s != benchmark]
    for i, sym in enumerate(names):
        if i % 4 == 0:  # dies: leaves the index, last bar the next session
            k = int(rng.integers(len(days) // 3, len(days) - 60))
            leave = days[k]
            intervals.append(Interval(sym, date(2000, 1, 3), leave))
            bars[sym] = bars[sym].loc[: pd.Timestamp(days[k])]
        elif i % 4 == 1:  # joins late
            k = int(rng.integers(300, len(days) // 2))
            intervals.append(Interval(sym, days[k], None))
        else:
            intervals.append(Interval(sym, date(2000, 1, 3), None))
    return bars, Membership(intervals)
