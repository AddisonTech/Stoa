"""Trial 10's entry: wait for the first spike after the news to cool, then buy.

From 9:30 am, r5 is the absolute 5 minute return (a minute's close over the
close 5 minutes earlier, less 1) and P the largest r5 so far. The buy fills at
the open of the first minute at or after 9:45 am that follows a completed
minute whose r5 is at most a third of P. No bar is used before it closes, so
there is no lookahead. If that never happens by 11:00 am, the buy fills at the
open of the first minute at or after 11:00. Sells stay at 9:45 am.

Minute bars come from the background pull, read from its compacted copy
(data/newsbrain/minute_c/): per month a few files, each symbol whole in one of them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from stoa import clock
from stoa.newsbrain import sealed
from stoa.newsbrain.alpaca_pull import MINUTE_STORE_DIR

COOL_SHARE = 1 / 3
FIRST_ENTRY = "09:45"
LAST_ENTRY = "11:00"


def cool_off_price(bars: pd.DataFrame, day: date) -> float | None:
    """The trial 10 entry price from one symbol's minute bars (columns t, o, c)."""
    if bars.empty:
        return None
    t = pd.to_datetime(bars["t"], utc=True)
    session_open = pd.Timestamp(clock.session(day).open)
    first = pd.Timestamp(clock.at_ny(day, FIRST_ENTRY))
    last = pd.Timestamp(clock.at_ny(day, LAST_ENTRY))
    df = bars.assign(t=t)
    df = df[df["t"] >= session_open].sort_values("t").reset_index(drop=True)
    if df.empty:
        return None
    tns = df["t"].dt.as_unit("ns").astype("int64").to_numpy()
    closes = df["c"].astype(float).to_numpy()
    minute = pd.Timedelta(minutes=1).value
    first_ns, last_ns = first.value, last.value
    peak = 0.0
    cooled_ns: int | None = None
    for k in range(len(df)):
        j = int(np.searchsorted(tns, tns[k] - 5 * minute, side="right")) - 1
        if j < 0 or closes[j] <= 0:
            continue
        r5 = abs(closes[k] / closes[j] - 1)
        peak = max(peak, r5)
        done = int(tns[k]) + minute  # this minute's close is known from here on
        if done >= first_ns and peak > 0 and r5 <= peak * COOL_SHARE:
            cooled_ns = done
            break
        if done >= last_ns:
            break
    start_ns = cooled_ns if cooled_ns is not None and cooled_ns <= last_ns else last_ns
    after = df[tns >= start_ns]
    if after.empty:
        return None
    price = float(after["o"].iloc[0])
    return price if math.isfinite(price) and price > 0 else None


class MinuteStore:
    """Read one symbol's bars for one session from the monthly group files."""

    def __init__(self, root: Path = MINUTE_STORE_DIR) -> None:
        self.root = root
        self._index: dict[str, dict[str, Path]] = {}

    def _month(self, stamp: str) -> dict[str, Path]:
        if stamp not in self._index:
            out: dict[str, Path] = {}
            for path in sorted((self.root / stamp).glob("*.parquet")):
                syms = pd.read_parquet(path, columns=["symbol"])["symbol"].astype(str).unique()
                for s in syms:
                    out[s] = path
            self._index[stamp] = out
        return self._index[stamp]

    def day(self, symbol: str, day: date) -> pd.DataFrame:
        sealed.check(day, "minute bars")
        path = self._month(f"{day.year:04d}-{day.month:02d}").get(symbol)
        if path is None:
            return pd.DataFrame(columns=["t", "o", "c"])
        lo = pd.Timestamp(clock.session(day).open) - pd.Timedelta(hours=6)
        hi = pd.Timestamp(clock.session(day).close)
        df = pd.read_parquet(
            path,
            columns=["symbol", "t", "o", "c"],
            filters=[("symbol", "==", symbol), ("t", ">=", lo), ("t", "<=", hi)],
        )
        for col in ("o", "c"):  # float32 in the compacted files: back to the price as stored
            df[col] = df[col].astype("float64").round(4)
        return df[["t", "o", "c"]]


class CoolOffPrices:
    """Fill prices for trial 10: buys at the cool-off, everything else at 9:45 am."""

    def __init__(
        self,
        store: MinuteStore,
        at_945: Callable[[date, set[str]], dict[str, Decimal]],
    ) -> None:
        self.store = store
        self.at_945 = at_945

    def __call__(self, day: date, buys: set[str], others: set[str]) -> dict[str, Decimal]:
        out = self.at_945(day, others) if others else {}
        for sym in sorted(buys):
            p = cool_off_price(self.store.day(sym, day), day)
            if p is not None:
                out[sym] = Decimal(repr(p))
        return out
