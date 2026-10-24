"""Local parquet cache of daily bars, one file per symbol and feed."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from stoa.data.base import Bars, DataError, validate_bars
from stoa.files import replace_file

DEFAULT_NAMESPACE = "default"


class BarCache:
    def __init__(self, root: Path, namespace: str) -> None:
        self.dir = root / namespace

    def path(self, symbol: str) -> Path:
        return self.dir / f"{symbol}.parquet"

    def write(self, bars: Bars) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for sym, df in bars.items():
            tmp = self.path(sym).with_suffix(".tmp")
            df.to_parquet(tmp)
            replace_file(tmp, self.path(sym))

    def read(self, symbols: list[str], start: date, end: date) -> Bars:
        out: Bars = {}
        for sym in symbols:
            p = self.path(sym)
            if not p.exists():
                raise DataError(f"{sym}: not in cache at {p}; run `stoa data fetch`")
            df = validate_bars(sym, pd.read_parquet(p))
            out[sym] = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        return out


def load_cached_bars(cache_dir: Path, symbols: list[str], start: date, end: date) -> Bars:
    """Bars saved by `stoa data fetch`. Backtests and scoring read only from here."""
    return BarCache(cache_dir, DEFAULT_NAMESPACE).read(symbols, start, end)
