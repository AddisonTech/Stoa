"""Bars from local CSV files: <dir>/<SYMBOL>.csv with date,open,high,low,close,volume."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from stoa.data.base import Bars, DataError, validate_bars


class CSVProvider:
    name = "csv"

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def fetch(self, symbols: list[str], start: date, end: date) -> Bars:
        out: Bars = {}
        for sym in symbols:
            p = self.directory / f"{sym}.csv"
            if not p.exists():
                raise DataError(f"{sym}: no CSV at {p}")
            df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
            out[sym] = validate_bars(sym, df).loc[pd.Timestamp(start) : pd.Timestamp(end)]
        return out
