"""Keyless fallback provider using yfinance. Install with `pip install stoa[yahoo]`."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from stoa.data.base import Bars, DataError, validate_bars


class YahooProvider:
    name = "yahoo"

    def fetch(self, symbols: list[str], start: date, end: date) -> Bars:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataError("yfinance is not installed; pip install 'stoa[yahoo]'") from exc
        out: Bars = {}
        for sym in symbols:
            raw = yf.Ticker(sym).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=True,
                actions=False,
            )
            if raw is None or raw.empty:
                raise DataError(f"{sym}: Yahoo returned no bars")
            df = raw.rename(columns=str.lower)
            idx = pd.DatetimeIndex(df.index)
            df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
            out[sym] = validate_bars(sym, df)
        return out
