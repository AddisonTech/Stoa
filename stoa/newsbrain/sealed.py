"""The holdout wall for trials 9 and 10.

Nothing in the news brain may read a session from the holdout start on: not news,
not filings, not daily bars, not minute bars, not 9:45 am prices. Every loader
calls ``check`` or filters with ``before_holdout`` first, and ``HoldoutBreach`` is
raised before any data from the window is read. The dates match
``backtest.holdout_start`` and ``holdout_end`` in config/default.yaml (a test
holds them together).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

HOLDOUT_START = date(2026, 7, 1)
HOLDOUT_END = date(2026, 10, 23)
# Midnight New York time on the first holdout day, for timestamps.
CUTOFF = pd.Timestamp(HOLDOUT_START.isoformat(), tz="America/New_York").tz_convert("UTC")


class HoldoutBreach(RuntimeError):
    """Something asked for data from inside the sealed holdout."""


def check(day: date, what: str) -> None:
    if day >= HOLDOUT_START:
        raise HoldoutBreach(
            f"{what} for {day} is inside the holdout ({HOLDOUT_START} to {HOLDOUT_END}), "
            "which trials 9 and 10 never read"
        )


def month_allowed(stem: str) -> bool:
    """Whether a monthly file named YYYY-MM is wholly before the holdout."""
    return stem < f"{HOLDOUT_START.year:04d}-{HOLDOUT_START.month:02d}"


def before_holdout(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Rows whose timestamp is before the holdout's first day, New York time."""
    ts = pd.to_datetime(df[column], utc=True)
    return df[ts < CUTOFF]
