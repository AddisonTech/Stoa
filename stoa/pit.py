"""The point-in-time rule, in one place.

Every record that is not a price bar (fundamentals, news scores, earnings
dates) carries ``known_at``: the UTC moment the information became public. A
decision made after the close of session t may use a record only if
``known_at <= decision_cutoff(t)``, the official close of t. Nothing else in
Stoa filters by time on its own; everything goes through :func:`visible`.

When a source gives only a date and no time (an SEC ``filed`` date, for
example), :func:`end_of_day` stamps it at 23:59:59 New York time, so it becomes
visible at the close of the next session, never the same day.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import pandas as pd

from stoa import clock

KNOWN_AT = "known_at"


def decision_cutoff(session: date) -> datetime:
    """The latest moment information may carry and still be used at the close of ``session``."""
    return clock.session(session).close.astimezone(UTC)


def end_of_day(day: date) -> datetime:
    """A date with no time: assume it became public at the very end of that New York day."""
    return datetime.combine(day, time(23, 59, 59), tzinfo=clock.NY).astimezone(UTC)


def eastern(stamp: datetime) -> datetime:
    """Read a naive timestamp as New York time and return it in UTC."""
    if stamp.tzinfo is not None:
        return stamp.astimezone(UTC)
    return stamp.replace(tzinfo=clock.NY).astimezone(UTC)


def as_utc(values: pd.Series) -> pd.Series:
    out = pd.to_datetime(values, utc=True)
    return out


def visible(frame: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    """Rows whose ``known_at`` is at or before ``cutoff``. The single enforcement point."""
    if frame.empty:
        return frame
    if KNOWN_AT not in frame.columns:
        raise KeyError("point-in-time frame has no known_at column")
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    known = as_utc(frame[KNOWN_AT])
    return frame.loc[known <= pd.Timestamp(cutoff).tz_convert(UTC)]


def latest_visible(frame: pd.DataFrame, cutoff: datetime, key: str = "symbol") -> pd.DataFrame:
    """For each ``key``, the most recent visible row (ties broken by row order)."""
    v = visible(frame, cutoff)
    if v.empty:
        return v
    v = v.assign(_k=as_utc(v[KNOWN_AT])).sort_values([key, "_k"], kind="stable")
    return v.groupby(key, sort=True).tail(1).drop(columns="_k").set_index(key)


def asof_panel(
    frame: pd.DataFrame,
    sessions: list[date],
    columns: list[str],
    key: str = "symbol",
) -> pd.DataFrame:
    """Latest visible value of ``columns`` for every (session, key), as a long frame.

    Equivalent to calling :func:`latest_visible` at each session's cutoff, but
    computed in one merge. The result is indexed by (date, key).
    """
    if frame.empty or not sessions:
        idx = pd.MultiIndex.from_tuples([], names=["date", key])
        return pd.DataFrame(columns=columns, index=idx, dtype=float)
    cut = pd.DataFrame(
        {
            "date": [pd.Timestamp(s) for s in sessions],
            "_cut": [pd.Timestamp(decision_cutoff(s)) for s in sessions],
        }
    ).sort_values("_cut")
    rec = frame.assign(_k=as_utc(frame[KNOWN_AT])).sort_values("_k", kind="stable")
    out = []
    for sym, g in rec.groupby(key, sort=True):
        m = pd.merge_asof(cut, g[["_k", *columns]], left_on="_cut", right_on="_k")
        m[key] = sym
        out.append(m.drop(columns=["_cut", "_k"]))
    panel = pd.concat(out, ignore_index=True)
    return panel.set_index(["date", key]).sort_index()
