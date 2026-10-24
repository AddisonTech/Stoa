"""New York time, NYSE sessions, early closes and job windows.

All session logic goes through the exchange_calendars XNYS calendar, which knows
holidays and early closes. Cron runs in UTC and ignores daylight saving time, so
every job asks this module whether it is inside its window before acting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xc
import pandas as pd

NY = ZoneInfo("America/New_York")
CALENDAR_START = "2015-01-01"


@dataclass(frozen=True)
class Session:
    day: date
    open: datetime  # timezone-aware, UTC
    close: datetime  # timezone-aware, UTC

    @property
    def early_close(self) -> bool:
        return self.close.astimezone(NY).time() < time(16, 0)


@lru_cache(maxsize=1)
def _calendar() -> xc.ExchangeCalendar:
    return xc.get_calendar("XNYS", start=CALENDAR_START)


def now_utc() -> datetime:
    return datetime.now(UTC)


def to_ny(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("naive datetime; pass a timezone-aware timestamp")
    return ts.astimezone(NY)


def is_session(day: date) -> bool:
    return bool(_calendar().is_session(pd.Timestamp(day)))


def session(day: date) -> Session:
    cal = _calendar()
    ts = pd.Timestamp(day)
    if not cal.is_session(ts):
        raise ValueError(f"{day} is not an NYSE session")
    return Session(
        day=day,
        open=cal.session_open(ts).to_pydatetime(),
        close=cal.session_close(ts).to_pydatetime(),
    )


def sessions_between(start: date, end: date) -> list[date]:
    """All NYSE sessions with start <= day <= end."""
    cal = _calendar()
    lo = max(pd.Timestamp(start), cal.first_session)
    hi = min(pd.Timestamp(end), cal.last_session)
    if hi < lo:
        return []
    return [d.date() for d in cal.sessions_in_range(lo, hi)]


def next_session(day: date) -> date:
    cal = _calendar()
    nxt: pd.Timestamp = cal.date_to_session(
        pd.Timestamp(day) + pd.Timedelta(days=1), direction="next"
    )
    return nxt.date()


def previous_session(day: date) -> date:
    cal = _calendar()
    prev: pd.Timestamp = cal.date_to_session(
        pd.Timestamp(day) - pd.Timedelta(days=1), direction="previous"
    )
    return prev.date()


def last_closed_session(ts: datetime) -> date:
    """The most recent session whose close is at or before ``ts``."""
    ny_day = to_ny(ts).date()
    day = ny_day if is_session(ny_day) else previous_session(ny_day)
    if session(day).close <= ts.astimezone(UTC):
        return day
    return previous_session(day)


def decide_session(ts: datetime, after_close_minutes: int, end_hhmm: str) -> date | None:
    """The session DECIDE may act for at ``ts``, or None when outside the window.

    The window runs from ``after_close_minutes`` after that day's close (early
    closes included) until ``end_hhmm`` New York time on the same date.
    """
    ny = to_ny(ts)
    day = ny.date()
    if not is_session(day):
        return None
    s = session(day)
    hh, mm = (int(x) for x in end_hhmm.split(":"))
    start = s.close + timedelta(minutes=after_close_minutes)
    end = datetime.combine(day, time(hh, mm, 59), tzinfo=NY)
    return day if start <= ts <= end else None


def execute_session(ts: datetime, after_open_minutes: int, window_minutes: int) -> date | None:
    """The session EXECUTE may act in at ``ts``, or None when outside the window."""
    day = to_ny(ts).date()
    if not is_session(day):
        return None
    s = session(day)
    start = s.open + timedelta(minutes=after_open_minutes)
    end = s.open + timedelta(minutes=after_open_minutes + window_minutes)
    return day if start <= ts <= end else None


@lru_cache(maxsize=1)
def _session_index() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(_calendar().sessions)


def sessions_after(start: date, end: date) -> int:
    """Number of sessions d with start < d <= end. Zero when end <= start."""
    idx = _session_index()
    lo = int(idx.searchsorted(pd.Timestamp(start), side="right"))
    hi = int(idx.searchsorted(pd.Timestamp(end), side="right"))
    return max(0, hi - lo)
