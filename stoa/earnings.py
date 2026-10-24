"""Earnings rules: never hold through a report, never open into one.

The calendar is a table of releases: ``symbol``, ``event_time`` (UTC, when the
numbers came out) and ``known_at`` (UTC, when the *date* of the release was
known). Visibility goes through :mod:`stoa.pit` like everything else.

Where the rows come from:
- History: SEC 8-K item 2.02 acceptance times (``stoa fundamentals fetch``).
  The date of a release is usually announced two to four weeks ahead, but
  EDGAR does not record when. Stoa assumes it was known
  ``earnings.confirm_lead_sessions`` sessions before the release. That is an
  assumption, stated in docs/TRAINING.md. A shorter lead makes the rules
  fire less often in backtests.
- Upcoming: a forward calendar (Finnhub with a free key, or Yahoo) fetched at
  DECIDE. ``known_at`` is the fetch time.
- Symbols with no SEC history (mostly companies that left the index before
  the SEC ticker map was taken): a volume spike proxy, flagged ``proxy``.

Timing. Stoa trades at the open (plus a few minutes). For a release at time
e, the *last safe session* is the last session whose execution time is before
e. A release before the open of day D has last safe session D - 1. A release
after the open of D (during the day or after the close) has last safe session D.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from stoa import clock, pit
from stoa.config import EarningsConfig
from stoa.data.base import Bars
from stoa.models import ExitSignal

EARNINGS_COLUMNS = ("symbol", "event_time", "known_at", "source")
EXEMPT_DEFAULT = frozenset({"SPY", "QQQ", "MTUM", "IWM", "DIA"})


def empty_calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype=str),
            "event_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "known_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "source": pd.Series(dtype=str),
        }
    )


def last_safe_session(event_time: datetime, after_open_minutes: int) -> date:
    """The last session whose execution time is strictly before ``event_time``."""
    ny_day = clock.to_ny(event_time).date()
    if clock.is_session(ny_day):
        execution = clock.session(ny_day).open + timedelta(minutes=after_open_minutes)
        if execution < event_time:
            return ny_day
    return clock.previous_session(ny_day)


def with_known_lead(table: pd.DataFrame, lead_sessions: int) -> pd.DataFrame:
    """Set ``known_at`` for historical rows: ``lead_sessions`` sessions before the release."""
    if table.empty:
        return table
    out = table.copy()
    known = []
    for et, kn, src in zip(out["event_time"], out["known_at"], out["source"], strict=True):
        if src in ("finnhub", "yahoo", "manual"):
            known.append(kn)
            continue
        day = clock.to_ny(pd.Timestamp(et).to_pydatetime()).date()
        if not clock.is_session(day):
            day = clock.next_session(day)
        for _ in range(lead_sessions):
            day = clock.previous_session(day)
        known.append(pit.end_of_day(clock.previous_session(day)))
    out["known_at"] = pd.to_datetime(known, utc=True)
    return out


def volume_proxy(
    bars: Bars, symbols: set[str], multiple: float, min_gap_sessions: int = 40
) -> pd.DataFrame:
    """Earnings-like sessions from volume spikes, for symbols with no SEC history.

    A session whose volume is at least ``multiple`` times its trailing 20 day
    average counts as a release after the previous close. Spikes closer than
    ``min_gap_sessions`` to the last one are ignored (one report a quarter).
    """
    rows = []
    for sym in sorted(symbols):
        df = bars.get(sym)
        if df is None or len(df) < 25:
            continue
        avg = df["volume"].rolling(20, min_periods=20).mean().shift(1)
        spikes = df.index[(df["volume"] >= multiple * avg).to_numpy()]
        last = -10_000
        pos = {ts: i for i, ts in enumerate(df.index)}
        for ts in spikes:
            i = pos[ts]
            if i - last < min_gap_sessions:
                continue
            last = i
            prev = df.index[i - 1].date() if i > 0 else ts.date()
            stamp = clock.session(prev).close + timedelta(minutes=5)
            rows.append({"symbol": sym, "event_time": stamp, "known_at": stamp, "source": "proxy"})
    if not rows:
        return empty_calendar()
    out = pd.DataFrame(rows)
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    out["known_at"] = pd.to_datetime(out["known_at"], utc=True)
    return out


@dataclass(frozen=True)
class EarningsView:
    """What is known about one symbol's earnings at one decision."""

    last_report: datetime | None
    next_event: datetime | None
    next_safe: date | None


class EarningsCalendar:
    def __init__(self, table: pd.DataFrame, after_open_minutes: int = 15) -> None:
        t = table if not table.empty else empty_calendar()
        t = t.assign(
            event_time=pd.to_datetime(t["event_time"], utc=True),
            known_at=pd.to_datetime(t["known_at"], utc=True),
        )
        self.after_open_minutes = after_open_minutes
        self._by_symbol: dict[str, pd.DataFrame] = {
            str(sym): g.sort_values("event_time").reset_index(drop=True)
            for sym, g in t.groupby("symbol")
        }

    @property
    def symbols(self) -> set[str]:
        return set(self._by_symbol)

    def view(self, symbol: str, session: date) -> EarningsView:
        g = self._by_symbol.get(symbol)
        if g is None or g.empty:
            return EarningsView(None, None, None)
        cutoff = pd.Timestamp(pit.decision_cutoff(session))
        past = g.loc[g["event_time"] <= cutoff]
        last = past["event_time"].iloc[-1].to_pydatetime() if not past.empty else None
        future = pit.visible(g.loc[g["event_time"] > cutoff], cutoff.to_pydatetime())
        if future.empty:
            return EarningsView(last, None, None)
        nxt = future["event_time"].iloc[0].to_pydatetime()
        return EarningsView(last, nxt, last_safe_session(nxt, self.after_open_minutes))


def sessions_until(start: date, end: date) -> int:
    """Signed number of sessions from ``start`` to ``end`` (0 when equal)."""
    if end >= start:
        return clock.sessions_after(start, end)
    return -clock.sessions_after(end, start)


def entry_block(
    cal: EarningsCalendar,
    symbol: str,
    session: date,
    cfg: EarningsConfig,
    exempt: frozenset[str] = EXEMPT_DEFAULT,
) -> tuple[str, str] | None:
    """Why a new position in ``symbol`` decided at the close of ``session`` is not allowed."""
    if not cfg.enabled or symbol in exempt:
        return None
    entry = clock.next_session(session)
    v = cal.view(symbol, session)
    if v.next_safe is not None:
        ahead = sessions_until(entry, v.next_safe)
        if ahead <= cfg.holding_window_sessions:
            assert v.next_event is not None
            when = clock.to_ny(v.next_event).strftime("%Y-%m-%d %H:%M ET")
            return (
                "earnings",
                f"earnings {when} inside the {cfg.holding_window_sessions} session holding window",
            )
        if ahead <= cfg.unknown_lookahead_sessions:
            return None
    if not cfg.block_when_unknown:
        return None
    if v.last_report is None:
        if v.next_safe is not None:
            return None
        return "earnings", "no earnings history and no confirmed date"
    age = (pit.decision_cutoff(session) - v.last_report).days
    if v.next_safe is None and age > cfg.unknown_stale_days:
        return (
            "earnings",
            f"no confirmed date in the next {cfg.unknown_lookahead_sessions} sessions and the "
            f"last report was {age} days ago",
        )
    return None


def exit_signals(
    cal: EarningsCalendar,
    held: list[str],
    session: date,
    cfg: EarningsConfig,
    exempt: frozenset[str] = EXEMPT_DEFAULT,
) -> list[ExitSignal]:
    """Sell at the next open any position with a confirmed release before the open after."""
    if not cfg.enabled or not cfg.exit_before:
        return []
    nxt = clock.next_session(session)
    out = []
    for sym in sorted(held):
        if sym in exempt:
            continue
        v = cal.view(sym, session)
        if v.next_safe is None:
            # No confirmed date. If the last report is old enough that the next
            # one could be close, leave rather than risk holding through it.
            if cfg.block_when_unknown and v.last_report is not None:
                age = (pit.decision_cutoff(session) - v.last_report).days
                if age > cfg.unknown_stale_days:
                    out.append(
                        ExitSignal(
                            symbol=sym,
                            reason="earnings_unknown",
                            detail=f"last report {age} days ago and no confirmed next date",
                        )
                    )
            continue
        if v.next_safe > nxt:
            continue
        assert v.next_event is not None
        when = clock.to_ny(v.next_event).strftime("%Y-%m-%d %H:%M ET")
        late = " (the date became known late)" if v.next_safe < nxt else ""
        out.append(ExitSignal(symbol=sym, reason="earnings_exit", detail=f"earnings {when}{late}"))
    return out


def synthetic_calendar(bars: Bars, exempt: frozenset[str], seed: int = 3) -> pd.DataFrame:
    """Quarterly releases on the bars' own calendar, for tests and the offline demo."""
    rows = []
    for k, (sym, df) in enumerate(sorted(bars.items())):
        if sym in exempt or len(df) < 70:
            continue
        rng = np.random.default_rng(seed * 1000 + k)
        i = int(rng.integers(5, 60))
        while i < len(df):
            day = df.index[i].date()
            after = bool(rng.integers(0, 2))
            sess = clock.session(day)
            stamp = sess.close + timedelta(minutes=5) if after else sess.open - timedelta(hours=1)
            rows.append(
                {
                    "symbol": sym,
                    "event_time": stamp.astimezone(UTC),
                    "known_at": stamp,
                    "source": "synthetic",
                }
            )
            i += int(rng.integers(60, 66))
    if not rows:
        return empty_calendar()
    out = pd.DataFrame(rows)
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    out["known_at"] = pd.to_datetime(out["known_at"], utc=True)
    return out
