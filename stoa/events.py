"""Macro event days (FOMC and CPI) from the vendored calendars.

"Go light" means new positions are sized down by a multiplier on an event day
and on the session before it, because the decision at the close before an
event executes at that event day's open. Exits are never affected.
"""

from __future__ import annotations

import csv
from datetime import date
from functools import lru_cache
from importlib import resources

from stoa import clock


def _dates(name: str) -> frozenset[date]:
    text = resources.files("stoa.reference").joinpath(name).read_text("utf-8")
    rows = csv.DictReader(line for line in text.splitlines() if not line.startswith("#"))
    return frozenset(date.fromisoformat(r["date"]) for r in rows)


@lru_cache(maxsize=1)
def fomc_days() -> frozenset[date]:
    return _dates("fomc.csv")


@lru_cache(maxsize=1)
def cpi_days() -> frozenset[date]:
    return _dates("cpi.csv")


def coverage() -> tuple[date, date]:
    every = fomc_days() | cpi_days()
    return min(every), max(every)


def event_on(day: date) -> list[str]:
    out = []
    if day in fomc_days():
        out.append("FOMC")
    if day in cpi_days():
        out.append("CPI")
    return out


@lru_cache(maxsize=1)
def _event_sessions() -> dict[date, list[str]]:
    """Each event mapped to the session it lands on (a weekend event moves to the next one)."""
    out: dict[date, list[str]] = {}
    for name, days in (("FOMC", fomc_days()), ("CPI", cpi_days())):
        for d in sorted(days):
            s = d if clock.is_session(d) else clock.next_session(d)
            out.setdefault(s, []).append(f"{name} {d.isoformat()}")
    return out


def light_reason(decision_session: date) -> str | None:
    """Why buys decided at this close should be sized down, or None.

    The plan executes at the next session's open and is held through it, so
    both the decision day (the day before an event) and the next session (the
    event day itself) count.
    """
    ev = _event_sessions()
    hits = ev.get(decision_session, []) + ev.get(clock.next_session(decision_session), [])
    return ", ".join(hits) if hits else None
