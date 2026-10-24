"""Point-in-time index membership, so backtests include companies that later left.

The tradable universe on session t is every symbol that was an S&P 500 member
on t and passes the liquidity filter on t. A company that went bankrupt, was
acquired or was dropped appears up to the day it left and never after. That is
what removes survivorship bias from the training data.

An interval's ``end`` is the first date the symbol was no longer a member, so a
symbol is a member on t when ``start <= t < end``.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from importlib import resources
from pathlib import Path

UNKNOWN_SECTOR = "Unknown"


@dataclass(frozen=True)
class Interval:
    symbol: str
    start: date
    end: date | None  # first day out; None while still a member

    def contains(self, day: date) -> bool:
        return self.start <= day and (self.end is None or day < self.end)


class Membership:
    def __init__(self, intervals: list[Interval]) -> None:
        self.intervals = sorted(intervals, key=lambda i: (i.symbol, i.start))

    @classmethod
    def from_csv(cls, path: Path) -> Membership:
        with path.open(encoding="utf-8") as fh:
            return cls._read(fh.read())

    @classmethod
    def _read(cls, text: str) -> Membership:
        rows = csv.DictReader(line for line in text.splitlines() if not line.startswith("#"))
        out = [
            Interval(
                symbol=r["symbol"].strip().upper(),
                start=date.fromisoformat(r["start"]),
                end=date.fromisoformat(r["end"]) if r["end"].strip() else None,
            )
            for r in rows
        ]
        return cls(out)

    def members(self, day: date) -> set[str]:
        return {i.symbol for i in self.intervals if i.contains(day)}

    def is_member(self, symbol: str, day: date) -> bool:
        return any(i.symbol == symbol and i.contains(day) for i in self.intervals)

    def ever_members(self, start: date, end: date) -> set[str]:
        """Every symbol that was a member on at least one day in [start, end]."""
        return {
            i.symbol for i in self.intervals if i.start <= end and (i.end is None or i.end > start)
        }

    def left_between(self, start: date, end: date) -> set[str]:
        """Symbols whose membership ended in (start, end] and never restarted by ``end``."""
        out = set()
        for sym in {i.symbol for i in self.intervals if i.end and start < i.end <= end}:
            if not self.is_member(sym, end):
                out.add(sym)
        return out


def parse_aliases(text: str) -> dict[str, str]:
    rows = csv.DictReader(line for line in text.splitlines() if not line.startswith("#"))
    return {r["symbol"].strip().upper(): r["fetch_as"].strip().upper() for r in rows}


@lru_cache(maxsize=1)
def ticker_aliases() -> dict[str, str]:
    """Membership symbol to the ticker the bar provider and the SEC file its history under."""
    text = resources.files("stoa.reference").joinpath("ticker_aliases.csv").read_text("utf-8")
    return parse_aliases(text)


def fetch_symbol(symbol: str) -> str:
    """The ticker to fetch ``symbol``'s bars or filings under (itself unless renamed)."""
    return ticker_aliases().get(symbol.upper(), symbol)


def fetch_renamed[T](
    fetch: Callable[[list[str]], dict[str, T]], symbols: list[str]
) -> dict[str, T]:
    """Fetch under each symbol's current ticker and key the results by the membership symbol."""
    tickers = {sym: fetch_symbol(sym) for sym in symbols}
    got = fetch(sorted(set(tickers.values())))
    return {sym: got[t] for sym, t in tickers.items() if t in got}


@lru_cache(maxsize=1)
def default_membership() -> Membership:
    text = resources.files("stoa.reference").joinpath("sp500_membership.csv").read_text("utf-8")
    return Membership._read(text)


@lru_cache(maxsize=1)
def sectors() -> dict[str, str]:
    text = resources.files("stoa.reference").joinpath("sectors.csv").read_text("utf-8")
    return {r["symbol"].upper(): r["sector"] for r in csv.DictReader(text.splitlines())}


def sector_of(symbol: str, overrides: dict[str, str] | None = None) -> str:
    if overrides and symbol in overrides:
        return overrides[symbol]
    return sectors().get(symbol.upper(), UNKNOWN_SECTOR)
