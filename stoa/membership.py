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


# SIC code ranges (inclusive) to the GICS sector names in sectors.csv, for symbols
# outside the S&P 500. Narrow ranges come first; the first match wins. It is an
# approximation of GICS, close enough for a sector cap.
SIC_SECTORS: tuple[tuple[int, int, str], ...] = (
    (1520, 1531, "Consumer Discretionary"),  # homebuilders
    (1200, 1399, "Energy"),  # coal, oil and gas
    (1000, 1499, "Materials"),  # metal and other mining
    (100, 999, "Consumer Staples"),  # agriculture
    (1500, 1799, "Industrials"),  # construction
    (2830, 2836, "Health Care"),  # drugs, biotech
    (2840, 2844, "Consumer Staples"),  # soap, cosmetics
    (2000, 2199, "Consumer Staples"),  # food, beverages, tobacco
    (2200, 2399, "Consumer Discretionary"),  # textiles, apparel
    (2500, 2599, "Consumer Discretionary"),  # furniture
    (2700, 2799, "Communication Services"),  # publishing
    (2900, 2999, "Energy"),  # refining
    (3010, 3011, "Consumer Discretionary"),  # tires
    (3100, 3199, "Consumer Discretionary"),  # leather
    (2400, 3399, "Materials"),  # lumber, paper, chemicals, plastics, glass, metals
    (3570, 3579, "Information Technology"),  # computers
    (3630, 3639, "Consumer Discretionary"),  # household appliances
    (3651, 3652, "Consumer Discretionary"),  # household audio and video
    (3660, 3679, "Information Technology"),  # communications equipment, semiconductors
    (3710, 3716, "Consumer Discretionary"),  # motor vehicles
    (3750, 3751, "Consumer Discretionary"),  # motorcycles, bicycles
    (3812, 3812, "Industrials"),  # defense electronics
    (3800, 3839, "Information Technology"),  # instruments
    (3840, 3851, "Health Care"),  # medical devices
    (3900, 3999, "Consumer Discretionary"),  # toys, sporting goods, jewelry
    (3400, 3899, "Industrials"),  # machinery, electrical equipment, aircraft
    (4800, 4899, "Communication Services"),  # telecom, broadcasting
    (4950, 4959, "Industrials"),  # waste
    (4900, 4999, "Utilities"),
    (4000, 4799, "Industrials"),  # transportation
    (5120, 5122, "Health Care"),  # drug wholesale
    (5140, 5149, "Consumer Staples"),  # grocery wholesale
    (5170, 5172, "Energy"),  # petroleum wholesale
    (5000, 5199, "Industrials"),  # other wholesale
    (5400, 5499, "Consumer Staples"),  # food stores
    (5912, 5912, "Consumer Staples"),  # drug stores
    (5200, 5999, "Consumer Discretionary"),  # retail
    (6798, 6798, "Real Estate"),  # REITs
    (6500, 6599, "Real Estate"),
    (6000, 6799, "Financials"),  # banks, brokers, insurance, holding and blank check
    (7370, 7379, "Information Technology"),  # software, IT services
    (7810, 7849, "Communication Services"),  # film
    (7000, 7299, "Consumer Discretionary"),  # hotels, personal services
    (7500, 7999, "Consumer Discretionary"),  # auto repair, amusement
    (7300, 7399, "Industrials"),  # other business services
    (8000, 8099, "Health Care"),  # health services
    (8731, 8731, "Health Care"),  # commercial biological research
    (8200, 8299, "Consumer Discretionary"),  # education
    (8100, 8999, "Industrials"),  # legal, engineering, management
    (9995, 9995, "Financials"),  # shells
)


def sic_sector(sic: int | str | None) -> str:
    """The GICS-style sector for an SEC SIC code, or Unknown."""
    try:
        code = int(str(sic))
    except (TypeError, ValueError):
        return UNKNOWN_SECTOR
    for lo, hi, name in SIC_SECTORS:
        if lo <= code <= hi:
            return name
    return UNKNOWN_SECTOR


def _read_sectors(name: str) -> dict[str, str]:
    text = resources.files("stoa.reference").joinpath(name).read_text("utf-8")
    return {r["symbol"].upper(): r["sector"] for r in csv.DictReader(text.splitlines())}


@lru_cache(maxsize=1)
def sectors() -> dict[str, str]:
    """Every known sector: the wide universe from SEC SIC codes, then GICS for the S&P 500."""
    out = _read_sectors("sectors_wide.csv")
    out.update(_read_sectors("sectors.csv"))  # GICS wins where both exist
    return out


def sector_of(symbol: str, overrides: dict[str, str] | None = None) -> str:
    if overrides and symbol in overrides:
        return overrides[symbol]
    return sectors().get(symbol.upper(), UNKNOWN_SECTOR)
