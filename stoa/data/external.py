"""Second sources: upcoming earnings dates and an independent price for the cross-check.

Finnhub (free key, https://finnhub.io): ``GET /api/v1/quote?symbol=`` returns
``c`` (current price) and ``t`` (unix time); ``GET /api/v1/calendar/earnings``
with ``from``, ``to`` and ``symbol`` returns ``earningsCalendar`` rows with
``date`` and ``hour`` (``bmo`` before the open, ``amc`` after the close,
``dmh`` during market hours). The free tier allows 60 calls a minute. The key
goes in the ``X-Finnhub-Token`` header, never in the URL, so it cannot end up
in a logged URL.

Yahoo through yfinance (``pip install stoa[yahoo]``): no key. Yahoo rate-limits
cloud IP ranges at times, so it is a fallback, never the only source.

Both are optional. Every function here returns None (or an empty result) when
a source is unavailable; the caller decides what that means.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Protocol

import httpx
import pandas as pd

from stoa import clock
from stoa.models import dec
from stoa.net import request_with_retry

log = logging.getLogger(__name__)

FINNHUB_URL = "https://finnhub.io/api/v1"
MAX_QUOTE_AGE = timedelta(minutes=30)


class PriceSource(Protocol):
    name: str

    def price(self, symbol: str) -> Decimal | None: ...


class EarningsSource(Protocol):
    name: str

    def upcoming(self, symbols: list[str], start: date, end: date) -> pd.DataFrame: ...


def _release_time(day: date, hour: str) -> datetime:
    """A release time from a date and a bmo/amc/dmh code. Unknown counts as before the open."""
    if clock.is_session(day):
        s = clock.session(day)
        if hour == "amc":
            return s.close + timedelta(minutes=5)
        if hour == "dmh":
            return s.open + timedelta(hours=2)
        return s.open - timedelta(hours=1)
    return datetime.combine(day, time(8, 0), tzinfo=clock.NY).astimezone(UTC)


class FinnhubClient:
    name = "finnhub"

    def __init__(
        self,
        token: str,
        client: httpx.Client | None = None,
        *,
        timeout: float = 10.0,
        now: Callable[[], datetime] = clock.now_utc,
    ) -> None:
        self.client = client or httpx.Client(base_url=FINNHUB_URL, timeout=timeout)
        self.client.headers["X-Finnhub-Token"] = token
        self.now = now

    def __repr__(self) -> str:
        return "FinnhubClient(token=<hidden>)"

    def price(self, symbol: str) -> Decimal | None:
        try:
            resp = request_with_retry(
                self.client, "GET", "/quote", params={"symbol": symbol}, max_retries=1
            )
        except httpx.HTTPError as exc:
            log.warning("finnhub quote failed for %s: %s", symbol, type(exc).__name__)
            return None
        if resp.status_code != 200:
            log.warning("finnhub quote for %s: HTTP %s", symbol, resp.status_code)
            return None
        body = resp.json()
        try:
            px = float(body.get("c") or 0)
            stamp = int(body.get("t") or 0)
        except (TypeError, ValueError):
            return None
        if px <= 0 or not math.isfinite(px):
            return None
        if stamp and self.now() - datetime.fromtimestamp(stamp, UTC) > MAX_QUOTE_AGE:
            log.warning("finnhub quote for %s is stale", symbol)
            return None
        return dec(px)

    def upcoming(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        rows = []
        fetched = self.now()
        for sym in symbols:
            try:
                resp = request_with_retry(
                    self.client,
                    "GET",
                    "/calendar/earnings",
                    params={"from": start.isoformat(), "to": end.isoformat(), "symbol": sym},
                    max_retries=1,
                )
            except httpx.HTTPError:
                continue
            if resp.status_code != 200:
                continue
            for r in resp.json().get("earningsCalendar") or []:
                try:
                    day = date.fromisoformat(str(r["date"]))
                except (KeyError, ValueError):
                    continue
                rows.append(
                    {
                        "symbol": sym,
                        "event_time": _release_time(day, str(r.get("hour") or "")),
                        "known_at": fetched,
                        "source": "finnhub",
                    }
                )
        return _frame(rows)


class YahooSource:
    """yfinance-backed price and earnings date. Imported lazily; absent means unavailable."""

    name = "yahoo"

    def __init__(self, now: Callable[[], datetime] = clock.now_utc) -> None:
        self.now = now

    @staticmethod
    def available() -> bool:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            return False
        return True

    def price(self, symbol: str) -> Decimal | None:
        try:
            import yfinance as yf

            px = float(yf.Ticker(symbol).fast_info["last_price"])
        except Exception as exc:
            log.warning("yahoo price failed for %s: %s", symbol, type(exc).__name__)
            return None
        return dec(px) if px > 0 and math.isfinite(px) else None

    def upcoming(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        rows = []
        fetched = self.now()
        try:
            import yfinance as yf
        except ImportError:
            return _frame([])
        for sym in symbols:
            try:
                cal = yf.Ticker(sym).calendar
            except Exception:
                continue
            dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
            for d in dates or []:
                day = d if isinstance(d, date) else pd.Timestamp(d).date()
                if start <= day <= end:
                    rows.append(
                        {
                            "symbol": sym,
                            "event_time": _release_time(day, ""),
                            "known_at": fetched,
                            "source": "yahoo",
                        }
                    )
        return _frame(rows)


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["symbol", "event_time", "known_at", "source"])
    if not df.empty:
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
        df["known_at"] = pd.to_datetime(df["known_at"], utc=True)
    return df


def pick_source(kind: str, finnhub_key: str | None) -> FinnhubClient | YahooSource | None:
    """The configured second source, or None when nothing is available."""
    if kind in ("finnhub", "auto") and finnhub_key:
        return FinnhubClient(finnhub_key)
    if kind in ("yahoo", "auto") and YahooSource.available():
        return YahooSource()
    return None
