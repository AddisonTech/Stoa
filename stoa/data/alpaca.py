"""Alpaca market data: daily adjusted bars with pagination and feed fallback.

The free plan allows SIP history except the latest 15 minutes. If a SIP request
is refused (403 or 422), the provider retries the whole request on the fallback
feed (IEX) and records which feed it used, because IEX volume is a fraction of
consolidated volume.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from datetime import time as time_of_day
from typing import Any

import httpx
import pandas as pd

from stoa.clock import NY
from stoa.data.base import Bars, DataError, validate_bars
from stoa.net import request_with_retry

log = logging.getLogger(__name__)

PAGE_LIMIT = 10000
SIP_DELAY_MINUTES = 16  # free plan: no SIP data from the latest 15 minutes
SYMBOLS_PER_REQUEST = 30


class AlpacaDataProvider:
    name = "alpaca"

    def __init__(
        self,
        client: httpx.Client,
        *,
        feed: str = "sip",
        fallback_feed: str | None = "iex",
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.client = client
        self.now = now
        self.feed = feed
        self.fallback_feed = fallback_feed
        self.max_retries = max_retries
        self.sleep = sleep
        self.last_feed_used: str | None = None

    def fetch(self, symbols: list[str], start: date, end: date) -> Bars:
        out: Bars = {}
        for i in range(0, len(symbols), SYMBOLS_PER_REQUEST):
            chunk = symbols[i : i + SYMBOLS_PER_REQUEST]
            out.update(self._fetch_chunk(chunk, start, end))
        missing = [s for s in symbols if s not in out]
        if missing:
            raise DataError(f"Alpaca returned no bars for {missing}")
        return out

    def _fetch_chunk(self, symbols: list[str], start: date, end: date) -> Bars:
        try:
            raw = self._paginate(symbols, start, end, self.feed)
            self.last_feed_used = self.feed
        except _FeedRefused as exc:
            if not self.fallback_feed:
                raise DataError(str(exc)) from exc
            log.warning(
                "feed %s refused (%s), falling back to %s", self.feed, exc, self.fallback_feed
            )
            try:
                raw = self._paginate(symbols, start, end, self.fallback_feed)
            except _FeedRefused as exc2:
                raise DataError(str(exc2)) from exc2
            self.last_feed_used = self.fallback_feed
        return {sym: validate_bars(sym, _to_frame(rows)) for sym, rows in raw.items() if rows}

    def _paginate(
        self, symbols: list[str], start: date, end: date, feed: str
    ) -> dict[str, list[dict[str, Any]]]:
        rows: dict[str, list[dict[str, Any]]] = {}
        token: str | None = None
        end_ts = min(
            datetime.combine(end, time_of_day(23, 59, 59), tzinfo=UTC),
            self.now() - timedelta(minutes=SIP_DELAY_MINUTES),
        )
        while True:
            params: dict[str, Any] = {
                "symbols": ",".join(symbols),
                "timeframe": "1Day",
                "start": f"{start.isoformat()}T00:00:00Z",
                "end": end_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "adjustment": "all",
                "feed": feed,
                "limit": PAGE_LIMIT,
                "sort": "asc",
            }
            if token:
                params["page_token"] = token
            resp = request_with_retry(
                self.client,
                "GET",
                "/v2/stocks/bars",
                params=params,
                max_retries=self.max_retries,
                sleep=self.sleep,
            )
            if resp.status_code in (403, 422):
                raise _FeedRefused(f"HTTP {resp.status_code} for feed {feed}: {resp.text[:200]}")
            if resp.status_code != 200:
                raise DataError(f"bars request failed: HTTP {resp.status_code}: {resp.text[:200]}")
            body = resp.json()
            for sym, bars in (body.get("bars") or {}).items():
                rows.setdefault(sym, []).extend(bars)
            token = body.get("next_page_token")
            if not token:
                return rows


class _FeedRefused(Exception):
    pass


def _session_date(stamp: str) -> pd.Timestamp:
    ts = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return pd.Timestamp(ts.astimezone(NY).date())


def _to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [r["o"] for r in rows],
            "high": [r["h"] for r in rows],
            "low": [r["l"] for r in rows],
            "close": [r["c"] for r in rows],
            "volume": [r["v"] for r in rows],
        },
        index=pd.DatetimeIndex([_session_date(r["t"]) for r in rows], name="date"),
    )
