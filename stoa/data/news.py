"""Historical news from Alpaca's News API (Benzinga), stored point in time.

Endpoint: ``GET https://data.alpaca.markets/v1beta1/news`` with ``symbols``,
``start``, ``end``, ``limit`` (50 per page), ``sort``, ``include_content`` and
``page_token``. Each article has ``id``, ``headline``, ``summary``, ``content``,
``symbols``, ``source``, ``created_at`` and ``updated_at``. History goes back to
2015. The free market data keys are enough.

Point in time: the text Stoa reads today is the latest version of the article.
If it was edited after it was first published, the edit was not known at
``created_at``. So ``known_at`` is ``updated_at`` (never earlier than
``created_at``), which can only delay an item, never leak one.
"""

from __future__ import annotations

import html
import logging
import re
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from datetime import time as time_of_day
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from stoa.data.base import DataError
from stoa.files import replace_file
from stoa.net import request_with_retry

log = logging.getLogger(__name__)

NEWS_PATH = "/v1beta1/news"
PAGE_LIMIT = 50
SYMBOLS_PER_REQUEST = 20
NEWS_COLUMNS = (
    "id",
    "symbol",
    "known_at",
    "created_at",
    "headline",
    "summary",
    "content",
    "source",
    "kind",
)
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """Strip HTML tags and entities, collapse whitespace."""
    return _SPACE.sub(" ", html.unescape(_TAG.sub(" ", raw or ""))).strip()


def _stamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def parse_articles(rows: list[dict[str, Any]], wanted: set[str]) -> list[dict[str, Any]]:
    """One record per (article, wanted symbol)."""
    out = []
    for a in rows:
        created = _stamp(a.get("created_at"))
        updated = _stamp(a.get("updated_at")) or created
        if created is None or updated is None:
            continue
        known = max(created, updated)
        for sym in a.get("symbols") or []:
            s = str(sym).upper()
            if s not in wanted:
                continue
            out.append(
                {
                    "id": f"alpaca:{a.get('id')}",
                    "symbol": s,
                    "known_at": known,
                    "created_at": created,
                    "headline": clean_text(str(a.get("headline", ""))),
                    "summary": clean_text(str(a.get("summary", ""))),
                    "content": clean_text(str(a.get("content", ""))),
                    "source": str(a.get("source", "")),
                    "kind": "news",
                }
            )
    return out


class AlpacaNewsClient:
    def __init__(
        self,
        client: httpx.Client,
        *,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        include_content: bool = True,
    ) -> None:
        self.client = client
        self.max_retries = max_retries
        self.sleep = sleep
        self.include_content = include_content

    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        wanted = {s.upper() for s in symbols}
        records: list[dict[str, Any]] = []
        for i in range(0, len(symbols), SYMBOLS_PER_REQUEST):
            chunk = symbols[i : i + SYMBOLS_PER_REQUEST]
            records.extend(self._chunk(chunk, start, end, wanted))
        if not records:
            return pd.DataFrame(columns=list(NEWS_COLUMNS))
        df = pd.DataFrame(records, columns=list(NEWS_COLUMNS))
        return df.drop_duplicates(["id", "symbol"]).sort_values("known_at").reset_index(drop=True)

    def _chunk(
        self, symbols: list[str], start: date, end: date, wanted: set[str]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        token: str | None = None
        end_ts = datetime.combine(end, time_of_day(23, 59, 59), tzinfo=UTC)
        while True:
            params: dict[str, Any] = {
                "symbols": ",".join(symbols),
                "start": f"{start.isoformat()}T00:00:00Z",
                "end": end_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit": PAGE_LIMIT,
                "sort": "asc",
                "include_content": str(self.include_content).lower(),
            }
            if token:
                params["page_token"] = token
            resp = request_with_retry(
                self.client,
                "GET",
                NEWS_PATH,
                params=params,
                max_retries=self.max_retries,
                sleep=self.sleep,
            )
            if resp.status_code != 200:
                raise DataError(f"news request failed: HTTP {resp.status_code}: {resp.text[:200]}")
            body = resp.json()
            out.extend(parse_articles(list(body.get("news") or []), wanted))
            token = body.get("next_page_token")
            if not token:
                return out


class NewsStore:
    """Parquet file of news items, merged on (id, symbol) so refetching never duplicates."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=list(NEWS_COLUMNS))
        return pd.read_parquet(self.path)

    def merge(self, new: pd.DataFrame) -> pd.DataFrame:
        old = self.load()
        both = pd.concat([old, new], ignore_index=True) if not old.empty else new
        both = both.drop_duplicates(["id", "symbol"], keep="last").sort_values("known_at")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        both.reset_index(drop=True).to_parquet(tmp)
        replace_file(tmp, self.path)
        return both
