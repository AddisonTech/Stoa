"""Background pulls from Alpaca: every news article, and minute bars.

News: the whole Benzinga feed from ``/v1beta1/news`` (no symbol filter, so no
article is fetched twice), one month at a time, kept when it tags at least one
symbol of the wide universe. Columns: id, created_at, updated_at, symbols,
headline, summary. One zstd parquet file per month; a month whose file exists
is done.

Minute bars: ``/v2/stocks/bars`` at 1Min, SIP, adjusted like the daily bars.
Each month is split into groups of symbols that traded that month (from the
daily bars), sized so liquid names share a request with few others and thin
names with many. One file per (month, group); a group whose file exists is
done. Newest month first.

Keys come from the environment (ALPACA_PAPER_KEY_ID, ALPACA_PAPER_SECRET) and
are never written anywhere.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from stoa.data.news import clean_text
from stoa.newsbrain import ROOT
from stoa.newsbrain.pace import Pace, log_progress, write_parquet

log = logging.getLogger(__name__)

DATA_URL = "https://data.alpaca.markets"
NEWS_DIR = ROOT / "news"
MINUTE_DIR = ROOT / "minute"
LIMITS = ROOT / "limits.json"
WIDE_BARS = Path("data/bars/wide")


def client_from_env() -> httpx.Client:
    key = os.environ.get("ALPACA_PAPER_KEY_ID", "")
    secret = os.environ.get("ALPACA_PAPER_SECRET", "")
    if not key or not secret:
        raise SystemExit("ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET must be set")
    return httpx.Client(
        base_url=DATA_URL,
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        timeout=60.0,
    )


def get(client: httpx.Client, pace: Pace, path: str, params: dict[str, Any]) -> dict[str, Any]:
    """One GET with pacing, backing off on 429 and server errors until it works."""
    wait = 5.0
    while True:
        pace.wait()
        try:
            resp = client.get(path, params=params)
        except httpx.HTTPError as exc:
            log.warning("%s: %s, retrying in %.0fs", path, type(exc).__name__, wait)
            pace.backoff(wait)
            wait = min(wait * 2, 300)
            continue
        if resp.status_code == 200:
            body: dict[str, Any] = resp.json()
            return body
        if resp.status_code == 429 or resp.status_code >= 500:
            log.warning("%s: HTTP %s, retrying in %.0fs", path, resp.status_code, wait)
            pace.backoff(wait)
            wait = min(wait * 2, 300)
            continue
        raise RuntimeError(f"{path}: HTTP {resp.status_code}: {resp.text[:200]}")


def universe() -> list[str]:
    """Every stock with daily bars in the wide universe (active and delisted)."""
    return sorted(p.stem for p in WIDE_BARS.glob("*.parquet"))


def months(start: date, end: date) -> list[date]:
    out, d = [], date(start.year, start.month, 1)
    while d <= end:
        out.append(d)
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return out


def month_end(m: date) -> date:
    nxt = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    return nxt - pd.Timedelta(days=1).to_pytimedelta()


def _stamp(m: date) -> str:
    return f"{m.year:04d}-{m.month:02d}"


# ---- news -----------------------------------------------------------------------------


def news_rows(articles: list[dict[str, Any]], wanted: set[str]) -> list[dict[str, Any]]:
    out = []
    for a in articles:
        syms = [str(s).upper() for s in (a.get("symbols") or [])]
        if not any(s in wanted for s in syms):
            continue
        out.append(
            {
                "id": int(a["id"]),
                "created_at": a.get("created_at"),
                "updated_at": a.get("updated_at") or a.get("created_at"),
                "symbols": syms,
                "headline": clean_text(str(a.get("headline", ""))),
                "summary": clean_text(str(a.get("summary", ""))),
            }
        )
    return out


def pull_news_month(
    client: httpx.Client, pace: Pace, m: date, wanted: set[str]
) -> tuple[pd.DataFrame, int]:
    """Every article created in month ``m`` that tags a wanted symbol, and the raw count."""
    start = datetime(m.year, m.month, 1, tzinfo=UTC)
    nxt = month_end(m)
    end = datetime(nxt.year, nxt.month, nxt.day, 23, 59, 59, tzinfo=UTC)
    params: dict[str, Any] = {
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 50,
        "sort": "asc",
        "include_content": "false",
        "exclude_contentless": "false",
    }
    rows: list[dict[str, Any]] = []
    raw = 0
    while True:
        body = get(client, pace, "/v1beta1/news", params)
        page = list(body.get("news") or [])
        raw += len(page)
        rows.extend(news_rows(page, wanted))
        token = body.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    df = pd.DataFrame(
        rows, columns=["id", "created_at", "updated_at", "symbols", "headline", "summary"]
    )
    if len(df):
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
        df = df.drop_duplicates("id").sort_values("created_at").reset_index(drop=True)
    return df, raw


def pull_news(start: date, end: date) -> None:
    client = client_from_env()
    pace = Pace("news", 110, LIMITS)
    wanted = set(universe())
    progress = ROOT / "logs" / "news.jsonl"
    for m in reversed(months(start, end)):
        path = NEWS_DIR / f"{_stamp(m)}.parquet"
        if path.exists():
            continue
        df, raw = pull_news_month(client, pace, m, wanted)
        write_parquet(df, path)
        log_progress(progress, month=_stamp(m), raw=raw, kept=len(df))
        log.info("news %s: %d of %d articles kept", _stamp(m), len(df), raw)
    log_progress(progress, done=True)


# ---- minute bars ------------------------------------------------------------------------

PLAN = ROOT / "minute_plan.parquet"
GROUP_BARS = 120_000  # estimated bars per request group and month
GROUP_MAX = 100


def bars_per_day(rank: np.ndarray) -> np.ndarray:
    """A rough count of minute bars a day by liquidity rank (1 is most liquid)."""
    return np.select(
        [rank <= 500, rank <= 1500, rank <= 3000, rank <= 5000], [900, 700, 450, 250], 80
    )


def build_plan() -> pd.DataFrame:
    """Per symbol and month: sessions with a daily bar and average dollar volume."""
    if PLAN.exists():
        return pd.read_parquet(PLAN)
    parts = []
    for sym in universe():
        df = pd.read_parquet(WIDE_BARS / f"{sym}.parquet", columns=["close", "volume"])
        if df.empty:
            continue
        idx = pd.DatetimeIndex(df.index)
        g = pd.DataFrame(
            {"days": 1, "dollars": df["close"].to_numpy() * df["volume"].to_numpy()},
            index=idx.to_period("M"),
        ).groupby(level=0)
        agg = g.agg(days=("days", "sum"), dollars=("dollars", "mean")).reset_index()
        agg.columns = ["month", "days", "dollars"]
        agg["symbol"] = sym
        parts.append(agg)
    plan = pd.concat(parts, ignore_index=True)
    plan["month"] = plan["month"].astype(str)
    write_parquet(plan, PLAN)
    return plan


def groups_for(month_rows: pd.DataFrame) -> list[list[str]]:
    """Pack the month's symbols into request groups, most liquid first."""
    rows = month_rows.sort_values("dollars", ascending=False).reset_index(drop=True)
    est = bars_per_day(np.arange(1, len(rows) + 1)) * rows["days"].to_numpy()
    out: list[list[str]] = []
    cur: list[str] = []
    load = 0
    for sym, n in zip(rows["symbol"], est, strict=True):
        if cur and (load + n > GROUP_BARS or len(cur) >= GROUP_MAX):
            out.append(cur)
            cur, load = [], 0
        cur.append(str(sym))
        load += int(n)
    if cur:
        out.append(cur)
    return out


def iter_bar_pages(
    client: httpx.Client, pace: Pace, symbols: list[str], m: date, end: date
) -> Iterator[dict[str, list[dict[str, Any]]]]:
    # The free plan refuses SIP queries that reach the last 15 minutes, so the
    # window never runs past the last full session.
    last = min(month_end(m), end)
    params: dict[str, Any] = {
        "symbols": ",".join(symbols),
        "timeframe": "1Min",
        "start": f"{m.isoformat()}T00:00:00Z",
        "end": f"{last.isoformat()}T23:59:59Z",
        "limit": 10000,
        "adjustment": "all",
        "feed": "sip",
        "sort": "asc",
    }
    while True:
        body = get(client, pace, "/v2/stocks/bars", params)
        yield dict(body.get("bars") or {})
        token = body.get("next_page_token")
        if not token:
            return
        params["page_token"] = token


def frame_of(pages: list[dict[str, list[dict[str, Any]]]]) -> pd.DataFrame:
    cols: dict[str, list[Any]] = {
        k: [] for k in ("symbol", "t", "o", "h", "l", "c", "v", "n", "vw")
    }
    for page in pages:
        for sym, bars in page.items():
            for b in bars:
                cols["symbol"].append(sym)
                cols["t"].append(b["t"])
                for k in ("o", "h", "l", "c", "v", "n", "vw"):
                    cols[k].append(b.get(k))
    df = pd.DataFrame(cols)
    if len(df):
        df["t"] = pd.to_datetime(df["t"], utc=True)
        df["symbol"] = df["symbol"].astype("category")
        for k in ("v", "n"):
            df[k] = pd.to_numeric(df[k]).astype("int64")
    return df


def pull_minutes(start: date, end: date) -> None:
    client = client_from_env()
    pace = Pace("minute", 70, LIMITS)
    plan = build_plan()
    progress = ROOT / "logs" / "minute.jsonl"
    for m in reversed(months(start, end)):
        stamp = _stamp(m)
        rows = plan[(plan["month"] == stamp) & (plan["days"] > 0)]
        groups = groups_for(rows)
        bars = 0
        for gi, group in enumerate(groups):
            path = MINUTE_DIR / stamp / f"g{gi:04d}.parquet"
            if path.exists():
                continue
            df = frame_of(list(iter_bar_pages(client, pace, group, m, end)))
            write_parquet(df, path)
            bars += len(df)
        log_progress(progress, month=stamp, symbols=len(rows), groups=len(groups), bars=bars)
        log.info("minute %s: %d symbols in %d groups, %d bars", stamp, len(rows), len(groups), bars)
    log_progress(progress, done=True)
