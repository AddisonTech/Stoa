"""Bring research bars up to the last full session, from Alpaca, for trials 17 and 18.

Daily: every symbol in ``data/bars/wide`` (active and delisted) gets the sessions after
its last stored bar up to ``end``. The bars are adjusted for splits and dividends
(``adjustment=all``), so a corporate action after the stored history changes the old
values too: each request starts two weeks before the stored end, the overlap is compared,
and a symbol whose overlap does not match (to 1e-6) has its whole history fetched again
and replaced, so no series is spliced across an adjustment. A symbol with no new bars is
left as it is (it stopped trading).

Minute: the four commodity funds of trial 18 (UNG, USO, BOIL, KOLD) and SPY, from the
month after the compacted store ends to ``end``, into ``data/commod/minute/<symbol>.parquet``
(the compacted store under data/newsbrain is not touched).

Keys come from ``.env`` (ALPACA_PAPER_KEY_ID, ALPACA_PAPER_SECRET). The live jobs share
that key, so every request waits out the live windows and keeps to the pace in
``data/newsbrain/limits.json`` (``extend``, default 60 a minute).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa import envfile
from stoa.newsbrain.alpaca_pull import LIMITS, client_from_env, get
from stoa.newsbrain.pace import Pace, log_progress, write_parquet

log = logging.getLogger(__name__)

WIDE = Path("data/bars/wide")
MINUTE_OUT = Path("data/commod/minute")
LOG = Path("data/commod/logs/extend.jsonl")
OVERLAP_DAYS = 14
GROUP = 100
HISTORY_START = date(2016, 1, 1)
MINUTE_SYMBOLS = ("UNG", "USO", "BOIL", "KOLD", "SPY")


def _client_and_pace() -> tuple[Any, Pace]:
    envfile.need("ALPACA_PAPER_KEY_ID", "ALPACA_PAPER_SECRET")
    return client_from_env(), Pace("extend", 60, LIMITS)


def daily_pages(
    client: Any, pace: Pace, symbols: list[str], start: date, end: date
) -> dict[str, list[dict[str, Any]]]:
    params: dict[str, Any] = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        "start": start.isoformat(),
        "end": f"{end.isoformat()}T23:59:59Z",
        "limit": 10000,
        "adjustment": "all",
        "feed": "sip",
        "sort": "asc",
    }
    out: dict[str, list[dict[str, Any]]] = {}
    while True:
        body = get(client, pace, "/v2/stocks/bars", params)
        for sym, bars in (body.get("bars") or {}).items():
            out.setdefault(sym, []).extend(bars)
        token = body.get("next_page_token")
        if not token:
            return out
        params["page_token"] = token


def frame(bars: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "open": [float(b["o"]) for b in bars],
            "high": [float(b["h"]) for b in bars],
            "low": [float(b["l"]) for b in bars],
            "close": [float(b["c"]) for b in bars],
            "volume": [float(b["v"]) for b in bars],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp(b["t"]).tz_convert("America/New_York").tz_localize(None).normalize()
                for b in bars
            ]
        ),
    )
    return df[~df.index.duplicated(keep="last")].sort_index()


def _read(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(WIDE / f"{sym}.parquet")
    df.index = pd.DatetimeIndex(df.index)
    return df


def _save(sym: str, df: pd.DataFrame) -> None:
    from stoa.files import replace_file

    p = WIDE / f"{sym}.parquet"
    tmp = p.with_name(p.name + ".tmp")
    df.to_parquet(tmp)
    replace_file(tmp, p)


def matches(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    common = old.index.intersection(new.index)
    if len(common) == 0:
        return False
    a = old.loc[common, "close"].to_numpy(dtype=float)
    b = new.loc[common, "close"].to_numpy(dtype=float)
    return bool(np.allclose(a, b, rtol=1e-6, atol=0))


def extend_daily(end: date) -> dict[str, int]:
    client, pace = _client_and_pace()
    syms = sorted(p.stem for p in WIDE.glob("*.parquet"))
    last = {s: _read(s).index.max() for s in syms}
    # Only symbols that traded in the stored history's last month can have new bars.
    stored_end = max(last.values())
    live = [s for s in syms if last[s] >= stored_end - pd.Timedelta(days=31)]
    counts = {
        "symbols": len(syms),
        "checked": len(live),
        "appended": 0,
        "refetched": 0,
        "no_new": 0,
    }
    start = (stored_end - pd.Timedelta(days=OVERLAP_DAYS)).date()
    for k in range(0, len(live), GROUP):
        group = live[k : k + GROUP]
        pages = daily_pages(client, pace, group, start, end)
        redo = []
        for sym in group:
            if sym not in pages:
                counts["no_new"] += 1
                continue
            old, new = _read(sym), frame(pages[sym])
            if not matches(old, new):
                redo.append(sym)
                continue
            fresh = new[new.index > old.index.max()]
            if len(fresh):
                _save(sym, pd.concat([old, fresh[old.columns]]))
                counts["appended"] += 1
            else:
                counts["no_new"] += 1
        for sym in redo:
            full = frame(daily_pages(client, pace, [sym], HISTORY_START, end).get(sym, []))
            if len(full):
                _save(sym, full[_read(sym).columns])
                counts["refetched"] += 1
        log_progress(LOG, step="daily", done=min(k + GROUP, len(live)), of=len(live), **counts)
    return counts


def extend_minutes(start: date, end: date) -> dict[str, int]:
    client, pace = _client_and_pace()
    out = {}
    for sym in MINUTE_SYMBOLS:
        params: dict[str, Any] = {
            "symbols": sym,
            "timeframe": "1Min",
            "start": f"{start.isoformat()}T00:00:00Z",
            "end": f"{end.isoformat()}T23:59:59Z",
            "limit": 10000,
            "adjustment": "all",
            "feed": "sip",
            "sort": "asc",
        }
        rows: list[dict[str, Any]] = []
        while True:
            body = get(client, pace, "/v2/stocks/bars", params)
            rows.extend((body.get("bars") or {}).get(sym, []))
            token = body.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        df = pd.DataFrame(rows)
        if len(df):
            df["t"] = pd.to_datetime(df["t"], utc=True)
            df.insert(0, "symbol", sym)
        write_parquet(df, MINUTE_OUT / f"{sym}.parquet")
        out[sym] = len(df)
        log_progress(LOG, step="minute", symbol=sym, bars=len(df), start=str(start), end=str(end))
    return out


def last_full_session(today: date) -> date:
    from stoa import clock

    d = today - timedelta(days=1)
    while not clock.is_session(d):
        d -= timedelta(days=1)
    return d


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    end = date.fromisoformat(argv[1]) if len(argv) > 1 else last_full_session(date.today())
    if argv[0] in ("daily", "all"):
        log.info("daily: %s", extend_daily(end))
    if argv[0] in ("minute", "all"):
        log.info("minute: %s", extend_minutes(date(2026, 10, 1), end))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
