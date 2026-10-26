"""Read-only live positions for the public dashboard (a Vercel Python function).

GET /api/positions returns the paper account's equity, cash, today's change and
every open position, as JSON. It holds the PAPER Alpaca keys as Vercel environment
variables (ALPACA_PAPER_KEY_ID, ALPACA_PAPER_SECRET) and makes exactly two kinds of
request, both GET: /v2/account and /v2/positions on the paper API. There is no
order, cancel or other write call anywhere in this file.

It refuses to run (HTTP 503) if it sees live keys (ALPACA_LIVE_KEY_ID,
ALPACA_LIVE_SECRET), STOA_MODE=live, or a key that is not a paper key.

Responses are cached for CACHE_SECONDS: in memory on a warm instance, and at
Vercel's edge through Cache-Control s-maxage, so dashboard polling costs at most
two Alpaca requests per CACHE_SECONDS however many tabs are open. Only
https://addisontech.github.io may read it from a browser (CORS).

Standard library only, so there is nothing to install.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from http.server import BaseHTTPRequestHandler
from typing import Any
from zoneinfo import ZoneInfo

PAPER_API = "https://paper-api.alpaca.markets"
ALLOWED_ORIGIN = "https://addisontech.github.io"
CACHE_SECONDS = 25
TIMEOUT_SECONDS = 8
GET_PATHS = ("/v2/account", "/v2/positions")  # the only requests this function makes
NY = ZoneInfo("America/New_York")
# NYSE full-day closures (a test checks them against the exchange calendar). Early
# closes (1:00 pm) are not listed; the dashboard keeps polling until 4:00 pm then.
HOLIDAYS = frozenset(
    date.fromisoformat(d)
    for d in (
        *("2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25"),
        *("2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"),
        *("2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31"),
        *("2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"),
    )
)

Fetch = Callable[[str, Mapping[str, str]], Any]


class Refused(RuntimeError):
    """The function will not run with this configuration."""


def paper_keys(env: Mapping[str, str]) -> tuple[str, str]:
    """The paper key pair, or Refused if anything points at a live account."""
    if env.get("STOA_MODE", "").strip().lower() == "live":
        raise Refused("STOA_MODE is live; this function only reads the paper account")
    if env.get("ALPACA_LIVE_KEY_ID") or env.get("ALPACA_LIVE_SECRET"):
        raise Refused("live Alpaca keys are set; remove them, this function is paper only")
    key, secret = env.get("ALPACA_PAPER_KEY_ID", ""), env.get("ALPACA_PAPER_SECRET", "")
    if not key or not secret:
        raise Refused("ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET are not set")
    if not key.startswith("PK"):
        raise Refused("ALPACA_PAPER_KEY_ID is not a paper key (paper keys start with PK)")
    return key, secret


def market_open(now: datetime) -> bool:
    """Regular hours, 9:30 am to 4:00 pm New York time, on NYSE trading days."""
    ny = now.astimezone(NY)
    if ny.weekday() >= 5 or ny.date() in HOLIDAYS:
        return False
    minutes = ny.hour * 60 + ny.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _pct(fraction: Any) -> float | None:
    f = _num(fraction)
    return None if f is None else round(f * 100, 4)


def summarize(
    account: Mapping[str, Any], positions: list[Mapping[str, Any]], now: datetime
) -> dict[str, Any]:
    equity = _num(account.get("equity"))
    last = _num(account.get("last_equity"))
    change = equity - last if equity is not None and last is not None else None
    rows = []
    for p in sorted(positions, key=lambda x: str(x.get("symbol", ""))):
        rows.append(
            {
                "symbol": str(p.get("symbol", "")),
                "qty": _num(p.get("qty")),
                "price": _num(p.get("current_price")),
                "market_value": _num(p.get("market_value")),
                "today_change": _num(p.get("unrealized_intraday_pl")),
                "today_change_pct": _pct(p.get("unrealized_intraday_plpc")),
                "total_pl": _num(p.get("unrealized_pl")),
                "total_pl_pct": _pct(p.get("unrealized_plpc")),
            }
        )
    return {
        "account": "paper",
        "equity": equity,
        "cash": _num(account.get("cash")),
        "today_change": round(change, 2) if change is not None else None,
        "today_change_pct": round(change / last * 100, 4) if change is not None and last else None,
        "positions": rows,
        "market_open": market_open(now),
        "as_of": now.astimezone(UTC).isoformat(timespec="seconds"),
    }


def http_get(url: str, headers: Mapping[str, str]) -> Any:
    req = urllib.request.Request(url, headers=dict(headers), method="GET")
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


class Snapshot:
    """The cached answer: rebuilt at most once per CACHE_SECONDS on this instance."""

    def __init__(
        self, fetch: Fetch = http_get, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.fetch = fetch
        self.clock = clock
        self._body: dict[str, Any] | None = None
        self._at = float("-inf")

    def get(self, env: Mapping[str, str], now: datetime) -> dict[str, Any]:
        key, secret = paper_keys(env)  # checked on every request, cached or not
        if self._body is not None and self.clock() - self._at < CACHE_SECONDS:
            return self._body
        headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        account = self.fetch(PAPER_API + GET_PATHS[0], headers)
        positions = self.fetch(PAPER_API + GET_PATHS[1], headers)
        if not isinstance(account, dict) or not isinstance(positions, list):
            raise RuntimeError("unexpected response from Alpaca")
        self._body = summarize(account, positions, now)
        self._at = self.clock()
        return self._body


SNAPSHOT = Snapshot()


def respond(
    method: str,
    origin: str | None,
    env: Mapping[str, str],
    now: datetime,
    snapshot: Snapshot = SNAPSHOT,
) -> tuple[int, dict[str, str], bytes]:
    """Status, headers and body for one request. Only GET and the CORS preflight."""
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Vary": "Origin",
        "X-Content-Type-Options": "nosniff",
    }
    if origin == ALLOWED_ORIGIN:
        headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
        headers["Access-Control-Allow-Methods"] = "GET"
    if method == "OPTIONS":
        return 204, headers, b""
    if method != "GET":
        headers["Allow"] = "GET"
        return 405, headers, b'{"error":"read only"}'
    try:
        body = snapshot.get(env, now)
    except Refused as exc:
        headers["Cache-Control"] = "no-store"
        return 503, headers, json.dumps({"error": str(exc)}).encode()
    except Exception as exc:  # Alpaca down or slow: the dashboard shows "unavailable"
        headers["Cache-Control"] = "no-store"
        return 502, headers, json.dumps({"error": type(exc).__name__}).encode()
    headers["Cache-Control"] = f"public, max-age=0, s-maxage={CACHE_SECONDS}"
    return 200, headers, json.dumps(body).encode()


class handler(BaseHTTPRequestHandler):  # noqa: N801 (Vercel looks for this name)
    def _send(self, method: str) -> None:
        status, headers, body = respond(
            method, self.headers.get("Origin"), os.environ, datetime.now(UTC)
        )
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        self._send("GET")

    def do_OPTIONS(self) -> None:
        self._send("OPTIONS")

    def do_POST(self) -> None:
        self._send("POST")

    def do_PUT(self) -> None:
        self._send("PUT")

    def do_PATCH(self) -> None:
        self._send("PATCH")

    def do_DELETE(self) -> None:
        self._send("DELETE")
