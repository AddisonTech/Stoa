from __future__ import annotations

import importlib.util
import json
import re
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from types import ModuleType
from typing import Any

import pytest

from stoa import clock
from tests.conftest import ROOT

SOURCE = ROOT / "live" / "api" / "positions.py"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("live_positions", SOURCE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


live = load()
PAPER = {"ALPACA_PAPER_KEY_ID": "PKTEST", "ALPACA_PAPER_SECRET": "secret"}
OPEN = datetime(2026, 10, 26, 15, 0, tzinfo=UTC)  # 11:00 am New York, a Monday
SITE = "https://addisontech.github.io"

ACCOUNT = {"equity": "100.04", "last_equity": "100.00", "cash": "2.05", "status": "ACTIVE"}
POSITIONS = [
    {
        "symbol": "MSFT",
        "qty": "0.037975789",
        "current_price": "526.60",
        "market_value": "19.99",
        "unrealized_intraday_pl": "0.01",
        "unrealized_intraday_plpc": "0.0004",
        "unrealized_pl": "0.01",
        "unrealized_plpc": "0.0004",
    },
    {
        "symbol": "AAPL",
        "qty": "0.0523315",
        "current_price": "344.04",
        "market_value": "18.00",
        "unrealized_intraday_pl": "-0.25",
        "unrealized_intraday_plpc": "-0.0137",
        "unrealized_pl": "-0.25",
        "unrealized_plpc": "-0.0137",
    },
]


class Alpaca:
    """Records every request; answers account and positions only."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.fail = fail

    def __call__(self, url: str, headers: Mapping[str, str]) -> Any:
        self.calls.append((url, dict(headers)))
        if self.fail:
            raise OSError("Alpaca is down")
        if url.endswith("/v2/account"):
            return dict(ACCOUNT)
        if url.endswith("/v2/positions"):
            return [dict(p) for p in POSITIONS]
        raise AssertionError(f"unexpected request {url}")


class Tick:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def snapshot(fetch: Alpaca, tick: Tick | None = None) -> Any:
    return live.Snapshot(fetch, tick or Tick())


def body(resp: tuple[int, dict[str, str], bytes]) -> dict[str, Any]:
    return json.loads(resp[2])


def test_returns_account_and_positions_from_the_paper_api() -> None:
    alpaca = Alpaca()
    status, headers, raw = live.respond("GET", SITE, PAPER, OPEN, snapshot(alpaca))
    assert status == 200
    data = json.loads(raw)
    assert data["account"] == "paper" and data["market_open"] is True
    assert data["equity"] == pytest.approx(100.04) and data["cash"] == pytest.approx(2.05)
    assert data["today_change"] == pytest.approx(0.04)
    assert data["today_change_pct"] == pytest.approx(0.04)
    assert [p["symbol"] for p in data["positions"]] == ["AAPL", "MSFT"]
    aapl = data["positions"][0]
    assert aapl == {
        "symbol": "AAPL",
        "qty": pytest.approx(0.0523315),
        "price": pytest.approx(344.04),
        "market_value": pytest.approx(18.0),
        "today_change": pytest.approx(-0.25),
        "today_change_pct": pytest.approx(-1.37),
        "total_pl": pytest.approx(-0.25),
        "total_pl_pct": pytest.approx(-1.37),
    }
    urls = [u for u, _ in alpaca.calls]
    assert urls == [
        "https://paper-api.alpaca.markets/v2/account",
        "https://paper-api.alpaca.markets/v2/positions",
    ]
    assert alpaca.calls[0][1] == {"APCA-API-KEY-ID": "PKTEST", "APCA-API-SECRET-KEY": "secret"}
    assert headers["Access-Control-Allow-Origin"] == SITE
    assert "s-maxage=25" in headers["Cache-Control"]


def test_cache_holds_25_seconds_so_polling_never_reaches_alpaca() -> None:
    alpaca, tick = Alpaca(), Tick()
    snap = snapshot(alpaca, tick)
    for _ in range(10):
        assert live.respond("GET", SITE, PAPER, OPEN, snap)[0] == 200
        tick.t += 2  # ten polls in 20 seconds
    assert len(alpaca.calls) == 2
    tick.t += 10
    live.respond("GET", SITE, PAPER, OPEN, snap)
    assert len(alpaca.calls) == 4
    assert 20 <= live.CACHE_SECONDS <= 30


@pytest.mark.parametrize(
    "env",
    [
        {**PAPER, "STOA_MODE": "live"},
        {**PAPER, "STOA_MODE": " LIVE "},
        {**PAPER, "ALPACA_LIVE_KEY_ID": "AKREAL"},
        {**PAPER, "ALPACA_LIVE_SECRET": "x"},
        {"ALPACA_PAPER_KEY_ID": "AKREAL", "ALPACA_PAPER_SECRET": "x"},  # a live key in the slot
        {},
    ],
)
def test_refuses_anything_but_paper_keys_before_calling_alpaca(env: dict[str, str]) -> None:
    alpaca = Alpaca()
    status, headers, raw = live.respond("GET", SITE, env, OPEN, snapshot(alpaca))
    assert status == 503 and "error" in json.loads(raw)
    assert headers["Cache-Control"] == "no-store"
    assert alpaca.calls == []


def test_a_cached_answer_is_not_served_once_live_keys_appear() -> None:
    alpaca = Alpaca()
    snap = snapshot(alpaca)
    assert live.respond("GET", SITE, PAPER, OPEN, snap)[0] == 200
    assert live.respond("GET", SITE, {**PAPER, "STOA_MODE": "live"}, OPEN, snap)[0] == 503


def test_cors_is_locked_to_the_dashboard_and_only_get_is_served() -> None:
    alpaca = Alpaca()
    snap = snapshot(alpaca)
    _, headers, _ = live.respond("GET", "https://evil.example", PAPER, OPEN, snap)
    assert "Access-Control-Allow-Origin" not in headers
    _, headers, _ = live.respond("GET", None, PAPER, OPEN, snap)
    assert "Access-Control-Allow-Origin" not in headers
    status, headers, raw = live.respond("OPTIONS", SITE, PAPER, OPEN, snap)
    assert status == 204 and raw == b"" and headers["Access-Control-Allow-Methods"] == "GET"
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        status, headers, _ = live.respond(method, SITE, PAPER, OPEN, snap)
        assert status == 405 and headers["Allow"] == "GET"
    assert alpaca.calls == [] or all(u.endswith(live.GET_PATHS) for u, _ in alpaca.calls)


def test_alpaca_down_is_a_502_the_dashboard_can_survive() -> None:
    status, headers, raw = live.respond("GET", SITE, PAPER, OPEN, snapshot(Alpaca(fail=True)))
    assert status == 502 and json.loads(raw) == {"error": "OSError"}
    assert headers["Cache-Control"] == "no-store"
    assert headers["Access-Control-Allow-Origin"] == SITE


def test_market_hours() -> None:
    ny = datetime(2026, 10, 26, 9, 29)  # naive New York wall times below
    at = [ny.replace(tzinfo=live.NY) + timedelta(minutes=m) for m in (0, 1, 390, 391)]
    assert [live.market_open(t) for t in at] == [False, True, True, False]  # 9:29 to 4:00
    assert not live.market_open(datetime(2026, 10, 24, 15, 0, tzinfo=UTC))  # Saturday
    assert not live.market_open(datetime(2026, 11, 26, 15, 0, tzinfo=UTC))  # Thanksgiving


def test_holidays_match_the_exchange_calendar() -> None:
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(730)]
    closed = {d for d in days if d.weekday() < 5 and not clock.is_session(d)}
    assert closed == set(live.HOLIDAYS)


def test_the_function_has_no_order_cancel_or_write_calls() -> None:
    text = SOURCE.read_text(encoding="utf-8").split('"""', 2)[2]  # the code, not the docstring
    assert re.findall(r'"/v2/[a-z_/]+"', text) == ['"/v2/account"', '"/v2/positions"']
    for word in ("/v2/orders", "orders", "cancel", "liquidat", "api.alpaca.markets/v2"):
        assert word not in text.replace("paper-api.alpaca.markets", "")
    assert 'method="GET"' in text and "urlopen" in text
    assert text.count("Request(") == 1


def test_vercel_serves_api_as_plain_functions() -> None:
    config = json.loads((ROOT / "live" / "vercel.json").read_text(encoding="utf-8"))
    assert config["framework"] is None  # otherwise Vercel wants a Python web app entrypoint
    assert config["functions"]["api/positions.py"]["maxDuration"] <= 15
    # Git pushes that leave live/ alone skip the build (exit 0 means skip). The
    # project's root directory is live/, so "." is live/.
    assert config["ignoreCommand"].startswith("git diff --quiet ")
    assert config["ignoreCommand"].endswith(" HEAD -- .")
