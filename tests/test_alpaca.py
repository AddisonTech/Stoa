from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal

import httpx
import pytest

from stoa.broker.alpaca import AlpacaBroker
from stoa.broker.base import BrokerError, OrderRequest
from stoa.data.alpaca import AlpacaDataProvider
from stoa.models import Side

D = Decimal
Handler = Callable[[httpx.Request], httpx.Response]


def client(handler: Handler, base: str = "https://paper-api.alpaca.markets") -> httpx.Client:
    return httpx.Client(
        base_url=base,
        transport=httpx.MockTransport(handler),
        headers={"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"},
    )


def broker(handler: Handler, data: Handler | None = None) -> AlpacaBroker:
    return AlpacaBroker(
        client(handler),
        client(data or handler, "https://data.alpaca.markets"),
        live=False,
        max_retries=3,
        fill_timeout=5,
        poll_interval=0,
        sleep=lambda _s: None,
    )


ORDER = {
    "id": "b-1",
    "client_order_id": "stoa-20260302-NVDA-buy-abc",
    "symbol": "NVDA",
    "side": "buy",
    "status": "filled",
    "filled_qty": "0.123456789",
    "filled_avg_price": "160.10",
    "submitted_at": "2026-03-02T14:46:01Z",
}


def test_account_and_headers() -> None:
    seen: list[httpx.Request] = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(
            200, json={"equity": "101.25", "cash": "50.5", "last_equity": "100", "status": "ACTIVE"}
        )

    acct = broker(h).account()
    assert acct.equity == D("101.25") and acct.cash == D("50.5") and acct.last_equity == D(100)
    assert seen[0].url.path == "/v2/account"
    assert seen[0].headers["APCA-API-KEY-ID"] == "k"


def test_positions_parsed_and_short_rejected() -> None:
    rows = [
        {
            "symbol": "AAPL",
            "qty": "0.5",
            "avg_entry_price": "180",
            "current_price": "190",
            "side": "long",
        }
    ]
    pos = broker(lambda r: httpx.Response(200, json=rows)).positions()
    assert pos["AAPL"].qty == D("0.5") and pos["AAPL"].market_value == D(95)
    short = [{**rows[0], "side": "short"}]
    with pytest.raises(BrokerError):
        broker(lambda r: httpx.Response(200, json=short)).positions()


def test_asset_flags() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/ZZZ"):
            return httpx.Response(404, json={"message": "not found"})
        return httpx.Response(
            200, json={"tradable": True, "status": "active", "fractionable": False}
        )

    b = broker(h)
    assert b.asset("BRK.A").tradable and not b.asset("BRK.A").fractionable
    assert not b.asset("ZZZ").eligible


def test_submit_notional_order_body() -> None:
    bodies: list[dict[str, object]] = []

    def h(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v2/orders:by_client_order_id":
            return httpx.Response(404, json={"message": "not found"})
        bodies.append(json.loads(req.content))
        return httpx.Response(
            200, json={**ORDER, "status": "accepted", "filled_qty": "0", "filled_avg_price": None}
        )

    st = broker(h).submit(
        OrderRequest(
            symbol="NVDA",
            side=Side.BUY,
            client_order_id=ORDER["client_order_id"],
            notional=D("19.4"),
        )
    )
    assert bodies == [
        {
            "symbol": "NVDA",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": ORDER["client_order_id"],
            "notional": "19.40",
        }
    ]
    assert st.status == "accepted" and st.filled_avg_price is None


def test_submit_qty_sell_rounds_down_to_nine_decimals() -> None:
    bodies: list[dict[str, object]] = []

    def h(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            return httpx.Response(404)
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json={**ORDER, "side": "sell"})

    broker(h).submit(
        OrderRequest(symbol="NVDA", side=Side.SELL, client_order_id="c", qty=D("0.1234567899"))
    )
    assert bodies[0]["qty"] == "0.123456789" and "notional" not in bodies[0]


def test_idempotent_client_order_id_never_resubmits() -> None:
    posts: list[httpx.Request] = []

    def h(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            posts.append(req)
            return httpx.Response(200, json=ORDER)
        assert req.url.params["client_order_id"] == ORDER["client_order_id"]
        return httpx.Response(200, json=ORDER)

    st = broker(h).submit(
        OrderRequest(
            symbol="NVDA", side=Side.BUY, client_order_id=ORDER["client_order_id"], notional=D(20)
        )
    )
    assert posts == []
    assert st.filled_qty == D("0.123456789") and st.session == date(2026, 3, 2)


def test_duplicate_client_order_id_race_resolves_to_existing() -> None:
    calls = {"get": 0}

    def h(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            return httpx.Response(422, json={"message": "client_order_id must be unique"})
        calls["get"] += 1
        return httpx.Response(404) if calls["get"] == 1 else httpx.Response(200, json=ORDER)

    st = broker(h).submit(
        OrderRequest(
            symbol="NVDA", side=Side.BUY, client_order_id=ORDER["client_order_id"], notional=D(20)
        )
    )
    assert st.status == "filled"


def test_rejected_order_reported_not_raised() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            return httpx.Response(404)
        return httpx.Response(403, json={"message": "insufficient buying power"})

    st = broker(h).submit(
        OrderRequest(symbol="NVDA", side=Side.BUY, client_order_id="c", notional=D(20))
    )
    assert st.status == "rejected" and "insufficient buying power" in st.detail


@pytest.mark.parametrize("status", [429, 503])
def test_retry_on_429_and_503(status: int) -> None:
    calls: list[int] = []

    def h(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(status, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"equity": "1", "cash": "1"})

    assert broker(h).account().equity == D(1)
    assert len(calls) == 3


@pytest.mark.parametrize("status", [400, 403])
def test_no_retry_on_400_and_403(status: int) -> None:
    calls: list[int] = []

    def h(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status, json={"message": "nope"})

    with pytest.raises(BrokerError, match=str(status)):
        broker(h).account()
    assert len(calls) == 1


def test_retries_give_up() -> None:
    calls: list[int] = []

    def h(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(BrokerError):
        broker(h).account()
    assert len(calls) == 4


def test_await_terminal_polls_until_filled() -> None:
    states = iter(["new", "partially_filled", "filled"])

    def h(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={**ORDER, "status": next(states)})

    out = broker(h).await_terminal([ORDER["client_order_id"]])
    assert out[0].status == "filled"


def test_latest_prices_from_data_api() -> None:
    def data(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v2/stocks/trades/latest"
        assert req.url.params["feed"] == "iex"
        return httpx.Response(200, json={"trades": {"AAPL": {"p": 190.5}, "MSFT": {"p": 0}}})

    prices = broker(lambda r: httpx.Response(500), data).latest_prices(["AAPL", "MSFT"])
    assert prices == {"AAPL": D("190.5")}


def test_transfer_activities_query() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        assert req.url.params["activity_types"] == "TRANS,CSD,CSW"
        return httpx.Response(
            200,
            json=[{"id": "a1", "activity_type": "CSW", "net_amount": "-100", "date": "2026-05-01"}],
        )

    assert broker(h).transfer_activities()[0]["id"] == "a1"


# ---- market data ------------------------------------------------------------------


def bar(t: str, c: float) -> dict[str, object]:
    return {"t": t, "o": c, "h": c + 1, "l": c - 1, "c": c, "v": 1000}


def provider(handler: Handler) -> AlpacaDataProvider:
    return AlpacaDataProvider(
        client(handler, "https://data.alpaca.markets"),
        sleep=lambda _s: None,
        now=lambda: datetime(2026, 3, 5, 23, 0, tzinfo=UTC),
    )


def test_bars_pagination_and_params() -> None:
    seen: list[httpx.Request] = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if "page_token" not in req.url.params:
            return httpx.Response(
                200,
                json={"bars": {"SPY": [bar("2026-03-02T05:00:00Z", 500)]}, "next_page_token": "p2"},
            )
        return httpx.Response(
            200,
            json={
                "bars": {
                    "SPY": [bar("2026-03-03T05:00:00Z", 501)],
                    "AAPL": [bar("2026-03-03T05:00:00Z", 200)],
                },
                "next_page_token": None,
            },
        )

    bars = provider(h).fetch(["SPY", "AAPL"], date(2026, 3, 1), date(2026, 3, 3))
    assert len(seen) == 2
    p = seen[0].url.params
    assert p["timeframe"] == "1Day" and p["adjustment"] == "all" and p["feed"] == "sip"
    assert seen[1].url.params["page_token"] == "p2"
    assert list(bars["SPY"]["close"]) == [500, 501]
    assert bars["SPY"].index[0].date() == date(2026, 3, 2)


def test_bars_end_respects_sip_delay() -> None:
    seen: list[httpx.Request] = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"bars": {"SPY": [bar("2026-03-05T05:00:00Z", 500)]}})

    provider(h).fetch(["SPY"], date(2026, 3, 1), date(2026, 3, 9))
    assert seen[0].url.params["end"] == "2026-03-05T22:44:00Z"


def test_bars_fall_back_to_iex() -> None:
    feeds: list[str] = []

    def h(req: httpx.Request) -> httpx.Response:
        feeds.append(req.url.params["feed"])
        if req.url.params["feed"] == "sip":
            return httpx.Response(
                403, json={"message": "subscription does not permit querying recent SIP data"}
            )
        return httpx.Response(200, json={"bars": {"SPY": [bar("2026-03-02T05:00:00Z", 500)]}})

    prov = provider(h)
    prov.fetch(["SPY"], date(2026, 3, 1), date(2026, 3, 3))
    assert feeds == ["sip", "iex"] and prov.last_feed_used == "iex"


def test_bars_missing_symbol_is_reported_not_fatal() -> None:
    """A delisted name returns nothing; the caller decides whether that is fatal."""

    def h(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bars": {"SPY": [bar("2026-03-02T05:00:00Z", 500)]}})

    p = provider(h)
    out = p.fetch(["SPY", "AAPL"], date(2026, 3, 1), date(2026, 3, 3))
    assert set(out) == {"SPY"} and p.last_missing == ["AAPL"]


def test_bars_retry_429() -> None:
    calls: list[int] = []

    def h(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"bars": {"SPY": [bar("2026-03-02T05:00:00Z", 500)]}})

    provider(h).fetch(["SPY"], date(2026, 3, 1), date(2026, 3, 3))
    assert len(calls) == 2
