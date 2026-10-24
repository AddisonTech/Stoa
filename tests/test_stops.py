from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from stoa import alerts, jobs, stops
from stoa.broker.alpaca import AlpacaBroker, parse_order
from stoa.config import ExitConfig, Secrets
from stoa.models import PlannedOrder, PositionMeta, Side
from stoa.planner import client_order_id
from stoa.state import KillSwitch, Plan, ProtectiveStop, State, StateStore
from tests.conftest import SMALL_WATCHLIST, make_settings

D = Decimal
PRIOR = date(2021, 10, 8)
DECIDE_DAY = date(2021, 10, 12)
EXEC_DAY = date(2021, 10, 13)
EXEC_AT = datetime(2021, 10, 13, 13, 50, tzinfo=UTC)
STAMP = "2021-10-13T13:51:00Z"
OPEN = {"new", "accepted", "pending_new", "partially_filled"}


class FakeAlpaca:
    """A tiny stateful Alpaca behind httpx.MockTransport. Records every call in order."""

    def __init__(self, positions: dict[str, tuple[str, str]], prices: dict[str, str]) -> None:
        self.positions = {s: {"qty": q, "avg": a} for s, (q, a) in positions.items()}
        self.prices = prices
        self.orders: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str, str]] = []  # method, path, symbol or client id
        self.fail: set[tuple[str, str]] = set()  # (method, symbol) answered with HTTP 500
        self.reject_market_sells: set[str] = set()

    # -- helpers ---------------------------------------------------------------------------

    def add_order(self, **fields: Any) -> dict[str, Any]:
        order = {
            "id": f"o-{len(self.orders) + 1}",
            "type": "market",
            "time_in_force": "day",
            "status": "new",
            "filled_qty": "0",
            "filled_avg_price": None,
            "submitted_at": STAMP,
            "filled_at": None,
            "qty": None,
            "stop_price": None,
            **fields,
        }
        self.orders.append(order)
        return order

    def by_cid(self, cid: str) -> dict[str, Any] | None:
        return next((o for o in self.orders if o["client_order_id"] == cid), None)

    # -- routing ---------------------------------------------------------------------------

    def handle(self, req: httpx.Request) -> httpx.Response:
        path, method = req.url.path, req.method
        if path == "/v2/account":
            cash = D(1000)
            return httpx.Response(
                200,
                json={"equity": str(cash + self._value()), "cash": str(cash), "status": "ACTIVE"},
            )
        if path == "/v2/clock":
            return httpx.Response(200, json={"is_open": True})
        if path == "/v2/account/activities":
            return httpx.Response(200, json=[])
        if path == "/v2/positions":
            return httpx.Response(200, json=[self._pos(s) for s in sorted(self.positions)])
        if path.startswith("/v2/assets/"):
            return httpx.Response(
                200, json={"tradable": True, "status": "active", "fractionable": True}
            )
        if path == "/v2/stocks/trades/latest":
            syms = req.url.params["symbols"].split(",")
            return httpx.Response(
                200,
                json={
                    "trades": {s: {"p": float(self.prices[s])} for s in syms if s in self.prices}
                },
            )
        if path == "/v2/orders:by_client_order_id":
            cid = req.url.params["client_order_id"]
            o = self.by_cid(cid)
            return httpx.Response(200, json=o) if o else httpx.Response(404, json={"message": "nf"})
        if path == "/v2/orders" and method == "GET":
            self.calls.append(("GET", path, req.url.params.get("status", "")))
            if req.url.params.get("status") == "open":
                return httpx.Response(200, json=[o for o in self.orders if o["status"] in OPEN])
            return httpx.Response(200, json=list(self.orders))
        if path == "/v2/orders" and method == "POST":
            return self._post(json.loads(req.content))
        if path.startswith("/v2/orders/") and method == "DELETE":
            return self._delete(path.rsplit("/", 1)[1])
        raise AssertionError(f"unexpected call {method} {path}")

    def _value(self) -> Decimal:
        return sum((D(p["qty"]) * D(self.prices[s]) for s, p in self.positions.items()), D(0))

    def _pos(self, sym: str) -> dict[str, Any]:
        p = self.positions[sym]
        return {
            "symbol": sym,
            "qty": p["qty"],
            "avg_entry_price": p["avg"],
            "current_price": self.prices[sym],
            "side": "long",
        }

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        sym, cid = body["symbol"], body["client_order_id"]
        self.calls.append(("POST", body["type"], f"{body['side']} {sym}"))
        if ("POST", sym) in self.fail:
            return httpx.Response(500, json={"message": "broker down"})
        if body["type"] == "market" and body["side"] == "sell" and sym in self.reject_market_sells:
            return httpx.Response(422, json={"message": "insufficient qty available"})
        order = self.add_order(
            client_order_id=cid,
            symbol=sym,
            side=body["side"],
            type=body["type"],
            qty=body.get("qty"),
            stop_price=body.get("stop_price"),
        )
        if body["type"] == "market":
            qty = D(body["qty"])
            order.update(
                status="filled",
                filled_qty=str(qty),
                filled_avg_price=self.prices[sym],
                filled_at=STAMP,
            )
            left = D(self.positions[sym]["qty"]) - qty
            if left <= 0:
                del self.positions[sym]
            else:
                self.positions[sym]["qty"] = str(left)
        return httpx.Response(200, json=order)

    def _delete(self, oid: str) -> httpx.Response:
        order = next(o for o in self.orders if o["id"] == oid)
        self.calls.append(("DELETE", "order", f"{order['symbol']} {order['client_order_id']}"))
        if ("DELETE", order["symbol"]) in self.fail:
            return httpx.Response(500, json={"message": "cancel failed"})
        if order["status"] not in OPEN:
            return httpx.Response(422, json={"message": "order is not cancelable"})
        order["status"] = "canceled"
        return httpx.Response(204)


def make_broker(fake: FakeAlpaca) -> AlpacaBroker:
    def client(base: str) -> httpx.Client:
        return httpx.Client(base_url=base, transport=httpx.MockTransport(fake.handle))

    return AlpacaBroker(
        client("https://paper-api.alpaca.markets"),
        client("https://data.alpaca.markets"),
        live=False,
        max_retries=0,
        fill_timeout=5,
        poll_interval=0,
        sleep=lambda _s: None,
    )


def meta(entry: date, price: str, stop: str | None) -> PositionMeta:
    return PositionMeta(
        entry_session=entry,
        entry_price=D(price),
        entry_atr=D(2),
        stop_price=D(stop) if stop else None,
    )


# ---- placing ------------------------------------------------------------------------------


def test_stops_placed_only_for_prior_day_positions() -> None:
    fake = FakeAlpaca(
        {"AAPL": ("1.5", "100"), "MSFT": ("0.25", "200")}, {"AAPL": "104", "MSFT": "205"}
    )
    broker = make_broker(fake)
    state = State(
        positions_meta={"AAPL": meta(PRIOR, "100", "95.01"), "MSFT": meta(EXEC_DAY, "200", "195")}
    )
    run = stops.place_stops(
        broker,
        state,
        broker.positions(),
        EXEC_DAY,
        managed=SMALL_WATCHLIST,
        bought_today=set(),
        prices={"AAPL": D(104), "MSFT": D(205)},
    )
    assert [p.symbol for p in run.placed] == ["AAPL"] and run.alerts == []
    order = fake.by_cid(run.placed[0].client_order_id)
    assert order is not None
    assert order["client_order_id"] == "stoa-stop-20211013-AAPL-1"
    assert (order["type"], order["time_in_force"], order["side"]) == ("stop", "day", "sell")
    assert order["qty"] == "1.500000000" and order["stop_price"] == "95.01"
    assert [p.symbol for p in state.protective_stops] == ["AAPL"]


def test_same_day_buy_gets_no_stop_even_with_an_older_entry() -> None:
    fake = FakeAlpaca({"AAPL": ("1", "100")}, {"AAPL": "104"})
    broker = make_broker(fake)
    state = State(positions_meta={"AAPL": meta(PRIOR, "100", "95")})
    run = stops.place_stops(
        broker,
        state,
        broker.positions(),
        EXEC_DAY,
        managed=SMALL_WATCHLIST,
        bought_today={"AAPL"},
        prices={},
    )
    assert run.placed == [] and not [c for c in fake.calls if c[0] == "POST"]


def test_no_stop_when_price_is_already_below_it() -> None:
    fake = FakeAlpaca({"AAPL": ("1", "100")}, {"AAPL": "94"})
    broker = make_broker(fake)
    state = State(positions_meta={"AAPL": meta(PRIOR, "100", "95")})
    run = stops.place_stops(
        broker,
        state,
        broker.positions(),
        EXEC_DAY,
        managed=SMALL_WATCHLIST,
        bought_today=set(),
        prices={"AAPL": D(94)},
    )
    assert run.placed == [] and run.alerts[0].title == "No stop placed: AAPL below its stop"


def test_an_open_matching_stop_is_kept_and_a_stale_one_replaced() -> None:
    fake = FakeAlpaca({"AAPL": ("1", "100"), "KO": ("2", "50")}, {"AAPL": "104", "KO": "52"})
    fake.add_order(
        client_order_id="stoa-stop-20211013-AAPL-1",
        symbol="AAPL",
        side="sell",
        type="stop",
        qty="1",
        stop_price="95",
    )
    fake.add_order(
        client_order_id="stoa-stop-20211013-KO-1",
        symbol="KO",
        side="sell",
        type="stop",
        qty="2",
        stop_price="45",
    )
    broker = make_broker(fake)
    state = State(
        positions_meta={"AAPL": meta(PRIOR, "100", "95"), "KO": meta(PRIOR, "50", "47.5")}
    )
    run = stops.place_stops(
        broker,
        state,
        broker.positions(),
        EXEC_DAY,
        managed=SMALL_WATCHLIST,
        bought_today=set(),
        prices={},
    )
    assert [p.client_order_id for p in run.kept] == ["stoa-stop-20211013-AAPL-1"]
    assert [p.client_order_id for p in run.placed] == ["stoa-stop-20211013-KO-2"]
    assert fake.by_cid("stoa-stop-20211013-KO-1")["status"] == "canceled"  # type: ignore[index]


def test_only_stoa_stops_are_ever_touched() -> None:
    fake = FakeAlpaca({"AAPL": ("1", "100")}, {"AAPL": "104"})
    fake.add_order(
        client_order_id="my-own-stop",
        symbol="AAPL",
        side="sell",
        type="stop",
        qty="1",
        stop_price="90",
    )
    broker = make_broker(fake)
    assert stops.cancel_stops(broker, ["AAPL"]) == []
    assert fake.by_cid("my-own-stop")["status"] == "new"  # type: ignore[index]
    assert not [c for c in fake.calls if c[0] == "DELETE"]


# ---- failures alert and continue ------------------------------------------------------


def test_place_failure_alerts_urgently_and_keeps_going() -> None:
    fake = FakeAlpaca({"AAPL": ("1", "100"), "KO": ("2", "50")}, {"AAPL": "104", "KO": "52"})
    fake.fail.add(("POST", "AAPL"))
    broker = make_broker(fake)
    state = State(positions_meta={"AAPL": meta(PRIOR, "100", "95"), "KO": meta(PRIOR, "50", "47")})
    run = stops.place_stops(
        broker,
        state,
        broker.positions(),
        EXEC_DAY,
        managed=SMALL_WATCHLIST,
        bought_today=set(),
        prices={},
    )
    assert [a.title for a in run.alerts] == ["Stop place failed: AAPL"]
    assert run.alerts[0].priority == alerts.URGENT
    assert [p.symbol for p in run.placed] == ["KO"]


def test_cancel_failure_alerts_urgently() -> None:
    fake = FakeAlpaca({"AAPL": ("1", "100")}, {"AAPL": "104"})
    fake.add_order(
        client_order_id="stoa-stop-20211012-AAPL-1",
        symbol="AAPL",
        side="sell",
        type="stop",
        qty="1",
        stop_price="95",
    )
    fake.fail.add(("DELETE", "AAPL"))
    out = stops.cancel_stops(make_broker(fake), ["AAPL"])
    assert [a.title for a in out] == ["Stop cancel failed: AAPL"] and out[
        0
    ].priority == alerts.URGENT


# ---- EXECUTE end to end -----------------------------------------------------------------


def settings_for(tmp: Path) -> Any:
    return make_settings(
        watchlist=",".join(SMALL_WATCHLIST),
        mode="paper",
        state__dir=str(tmp / "state"),
        scout__kind="rule",
        strategy__kind="rule",
        crosscheck__enabled="false",
        data__provider="csv",
    )


def plan_selling(symbols: list[str]) -> Plan:
    pid = "p1"
    orders = tuple(
        PlannedOrder(
            symbol=s,
            side=Side.SELL,
            qty=D(1),
            ref_price=D(100),
            reason="trend_break",
            client_order_id=client_order_id(pid, EXEC_DAY, s, Side.SELL),
        )
        for s in symbols
    )
    return Plan(
        plan_id=pid,
        strategy="rule",
        mode="paper",
        decided_session=DECIDE_DAY,
        execute_session=EXEC_DAY,
        created_at=datetime(2021, 10, 12, 22, 20, tzinfo=UTC),
        orders=orders,
        equity_at_decide=D(1500),
    )


def run_execute(tmp: Path, fake: FakeAlpaca, state: State) -> alerts.RecordingNotifier:
    s = settings_for(tmp)
    StateStore(s.state.dir).save(state)
    rec = alerts.RecordingNotifier()
    rt = jobs.Runtime(
        settings=s,
        secrets=Secrets.from_env({"ALPACA_PAPER_KEY_ID": "pk", "ALPACA_PAPER_SECRET": "ps"}),
        notifier=rec,
        now=lambda: EXEC_AT,
        broker=make_broker(fake),
        use_external=False,
        sleep=lambda _s: None,
    )
    res = jobs.execute(rt)
    assert res.code == 0, (res.status, [a.title for a in rec.alerts])
    return rec


def test_execute_cancels_before_selling_then_protects_prior_day_positions(tmp_path: Path) -> None:
    # AAPL is being sold and has an open Stoa stop; KO is kept; MSFT was bought this morning.
    fake = FakeAlpaca(
        {"AAPL": ("1", "100"), "KO": ("2", "50"), "MSFT": ("0.5", "200")},
        {"AAPL": "101", "KO": "52", "MSFT": "201"},
    )
    fake.add_order(
        client_order_id="stoa-stop-20211013-AAPL-1",
        symbol="AAPL",
        side="sell",
        type="stop",
        qty="1",
        stop_price="95",
    )
    fake.add_order(
        client_order_id="stoa-20211013-MSFT-buy-p0",
        symbol="MSFT",
        side="buy",
        status="filled",
        filled_qty="0.5",
        filled_avg_price="200",
    )
    state = State(
        pending_plan=plan_selling(["AAPL"]),
        positions_meta={
            "AAPL": meta(PRIOR, "100", "95"),
            "KO": meta(PRIOR, "50", "47.25"),
            "MSFT": meta(PRIOR, "200", "190"),
        },
    )
    rec = run_execute(tmp_path, fake, state)
    cancel = fake.calls.index(("DELETE", "order", "AAPL stoa-stop-20211013-AAPL-1"))
    sell = fake.calls.index(("POST", "market", "sell AAPL"))
    assert cancel < sell
    stop_posts = [c for c in fake.calls if c[:2] == ("POST", "stop")]
    assert stop_posts == [("POST", "stop", "sell KO")]
    saved = StateStore(settings_for(tmp_path).state.dir).load()
    assert [(p.symbol, p.stop_price) for p in saved.protective_stops] == [("KO", D("47.25"))]
    assert not [a for a in rec.alerts if a.priority == alerts.URGENT]


def test_kill_switch_keeps_protective_stops_in_place(tmp_path: Path) -> None:
    # The kill switch flattens everything; a flatten sell that the broker refuses
    # leaves the position open, and it still gets its protective stop.
    fake = FakeAlpaca({"AAPL": ("1", "100"), "KO": ("2", "50")}, {"AAPL": "101", "KO": "52"})
    fake.reject_market_sells.add("KO")
    state = State(
        pending_plan=plan_selling([]),
        kill_switch=KillSwitch(active=True, reason="drawdown limit"),
        positions_meta={"AAPL": meta(PRIOR, "100", "95"), "KO": meta(PRIOR, "50", "47")},
    )
    run_execute(tmp_path, fake, state)
    posts = [c for c in fake.calls if c[0] == "POST"]
    assert ("POST", "market", "sell AAPL") in posts
    assert posts[-1] == ("POST", "stop", "sell KO")
    assert "AAPL" not in fake.positions and "KO" in fake.positions


def test_execute_survives_a_broken_open_orders_call(tmp_path: Path) -> None:
    fake = FakeAlpaca({"KO": ("2", "50")}, {"KO": "52"})
    real = fake.handle

    def flaky(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v2/orders" and req.url.params.get("status") == "open":
            return httpx.Response(500, json={"message": "down"})
        return real(req)

    fake.handle = flaky  # type: ignore[method-assign]
    state = State(pending_plan=plan_selling([]), positions_meta={"KO": meta(PRIOR, "50", "47")})
    rec = run_execute(tmp_path, fake, state)
    urgent = [a.title for a in rec.alerts if a.priority == alerts.URGENT]
    assert urgent == ["Stop place failed: KO"]
    assert EXEC_DAY in StateStore(settings_for(tmp_path).state.dir).load().executed_sessions


# ---- DECIDE: reconcile and raise ----------------------------------------------------------


def test_reconcile_records_a_stop_fill_as_a_stop_out() -> None:
    fake = FakeAlpaca({"AAPL": ("1", "100")}, {"AAPL": "90", "KO": "44"})
    fake.add_order(
        client_order_id="stoa-stop-20211013-KO-1",
        symbol="KO",
        side="sell",
        type="stop",
        qty="2",
        stop_price="47",
        status="filled",
        filled_qty="2",
        filled_avg_price="46.80",
        filled_at="2021-10-13T17:02:11Z",
    )
    fake.add_order(
        client_order_id="stoa-stop-20211013-AAPL-1",
        symbol="AAPL",
        side="sell",
        type="stop",
        qty="1",
        stop_price="95",
        status="expired",
    )
    broker = make_broker(fake)
    state = State(
        positions_meta={"AAPL": meta(PRIOR, "100", "95"), "KO": meta(PRIOR, "50", "47")},
        protective_stops=[
            ProtectiveStop(
                symbol="KO",
                client_order_id="stoa-stop-20211013-KO-1",
                stop_price=D(47),
                qty=D(2),
                session=EXEC_DAY,
            ),
            ProtectiveStop(
                symbol="AAPL",
                client_order_id="stoa-stop-20211013-AAPL-1",
                stop_price=D(95),
                qty=D(1),
                session=EXEC_DAY,
            ),
        ],
    )
    out = stops.reconcile(
        broker, state, broker.positions(), EXEC_DAY, make_settings().costs, "paper"
    )
    assert [s.symbol for s in out.stop_outs] == ["KO"]
    so = state.stop_outs[0]
    assert so.fill_price == D("46.80") and so.filled_at == datetime(
        2021, 10, 13, 17, 2, 11, tzinfo=UTC
    )
    assert out.trades[0].reason == "stop_out" and out.trades[0].pnl == D("-6.40")
    assert "KO" not in state.positions_meta and "AAPL" in state.positions_meta
    assert state.protective_stops == []
    assert out.alerts[0].title == "Stopped out: KO" and out.cost > 0


def test_reconcile_check_failure_alerts_and_keeps_the_record() -> None:
    fake = FakeAlpaca({}, {})
    real = fake.handle

    def broken(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v2/orders:by_client_order_id":
            return httpx.Response(500, json={"message": "down"})
        return real(req)

    fake.handle = broken  # type: ignore[method-assign]
    ps = ProtectiveStop(
        symbol="KO",
        client_order_id="stoa-stop-20211013-KO-1",
        stop_price=D(47),
        qty=D(2),
        session=EXEC_DAY,
    )
    state = State(protective_stops=[ps])
    out = stops.reconcile(make_broker(fake), state, {}, EXEC_DAY, make_settings().costs, "paper")
    assert out.alerts[0].priority == alerts.URGENT and state.protective_stops == [ps]


def history(highs: list[float], start: date = PRIOR) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(highs))
    return pd.DataFrame({"high": highs, "close": highs}, index=idx)


def test_stops_only_tighten_and_never_go_below_the_engine_stop() -> None:
    cfg = ExitConfig(stop_atr_multiple=2.5, trailing_atr_multiple=3.0)
    m = meta(PRIOR, "100", None)  # ATR 2: initial stop 95, trailing from a 110 high is 104
    state = State(positions_meta={"AAPL": m})
    stops.raise_stops(state, ["AAPL"], {"AAPL": history([100, 101])}, cfg)
    assert state.positions_meta["AAPL"].stop_price == D("95.00")
    stops.raise_stops(state, ["AAPL"], {"AAPL": history([100, 110, 103])}, cfg)
    assert state.positions_meta["AAPL"].stop_price == D("104.00")
    # A lower engine stop later (say the history was trimmed) never loosens it.
    stops.raise_stops(state, ["AAPL"], {"AAPL": history([100, 101])}, cfg)
    assert state.positions_meta["AAPL"].stop_price == D("104.00")


def test_stop_rounds_up_to_the_tick() -> None:
    assert stops.round_stop(D("95.001")) == D("95.01")
    assert stops.round_stop(D("0.12341")) == D("0.1235")


@pytest.mark.parametrize("cid", ["stoa-stop-20211013-KO-1", "stoa-stop-x"])
def test_stoa_stop_prefix(cid: str) -> None:
    assert stops.is_stoa_stop(cid)
    assert not stops.is_stoa_stop("stoa-20211013-KO-sell-p1")


def test_parse_order_reads_stop_fields() -> None:
    st = parse_order(
        {
            "id": "1",
            "client_order_id": "stoa-stop-20211013-KO-1",
            "symbol": "KO",
            "side": "sell",
            "status": "filled",
            "qty": "2.5",
            "stop_price": "47.10",
            "filled_qty": "2.5",
            "filled_avg_price": "46.9",
            "submitted_at": STAMP,
            "filled_at": "2021-10-13T17:02:11Z",
        }
    )
    assert st.qty == D("2.5") and st.stop_price == D("47.10")
    assert st.filled_at == datetime(2021, 10, 13, 17, 2, 11, tzinfo=UTC)
