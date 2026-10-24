"""Alpaca REST client for paper and live trading, built on httpx directly.

Orders are notional (buys) or quantity (sells) market orders with time in
force DAY, the only time in force Alpaca allows for fractional orders. Every
order has a deterministic client order id, and the client looks it up before
submitting, so a retried job can never double-submit.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import ROUND_DOWN, Decimal
from typing import Any

import httpx

from stoa.broker.base import Account, BrokerError, BrokerPosition, OrderRequest, OrderStatus
from stoa.clock import NY
from stoa.models import ZERO, Side, dec
from stoa.net import request_with_retry
from stoa.universe import AssetInfo

log = logging.getLogger(__name__)

QTY_PLACES = 9
NOTIONAL_PLACES = 2
TRANSFER_ACTIVITY_TYPES = ("TRANS", "CSD", "CSW")


def _d(value: Any) -> Decimal:
    if value is None or value == "":
        return ZERO
    return Decimal(str(value))


def _fmt(value: Decimal, places: int) -> str:
    """Round toward zero so a sell can never ask for more than is held."""
    q = Decimal(1).scaleb(-places)
    return format(value.quantize(q, rounding=ROUND_DOWN), "f")


def _session_of(stamp: str | None) -> date | None:
    if not stamp:
        return None
    ts = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return ts.astimezone(NY).date()


def parse_order(body: dict[str, Any]) -> OrderStatus:
    filled_qty = _d(body.get("filled_qty"))
    avg = body.get("filled_avg_price")
    return OrderStatus(
        client_order_id=str(body.get("client_order_id", "")),
        broker_order_id=str(body.get("id", "")),
        symbol=str(body.get("symbol", "")),
        side=Side(str(body.get("side", "buy"))),
        status=str(body.get("status", "")),
        filled_qty=filled_qty,
        filled_avg_price=_d(avg) if avg not in (None, "") else None,
        fees=ZERO,
        session=_session_of(body.get("submitted_at") or body.get("created_at")),
    )


class AlpacaBroker:
    def __init__(
        self,
        client: httpx.Client,
        data_client: httpx.Client,
        *,
        live: bool,
        quote_feed: str = "iex",
        max_retries: int = 4,
        fill_timeout: float = 90.0,
        poll_interval: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.data_client = data_client
        self.live = live
        self.name = "alpaca-live" if live else "alpaca-paper"
        self.quote_feed = quote_feed
        self.max_retries = max_retries
        self.fill_timeout = fill_timeout
        self.poll_interval = poll_interval
        self.sleep = sleep
        self.monotonic = monotonic

    # -- plumbing ---------------------------------------------------------------

    def _call(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        data: bool = False,
    ) -> httpx.Response:
        client = self.data_client if data else self.client
        return request_with_retry(
            client,
            method,
            path,
            params=params,
            json=json,
            max_retries=self.max_retries,
            sleep=self.sleep,
        )

    def _json(self, resp: httpx.Response, what: str) -> Any:
        if resp.status_code >= 400:
            raise BrokerError(f"{what}: HTTP {resp.status_code}: {_error_text(resp)}")
        return resp.json()

    # -- account and market -----------------------------------------------------

    def account(self) -> Account:
        body = self._json(self._call("GET", "/v2/account"), "account")
        return Account(
            equity=_d(body.get("equity")),
            cash=_d(body.get("cash")),
            last_equity=_d(body.get("last_equity")) if body.get("last_equity") else None,
            status=str(body.get("status", "")),
            trading_blocked=bool(body.get("trading_blocked") or body.get("account_blocked")),
        )

    def positions(self) -> dict[str, BrokerPosition]:
        rows = self._json(self._call("GET", "/v2/positions"), "positions")
        out = {}
        for r in rows:
            if str(r.get("side", "long")) != "long":
                raise BrokerError(f"unexpected short position in {r.get('symbol')}")
            sym = str(r["symbol"])
            out[sym] = BrokerPosition(
                symbol=sym,
                qty=_d(r.get("qty")),
                avg_entry_price=_d(r.get("avg_entry_price")),
                current_price=_d(r.get("current_price")),
            )
        return out

    def asset(self, symbol: str) -> AssetInfo:
        resp = self._call("GET", f"/v2/assets/{symbol}")
        if resp.status_code == 404:
            return AssetInfo(symbol, tradable=False, fractionable=False)
        body = self._json(resp, f"asset {symbol}")
        return AssetInfo(
            symbol,
            tradable=bool(body.get("tradable")) and str(body.get("status")) == "active",
            fractionable=bool(body.get("fractionable")),
        )

    def latest_prices(self, symbols: list[str]) -> dict[str, Decimal]:
        if not symbols:
            return {}
        params = {"symbols": ",".join(symbols), "feed": self.quote_feed}
        body = self._json(
            self._call("GET", "/v2/stocks/trades/latest", params=params, data=True), "latest trades"
        )
        trades = body.get("trades") or {}
        out = {s: dec(float(t["p"])) for s, t in trades.items() if t and float(t.get("p", 0)) > 0}
        missing = [s for s in symbols if s not in out]
        if missing:
            log.warning("no latest trade for %s", missing)
        return out

    def market_is_open(self) -> bool:
        body = self._json(self._call("GET", "/v2/clock"), "clock")
        return bool(body.get("is_open"))

    def transfer_activities(self, after: date | None = None) -> list[dict[str, Any]]:
        """Cash deposits and withdrawals, used to cross-check the money plan."""
        params: dict[str, Any] = {
            "activity_types": ",".join(TRANSFER_ACTIVITY_TYPES),
            "direction": "asc",
        }
        if after is not None:
            params["after"] = after.isoformat()
        body = self._json(self._call("GET", "/v2/account/activities", params=params), "activities")
        return [b for b in body if isinstance(b, dict)]

    # -- orders -------------------------------------------------------------------

    def find_order(self, client_order_id: str) -> OrderStatus | None:
        resp = self._call(
            "GET", "/v2/orders:by_client_order_id", params={"client_order_id": client_order_id}
        )
        if resp.status_code == 404:
            return None
        return parse_order(self._json(resp, "order lookup"))

    def submit(self, request: OrderRequest) -> OrderStatus:
        existing = self.find_order(request.client_order_id)
        if existing is not None:
            return existing
        body: dict[str, Any] = {
            "symbol": request.symbol,
            "side": request.side.value,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": request.client_order_id,
        }
        if request.notional is not None:
            body["notional"] = _fmt(request.notional, NOTIONAL_PLACES)
        else:
            assert request.qty is not None
            body["qty"] = _fmt(request.qty, QTY_PLACES)
        resp = self._call("POST", "/v2/orders", json=body)
        if resp.status_code == 422 and "client_order_id" in resp.text.lower():
            # Raced with an earlier attempt that did reach the broker.
            found = self.find_order(request.client_order_id)
            if found is not None:
                return found
        if resp.status_code in (403, 422):
            return OrderStatus(
                client_order_id=request.client_order_id,
                broker_order_id="",
                symbol=request.symbol,
                side=request.side,
                status="rejected",
                detail=f"HTTP {resp.status_code}: {_error_text(resp)}",
            )
        return parse_order(self._json(resp, f"submit {request.client_order_id}"))

    def await_terminal(self, client_order_ids: list[str]) -> list[OrderStatus]:
        deadline = self.monotonic() + self.fill_timeout
        latest: dict[str, OrderStatus] = {}
        pending = list(client_order_ids)
        while pending:
            still = []
            for cid in pending:
                status = self.find_order(cid)
                if status is None:
                    continue
                latest[cid] = status
                if not status.terminal:
                    still.append(cid)
            pending = still
            if not pending or self.monotonic() >= deadline:
                break
            self.sleep(self.poll_interval)
        return [latest[c] for c in client_order_ids if c in latest]

    def orders_for_session(self, session: date) -> list[OrderStatus]:
        start = datetime.combine(session, datetime.min.time(), tzinfo=NY).astimezone(UTC)
        params = {
            "status": "all",
            "after": start.isoformat().replace("+00:00", "Z"),
            "limit": 500,
            "direction": "asc",
        }
        rows = self._json(self._call("GET", "/v2/orders", params=params), "orders")
        return [o for o in (parse_order(r) for r in rows) if o.session == session]


def _error_text(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and "message" in body:
            return str(body["message"])[:300]
    except ValueError:
        pass
    return resp.text[:300]


def make_trading_client(
    base_url: str, key_id: str, secret: str, timeout: float = 30.0
) -> httpx.Client:
    return httpx.Client(
        base_url=base_url,
        headers={"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret},
        timeout=timeout,
    )
