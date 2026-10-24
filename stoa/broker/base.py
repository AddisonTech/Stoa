"""The broker interface shared by the simulator and Alpaca."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from stoa.models import ZERO, Side
from stoa.universe import AssetInfo

TERMINAL_STATUSES = frozenset(
    {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced"}
)


class BrokerError(RuntimeError):
    """The broker refused a request or answered with something unusable."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Account(_Model):
    equity: Decimal
    cash: Decimal
    buying_power: Decimal | None = None
    last_equity: Decimal | None = None
    status: str = "ACTIVE"
    trading_blocked: bool = False


class BrokerPosition(_Model):
    symbol: str
    qty: Decimal
    avg_entry_price: Decimal
    current_price: Decimal

    @property
    def market_value(self) -> Decimal:
        return self.qty * self.current_price


class OrderRequest(_Model):
    """A notional or quantity market order, time in force DAY. Exactly one of the two."""

    symbol: str
    side: Side
    client_order_id: str
    notional: Decimal | None = None
    qty: Decimal | None = None
    reason: str = ""  # kept locally for the journal, never sent to the broker

    @model_validator(mode="after")
    def _one_of(self) -> OrderRequest:
        if (self.notional is None) == (self.qty is None):
            raise ValueError("set exactly one of notional or qty")
        amount = self.notional if self.notional is not None else self.qty
        if amount is None or amount <= 0:
            raise ValueError("order amount must be positive")
        if self.side is Side.SELL and self.notional is not None:
            raise ValueError("Stoa sells by quantity so a position closes exactly")
        if not self.client_order_id or len(self.client_order_id) > 128:
            raise ValueError("client_order_id must be 1 to 128 characters")
        return self


class OrderStatus(_Model):
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: Side
    status: str
    filled_qty: Decimal = ZERO
    filled_avg_price: Decimal | None = None
    fees: Decimal = ZERO
    session: date | None = None
    detail: str = ""

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def filled(self) -> bool:
        return self.filled_qty > 0 and self.filled_avg_price is not None


class Broker(Protocol):
    name: str

    def account(self) -> Account: ...

    def positions(self) -> dict[str, BrokerPosition]: ...

    def asset(self, symbol: str) -> AssetInfo: ...

    def latest_prices(self, symbols: list[str]) -> dict[str, Decimal]: ...

    def find_order(self, client_order_id: str) -> OrderStatus | None: ...

    def submit(self, request: OrderRequest) -> OrderStatus: ...

    def await_terminal(self, client_order_ids: list[str]) -> list[OrderStatus]: ...

    def orders_for_session(self, session: date) -> list[OrderStatus]: ...
