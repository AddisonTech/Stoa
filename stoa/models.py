"""Shared value types. Money and quantities are Decimal end to end.

Features and indicators are floats. Anything that touches cash, quantity or a
price used for accounting is converted once at the boundary with :func:`dec`.
"""

from __future__ import annotations

import math
from datetime import date
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

CENT = Decimal("0.01")
QTY_STEP = Decimal("0.000000001")  # broker accepts up to 9 decimals
ZERO = Decimal(0)


def dec(value: float | int | str | Decimal) -> Decimal:
    """Exact conversion to Decimal. Floats use their shortest repr, so 0.1 stays 0.1.

    numpy floats subclass float but repr as "np.float64(...)", so convert to a
    plain float first.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        f = float(value)
        if not math.isfinite(f):
            raise ValueError(f"non-finite value {value!r}")
        return Decimal(repr(f))
    return Decimal(value)


def cents_down(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_DOWN)


def cents_up(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_CEILING)


def qty_down(value: Decimal) -> Decimal:
    return value.quantize(QTY_STEP, rounding=ROUND_DOWN)


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Candidate(_Model):
    """An idea from the scout. Not an order."""

    symbol: str
    conviction: int
    thesis: str
    score: float
    source: str


class QuantKill(_Model):
    symbol: str
    gate: str
    detail: str


class ExitSignal(_Model):
    symbol: str
    reason: str
    detail: str


class Rejection(_Model):
    symbol: str
    side: Side
    reason: str
    detail: str


class PlannedOrder(_Model):
    """One order in a plan. Buys carry a notional, sells carry a quantity."""

    symbol: str
    side: Side
    notional: Decimal | None = None
    qty: Decimal | None = None
    ref_price: Decimal
    entry_atr: Decimal | None = None
    stop_price: Decimal | None = None
    reason: str
    conviction: int | None = None
    thesis: str = ""
    client_order_id: str = ""


class Fill(_Model):
    symbol: str
    side: Side
    qty: Decimal
    price: Decimal
    fees: Decimal
    session: date
    client_order_id: str

    @property
    def gross(self) -> Decimal:
        return self.qty * self.price


class ClosedTrade(_Model):
    symbol: str
    entry_session: date
    exit_session: date
    qty: Decimal
    entry_price: Decimal
    exit_price: Decimal
    fees: Decimal
    pnl: Decimal
    reason: str

    @property
    def return_pct(self) -> float:
        cost = self.qty * self.entry_price
        return float(self.pnl / cost) if cost > 0 else 0.0


class PositionMeta(_Model):
    """What Stoa remembers about an entry. The broker owns the quantity."""

    entry_session: date
    entry_price: Decimal
    entry_atr: Decimal


class Holding(_Model):
    """A position as seen at decision time: broker quantity plus Stoa's entry metadata."""

    symbol: str
    qty: Decimal
    avg_entry_price: Decimal
    meta: PositionMeta
