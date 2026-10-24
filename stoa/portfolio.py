"""Positions, cash and equity with exact Decimal accounting.

Invariants, enforced here and by tests:
- cash is never negative
- equity == cash + market value
- fees and slippage are applied exactly once, at fill time
- fractional quantities are carried at full precision
- a symbol is never both bought and sold in the same session
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from stoa.models import ZERO, ClosedTrade, Fill, Side


class AccountingError(RuntimeError):
    """An operation would break an accounting invariant."""


@dataclass
class Lot:
    qty: Decimal
    avg_cost: Decimal  # per share cost basis: fill price (slippage included) plus buy fees
    entry_session: date
    realized: Decimal = ZERO  # profit booked by sells of this lot so far
    sell_fees: Decimal = ZERO
    sold_qty: Decimal = ZERO
    sold_value: Decimal = ZERO


@dataclass
class Portfolio:
    cash: Decimal
    positions: dict[str, Lot] = field(default_factory=dict)
    realized_pnl: Decimal = ZERO
    fees_paid: Decimal = ZERO
    _traded: dict[date, dict[str, set[Side]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cash < 0:
            raise AccountingError("starting cash cannot be negative")

    def qty(self, symbol: str) -> Decimal:
        lot = self.positions.get(symbol)
        return lot.qty if lot else ZERO

    def market_value(self, prices: Mapping[str, Decimal]) -> Decimal:
        total = ZERO
        for sym, lot in self.positions.items():
            if sym not in prices:
                raise AccountingError(f"no price for held symbol {sym}")
            total += lot.qty * prices[sym]
        return total

    def equity(self, prices: Mapping[str, Decimal]) -> Decimal:
        return self.cash + self.market_value(prices)

    def sides_traded(self, session: date, symbol: str) -> set[Side]:
        return set(self._traded.get(session, {}).get(symbol, set()))

    def apply_fill(self, fill: Fill, reason: str = "") -> ClosedTrade | None:
        """Apply a fill. Returns the closed trade when a sell closes a position."""
        if fill.qty <= 0 or fill.price <= 0 or fill.fees < 0:
            raise AccountingError(f"invalid fill {fill}")
        opposite = Side.SELL if fill.side is Side.BUY else Side.BUY
        if opposite in self.sides_traded(fill.session, fill.symbol):
            raise AccountingError(f"same-day round trip in {fill.symbol} on {fill.session}")
        if fill.side is Side.BUY:
            closed = None
            self._buy(fill)
        else:
            closed = self._sell(fill, reason)
        self._traded.setdefault(fill.session, {}).setdefault(fill.symbol, set()).add(fill.side)
        return closed

    def _buy(self, fill: Fill) -> None:
        cost = fill.gross + fill.fees
        if cost > self.cash:
            raise AccountingError(f"buy {fill.symbol} costs {cost} but cash is {self.cash}")
        self.cash -= cost
        self.fees_paid += fill.fees
        lot = self.positions.get(fill.symbol)
        if lot is None:
            self.positions[fill.symbol] = Lot(fill.qty, cost / fill.qty, fill.session)
        else:
            total = lot.qty + fill.qty
            lot.avg_cost = (lot.qty * lot.avg_cost + cost) / total
            lot.qty = total

    def _sell(self, fill: Fill, reason: str) -> ClosedTrade | None:
        lot = self.positions.get(fill.symbol)
        if lot is None or fill.qty > lot.qty:
            held = lot.qty if lot else ZERO
            raise AccountingError(f"sell {fill.qty} {fill.symbol} but hold {held}; no shorting")
        proceeds = fill.gross - fill.fees
        if self.cash + proceeds < 0:
            raise AccountingError("fees exceed cash plus proceeds")
        self.cash += proceeds
        self.fees_paid += fill.fees
        pnl = fill.qty * (fill.price - lot.avg_cost) - fill.fees
        self.realized_pnl += pnl
        lot.realized += pnl
        lot.sell_fees += fill.fees
        lot.sold_qty += fill.qty
        lot.sold_value += fill.gross
        lot.qty -= fill.qty
        if lot.qty == 0:
            del self.positions[fill.symbol]
            return ClosedTrade(
                symbol=fill.symbol,
                entry_session=lot.entry_session,
                exit_session=fill.session,
                qty=lot.sold_qty,
                entry_price=lot.avg_cost,
                exit_price=lot.sold_value / lot.sold_qty,
                fees=lot.sell_fees,
                pnl=lot.realized,
                reason=reason,
            )
        return None
