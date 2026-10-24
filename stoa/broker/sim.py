"""Simulated broker with exact accounting.

Fills market orders at the session reference price (the open in backtests) plus
slippage against the trader on each side, and charges regulatory fees on sells.
It applies the same rejections a real broker would: no borrowing, no shorting,
exactly one of notional or quantity.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from decimal import Decimal

from stoa.broker.base import Account, BrokerError, BrokerPosition, OrderRequest, OrderStatus
from stoa.config import CostConfig
from stoa.models import ZERO, ClosedTrade, Fill, Side, cents_up, dec, qty_down
from stoa.portfolio import Portfolio
from stoa.universe import AssetInfo, Universe

BPS = Decimal(10000)


def buy_fill_price(ref: Decimal, slippage_bps: float) -> Decimal:
    return ref * (1 + dec(slippage_bps) / BPS)


def sell_fill_price(ref: Decimal, slippage_bps: float) -> Decimal:
    return ref * (1 - dec(slippage_bps) / BPS)


def sell_fees(gross: Decimal, costs: CostConfig) -> Decimal:
    if gross <= 0:
        return ZERO
    fee = cents_up(gross * dec(costs.sell_fee_bps) / BPS)
    return max(fee, costs.min_sell_fee_usd)


Slippage = Callable[[str, date | None], float]  # bps one way, by symbol and session


class SimBroker:
    name = "sim"

    def __init__(
        self,
        portfolio: Portfolio,
        costs: CostConfig,
        universe: Universe,
        slippage: Slippage | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.costs = costs
        self.universe = universe
        self.slippage = slippage  # per symbol and session; None means costs.slippage_bps
        self.session: date | None = None
        self.prices: dict[str, Decimal] = {}
        self.tradable_today: frozenset[str] = frozenset()
        self.orders: dict[str, OrderStatus] = {}
        self.fills: list[Fill] = []
        self.closed_trades: list[ClosedTrade] = []
        self._seq = 0

    def begin_session(self, session: date, prices: dict[str, Decimal]) -> None:
        """Start a session. Orders fill only against symbols priced for this session.

        Symbols without a fresh price keep their last price for valuation only.
        """
        self.session = session
        self.prices = {**self.prices, **prices}
        self.tradable_today = frozenset(prices)

    def mark(self, prices: dict[str, Decimal]) -> None:
        """Update valuation prices without changing the session (for example the close)."""
        self.prices = {**self.prices, **prices}

    def account(self) -> Account:
        equity = self.portfolio.equity(self.prices)
        return Account(equity=equity, cash=self.portfolio.cash)

    def positions(self) -> dict[str, BrokerPosition]:
        return {
            sym: BrokerPosition(
                symbol=sym,
                qty=lot.qty,
                avg_entry_price=lot.avg_cost,
                current_price=self.prices[sym],
            )
            for sym, lot in self.portfolio.positions.items()
        }

    def asset(self, symbol: str) -> AssetInfo:
        info = self.universe.assets.get(symbol)
        if info is None:
            return AssetInfo(symbol, tradable=False, fractionable=False)
        return info

    def latest_prices(self, symbols: list[str]) -> dict[str, Decimal]:
        """Prices for symbols that trade this session. Like Alpaca, a missing symbol
        is left out instead of failing the whole request; the risk check then
        skips that buy for having no live price."""
        return {s: self.prices[s] for s in symbols if s in self.tradable_today}

    def find_order(self, client_order_id: str) -> OrderStatus | None:
        return self.orders.get(client_order_id)

    def submit(self, request: OrderRequest) -> OrderStatus:
        existing = self.orders.get(request.client_order_id)
        if existing is not None:
            return existing
        if self.session is None:
            raise BrokerError("sim broker has no session")
        status = self._fill(request, self.session)
        self.orders[request.client_order_id] = status
        return status

    def await_terminal(self, client_order_ids: list[str]) -> list[OrderStatus]:
        return [self.orders[c] for c in client_order_ids if c in self.orders]

    def orders_for_session(self, session: date) -> list[OrderStatus]:
        return [o for o in self.orders.values() if o.session == session]

    def force_close(self, symbol: str, price: Decimal, session: date, reason: str) -> Fill | None:
        """Close a position in a symbol that no longer trades, at its last close less slippage.

        Models a delisting or a cash takeover in backtests: the money comes back
        at the last traded price. Returns the fill, or None when not held.
        """
        qty = self.portfolio.qty(symbol)
        if qty <= 0:
            return None
        px = sell_fill_price(price, self._slip(symbol))
        fill = Fill(
            symbol=symbol,
            side=Side.SELL,
            qty=qty,
            price=px,
            fees=sell_fees(qty * px, self.costs),
            session=session,
            client_order_id=f"delist-{symbol}-{session:%Y%m%d}",
        )
        closed = self.portfolio.apply_fill(fill, reason)
        self.fills.append(fill)
        if closed is not None:
            self.closed_trades.append(closed)
        return fill

    def _slip(self, symbol: str) -> float:
        if self.slippage is None:
            return self.costs.slippage_bps
        return self.slippage(symbol, self.session)

    def _next_id(self) -> str:
        self._seq += 1
        return f"sim-{self._seq}"

    def _reject(self, request: OrderRequest, session: date, detail: str) -> OrderStatus:
        return OrderStatus(
            client_order_id=request.client_order_id,
            broker_order_id=self._next_id(),
            symbol=request.symbol,
            side=request.side,
            status="rejected",
            session=session,
            detail=detail,
        )

    def _fill(self, request: OrderRequest, session: date) -> OrderStatus:
        ref = self.prices.get(request.symbol)
        if request.symbol not in self.tradable_today or ref is None or ref <= 0:
            return self._reject(request, session, "no price this session")
        info = self.asset(request.symbol)
        if not info.tradable:
            return self._reject(request, session, "asset not tradable")
        if request.side is Side.BUY:
            price = buy_fill_price(ref, self._slip(request.symbol))
            if request.notional is not None:
                if not info.fractionable:
                    return self._reject(request, session, "asset not fractionable")
                if request.notional > self.portfolio.cash:
                    return self._reject(request, session, "insufficient buying power")
                qty = qty_down(request.notional / price)
            else:
                assert request.qty is not None
                qty = request.qty
                if qty * price > self.portfolio.cash:
                    return self._reject(request, session, "insufficient buying power")
            fees = ZERO
        else:
            assert request.qty is not None
            qty = request.qty
            held = self.portfolio.qty(request.symbol)
            if qty > held:
                return self._reject(request, session, "insufficient qty; no shorting")
            if qty != qty.to_integral_value() and not info.fractionable:
                return self._reject(request, session, "fractional sell of non-fractionable asset")
            price = sell_fill_price(ref, self._slip(request.symbol))
            fees = sell_fees(qty * price, self.costs)
        if qty <= 0:
            return self._reject(request, session, "order too small")
        fill = Fill(
            symbol=request.symbol,
            side=request.side,
            qty=qty,
            price=price,
            fees=fees,
            session=session,
            client_order_id=request.client_order_id,
        )
        closed = self.portfolio.apply_fill(fill, request.reason)
        self.fills.append(fill)
        if closed is not None:
            self.closed_trades.append(closed)
        return OrderStatus(
            client_order_id=request.client_order_id,
            broker_order_id=self._next_id(),
            symbol=request.symbol,
            side=request.side,
            status="filled",
            filled_qty=qty,
            filled_avg_price=price,
            fees=fees,
            session=session,
        )
