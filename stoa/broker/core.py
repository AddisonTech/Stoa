"""The SPY core at a real broker: idle money sits in one fund (the big paper account).

The live twin of ``SweepBroker`` (trial 12 and 13's backtests). The strategy and the risk
engine see the fund as cash: ``account().cash`` includes it and ``positions()`` leaves it
out, so it takes no slot, no size cap and no sector. A buy that cash cannot cover first
sells enough of the fund at market and waits for that fill. After the session's orders,
``sweep`` puts cash above the reserve back into the fund, never above the overlay's
exposure cap, and sells the fund down to the cap when it is over. Each buy waits for its own
fill before the next one, so every buy sees the cash the last one left. The fund is never
bought and sold in the same session. When the kill switch is on, ``sweep`` sells it all.

Every other call goes straight to the wrapped broker, so stops, order lookups and the
kill switch's cancel-everything see the real account.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_DOWN, Decimal
from typing import Any

from stoa.broker.base import Account, Broker, BrokerPosition, OrderRequest, OrderStatus
from stoa.models import Side

ZERO = Decimal(0)
QTY = Decimal("0.000000001")
MIN_SWAP = Decimal("1.00")  # dollars; Alpaca's smallest notional order
CUSHION = Decimal("1.003")  # sell a little extra fund so the buy is covered after costs


class CoreBroker:
    def __init__(self, broker: Broker, *, symbol: str, reserve_pct: Decimal, session: date) -> None:
        self.broker = broker
        self.symbol = symbol
        self.reserve_pct = reserve_pct
        self.session = session
        self.swaps: list[OrderStatus] = []
        self._seq = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.broker, name)

    # -- what the strategy and the risk engine see ---------------------------------------------

    def _fund(self) -> BrokerPosition | None:
        return self.broker.positions().get(self.symbol)

    def fund_value(self) -> Decimal:
        p = self._fund()
        return p.market_value if p is not None and p.qty > 0 else ZERO

    def account(self) -> Account:
        a = self.broker.account()
        return a.model_copy(update={"cash": a.cash + self.fund_value()})

    def positions(self) -> dict[str, BrokerPosition]:
        return {s: p for s, p in self.broker.positions().items() if s != self.symbol}

    # -- fund trades ---------------------------------------------------------------------------

    def _traded(self, side: Side) -> bool:
        return any(
            o.symbol == self.symbol and o.side is side and (o.filled or not o.terminal)
            for o in self.broker.orders_for_session(self.session)
        )

    def _swap(
        self, side: Side, *, qty: Decimal | None = None, notional: Decimal | None = None
    ) -> None:
        self._seq += 1
        req = OrderRequest(
            symbol=self.symbol,
            side=side,
            qty=qty.quantize(QTY, rounding=ROUND_DOWN) if qty is not None else None,
            notional=notional.quantize(Decimal("0.01"), rounding=ROUND_DOWN) if notional else None,
            client_order_id=f"core-{self.session:%Y%m%d}-{side.value}-{self._seq}",
            reason="sweep",
        )
        st = self.broker.submit(req)
        done = self.broker.await_terminal([req.client_order_id]) if not st.terminal else [st]
        self.swaps.append(done[-1] if done else st)

    def _sell_for(self, dollars: Decimal) -> None:
        fund = self._fund()
        if fund is None or fund.qty <= 0 or fund.current_price <= 0 or self._traded(Side.BUY):
            return  # nothing to sell, or bought this session: the buy waits for cash
        need = dollars * CUSHION / fund.current_price
        self._swap(Side.SELL, qty=min(fund.qty, need))

    def submit(self, request: OrderRequest) -> OrderStatus:
        if request.symbol == self.symbol:
            raise ValueError("the core fund is not a pick")
        if request.side is not Side.BUY or self.broker.find_order(request.client_order_id):
            return self.broker.submit(request)
        ref = self.broker.latest_prices([request.symbol]).get(request.symbol) or ZERO
        cost = request.notional if request.notional is not None else (request.qty or ZERO) * ref
        short = cost - self.broker.account().cash
        if short > 0:
            self._sell_for(short)
        st = self.broker.submit(request)
        if st.terminal:
            return st
        done = self.broker.await_terminal([request.client_order_id])
        return done[-1] if done else st

    def sweep(self, kill_switch: bool, cap: Decimal | None = None) -> None:
        """After the orders: spare cash into the fund, the fund down to the cap, or all out."""
        fund = self._fund()
        held = fund.qty if fund is not None else ZERO
        if kill_switch:
            if held > 0:
                self._swap(Side.SELL, qty=held)
            return
        acct = self.broker.account()
        equity = acct.equity
        invested = equity - acct.cash
        if cap is not None:
            over = invested - cap * equity
            if over >= MIN_SWAP and held > 0 and fund is not None:
                if not self._traded(Side.BUY):
                    self._swap(Side.SELL, qty=min(held, over / fund.current_price))
                return
        if self._traded(Side.SELL):
            return  # no same-session round trip: today's spare cash waits a session
        spare = acct.cash - self.reserve_pct * equity
        if cap is not None:
            spare = min(spare, cap * equity - invested)
        if spare >= MIN_SWAP:
            self._swap(Side.BUY, notional=spare)
