"""A simulated account whose idle money sits in one fund (trial 12's SPY core). Backtest only.

The risk engine sees the fund as idle money: it counts in cash and equity but is not a
position, so it takes no position slot, no size cap and no sector. Buys are funded by
selling the fund at the same session price when cash falls short; after the session's
orders, cash above the reserve buys the fund. Every fund trade pays the same slippage and
sell fee as any other fill. Like every holding, the fund is never bought and sold in the
same session: spare cash on a session the fund was sold waits for the next one. When the
kill switch is on, the fund is sold and nothing is swept until it is off. Fund trades are
swaps, not trades: they are kept out of the closed trades the strategy is graded on and
counted on their own.

Trial 13's overlay adds an exposure cap: picks plus the fund at most a share of equity.
Above it the fund is sold down to cash (never on a session it was bought), and the sweep
buys only up to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_DOWN, Decimal

from stoa.broker.base import Account, BrokerPosition, OrderRequest, OrderStatus
from stoa.broker.sim import SimBroker, Slippage, buy_fill_price
from stoa.config import CostConfig
from stoa.models import Fill, Side
from stoa.portfolio import Portfolio
from stoa.universe import Universe

ZERO = Decimal(0)
QTY = Decimal("0.000000001")
MIN_SWAP = Decimal("0.01")  # dollars; smaller sweeps are skipped


@dataclass
class SweepStats:
    swaps: int = 0
    bought: Decimal = ZERO  # gross dollars of the fund bought
    sold: Decimal = ZERO
    cost: Decimal = ZERO  # slippage plus fees paid on fund trades
    sessions: int = 0
    full_sessions: int = 0  # sessions that ended with no pick held, all idle money in the fund
    fills: list[Fill] = field(default_factory=list)


class SweepBroker(SimBroker):
    def __init__(
        self,
        portfolio: Portfolio,
        costs: CostConfig,
        universe: Universe,
        slippage: Slippage | None = None,
        *,
        symbol: str,
        reserve_pct: Decimal,
    ) -> None:
        super().__init__(portfolio, costs, universe, slippage)
        self.symbol = symbol
        self.reserve_pct = reserve_pct
        self.stats = SweepStats()
        self._swap_seq = 0

    # -- what the risk engine sees -----------------------------------------------------------

    def fund_value(self) -> Decimal:
        qty = self.portfolio.qty(self.symbol)
        price = self.prices.get(self.symbol)
        return qty * price if qty > 0 and price is not None else ZERO

    def account(self) -> Account:
        acct = super().account()
        return Account(equity=acct.equity, cash=acct.cash + self.fund_value())

    def positions(self) -> dict[str, BrokerPosition]:
        return {s: p for s, p in super().positions().items() if s != self.symbol}

    # -- fund trades ---------------------------------------------------------------------------

    def _swap(self, side: Side, qty: Decimal) -> Fill | None:
        qty = qty.quantize(QTY, rounding=ROUND_DOWN)
        if qty <= 0 or self.session is None:
            return None
        self._swap_seq += 1
        req = OrderRequest(
            symbol=self.symbol,
            side=side,
            qty=qty,
            client_order_id=f"sweep-{self.session}-{self._swap_seq}",
            reason="sweep",
        )
        before = len(self.closed_trades)
        status = self._fill(req, self.session)
        del self.closed_trades[before:]  # a swap is not a trade
        if status.status != "filled":
            return None
        fill = self.fills[-1]
        ref = self.prices[self.symbol]
        self.stats.swaps += 1
        self.stats.fills.append(fill)
        self.stats.cost += abs(fill.price - ref) * fill.qty + fill.fees
        if side is Side.BUY:
            self.stats.bought += fill.qty * fill.price
        else:
            self.stats.sold += fill.qty * fill.price
        return fill

    def _sell_for(self, dollars: Decimal) -> None:
        """Sell enough of the fund to raise ``dollars`` of cash after costs."""
        ref = self.prices.get(self.symbol)
        held = self.portfolio.qty(self.symbol)
        if ref is None or ref <= 0 or held <= 0 or self.symbol not in self.tradable_today:
            return
        if self.session is not None and Side.BUY in self.portfolio.sides_traded(
            self.session, self.symbol
        ):
            return  # bought this session: no same-session round trip, so the buy waits for cash
        net = ref * (1 - Decimal(str(self._slip(self.symbol))) / Decimal(10_000))
        net = net * (1 - Decimal(str(self.costs.sell_fee_bps)) / Decimal(10_000))
        dollars = dollars + self.costs.min_sell_fee_usd + Decimal("0.01")  # fee floor and rounding
        need = dollars / net if net > 0 else held
        self._swap(Side.SELL, min(held, need))

    def submit(self, request: OrderRequest) -> OrderStatus:
        if request.symbol == self.symbol:
            return self._reject(request, self.session or date.min, "the sweep fund is not a pick")
        if request.side is Side.BUY and request.client_order_id not in self.orders:
            ref = self.prices.get(request.symbol)
            if ref is not None and ref > 0:
                price = buy_fill_price(ref, self._slip(request.symbol))
                cost = (
                    request.notional
                    if request.notional is not None
                    else (request.qty or ZERO) * price
                )
                short = cost - self.portfolio.cash
                if short > 0:
                    self._sell_for(short)
        return super().submit(request)

    def sweep(self, kill_switch: bool, cap: Decimal | None = None) -> None:
        """After the session's orders: idle cash above the reserve into the fund, or all out.

        ``cap`` (trial 13's overlay) is the most of equity in picks plus the fund; None
        means no cap beyond the cash reserve.
        """
        if self.session is None:
            return
        self.stats.sessions += 1
        if not self.positions():
            self.stats.full_sessions += 1
        if self.symbol not in self.tradable_today:
            return
        held = self.portfolio.qty(self.symbol)
        if kill_switch:
            if held > 0:
                self._swap(Side.SELL, held)
            return
        if cap is not None and self._trim(cap):
            return
        if Side.SELL in self.portfolio.sides_traded(self.session, self.symbol):
            return  # no same-session round trip in the fund: today's spare cash waits a session
        equity = self.portfolio.equity(self.prices)
        spare = self.portfolio.cash - self.reserve_pct * equity
        if cap is not None:
            spare = min(spare, cap * equity - (equity - self.portfolio.cash))
        if spare < MIN_SWAP:
            return
        price = buy_fill_price(self.prices[self.symbol], self._slip(self.symbol))
        self._swap(Side.BUY, spare / price)

    def _trim(self, cap: Decimal) -> bool:
        """Sell the fund down to the exposure cap. True when it is over the cap."""
        assert self.session is not None
        equity = self.portfolio.equity(self.prices)
        over = (equity - self.portfolio.cash) - cap * equity
        held = self.portfolio.qty(self.symbol)
        if over < MIN_SWAP or held <= 0:
            return False
        if Side.BUY in self.portfolio.sides_traded(self.session, self.symbol):
            return True  # bought this session: the cut waits a session
        price = self.prices[self.symbol]
        self._swap(Side.SELL, min(held, over / price))
        return True
