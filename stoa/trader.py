"""Executes a plan against any broker: sells first, then buys resized to real cash.

The same function runs in backtests (simulated broker at the open) and in paper
and live (Alpaca after the open). It never submits an order twice: every order
carries a deterministic client order id and is looked up before submission.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from stoa import risk
from stoa.broker.base import Broker, BrokerPosition, OrderRequest, OrderStatus
from stoa.config import Settings
from stoa.models import ZERO, PlannedOrder, PositionMeta, Rejection, Side, cents_down
from stoa.planner import client_order_id
from stoa.state import KillSwitch, Plan, State
from stoa.universe import Universe

log = logging.getLogger(__name__)

BROKER_MIN_NOTIONAL = Decimal("1.00")


@dataclass(frozen=True)
class FillReport:
    order: PlannedOrder
    status: OrderStatus
    realized_pnl: Decimal | None = None  # set on sells: proceeds minus cost basis minus fees


@dataclass
class ExecutionReport:
    session: date
    fills: list[FillReport] = field(default_factory=list)
    unfilled: list[FillReport] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    flattened: bool = False
    kill_switch_tripped: bool = False


def traded_sides(orders: list[OrderStatus]) -> dict[str, frozenset[Side]]:
    out: dict[str, set[Side]] = {}
    for o in orders:
        if o.filled_qty > 0 or not o.terminal:
            out.setdefault(o.symbol, set()).add(o.side)
    return {k: frozenset(v) for k, v in out.items()}


def _await(
    broker: Broker, submitted: list[tuple[PlannedOrder, OrderStatus]]
) -> list[tuple[PlannedOrder, OrderStatus]]:
    if not submitted:
        return []
    final = {
        s.client_order_id: s
        for s in broker.await_terminal([s.client_order_id for _, s in submitted])
    }
    return [(o, final.get(s.client_order_id, s)) for o, s in submitted]


def _submit_all(
    broker: Broker, orders: list[PlannedOrder]
) -> list[tuple[PlannedOrder, OrderStatus]]:
    submitted: list[tuple[PlannedOrder, OrderStatus]] = []
    for o in orders:
        existing = broker.find_order(o.client_order_id)
        if existing is not None:
            log.info(
                "order %s already exists (%s); not resubmitting", o.client_order_id, existing.status
            )
            submitted.append((o, existing))
            continue
        request = OrderRequest(
            symbol=o.symbol,
            side=o.side,
            client_order_id=o.client_order_id,
            notional=o.notional if o.side is Side.BUY else None,
            qty=o.qty if o.side is Side.SELL else None,
            reason=o.reason,
        )
        submitted.append((o, broker.submit(request)))
    return _await(broker, submitted)


def _record(
    report: ExecutionReport,
    results: list[tuple[PlannedOrder, OrderStatus]],
    basis: dict[str, Decimal],
) -> None:
    for order, status in results:
        if status.filled and status.filled_avg_price is not None:
            realized = None
            if order.side is Side.SELL and order.symbol in basis:
                realized = (
                    status.filled_qty * (status.filled_avg_price - basis[order.symbol])
                    - status.fees
                )
            report.fills.append(FillReport(order, status, realized))
        else:
            report.unfilled.append(FillReport(order, status))


def cost_basis(positions: dict[str, BrokerPosition], state: State) -> dict[str, Decimal]:
    """Per-share basis for realized P&L: the broker's average entry, else Stoa's entry record."""
    out = {s: m.entry_price for s, m in state.positions_meta.items()}
    out.update({s: p.avg_entry_price for s, p in positions.items()})
    return out


def scaled_atr(atr: Decimal | None, ref_price: Decimal | None, fill: Decimal) -> Decimal:
    """ATR from the decision close carried to the fill's price scale.

    The plan's ATR is measured on the bars as adjusted at DECIDE. If a split or
    dividend takes effect at the next open, the fill is on the new scale. Keeping
    the ATR as a fraction of the reference close and applying it to the fill
    price keeps stop and entry on the same scale.
    """
    if atr is None or ref_price is None or ref_price <= 0:
        return atr if atr is not None else ZERO
    return atr / ref_price * fill


def update_meta(
    state: State, report: ExecutionReport, held_after: dict[str, BrokerPosition]
) -> None:
    """Record entry metadata for new positions and drop it for closed ones."""
    meta = dict(state.positions_meta)
    for f in report.fills:
        sym = f.order.symbol
        if f.order.side is Side.BUY and sym not in meta and f.status.filled_avg_price is not None:
            fill = f.status.filled_avg_price
            meta[sym] = PositionMeta(
                entry_session=report.session,
                entry_price=fill,
                entry_atr=scaled_atr(f.order.entry_atr, f.order.ref_price, fill),
            )
    for sym in list(meta):
        if sym not in held_after:
            del meta[sym]
    state.positions_meta = meta


def execute_plan(
    plan: Plan,
    broker: Broker,
    state: State,
    settings: Settings,
    universe: Universe,
    session: date,
    prev_equity: Decimal | None,
) -> ExecutionReport:
    """Run one plan. Safe to call again for the same session after a crash.

    Orders that already exist at the broker (same client order id) are adopted,
    not resubmitted, and skip the pre-trade checks they already passed.
    """
    report = ExecutionReport(session=session)
    positions = broker.positions()
    basis = cost_basis(positions, state)

    equity = broker.account().equity
    hwm = state.high_water_mark or equity
    if not state.kill_switch.active and risk.kill_switch_breached(equity, hwm, settings.risk):
        state.kill_switch = KillSwitch(
            active=True, tripped_session=session, reason="drawdown at the open"
        )
        report.kill_switch_tripped = True

    if state.kill_switch.active:
        # Only symbols Stoa manages are flattened; anything else in the account is left alone.
        held = {
            s: risk.HeldPosition(s, p.qty, p.current_price)
            for s, p in positions.items()
            if s in settings.watchlist
        }
        flatten = [
            o.model_copy(
                update={
                    "client_order_id": client_order_id(plan.plan_id, session, o.symbol, Side.SELL)
                }
            )
            for o in risk.flatten_orders(held)
        ]
        report.flattened = True
        report.rejections.extend(
            Rejection(symbol=o.symbol, side=o.side, reason="kill_switch", detail="entries halted")
            for o in plan.orders
            if o.side is Side.BUY
        )
        _record(report, _submit_all(broker, flatten), basis)
        update_meta(state, report, broker.positions())
        return report

    adopted: list[tuple[PlannedOrder, OrderStatus]] = []
    fresh: list[PlannedOrder] = []
    for o in plan.orders:
        existing = broker.find_order(o.client_order_id)
        if existing is None:
            fresh.append(o)
        else:
            adopted.append((o, existing))
    _record(report, _await(broker, adopted), basis)

    def context(positions_now: dict[str, BrokerPosition]) -> risk.ExecutionContext:
        account = broker.account()
        today = broker.orders_for_session(session)
        return risk.ExecutionContext(
            equity=account.equity,
            cash=account.cash,
            invested=sum((p.market_value for p in positions_now.values()), ZERO),
            prev_equity=prev_equity,
            high_water_mark=state.high_water_mark or account.equity,
            kill_switch_active=state.kill_switch.active,
            held=frozenset(s for s, p in positions_now.items() if p.qty > 0),
            open_positions=len([p for p in positions_now.values() if p.qty > 0]),
            orders_today=len(today),
            traded_today=traded_sides(today),
            universe=universe,
            limits=settings.risk,
        )

    sells = [o for o in fresh if o.side is Side.SELL]
    buys = [o for o in fresh if o.side is Side.BUY]
    if sells:
        ok, rej = risk.check_sells(
            sells, {s: p.qty for s, p in positions.items()}, context(positions)
        )
        report.rejections.extend(rej)
        _record(report, _submit_all(broker, ok), basis)

    if buys:
        positions_mid = broker.positions()
        prices = broker.latest_prices(sorted({o.symbol for o in buys}))
        ok, rej = risk.check_buys(buys, prices, context(positions_mid))
        report.rejections.extend(rej)
        _record(report, _submit_all(broker, ok), basis)

    update_meta(state, report, broker.positions())
    return report


def execute_baseline(plan: Plan, broker: Broker, state: State, session: date) -> ExecutionReport:
    """Execution for the baselines: no risk sizing, buys resized only to available cash.

    Baselines exist to be measured against, so they stay fully invested when
    their rule says so. They still get the same fills, slippage and fees.
    """
    report = ExecutionReport(session=session)
    positions = broker.positions()
    basis = cost_basis(positions, state)
    sells = [
        o.model_copy(update={"qty": min(o.qty or ZERO, positions[o.symbol].qty)})
        for o in plan.orders
        if o.side is Side.SELL and o.symbol in positions
    ]
    _record(report, _submit_all(broker, sells), basis)
    buys = []
    cash = broker.account().cash
    for o in plan.orders:
        if o.side is not Side.BUY:
            continue
        notional = cents_down(min(o.notional or ZERO, cash))
        if notional < BROKER_MIN_NOTIONAL:
            continue
        cash -= notional
        buys.append(o.model_copy(update={"notional": notional}))
    _record(report, _submit_all(broker, buys), basis)
    update_meta(state, report, broker.positions())
    return report
