"""Broker-side protective stops: a real stop order at Alpaca for every position held overnight.

The exit rules in exits.py only look at the close, so a crash during the day
would not be acted on until the next open. A stop order at the broker closes
that gap. The stop price is the one the risk engine already uses for the
position (the initial ATR stop, or the trailing stop once it is higher).

- DECIDE (after the close) works out each position's stop and stores it on the
  position. It only ever goes up, so it is never looser than the engine's stop.
- EXECUTE (after the open) cancels a position's open Stoa stop before any sell
  of it, runs the plan, then places DAY stop sells for every position held
  from a prior day. A position bought this morning gets no stop, so nothing is
  ever bought and sold on the same day. Positions are fractional at this
  account size and Alpaca only takes fractional orders as DAY orders, so the
  stops expire at the close and are placed again each morning.
- DECIDE then reconciles: a stop that filled during the day is recorded as a
  stop-out with its fill price and time.

Every stop carries a client order id starting with ``stoa-stop-``. Stoa only
ever lists, cancels or replaces orders with that prefix, with one exception:
when the kill switch is on, every working order in the account is cancelled
before the flatten. A failure to place or cancel a stop is alerted urgently and
never stops the rest of the run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_CEILING, Decimal
from typing import Protocol, runtime_checkable

import pandas as pd

from stoa import alerts
from stoa.broker.base import BrokerError, BrokerPosition, OrderStatus
from stoa.config import CostConfig, ExitConfig
from stoa.exits import initial_stop, trailing_stop
from stoa.models import ClosedTrade, PositionMeta, Side, cents_up, dec
from stoa.state import ProtectiveStop, State, StopOut
from stoa.tracking import fill_cost

log = logging.getLogger(__name__)

STOP_PREFIX = "stoa-stop-"
SUB_DOLLAR_STEP = Decimal("0.0001")


@runtime_checkable
class StopBroker(Protocol):
    """What a broker needs to hold protective stops. Alpaca has it; the simulator does not."""

    def positions(self) -> dict[str, BrokerPosition]: ...

    def latest_prices(self, symbols: list[str]) -> dict[str, Decimal]: ...

    def open_orders(self) -> list[OrderStatus]: ...

    def orders_for_session(self, session: date) -> list[OrderStatus]: ...

    def find_order(self, client_order_id: str) -> OrderStatus | None: ...

    def submit_stop(
        self, symbol: str, qty: Decimal, stop_price: Decimal, client_order_id: str
    ) -> OrderStatus: ...

    def cancel_order(self, order: OrderStatus) -> OrderStatus: ...


def is_stoa_stop(client_order_id: str) -> bool:
    return client_order_id.startswith(STOP_PREFIX)


def stop_id(session: date, symbol: str, attempt: int) -> str:
    return f"{STOP_PREFIX}{session:%Y%m%d}-{symbol}-{attempt}"


def round_stop(price: Decimal) -> Decimal:
    """Up to the broker's tick: a cent from $1, else 0.0001. Up, so never looser."""
    if price >= 1:
        return cents_up(price)
    return price.quantize(SUB_DOLLAR_STEP, rounding=ROUND_CEILING)


# ---- DECIDE: work out and raise the stop -------------------------------------------------


def engine_stop(
    meta: PositionMeta, history: pd.DataFrame | None, cfg: ExitConfig
) -> Decimal | None:
    """The stop the exit rules use for this position today. None if it cannot be known."""
    entry, atr = float(meta.entry_price), float(meta.entry_atr)
    if entry <= 0 or atr <= 0:
        return None
    stop = initial_stop(entry, atr, cfg)
    if cfg.trailing_enabled and history is not None and not history.empty:
        since = history.loc[pd.Timestamp(meta.entry_session) :]
        if not since.empty:
            stop = trailing_stop(float(since["high"].max()), entry, atr, cfg)
    if stop <= 0:
        return None
    return round_stop(dec(stop))


def raise_stops(
    state: State, symbols: Iterable[str], features: Mapping[str, pd.DataFrame], cfg: ExitConfig
) -> None:
    """Store each held position's stop. It only tightens: the higher of today's and the last."""
    meta = dict(state.positions_meta)
    for sym in symbols:
        m = meta.get(sym)
        if m is None:
            continue
        today = engine_stop(m, features.get(sym), cfg)
        if today is None:
            continue
        new = today if m.stop_price is None else max(m.stop_price, today)
        if new != m.stop_price:
            meta[sym] = m.model_copy(update={"stop_price": new})
    state.positions_meta = meta


# ---- EXECUTE: cancel before sells, place after the plan ----------------------------------


@dataclass
class StopRun:
    placed: list[ProtectiveStop] = field(default_factory=list)
    kept: list[ProtectiveStop] = field(default_factory=list)
    alerts: list[alerts.Alert] = field(default_factory=list)


def failure_alert(action: str, symbol: str, detail: str) -> alerts.Alert:
    return alerts.Alert(
        title=f"Stop {action} failed: {symbol}",
        message=(
            f"Could not {action} the protective stop for {symbol}: {detail[:300]}. "
            "The rest of the run carried on. Check the position in the Alpaca app."
        ),
        priority=alerts.URGENT,
        tags=("warning",),
    )


def _stoa_stops(broker: StopBroker) -> list[OrderStatus]:
    return [o for o in broker.open_orders() if is_stoa_stop(o.client_order_id)]


def cancel_stops(broker: StopBroker, symbols: Iterable[str]) -> list[alerts.Alert]:
    """Cancel every open Stoa stop on ``symbols`` and confirm each one. Returns alerts.

    Runs before any sell of those symbols, so the sell is never rejected for
    shares held by a stop order.
    """
    wanted = set(symbols)
    if not wanted:
        return []
    out: list[alerts.Alert] = []
    try:
        open_stops = [o for o in _stoa_stops(broker) if o.symbol in wanted]
    except BrokerError as exc:
        return [failure_alert("cancel", ", ".join(sorted(wanted)), str(exc))]
    for o in open_stops:
        try:
            final = broker.cancel_order(o)
        except BrokerError as exc:
            out.append(failure_alert("cancel", o.symbol, str(exc)))
            continue
        if final.filled_qty > 0:
            out.append(
                alerts.Alert(
                    title=f"Stop filled before cancel: {o.symbol}",
                    message=(
                        f"The protective stop on {o.symbol} filled "
                        f"{alerts.shares(final.filled_qty)} sh before it could be cancelled. "
                        "The planned sell may now be smaller or rejected."
                    ),
                    priority=alerts.URGENT,
                    tags=("warning",),
                )
            )
        log.info("cancelled stop %s (%s)", o.client_order_id, final.status)
    return out


def cancel_all_orders(broker: StopBroker) -> list[alerts.Alert]:
    """Kill switch: cancel every working order in the account, Stoa's or not.

    Each cancel is confirmed, so no order can hold shares the flatten needs or
    buy something after the switch tripped. Returns alerts for anything that
    could not be cancelled or filled first; never raises.
    """
    try:
        working = broker.open_orders()
    except BrokerError as exc:
        return [cancel_failure_alert("all open orders", str(exc))]
    out: list[alerts.Alert] = []
    for o in working:
        try:
            final = broker.cancel_order(o)
        except BrokerError as exc:
            out.append(cancel_failure_alert(f"{o.side.value} {o.symbol}", str(exc)))
            continue
        if final.filled_qty > 0:
            out.append(
                alerts.Alert(
                    title=f"Order filled before cancel: {o.symbol}",
                    message=(
                        f"The {o.side.value} order on {o.symbol} filled "
                        f"{alerts.shares(final.filled_qty)} sh before the kill switch could "
                        "cancel it. The flatten uses the position as it is now."
                    ),
                    priority=alerts.URGENT,
                    tags=("warning",),
                )
            )
        log.info("kill switch cancelled %s (%s)", o.client_order_id, final.status)
    return out


def cancel_failure_alert(what: str, detail: str) -> alerts.Alert:
    return alerts.Alert(
        title=f"Order cancel failed: {what}",
        message=(
            f"The kill switch could not cancel {what}: {detail[:300]}. "
            "Open the Alpaca app, cancel it by hand and check the positions."
        ),
        priority=alerts.URGENT,
        tags=("warning",),
    )


def place_stops(
    broker: StopBroker,
    state: State,
    positions: Mapping[str, BrokerPosition],
    session: date,
    *,
    managed: Iterable[str],
    bought_today: Iterable[str],
    prices: Mapping[str, Decimal],
) -> StopRun:
    """Place a DAY stop sell for every managed position held from a prior day."""
    run = StopRun()
    skip = set(bought_today)
    eligible: dict[str, tuple[Decimal, Decimal]] = {}
    for sym in sorted(set(managed) & set(positions)):
        meta = state.positions_meta.get(sym)
        qty = positions[sym].qty
        if meta is None or meta.stop_price is None or qty <= 0:
            continue
        if meta.entry_session >= session or sym in skip:
            continue  # bought this morning: no same-day sell, not even a stop
        eligible[sym] = (meta.stop_price, qty)
    if not eligible:
        state.protective_stops = []
        return run
    try:
        open_stops = _stoa_stops(broker)
        tried = [o for o in broker.orders_for_session(session) if is_stoa_stop(o.client_order_id)]
    except BrokerError as exc:
        run.alerts.append(failure_alert("place", ", ".join(sorted(eligible)), str(exc)))
        return run
    for sym, (stop, qty) in eligible.items():
        price = prices.get(sym, positions[sym].current_price)
        if price <= stop:
            run.alerts.append(
                alerts.Alert(
                    title=f"No stop placed: {sym} below its stop",
                    message=(
                        f"{sym} trades at {alerts.money(price)}, at or below its stop of "
                        f"{alerts.money(stop)}, so a stop order would fill at once. None was "
                        "placed. The exit rules sell it at the next open if it closes there."
                    ),
                    priority=alerts.HIGH,
                    tags=("warning",),
                )
            )
            continue
        mine = [o for o in open_stops if o.symbol == sym]
        if len(mine) == 1 and _matches(mine[0], stop, qty):
            run.kept.append(
                ProtectiveStop(
                    symbol=sym,
                    client_order_id=mine[0].client_order_id,
                    stop_price=stop,
                    qty=qty,
                    session=session,
                )
            )
            continue
        cancel_failed = False
        for o in mine:
            try:
                broker.cancel_order(o)
            except BrokerError as exc:
                run.alerts.append(failure_alert("replace", sym, str(exc)))
                cancel_failed = True
        if cancel_failed:
            continue
        attempt = 1 + len([o for o in tried if o.symbol == sym])
        cid = stop_id(session, sym, attempt)
        try:
            status = broker.submit_stop(sym, qty, stop, cid)
        except BrokerError as exc:
            run.alerts.append(failure_alert("place", sym, str(exc)))
            continue
        if status.status == "rejected":
            run.alerts.append(failure_alert("place", sym, status.detail or "rejected"))
            continue
        run.placed.append(
            ProtectiveStop(
                symbol=sym, client_order_id=cid, stop_price=stop, qty=qty, session=session
            )
        )
    state.protective_stops = [*run.kept, *run.placed]
    return run


def _matches(order: OrderStatus, stop: Decimal, qty: Decimal) -> bool:
    """An open stop already at the right price for the whole position: keep it."""
    return order.stop_price == stop and order.qty == qty and order.filled_qty == 0


# ---- DECIDE: reconcile fills --------------------------------------------------------------


@dataclass
class Reconciled:
    stop_outs: list[StopOut] = field(default_factory=list)
    trades: list[ClosedTrade] = field(default_factory=list)
    cost: Decimal = Decimal(0)
    alerts: list[alerts.Alert] = field(default_factory=list)


def reconcile(
    broker: StopBroker,
    state: State,
    positions: Mapping[str, BrokerPosition],
    session: date,
    costs: CostConfig,
    mode: str,
) -> Reconciled:
    """Find stops that filled since EXECUTE placed them and record each as a stop-out."""
    out = Reconciled()
    still: list[ProtectiveStop] = []
    meta = dict(state.positions_meta)
    for ps in state.protective_stops:
        try:
            st = broker.find_order(ps.client_order_id)
        except BrokerError as exc:
            out.alerts.append(failure_alert("check", ps.symbol, str(exc)))
            still.append(ps)
            continue
        if st is None:
            continue
        if st.filled_qty > 0 and st.filled_avg_price is not None:
            m = meta.get(ps.symbol)
            entry = m.entry_price if m is not None else st.filled_avg_price
            pnl = st.filled_qty * (st.filled_avg_price - entry) - st.fees
            out.stop_outs.append(
                StopOut(
                    symbol=ps.symbol,
                    client_order_id=ps.client_order_id,
                    session=ps.session,
                    qty=st.filled_qty,
                    stop_price=ps.stop_price,
                    fill_price=st.filled_avg_price,
                    filled_at=st.filled_at,
                )
            )
            if m is not None:
                out.trades.append(
                    ClosedTrade(
                        symbol=ps.symbol,
                        entry_session=m.entry_session,
                        exit_session=ps.session,
                        qty=st.filled_qty,
                        entry_price=m.entry_price,
                        exit_price=st.filled_avg_price,
                        fees=st.fees,
                        pnl=pnl,
                        reason="stop_out",
                    )
                )
            out.cost += fill_cost(Side.SELL, st.filled_qty, st.filled_avg_price, st.fees, costs)
            if ps.symbol not in positions:
                meta.pop(ps.symbol, None)
            out.alerts.append(stop_out_alert(ps, st, pnl, mode))
        elif not st.terminal:
            still.append(ps)
    state.positions_meta = meta
    state.protective_stops = still
    state.add_stop_outs(out.stop_outs)
    return out


def stop_out_alert(ps: ProtectiveStop, st: OrderStatus, pnl: Decimal, mode: str) -> alerts.Alert:
    assert st.filled_avg_price is not None
    when = f" at {st.filled_at:%H:%M} UTC" if st.filled_at else ""
    return alerts.Alert(
        title=f"Stopped out: {ps.symbol}",
        message=(
            f"The protective stop on {ps.symbol} filled{when}: "
            f"{alerts.shares(st.filled_qty)} sh at {alerts.money(st.filled_avg_price)} "
            f"(stop {alerts.money(ps.stop_price)}), P&L {alerts.money(pnl)} ({mode})."
        ),
        priority=alerts.HIGH,
        tags=("octagonal_sign",),
    )
