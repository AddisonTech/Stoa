"""The final word on every order. Pure functions, no network, no model.

Every rule from the architecture table lives here. Functions take plain values
and return approved orders plus rejections with reason codes, so each rule can
be tested in isolation and the same code runs in backtest, paper and live.

Reason codes:
  kill_switch          kill switch active or tripped, entries halted, positions flattened
  daily_loss_limit     no new entries after a day down by the limit or more
  risk_off             benchmark below its 200 day average, no new entries
  not_in_watchlist     symbol is not on the watchlist
  not_fractionable     broker says the asset is not tradable or not fractionable
  already_held         a position in the symbol is already open
  max_positions        the position count limit is reached
  insufficient_cash    nothing left after the cash buffer and gross exposure cap
  min_order            order would be below the minimum notional
  max_orders           the per-day order count limit is reached
  same_day_round_trip  the symbol was or would be traded the other way the same day
  price_sanity         execution price moved too far from the plan reference
  no_position          exit for a symbol that is not held
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from stoa.config import RiskLimits
from stoa.models import (
    ZERO,
    Candidate,
    ExitSignal,
    PlannedOrder,
    Rejection,
    Side,
    cents_down,
)
from stoa.universe import Universe


@dataclass(frozen=True)
class HeldPosition:
    symbol: str
    qty: Decimal
    price: Decimal

    @property
    def value(self) -> Decimal:
        return self.qty * self.price


@dataclass(frozen=True)
class EntryIdea:
    """A candidate that passed the quant gates, with the numbers sizing needs."""

    candidate: Candidate
    ref_price: Decimal
    atr: Decimal
    stop_price: Decimal


@dataclass(frozen=True)
class RiskContext:
    session: date
    equity: Decimal
    cash: Decimal
    positions: Mapping[str, HeldPosition]
    prev_equity: Decimal | None
    high_water_mark: Decimal
    kill_switch_active: bool
    regime_bull: bool
    universe: Universe
    limits: RiskLimits


@dataclass(frozen=True)
class RiskResult:
    orders: tuple[PlannedOrder, ...]
    rejections: tuple[Rejection, ...]
    kill_switch_tripped: bool


def drawdown(equity: Decimal, high_water_mark: Decimal) -> Decimal:
    """Fractional drawdown from the high-water mark, zero at or above it."""
    if high_water_mark <= 0 or equity >= high_water_mark:
        return ZERO
    return (high_water_mark - equity) / high_water_mark


def next_high_water_mark(current: Decimal | None, equity: Decimal) -> Decimal:
    return equity if current is None else max(current, equity)


def kill_switch_breached(equity: Decimal, high_water_mark: Decimal, limits: RiskLimits) -> bool:
    return drawdown(equity, high_water_mark) >= limits.kill_switch_drawdown_pct


def daily_loss_breached(equity: Decimal, prev_equity: Decimal | None, limits: RiskLimits) -> bool:
    if prev_equity is None or prev_equity <= 0:
        return False
    return (prev_equity - equity) / prev_equity >= limits.daily_loss_limit_pct


def cash_buffer(equity: Decimal, limits: RiskLimits) -> Decimal:
    return equity * limits.cash_buffer_pct


def buying_room(equity: Decimal, cash: Decimal, invested: Decimal, limits: RiskLimits) -> Decimal:
    """Most new money that may go into positions: cash after the buffer, under the gross cap."""
    after_buffer = cash - cash_buffer(equity, limits)
    gross_room = equity * limits.max_gross_exposure_pct - invested
    return max(ZERO, min(after_buffer, gross_room))


def flatten_orders(
    positions: Mapping[str, HeldPosition], reason: str = "kill_switch"
) -> list[PlannedOrder]:
    return [
        PlannedOrder(symbol=p.symbol, side=Side.SELL, qty=p.qty, ref_price=p.price, reason=reason)
        for p in sorted(positions.values(), key=lambda p: p.symbol)
        if p.qty > 0
    ]


def _reject(symbol: str, side: Side, reason: str, detail: str) -> Rejection:
    return Rejection(symbol=symbol, side=side, reason=reason, detail=detail)


def exit_orders(
    exits: Sequence[ExitSignal], positions: Mapping[str, HeldPosition]
) -> tuple[list[PlannedOrder], list[Rejection]]:
    orders: list[PlannedOrder] = []
    rejections: list[Rejection] = []
    seen: set[str] = set()
    for sig in exits:
        if sig.symbol in seen:
            continue
        seen.add(sig.symbol)
        pos = positions.get(sig.symbol)
        if pos is None or pos.qty <= 0:
            rejections.append(_reject(sig.symbol, Side.SELL, "no_position", sig.reason))
            continue
        orders.append(
            PlannedOrder(
                symbol=sig.symbol,
                side=Side.SELL,
                qty=pos.qty,
                ref_price=pos.price,
                reason=sig.reason,
                thesis=sig.detail,
            )
        )
    return orders, rejections


def entry_block_reason(ctx: RiskContext) -> tuple[str, str] | None:
    """Why no new entries are allowed at all this session, or None."""
    if ctx.kill_switch_active:
        return "kill_switch", "kill switch is active; reset it by hand"
    if daily_loss_breached(ctx.equity, ctx.prev_equity, ctx.limits):
        assert ctx.prev_equity is not None
        change = (ctx.equity - ctx.prev_equity) / ctx.prev_equity
        return (
            "daily_loss_limit",
            f"day change {change:.2%} breaches the {ctx.limits.daily_loss_limit_pct:.0%} limit",
        )
    if ctx.limits.block_entries_when_risk_off and not ctx.regime_bull:
        return "risk_off", "benchmark is below its 200 day average"
    return None


def evaluate(
    ctx: RiskContext, exits: Sequence[ExitSignal], ideas: Sequence[EntryIdea]
) -> RiskResult:
    """Approve exits and size entries for one decision. Sells always come first."""
    limits = ctx.limits
    rejections: list[Rejection] = []

    tripped = not ctx.kill_switch_active and kill_switch_breached(
        ctx.equity, ctx.high_water_mark, limits
    )
    if ctx.kill_switch_active or tripped:
        dd = drawdown(ctx.equity, ctx.high_water_mark)
        rejections.extend(
            _reject(i.candidate.symbol, Side.BUY, "kill_switch", f"drawdown {dd:.2%}")
            for i in ideas
        )
        return RiskResult(tuple(flatten_orders(ctx.positions)), tuple(rejections), tripped)

    sells, sell_rejections = exit_orders(exits, ctx.positions)
    rejections.extend(sell_rejections)
    if len(sells) > limits.max_orders_per_day:
        rejections.extend(
            _reject(o.symbol, Side.SELL, "max_orders", "per-day order limit")
            for o in sells[limits.max_orders_per_day :]
        )
        sells = sells[: limits.max_orders_per_day]

    block = entry_block_reason(ctx)
    if block is not None:
        rejections.extend(_reject(i.candidate.symbol, Side.BUY, block[0], block[1]) for i in ideas)
        return RiskResult(tuple(sells), tuple(rejections), False)

    selling = {o.symbol for o in sells}
    sell_value = sum((ctx.positions[s].value for s in selling), ZERO)
    invested = sum((p.value for p in ctx.positions.values()), ZERO) - sell_value
    # Sell proceeds are not counted as cash here. Buys are resized at execution
    # from the real cash after sells fill, so the plan never assumes money it lacks.
    room = buying_room(ctx.equity, ctx.cash, invested, limits)
    open_count = len([p for p in ctx.positions.values() if p.qty > 0 and p.symbol not in selling])
    target = cents_down(ctx.equity * limits.max_position_pct)
    buys: list[PlannedOrder] = []

    for idea in ideas:
        sym = idea.candidate.symbol
        reason = _entry_rule_violation(sym, ctx, selling, {b.symbol for b in buys})
        if reason is not None:
            rejections.append(_reject(sym, Side.BUY, reason[0], reason[1]))
            continue
        if open_count + len(buys) >= limits.max_positions:
            rejections.append(
                _reject(sym, Side.BUY, "max_positions", f"limit {limits.max_positions}")
            )
            continue
        if len(sells) + len(buys) >= limits.max_orders_per_day:
            rejections.append(
                _reject(sym, Side.BUY, "max_orders", f"limit {limits.max_orders_per_day}")
            )
            continue
        notional = cents_down(min(target, room))
        if notional <= 0:
            rejections.append(_reject(sym, Side.BUY, "insufficient_cash", f"room {room:.2f}"))
            continue
        if notional < limits.min_order_usd:
            rejections.append(
                _reject(sym, Side.BUY, "min_order", f"{notional} < {limits.min_order_usd}")
            )
            continue
        room -= notional
        buys.append(
            PlannedOrder(
                symbol=sym,
                side=Side.BUY,
                notional=notional,
                ref_price=idea.ref_price,
                entry_atr=idea.atr,
                stop_price=idea.stop_price,
                reason="entry",
                conviction=idea.candidate.conviction,
                thesis=idea.candidate.thesis,
            )
        )
    return RiskResult(tuple(sells + buys), tuple(rejections), False)


def _entry_rule_violation(
    symbol: str, ctx: RiskContext, selling: set[str], buying: set[str]
) -> tuple[str, str] | None:
    if not ctx.universe.is_listed(symbol):
        return "not_in_watchlist", "symbol is not on the watchlist"
    if not ctx.universe.is_fractionable(symbol):
        return "not_fractionable", "asset is not tradable or not fractionable"
    if symbol in selling:
        return "same_day_round_trip", "symbol is being sold in this plan"
    if symbol in buying:
        return "already_held", "duplicate buy in this plan"
    pos = ctx.positions.get(symbol)
    if pos is not None and pos.qty > 0:
        return "already_held", "position already open"
    return None


# ---- execution-time checks ------------------------------------------------


def price_sanity_ok(ref_price: Decimal, live_price: Decimal, limits: RiskLimits) -> bool:
    if ref_price <= 0 or live_price <= 0:
        return False
    return abs(live_price - ref_price) / ref_price <= limits.price_sanity_pct


def round_trip_blocked(
    symbol: str, side: Side, traded_today: Mapping[str, frozenset[Side]]
) -> bool:
    opposite = Side.SELL if side is Side.BUY else Side.BUY
    return opposite in traded_today.get(symbol, frozenset())


@dataclass(frozen=True)
class ExecutionContext:
    equity: Decimal
    cash: Decimal
    invested: Decimal
    prev_equity: Decimal | None
    high_water_mark: Decimal
    kill_switch_active: bool
    held: frozenset[str]
    open_positions: int
    orders_today: int
    traded_today: Mapping[str, frozenset[Side]]
    universe: Universe
    limits: RiskLimits


def check_sells(
    sells: Sequence[PlannedOrder], held_qty: Mapping[str, Decimal], ctx: ExecutionContext
) -> tuple[list[PlannedOrder], list[Rejection]]:
    """Sells run unless they would round-trip or the symbol is no longer held.

    Price sanity does not apply to sells: skipping a stop after a gap down would
    keep the account in the position it is trying to leave.
    """
    out: list[PlannedOrder] = []
    rejections: list[Rejection] = []
    count = ctx.orders_today
    for o in sells:
        qty = held_qty.get(o.symbol, ZERO)
        if qty <= 0:
            rejections.append(_reject(o.symbol, Side.SELL, "no_position", "not held at execution"))
            continue
        if round_trip_blocked(o.symbol, Side.SELL, ctx.traded_today):
            rejections.append(_reject(o.symbol, Side.SELL, "same_day_round_trip", "bought today"))
            continue
        if count >= ctx.limits.max_orders_per_day:
            rejections.append(_reject(o.symbol, Side.SELL, "max_orders", "per-day order limit"))
            continue
        count += 1
        out.append(o.model_copy(update={"qty": min(qty, o.qty or qty)}))
    return out, rejections


def check_buys(
    buys: Sequence[PlannedOrder], live_prices: Mapping[str, Decimal], ctx: ExecutionContext
) -> tuple[list[PlannedOrder], list[Rejection]]:
    """Re-check every buy against the account as it is now and resize to real cash."""
    limits = ctx.limits
    rejections: list[Rejection] = []
    if ctx.kill_switch_active or kill_switch_breached(ctx.equity, ctx.high_water_mark, limits):
        return [], [_reject(o.symbol, Side.BUY, "kill_switch", "kill switch") for o in buys]
    if daily_loss_breached(ctx.equity, ctx.prev_equity, limits):
        return [], [
            _reject(o.symbol, Side.BUY, "daily_loss_limit", "down day at execution") for o in buys
        ]

    room = buying_room(ctx.equity, ctx.cash, ctx.invested, limits)
    target = cents_down(ctx.equity * limits.max_position_pct)
    count = ctx.orders_today
    positions = ctx.open_positions
    out: list[PlannedOrder] = []
    for o in buys:
        sym = o.symbol
        live = live_prices.get(sym)
        if not ctx.universe.is_listed(sym):
            rejections.append(_reject(sym, Side.BUY, "not_in_watchlist", "not on the watchlist"))
            continue
        if not ctx.universe.is_fractionable(sym):
            rejections.append(_reject(sym, Side.BUY, "not_fractionable", "broker asset flags"))
            continue
        if sym in ctx.held:
            rejections.append(_reject(sym, Side.BUY, "already_held", "held at execution"))
            continue
        if round_trip_blocked(sym, Side.BUY, ctx.traded_today):
            rejections.append(_reject(sym, Side.BUY, "same_day_round_trip", "sold today"))
            continue
        if live is None or not price_sanity_ok(o.ref_price, live, limits):
            detail = "no live price" if live is None else f"ref {o.ref_price:.2f} live {live:.2f}"
            rejections.append(_reject(sym, Side.BUY, "price_sanity", detail))
            continue
        if positions >= limits.max_positions:
            rejections.append(
                _reject(sym, Side.BUY, "max_positions", f"limit {limits.max_positions}")
            )
            continue
        if count >= limits.max_orders_per_day:
            rejections.append(_reject(sym, Side.BUY, "max_orders", "per-day order limit"))
            continue
        planned = o.notional or ZERO
        notional = cents_down(min(planned, target, room))
        if notional < limits.min_order_usd:
            reason = "insufficient_cash" if room < limits.min_order_usd else "min_order"
            rejections.append(_reject(sym, Side.BUY, reason, f"notional {notional}"))
            continue
        room -= notional
        count += 1
        positions += 1
        out.append(o.model_copy(update={"notional": notional}))
    return out, rejections
