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
  sector_cap           the position's sector is already at its share of equity
  correlation          too correlated with a position already held or planned
  earnings             an earnings release falls inside the holding window, or no
                       confirmed date is known and the last report is stale
  halted               new entries are halted (the drift fallback or the quit rule)
  zero_size            sizing gave the idea no weight (no edge after costs)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from stoa.config import RiskLimits
from stoa.membership import UNKNOWN_SECTOR
from stoa.models import (
    ZERO,
    Candidate,
    ExitSignal,
    PlannedOrder,
    Rejection,
    Side,
    cents_down,
)
from stoa.sizing import correlation
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
    """A candidate that passed the quant gates, with the numbers sizing needs.

    ``target_weight`` comes from the sizing module (None means the v1 rule:
    every entry asks for the full position cap). ``max_weight`` may raise the
    per-position cap only for the benchmark while the SPY-hold fallback is
    active; the risk engine ignores it for any other symbol.
    """

    candidate: Candidate
    ref_price: Decimal
    atr: Decimal
    stop_price: Decimal
    target_weight: Decimal | None = None
    max_weight: Decimal | None = None
    probability: float | None = None


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
    sectors: Mapping[str, str] = field(default_factory=dict)
    returns: Mapping[str, Sequence[float]] = field(default_factory=dict)
    entry_blocks: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    size_multiplier: Decimal = Decimal(1)
    size_note: str = ""
    halt_reason: str | None = None
    fallback_active: bool = False


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


def sector_of(symbol: str, sectors: Mapping[str, str]) -> str:
    return sectors.get(symbol, UNKNOWN_SECTOR)


def position_cap(
    symbol: str, max_weight: Decimal | None, benchmark: str, fallback: bool, limits: RiskLimits
) -> Decimal:
    """The per-position cap as a fraction of equity.

    Only the benchmark, and only while the SPY-hold fallback is active, may go
    above ``max_position_pct``, and never above ``max_benchmark_pct``.
    """
    if fallback and symbol == benchmark and max_weight is not None:
        return min(max_weight, limits.max_benchmark_pct)
    return limits.max_position_pct


def too_correlated(
    symbol: str,
    others: set[str],
    returns: Mapping[str, Sequence[float]],
    limit: float,
) -> tuple[str, str] | None:
    """The first held or planned symbol whose return correlation with ``symbol`` is over ``limit``.

    With no return history supplied at all the check is skipped (unit tests and
    baselines). With history supplied, a symbol that has none is rejected,
    because the check cannot be done.
    """
    if not returns or not others:
        return None
    mine = returns.get(symbol)
    if mine is None:
        return symbol, "no return history to check correlation"
    for other in sorted(others):
        theirs = returns.get(other)
        if theirs is None:
            continue
        n = min(len(mine), len(theirs))
        c = correlation(list(mine)[-n:], list(theirs)[-n:]) if n >= 3 else None
        if c is not None and c > limit:
            return other, f"{c:.2f} correlation with {other} (limit {limit:.2f})"
    return None


def entry_block_reason(ctx: RiskContext) -> tuple[str, str] | None:
    """Why no new entries are allowed at all this session, or None."""
    if ctx.kill_switch_active:
        return "kill_switch", "kill switch is active; reset it by hand"
    if ctx.halt_reason is not None and not ctx.fallback_active:
        return "halted", ctx.halt_reason
    if daily_loss_breached(ctx.equity, ctx.prev_equity, ctx.limits):
        assert ctx.prev_equity is not None
        change = (ctx.equity - ctx.prev_equity) / ctx.prev_equity
        return (
            "daily_loss_limit",
            f"day change {change:.2%} breaches the {ctx.limits.daily_loss_limit_pct:.0%} limit",
        )
    if ctx.limits.block_entries_when_risk_off and not ctx.regime_bull and not ctx.fallback_active:
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
    buys: list[PlannedOrder] = []
    kept = {s for s, p in ctx.positions.items() if p.qty > 0 and s not in selling}
    sector_value: dict[str, Decimal] = {}
    for s_ in kept:
        sec = sector_of(s_, ctx.sectors)
        sector_value[sec] = sector_value.get(sec, ZERO) + ctx.positions[s_].value
    sector_cap = ctx.equity * limits.max_sector_pct

    for idea in ideas:
        sym = idea.candidate.symbol
        reason = _entry_rule_violation(sym, ctx, selling, {b.symbol for b in buys})
        if reason is None and sym in ctx.entry_blocks:
            reason = ctx.entry_blocks[sym]
        if reason is None and ctx.fallback_active and sym != ctx.universe.benchmark:
            reason = ("halted", ctx.halt_reason or "SPY-hold fallback is active")
        fallback_buy = ctx.fallback_active and sym == ctx.universe.benchmark
        if reason is None and not fallback_buy:
            hit = too_correlated(
                sym, kept | {b.symbol for b in buys}, ctx.returns, limits.max_pair_correlation
            )
            if hit is not None:
                reason = ("correlation", hit[1])
        if reason is not None:
            rejections.append(_reject(sym, Side.BUY, reason[0], reason[1]))
            continue
        topping_up = fallback_buy and sym in kept
        if not topping_up and open_count + len(buys) >= limits.max_positions:
            rejections.append(
                _reject(sym, Side.BUY, "max_positions", f"limit {limits.max_positions}")
            )
            continue
        if len(sells) + len(buys) >= limits.max_orders_per_day:
            rejections.append(
                _reject(sym, Side.BUY, "max_orders", f"limit {limits.max_orders_per_day}")
            )
            continue
        cap_w = position_cap(
            sym, idea.max_weight, ctx.universe.benchmark, ctx.fallback_active, limits
        )
        want_w = cap_w if idea.target_weight is None else min(idea.target_weight, cap_w)
        want_w = want_w * ctx.size_multiplier
        if topping_up:
            want_w = want_w - ctx.positions[sym].value / ctx.equity
        if want_w <= 0:
            rejections.append(_reject(sym, Side.BUY, "zero_size", "sizing gave no weight"))
            continue
        sec = sector_of(sym, ctx.sectors)
        # No sector map at all (unit tests, baselines) means no sector cap.
        exempt_sector = not ctx.sectors or (ctx.fallback_active and sym == ctx.universe.benchmark)
        sector_room = room if exempt_sector else sector_cap - sector_value.get(sec, ZERO)
        if room >= limits.min_order_usd and sector_room < limits.min_order_usd:
            rejections.append(
                _reject(
                    sym,
                    Side.BUY,
                    "sector_cap",
                    f"{sec} already {sector_value.get(sec, ZERO) / ctx.equity:.0%} of equity "
                    f"(cap {limits.max_sector_pct:.0%})",
                )
            )
            continue
        notional = cents_down(min(ctx.equity * want_w, room, sector_room))
        if notional <= 0:
            rejections.append(_reject(sym, Side.BUY, "insufficient_cash", f"room {room:.2f}"))
            continue
        if notional < limits.min_order_usd:
            rejections.append(
                _reject(sym, Side.BUY, "min_order", f"{notional} < {limits.min_order_usd}")
            )
            continue
        room -= notional
        sector_value[sec] = sector_value.get(sec, ZERO) + notional
        thesis = idea.candidate.thesis
        if ctx.size_multiplier < 1 and ctx.size_note:
            thesis = f"{thesis} Sized x{ctx.size_multiplier} for {ctx.size_note}."
        buys.append(
            PlannedOrder(
                symbol=sym,
                side=Side.BUY,
                notional=notional,
                ref_price=idea.ref_price,
                entry_atr=idea.atr,
                stop_price=idea.stop_price,
                reason="entry" if not ctx.fallback_active else "spy_fallback",
                conviction=idea.candidate.conviction,
                thesis=thesis,
                probability=idea.probability,
                sector=sec,
                max_weight=cap_w if cap_w != limits.max_position_pct else None,
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
    if (
        pos is not None
        and pos.qty > 0
        and not (ctx.fallback_active and symbol == ctx.universe.benchmark)
    ):
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
    position_values: Mapping[str, Decimal] = field(default_factory=dict)
    sectors: Mapping[str, str] = field(default_factory=dict)
    halt_reason: str | None = None
    fallback_active: bool = False


def check_sells(
    sells: Sequence[PlannedOrder], held_qty: Mapping[str, Decimal], ctx: ExecutionContext
) -> tuple[list[PlannedOrder], list[Rejection]]:
    """Sells run unless they would round-trip or the symbol is no longer held.

    Every Stoa exit closes the whole position, so the quantity is whatever is
    held right now (a split overnight changes it). Price sanity does not apply
    to sells: skipping a stop after a gap down would keep the account in the
    position it is trying to leave.
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
        out.append(o.model_copy(update={"qty": qty}))
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
    count = ctx.orders_today
    positions = ctx.open_positions
    out: list[PlannedOrder] = []
    sector_value: dict[str, Decimal] = {}
    for s_, v in ctx.position_values.items():
        sec = sector_of(s_, ctx.sectors)
        sector_value[sec] = sector_value.get(sec, ZERO) + v
    for o in buys:
        sym = o.symbol
        live = live_prices.get(sym)
        fallback_buy = ctx.fallback_active and sym == ctx.universe.benchmark
        if ctx.halt_reason is not None and not fallback_buy:
            rejections.append(_reject(sym, Side.BUY, "halted", ctx.halt_reason))
            continue
        if not ctx.universe.is_listed(sym):
            rejections.append(_reject(sym, Side.BUY, "not_in_watchlist", "not on the watchlist"))
            continue
        if not ctx.universe.is_fractionable(sym):
            rejections.append(_reject(sym, Side.BUY, "not_fractionable", "broker asset flags"))
            continue
        topping_up = fallback_buy and sym in ctx.held
        if sym in ctx.held and not topping_up:
            rejections.append(_reject(sym, Side.BUY, "already_held", "held at execution"))
            continue
        if round_trip_blocked(sym, Side.BUY, ctx.traded_today):
            rejections.append(_reject(sym, Side.BUY, "same_day_round_trip", "sold today"))
            continue
        if live is None or not price_sanity_ok(o.ref_price, live, limits):
            detail = "no live price" if live is None else f"ref {o.ref_price:.2f} live {live:.2f}"
            rejections.append(_reject(sym, Side.BUY, "price_sanity", detail))
            continue
        if positions >= limits.max_positions and not topping_up:
            rejections.append(
                _reject(sym, Side.BUY, "max_positions", f"limit {limits.max_positions}")
            )
            continue
        if count >= limits.max_orders_per_day:
            rejections.append(_reject(sym, Side.BUY, "max_orders", "per-day order limit"))
            continue
        cap_w = position_cap(sym, o.max_weight, ctx.universe.benchmark, ctx.fallback_active, limits)
        already = ctx.position_values.get(sym, ZERO) if topping_up else ZERO
        target = cents_down(max(ZERO, ctx.equity * cap_w - already))
        sec = sector_of(sym, ctx.sectors)
        sector_room = (
            room
            if fallback_buy or not ctx.sectors
            else ctx.equity * limits.max_sector_pct - sector_value.get(sec, ZERO)
        )
        planned = o.notional or ZERO
        notional = cents_down(min(planned, target, room, sector_room))
        if notional < limits.min_order_usd:
            if room < limits.min_order_usd:
                reason = "insufficient_cash"
            elif sector_room < limits.min_order_usd:
                reason = "sector_cap"
            else:
                reason = "min_order"
            rejections.append(_reject(sym, Side.BUY, reason, f"notional {notional}"))
            continue
        room -= notional
        sector_value[sec] = sector_value.get(sec, ZERO) + notional
        count += 1
        positions += 0 if topping_up else 1
        out.append(o.model_copy(update={"notional": notional}))
    return out, rejections
