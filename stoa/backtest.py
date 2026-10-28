"""Event loop over NYSE sessions using the same decide, risk, exit and trader code as live.

Timeline for each session t in the window:
  1. open of t:  execute the plan decided at the close of t-1 (fills at the open
                 plus slippage, sell fees charged)
  2. close of t: mark to market, update the high-water mark
  3. close of t: decide with features cut at t, write the plan for t+1

The first decision is made at the close of the session before the window, so
every strategy starts the window with the same cash and the same first fill day.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Any

import pandas as pd

from stoa.broker.sim import SimBroker, Slippage
from stoa.clock import previous_session, sessions_between
from stoa.config import Settings
from stoa.data.base import Bars, DataError
from stoa.features import build_features, cell, slice_features
from stoa.journal import Journal, hash_obj
from stoa.models import ClosedTrade, Fill, Holding, PositionMeta, Side, dec
from stoa.overlay import Overlay
from stoa.planner import Decision, DecisionInput, Strategy, make_plan
from stoa.portfolio import Portfolio
from stoa.risk import next_high_water_mark
from stoa.state import Plan, State
from stoa.stops import engine_stop
from stoa.trader import execute_baseline, execute_plan
from stoa.universe import Universe


@dataclass
class BacktestResult:
    strategy: str
    start: date
    end: date
    starting_equity: Decimal
    equity: pd.Series  # close-of-session equity, first point is the starting equity
    exposure: pd.Series  # invested value / equity at each close
    trades: list[ClosedTrade]
    fills: list[Fill]
    final_cash: Decimal
    final_positions: dict[str, Decimal]
    llm_cost_by_month: dict[str, float] = field(default_factory=dict)
    llm_fallbacks: int = 0
    kill_switch_trips: int = 0
    decisions: int = 0
    kills_by_gate: dict[str, int] = field(default_factory=dict)
    rejections_by_reason: dict[str, int] = field(default_factory=dict)
    sweep: Any = None  # SweepStats when idle money sat in a fund (trial 12), else None
    stop_outs: int = 0  # broker-side stops hit (trial 13's overlay)

    @property
    def final_equity(self) -> Decimal:
        return dec(float(self.equity.iloc[-1])) if len(self.equity) else self.starting_equity


def _prices(bars: Bars, ts: pd.Timestamp, column: str) -> dict[str, Decimal]:
    return {sym: dec(cell(df, ts, column)) for sym, df in bars.items() if ts in df.index}


def _holdings(portfolio: Portfolio, state: State) -> dict[str, Holding]:
    out = {}
    for sym, lot in portfolio.positions.items():
        meta = state.positions_meta.get(sym) or PositionMeta(
            entry_session=lot.entry_session, entry_price=lot.avg_cost, entry_atr=Decimal(0)
        )
        out[sym] = Holding(symbol=sym, qty=lot.qty, avg_entry_price=lot.avg_cost, meta=meta)
    return out


def run_backtest(
    settings: Settings,
    bars: Bars,
    strategy: Strategy,
    start: date,
    end: date,
    *,
    starting_equity: Decimal | None = None,
    universe: Universe | None = None,
    journal: Journal | None = None,
    slippage: Slippage | None = None,
    fill_prices: Callable[[date, set[str], set[str]], dict[str, Decimal]] | None = None,
    sweep_symbol: str | None = None,
    overlay: Overlay | None = None,
) -> BacktestResult:
    """Run ``strategy`` over the sessions from ``start`` to ``end``.

    ``fill_prices`` replaces the open as the fill price: it gets the session, the
    planned buys and the other symbols that may trade (holdings, planned sells)
    and returns a price for those it knows (the 9:45 am price in the news
    trials). Symbols it leaves out fill at the open.

    ``sweep_symbol`` (trial 12's SPY core, backtest only) holds idle money in that fund
    through a ``SweepBroker``: the strategy sees the fund as cash, never as a holding.

    ``overlay`` (trial 13) is the one the strategy sizes with: the sweep keeps picks plus
    the fund under its exposure cap, and its broker-side stops are simulated.
    """
    bench = settings.benchmark
    if bench not in bars:
        raise DataError(f"benchmark {bench} missing")
    window = sessions_between(start, end)
    if not window:
        raise DataError(f"no sessions between {start} and {end}")
    first_decide = previous_session(window[0])
    bench_idx = bars[bench].index
    needed = [first_decide, *window]
    missing = [d for d in needed if pd.Timestamp(d) not in bench_idx]
    if missing:
        raise DataError(f"benchmark bars missing for {len(missing)} sessions, first {missing[0]}")
    warm = int((bench_idx < pd.Timestamp(first_decide)).sum())
    if warm < settings.backtest.warmup_sessions:
        raise DataError(
            f"only {warm} warmup sessions before {first_decide}; "
            f"need {settings.backtest.warmup_sessions}"
        )

    equity0 = starting_equity if starting_equity is not None else settings.backtest.starting_equity
    universe = universe or Universe.assume_all_fractionable(settings.watchlist, bench)
    features = build_features(bars, bench)
    portfolio = Portfolio(cash=equity0)
    broker: SimBroker
    if sweep_symbol is None:
        broker = SimBroker(portfolio, settings.costs, universe, slippage)
    else:
        from stoa.broker.sweep import SweepBroker

        broker = SweepBroker(
            portfolio,
            settings.costs,
            universe,
            slippage,
            symbol=sweep_symbol,
            reserve_pct=settings.risk.cash_buffer_pct,
        )
    state = State(high_water_mark=equity0)

    equity_points: dict[pd.Timestamp, float] = {pd.Timestamp(first_decide): float(equity0)}
    exposure_points: dict[pd.Timestamp, float] = {pd.Timestamp(first_decide): 0.0}
    llm_cost: dict[str, float] = defaultdict(float)
    kills: dict[str, int] = defaultdict(int)
    rejections: dict[str, int] = defaultdict(int)
    fallbacks = trips = decisions = stop_outs = 0
    pending: Plan | None = None
    prev_equity: Decimal = equity0

    def decide(day: date, next_day: date, equity: Decimal) -> Plan:
        nonlocal fallbacks, trips, decisions
        inp = DecisionInput(
            session=day,
            execute_session=next_day,
            features=slice_features(features, day),
            holdings={k: v for k, v in _holdings(portfolio, state).items() if k != sweep_symbol},
            equity=equity,
            cash=broker.account().cash,
            prev_equity=prev_equity if day != first_decide else None,
            high_water_mark=state.high_water_mark or equity,
            kill_switch_active=state.kill_switch.active,
            universe=universe,
            recent_trades=tuple(broker.closed_trades[-settings.llm.recent_trades_in_prompt :]),
        )
        decision: Decision = strategy.decide(inp)
        decisions += 1
        if decision.kill_switch_tripped:
            trips += 1
            state.kill_switch.active = True
            state.kill_switch.tripped_session = day
            state.kill_switch.reason = "drawdown limit"
        if decision.scout is not None:
            if decision.scout.fallback_reason:
                fallbacks += 1
            llm_cost[day.strftime("%Y-%m")] += decision.scout.cost_usd
        for k in decision.kills:
            kills[k.gate] += 1
        for r in decision.rejections:
            rejections[r.reason] += 1
        plan = make_plan(
            decision,
            strategy=strategy.name,
            mode="backtest",
            session=day,
            execute_session=next_day,
            created_at=datetime.combine(day, time(21, 0), tzinfo=UTC),
            equity=equity,
        )
        if journal is not None:
            journal.write(
                plan.created_at,
                "backtest",
                day,
                "decide",
                strategy=strategy.name,
                equity=str(equity),
                features_hash=hash_obj(
                    {s: f.iloc[-1].to_dict() for s, f in inp.features.items() if len(f)}
                ),
                candidates=[c.model_dump(mode="json") for c in decision.candidates],
                kills=[k.model_dump(mode="json") for k in decision.kills],
                rejections=[r.model_dump(mode="json") for r in decision.rejections],
                orders=[o.model_dump(mode="json") for o in plan.orders],
            )
        return plan

    pending = decide(first_decide, window[0], equity0)

    last_bar = {sym: df.index[-1] for sym, df in bars.items() if len(df)}
    for i, day in enumerate(window):
        ts = pd.Timestamp(day)
        for sym in sorted(portfolio.positions):
            df = bars.get(sym)
            if df is not None and ts not in df.index and last_bar.get(sym, ts) < ts:
                # The stock stopped trading while held: settle at its last close.
                last = last_bar[sym]
                broker.force_close(sym, dec(cell(df, last, "close")), day, "delisted")
                state.positions_meta = {k: v for k, v in state.positions_meta.items() if k != sym}
        fills = _prices(bars, ts, "open")
        if fill_prices is not None and pending is not None:
            buys = {o.symbol for o in pending.orders if o.side is Side.BUY}
            others = {o.symbol for o in pending.orders if o.side is not Side.BUY}
            fills.update(fill_prices(day, buys, others | set(portfolio.positions)))
        broker.begin_session(day, fills)
        due = pending is not None and (
            pending.orders or (strategy.risk_managed and portfolio.positions)
        )
        if pending is not None and due:
            if strategy.risk_managed:
                report = execute_plan(pending, broker, state, settings, universe, day, prev_equity)
                trips += int(report.kill_switch_tripped)
            else:
                execute_baseline(pending, broker, state, day)
        if overlay is not None and overlay.cfg.broker_stops:
            stop_outs += _broker_stops(broker, state, bars, settings, day, fills, sweep_symbol)
        if sweep_symbol is not None:
            from stoa.broker.sweep import SweepBroker

            assert isinstance(broker, SweepBroker)
            cap = overlay.current.exposure_cap if overlay and overlay.current else None
            broker.sweep(state.kill_switch.active, cap)
        broker.mark(_prices(bars, ts, "close"))
        equity = portfolio.equity(broker.prices)
        invested = portfolio.market_value(broker.prices)
        equity_points[ts] = float(equity)
        exposure_points[ts] = float(invested / equity) if equity > 0 else 0.0
        state.high_water_mark = next_high_water_mark(state.high_water_mark, equity)
        if i + 1 < len(window):
            pending = decide(day, window[i + 1], equity)
        else:
            pending = None
        prev_equity = equity

    return BacktestResult(
        strategy=strategy.name,
        start=window[0],
        end=window[-1],
        starting_equity=equity0,
        equity=pd.Series(equity_points, dtype=float).sort_index(),
        exposure=pd.Series(exposure_points, dtype=float).sort_index(),
        trades=list(broker.closed_trades),
        fills=list(broker.fills),
        final_cash=portfolio.cash,
        final_positions={s: lot.qty for s, lot in portfolio.positions.items()},
        llm_cost_by_month=dict(llm_cost),
        llm_fallbacks=fallbacks,
        kill_switch_trips=trips,
        decisions=decisions,
        kills_by_gate=dict(kills),
        rejections_by_reason=dict(rejections),
        sweep=getattr(broker, "stats", None) if sweep_symbol is not None else None,
        stop_outs=stop_outs,
    )


def _broker_stops(
    broker: SimBroker,
    state: State,
    bars: Bars,
    settings: Settings,
    day: date,
    fills: dict[str, Decimal],
    sweep_symbol: str | None,
) -> int:
    """The overlay's broker-side stops for one session, after the session's orders.

    As live (stops.py): every position held from a prior session has a stop from the
    close before (the ATR stop, or the trailing stop once higher); a position bought
    today has none. A stop already crossed at the 9:45 am price fills there; else it
    fills at the stop when the day's low reaches it. Slippage and fees apply as to any sell.
    """
    ts = pd.Timestamp(day)
    hit = 0
    for sym in sorted(broker.portfolio.positions):
        meta = state.positions_meta.get(sym)
        df = bars.get(sym)
        if sym == sweep_symbol or meta is None or meta.entry_session >= day or df is None:
            continue
        if ts not in df.index:
            continue
        stop = engine_stop(meta, df.loc[: ts - pd.Timedelta(days=1)], settings.exits)
        if stop is None:
            continue
        placed_at = fills.get(sym)
        if placed_at is not None and placed_at <= stop:
            price = placed_at
        elif dec(cell(df, ts, "low")) <= stop:
            price = stop
        else:
            continue
        broker.force_close(sym, price, day, "broker_stop")
        state.positions_meta = {k: v for k, v in state.positions_meta.items() if k != sym}
        hit += 1
    return hit
