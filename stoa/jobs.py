"""DECIDE and EXECUTE entry points with every guard.

Guards, in order: the job window on the NYSE calendar (unless forced), the
mode and live-money triple lock, once per session, data freshness, the kill
switch, and the risk engine. Any refusal or failure sends an urgent alert and
returns a non-zero exit code. Alerts never raise.

Signal mode never touches a broker. It keeps a shadow portfolio in state and
settles last night's plan at today's open (taken from today's completed bar)
through the same execution code as paper, live and backtests.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Literal

import pandas as pd

from stoa import alerts, clock, money, risk
from stoa.broker.alpaca import AlpacaBroker, make_trading_client
from stoa.broker.base import Broker, BrokerPosition, OrderStatus
from stoa.broker.sim import SimBroker
from stoa.config import Secrets, Settings
from stoa.data.base import BarProvider, Bars, DataError, require_fresh
from stoa.data.factory import alpaca_data_client, make_provider
from stoa.features import build_features, cell, slice_features, snapshot
from stoa.journal import Journal, hash_obj
from stoa.models import ZERO, ClosedTrade, Holding, PositionMeta, Side, dec
from stoa.planner import Decision, DecisionInput, make_plan
from stoa.portfolio import Lot, Portfolio
from stoa.roles.llm import Budget
from stoa.roles.scout import Scout
from stoa.state import Inception, KillSwitch, OpenOrder, Plan, State, StateStore
from stoa.strategy import AgentStrategy, build_scout
from stoa.trader import ExecutionReport, execute_plan
from stoa.universe import AssetInfo, Universe

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2

JobName = Literal["decide", "execute"]


class Refusal(RuntimeError):
    """A guard refused to let the job run. Always alerted."""


@dataclass
class Runtime:
    """Everything a job touches from the outside world, injectable for tests."""

    settings: Settings
    secrets: Secrets
    notifier: alerts.Notifier
    now: Callable[[], datetime] = clock.now_utc
    provider: BarProvider | None = None
    broker: Broker | None = None
    scout: Scout | None = None

    @property
    def store(self) -> StateStore:
        return StateStore(self.settings.state.dir)

    @property
    def journal(self) -> Journal:
        return Journal(self.settings.state.dir / "journal")


@dataclass
class JobResult:
    code: int
    status: str
    session: date | None = None
    alerts_sent: list[str] = field(default_factory=list)


# ---- guards -------------------------------------------------------------------


def live_lock_problems(settings: Settings, secrets: Secrets) -> list[str]:
    """Empty when live trading is fully unlocked. Each entry is one missing lock."""
    problems = []
    if settings.mode != "live":
        problems.append("config mode is not live")
    if not secrets.live_confirmed():
        problems.append("STOA_LIVE_CONFIRM is not I_UNDERSTAND")
    if not secrets.has_live_keys():
        problems.append("ALPACA_LIVE_KEY_ID / ALPACA_LIVE_SECRET are not set")
    return problems


def check_mode(settings: Settings, secrets: Secrets) -> None:
    mode = settings.mode
    if mode == "backtest":
        raise Refusal("mode is backtest; use `stoa backtest` instead of the daily jobs")
    if mode == "live":
        problems = live_lock_problems(settings, secrets)
        if problems:
            raise Refusal(
                "Live trading is locked: " + "; ".join(problems) + ". Nothing was traded."
            )
    if mode == "paper" and not secrets.has_paper_keys():
        raise Refusal("paper mode needs ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET")
    if mode in ("paper", "live") and settings.data.provider == "synthetic":
        raise Refusal(f"{mode} mode cannot run on synthetic data")


def make_broker(rt: Runtime) -> Broker:
    if rt.broker is not None:
        return rt.broker
    s, sec = rt.settings, rt.secrets
    live = s.mode == "live"
    if live:
        assert sec.alpaca_live_key_id and sec.alpaca_live_secret
        key, secret, url = sec.alpaca_live_key_id, sec.alpaca_live_secret, s.alpaca.live_url
    else:
        assert sec.alpaca_paper_key_id and sec.alpaca_paper_secret
        key, secret, url = sec.alpaca_paper_key_id, sec.alpaca_paper_secret, s.alpaca.paper_url
    trading = make_trading_client(url, key.get_secret_value(), secret.get_secret_value())
    broker = AlpacaBroker(
        trading,
        alpaca_data_client(s, sec),
        live=live,
        quote_feed=s.data.quote_feed,
        max_retries=s.data.max_retries,
        fill_timeout=s.alpaca.fill_timeout_seconds,
        poll_interval=s.alpaca.poll_interval_seconds,
    )
    rt.broker = broker
    return broker


def make_scout(rt: Runtime, state: State) -> Scout:
    if rt.scout is not None:
        return rt.scout
    s = rt.settings
    use_llm = s.scout.kind == "llm" or (
        s.scout.kind == "auto" and rt.secrets.llm_api_key is not None
    )
    budget = Budget(s.llm.monthly_cost_cap_usd, state.llm_spend_by_month, clock=rt.now)
    return build_scout(s, rt.secrets, llm=use_llm, budget=budget)


# ---- shared helpers -------------------------------------------------------------


def send(rt: Runtime, result: JobResult, alert: alerts.Alert) -> None:
    alerts.safe_send(rt.notifier, alert)
    result.alerts_sent.append(alert.title)


def fetch_bars(rt: Runtime, session: date) -> Bars:
    provider = rt.provider or make_provider(rt.settings, rt.secrets)
    start = session - timedelta(days=rt.settings.data.live_lookback_days)
    symbols = list(rt.settings.watchlist)
    bars = provider.fetch(symbols, start, session)
    require_fresh(bars, symbols, session)
    return bars


def universe_from_broker(broker: Broker, settings: Settings) -> Universe:
    assets: dict[str, AssetInfo] = {s: broker.asset(s) for s in settings.watchlist}
    return Universe(watchlist=settings.watchlist, benchmark=settings.benchmark, assets=assets)


def reconcile_meta(
    state: State, positions: dict[str, BrokerPosition], session: date, atr: dict[str, float]
) -> list[str]:
    """Keep entry metadata in line with what the broker actually holds."""
    notes = []
    meta = {k: v for k, v in state.positions_meta.items() if k in positions}
    for sym in sorted(set(state.positions_meta) - set(positions)):
        notes.append(f"{sym} is no longer held at the broker")
    for sym, p in positions.items():
        if sym not in meta:
            meta[sym] = PositionMeta(
                entry_session=session,
                entry_price=p.avg_entry_price,
                entry_atr=dec(atr.get(sym, 0.0)),
            )
            notes.append(f"{sym} found at the broker without a Stoa entry; tracking it from today")
    state.positions_meta = meta
    return notes


def settle_open_orders(state: State, broker: Broker) -> list[ClosedTrade | str]:
    """Follow up orders EXECUTE left working. Returns notes for the summary."""
    notes: list[ClosedTrade | str] = []
    still: list[OpenOrder] = []
    for o in state.open_orders:
        status = broker.find_order(o.client_order_id)
        if status is None or not status.terminal:
            still.append(o)
            continue
        if status.filled and status.filled_avg_price is not None:
            if o.side == Side.BUY.value and o.symbol not in state.positions_meta:
                state.positions_meta = {
                    **state.positions_meta,
                    o.symbol: PositionMeta(
                        entry_session=o.session,
                        entry_price=status.filled_avg_price,
                        entry_atr=o.entry_atr or ZERO,
                    ),
                }
            if o.side == Side.SELL.value and o.basis is not None:
                pnl = status.filled_qty * (status.filled_avg_price - o.basis) - status.fees
                state.add_realized(o.session.year, pnl)
            notes.append(f"late fill: {o.side} {o.symbol} {alerts.shares(status.filled_qty)} sh")
        else:
            notes.append(f"order {o.side} {o.symbol} ended {status.status}")
    state.open_orders = still
    return notes


def prev_close_equity(state: State, session: date, broker_value: Decimal | None) -> Decimal | None:
    """Equity at the close before ``session``, the daily loss limit's baseline.

    Stoa's own record of that close wins; the broker's last_equity is the fallback.
    """
    if state.last_close_equity is not None and state.last_close_session == clock.previous_session(
        session
    ):
        return state.last_close_equity
    return broker_value


# ---- shadow portfolio for signal mode ------------------------------------------


def load_shadow(state: State, settings: Settings) -> Portfolio:
    cash = state.sim_cash if state.sim_cash is not None else settings.money.deposit
    p = Portfolio(cash=cash)
    for sym, qty in state.sim_positions.items():
        meta = state.positions_meta.get(sym)
        entry = meta.entry_session if meta else date.min
        p.positions[sym] = Lot(qty=qty, avg_cost=state.sim_avg_cost[sym], entry_session=entry)
    return p


def save_shadow(state: State, p: Portfolio) -> None:
    state.sim_cash = p.cash
    state.sim_positions = {s: lot.qty for s, lot in p.positions.items()}
    state.sim_avg_cost = {s: lot.avg_cost for s, lot in p.positions.items()}


def _prices(bars: Bars, day: date, column: str) -> dict[str, Decimal]:
    ts = pd.Timestamp(day)
    return {s: dec(cell(df, ts, column)) for s, df in bars.items() if ts in df.index}


# ---- DECIDE ---------------------------------------------------------------------


@dataclass
class Book:
    """The account as seen at decision time."""

    equity: Decimal
    cash: Decimal
    positions: dict[str, Holding]
    prev_equity: Decimal | None
    universe: Universe
    settled: ExecutionReport | None = None
    unmanaged: list[str] = field(default_factory=list)


def _book_signal(rt: Runtime, state: State, bars: Bars, session: date) -> Book:
    s = rt.settings
    portfolio = load_shadow(state, s)
    universe = Universe.assume_all_fractionable(s.watchlist, s.benchmark)
    broker = SimBroker(portfolio, s.costs, universe)
    prev = state.last_close_equity
    settled = None
    plan = state.pending_plan
    held_prices = _prices(bars, session, "close")
    broker.mark(held_prices)
    if plan is not None and plan.execute_session == session:
        broker.begin_session(session, _prices(bars, session, "open"))
        settled = execute_plan(plan, broker, state, s, universe, session, prev)
        for t in broker.closed_trades:
            state.add_realized(session.year, t.pnl)
        state.add_trades(broker.closed_trades)
        state.pending_plan = None
        state.mark_executed(session)
    broker.mark(held_prices)
    save_shadow(state, portfolio)
    holdings = {
        sym: Holding(
            symbol=sym, qty=lot.qty, avg_entry_price=lot.avg_cost, meta=state.positions_meta[sym]
        )
        for sym, lot in portfolio.positions.items()
    }
    return Book(
        equity=portfolio.equity(broker.prices),
        cash=portfolio.cash,
        positions=holdings,
        prev_equity=prev,
        universe=universe,
        settled=settled,
    )


def close_equity(
    cash: Decimal, positions: dict[str, BrokerPosition], closes: dict[str, Decimal]
) -> Decimal:
    """Equity marked at the official close, like the backtest.

    The broker marks positions at the latest trade, which after 4 pm can be an
    after-hours print. Positions with a session close use it instead.
    """
    marks = (p.qty * closes.get(sym, p.current_price) for sym, p in positions.items())
    return cash + sum(marks, ZERO)


def _book_broker(
    rt: Runtime, state: State, session: date, atr: dict[str, float], closes: dict[str, Decimal]
) -> tuple[Book, list[str]]:
    s = rt.settings
    broker = make_broker(rt)
    notes = [str(n) for n in settle_open_orders(state, broker)]
    account = broker.account()
    if account.trading_blocked:
        raise Refusal(f"Alpaca says trading is blocked on this account (status {account.status})")
    positions = broker.positions()
    notes += reconcile_meta(state, positions, session, atr)
    universe = universe_from_broker(broker, s)
    managed = {sym: p for sym, p in positions.items() if sym in s.watchlist}
    holdings = {
        sym: Holding(
            symbol=sym, qty=p.qty, avg_entry_price=p.avg_entry_price, meta=state.positions_meta[sym]
        )
        for sym, p in managed.items()
    }
    if isinstance(broker, AlpacaBroker):
        since = state.inception.session if state.inception else session
        flows = money.apply_flows(
            state, money.parse_flows(broker.transfer_activities(after=since)), since
        )
        flow_alert = money.flows_alert(flows, state.principal_withdrawn)
        if flow_alert is not None:
            alerts.safe_send(rt.notifier, flow_alert)
    book = Book(
        equity=close_equity(account.cash, positions, closes),
        cash=account.cash,
        positions=holdings,
        prev_equity=prev_close_equity(state, session, account.last_equity),
        universe=universe,
        unmanaged=sorted(set(positions) - set(managed)),
    )
    return book, notes


def decide(rt: Runtime, force: bool = False) -> JobResult:
    return _guarded(rt, "decide", lambda: _decide(rt, force))


def _decide(rt: Runtime, force: bool) -> JobResult:
    s = rt.settings
    now = rt.now()
    session = clock.decide_session(
        now, s.job_windows.decide_after_close_minutes, s.job_windows.decide_end
    )
    if session is None:
        if not force:
            log.info(
                "DECIDE: outside the window at %s ET; nothing to do",
                clock.to_ny(now).strftime("%Y-%m-%d %H:%M"),
            )
            return JobResult(EXIT_OK, "outside_window")
        session = clock.last_closed_session(now)
    check_mode(s, rt.secrets)
    store = rt.store
    state = store.load()
    if session in state.decided_sessions and not force:
        log.info("DECIDE: session %s already decided", session)
        return JobResult(EXIT_OK, "already_done", session)

    result = JobResult(EXIT_OK, "decided", session)
    journal = rt.journal
    bars = fetch_bars(rt, session)
    features = build_features(bars, s.benchmark)
    snap = snapshot(features, session)
    atr = {sym: cell(snap, sym, "atr_14") for sym in snap.index}
    journal.write(
        now, "decide", session, "data", symbols=sorted(bars), features_hash=hash_obj(snap.to_dict())
    )

    notes: list[str] = []
    if s.mode == "signal":
        book = _book_signal(rt, state, bars, session)
    else:
        book, notes = _book_broker(rt, state, session, atr, _prices(bars, session, "close"))

    if state.inception is None:
        state.inception = Inception(
            session=session,
            equity=book.equity,
            benchmark_close=dec(cell(snap, s.benchmark, "close")),
        )
        mismatch = money.deposit_mismatch_note(book.equity, s.money)
        if mismatch:
            notes.append(mismatch)
    state.high_water_mark = risk.next_high_water_mark(state.high_water_mark, book.equity)
    hwm = state.high_water_mark

    scout = make_scout(rt, state)
    strategy = AgentStrategy(s, scout)
    decision: Decision = strategy.decide(
        DecisionInput(
            session=session,
            execute_session=clock.next_session(session),
            features=slice_features(features, session),
            holdings=book.positions,
            equity=book.equity,
            cash=book.cash,
            prev_equity=book.prev_equity,
            high_water_mark=hwm,
            kill_switch_active=state.kill_switch.active,
            universe=book.universe,
            recent_trades=tuple(state.recent_trades[-s.llm.recent_trades_in_prompt :]),
        )
    )
    journal.write(
        now,
        "decide",
        session,
        "decision",
        candidates=[c.model_dump(mode="json") for c in decision.candidates],
        quant_kills=[k.model_dump(mode="json") for k in decision.kills],
        exits=[e.model_dump(mode="json") for e in decision.exits],
        rejections=[r.model_dump(mode="json") for r in decision.rejections],
        regime_bull=decision.regime_bull,
        scout=None if decision.scout is None else decision.scout.__dict__,
    )

    if decision.kill_switch_tripped:
        state.kill_switch = KillSwitch(
            active=True, tripped_session=session, reason="drawdown limit"
        )
        dd = float(risk.drawdown(book.equity, hwm))
        send(rt, result, alerts.kill_switch_alert(book.equity, hwm, dd))
    if decision.scout is not None and decision.scout.fallback_reason and s.scout.kind != "rule":
        spent = float(state.llm_spend_by_month.get(now.strftime("%Y-%m"), ZERO))
        send(
            rt,
            result,
            alerts.llm_fallback_alert(
                decision.scout.fallback_reason, spent, s.llm.monthly_cost_cap_usd
            ),
        )

    plan = make_plan(
        decision,
        strategy=strategy.name,
        mode=s.mode,
        session=session,
        execute_session=clock.next_session(session),
        created_at=now,
        equity=book.equity,
    )
    stale = state.pending_plan
    if stale is not None and stale.orders and stale.execute_session <= session:
        notes.append(f"plan for {stale.execute_session} was never executed and is dropped")
    # An empty plan is kept too, so EXECUTE still checks the kill switch at the open.
    state.pending_plan = plan
    if plan.orders:
        store.archive_plan(plan)
    journal.write(now, "decide", session, "plan", plan=plan.model_dump(mode="json"))

    for order in plan.orders:
        send(rt, result, alerts.signal_alert(order, book.equity, s.mode))

    milestone = money.check_milestone(state, book.equity, s.money)
    if milestone is not None:
        send(rt, result, milestone)

    send(rt, result, alerts.daily_summary_alert(_summary(rt, state, book, snap, plan, notes, now)))
    state.last_close_session = session
    state.last_close_equity = book.equity
    state.mark_decided(session)
    store.save(state)
    return result


def _summary(
    rt: Runtime,
    state: State,
    book: Book,
    snap: pd.DataFrame,
    plan: Plan,
    notes: list[str],
    now: datetime,
) -> alerts.Summary:
    s = rt.settings
    day_change = None
    if book.prev_equity and book.prev_equity > 0:
        day_change = float(book.equity / book.prev_equity - 1)
    since = spy_since = None
    if state.inception is not None and state.inception.equity > 0:
        since = float((book.equity - state.net_flows) / state.inception.equity - 1)
        bench_now = cell(snap, s.benchmark, "close")
        spy_since = bench_now / float(state.inception.benchmark_close) - 1
    lines = []
    for sym, h in sorted(book.positions.items()):
        close = dec(cell(snap, sym, "close")) if sym in snap.index else h.avg_entry_price
        gain = float(close / h.avg_entry_price - 1) if h.avg_entry_price > 0 else 0.0
        lines.append(alerts.PositionLine(sym, h.qty * close, gain))
    actions = [
        f"{o.side.value.upper()} {o.symbol} "
        + (alerts.money(o.notional) if o.notional else f"{alerts.shares(o.qty or ZERO)} sh")
        for o in plan.orders
    ]
    realized = state.realized_for_year(now.year)
    extra = list(notes)
    if book.unmanaged:
        extra.append("Not managed by Stoa: " + ", ".join(book.unmanaged))
    if book.settled is not None and book.settled.fills:
        extra.append(f"Settled {len(book.settled.fills)} paper fills at today's open")
    return alerts.Summary(
        session=str(plan.decided_session),
        mode=s.mode,
        equity=book.equity,
        day_change=day_change,
        since_start=since,
        spy_since_start=spy_since,
        positions=lines,
        cash=book.cash,
        next_actions=actions,
        api_spend=float(state.llm_spend_by_month.get(now.strftime("%Y-%m"), ZERO)),
        api_cap=s.llm.monthly_cost_cap_usd,
        realized_ytd=realized,
        tax_reserve=money.tax_reserve(realized, s.money.tax_rate),
        kill_switch=state.kill_switch.active,
        notes=extra,
    )


# ---- EXECUTE --------------------------------------------------------------------


def execute(rt: Runtime, force: bool = False) -> JobResult:
    return _guarded(rt, "execute", lambda: _execute(rt, force))


def _market_open(now: datetime) -> date | None:
    day = clock.to_ny(now).date()
    if not clock.is_session(day):
        return None
    sess = clock.session(day)
    return day if sess.open <= now < sess.close else None


def _execute(rt: Runtime, force: bool) -> JobResult:
    s = rt.settings
    now = rt.now()
    w = s.job_windows
    session = clock.execute_session(now, w.execute_after_open_minutes, w.execute_window_minutes)
    if session is None:
        if not force:
            log.info(
                "EXECUTE: outside the window at %s ET; nothing to do",
                clock.to_ny(now).strftime("%Y-%m-%d %H:%M"),
            )
            return JobResult(EXIT_OK, "outside_window")
        session = _market_open(now)
        if session is None:
            raise Refusal("the market is closed; orders would queue blind, so EXECUTE will not run")
    if s.mode == "signal":
        log.info("EXECUTE: signal mode never touches a broker")
        return JobResult(EXIT_OK, "signal_mode", session)
    check_mode(s, rt.secrets)
    store = rt.store
    state = store.load()
    if session in state.executed_sessions and not force:
        log.info("EXECUTE: session %s already executed", session)
        return JobResult(EXIT_OK, "already_done", session)
    result = JobResult(EXIT_OK, "executed", session)
    plan = state.pending_plan
    if plan is None:
        log.info("EXECUTE: no plan for %s", session)
        state.mark_executed(session)
        store.save(state)
        result.status = "no_plan"
        return result
    if plan.execute_session != session:
        if plan.execute_session < session:
            state.pending_plan = None
            store.save(state)
            if not plan.orders:
                result.status = "stale_plan"
                return result
            send(
                rt,
                result,
                alerts.Alert(
                    title="Stale plan dropped",
                    message=(
                        f"The plan for {plan.execute_session} was not executed in time "
                        f"and will not run on {session}."
                    ),
                    priority=alerts.HIGH,
                ),
            )
            result.status = "stale_plan"
            return result
        log.info("EXECUTE: plan is for %s, not %s", plan.execute_session, session)
        result.status = "plan_not_due"
        return result

    broker = make_broker(rt)
    account = broker.account()
    if account.trading_blocked:
        raise Refusal(f"Alpaca says trading is blocked on this account (status {account.status})")
    if isinstance(broker, AlpacaBroker) and not broker.market_is_open():
        raise Refusal("Alpaca says the market is closed right now; orders would queue blind")
    # The high-water mark only moves on closes, as in backtests. execute_plan checks it.
    hwm = state.high_water_mark or account.equity

    buy_symbols = sorted({o.symbol for o in plan.orders if o.side is Side.BUY})
    assets = {sym: broker.asset(sym) for sym in buy_symbols}
    universe = Universe(watchlist=s.watchlist, benchmark=s.benchmark, assets=assets)
    before = broker.positions()
    meta_before = dict(state.positions_meta)
    prev = prev_close_equity(state, session, account.last_equity)
    report = execute_plan(plan, broker, state, s, universe, session, prev)
    state.add_trades(closed_trades(report, meta_before))
    if report.kill_switch_tripped:
        dd = float(risk.drawdown(account.equity, hwm))
        send(rt, result, alerts.kill_switch_alert(account.equity, hwm, dd))

    journal = rt.journal
    journal.write(
        now,
        "execute",
        session,
        "execution",
        plan_id=plan.plan_id,
        fills=[f.status.model_dump(mode="json") for f in report.fills],
        unfilled=[f.status.model_dump(mode="json") for f in report.unfilled],
        rejections=[r.model_dump(mode="json") for r in report.rejections],
        flattened=report.flattened,
    )
    for f in report.fills:
        st = f.status
        assert st.filled_avg_price is not None
        send(
            rt,
            result,
            alerts.fill_alert(st.symbol, st.side, st.filled_qty, st.filled_avg_price, s.mode),
        )
        if f.realized_pnl is not None:
            state.add_realized(session.year, f.realized_pnl)
    for f in report.unfilled:
        st = f.status
        if st.terminal:
            send(
                rt,
                result,
                alerts.rejected_order_alert(
                    st.symbol, st.side, st.detail or st.status, s.mode == "live"
                ),
            )
        else:
            state.open_orders = [
                *state.open_orders,
                _open_order(f.order.symbol, st, session, before, f.order.entry_atr),
            ]
            send(
                rt,
                result,
                alerts.Alert(
                    title=f"Order still working: {st.side.value} {st.symbol}",
                    message=f"Status {st.status}. Stoa will check it again at DECIDE.",
                    priority=alerts.HIGH,
                ),
            )
    for r in report.rejections:
        send(rt, result, alerts.skipped_order_alert(r.symbol, r.side, r.reason, r.detail))
    state.pending_plan = None
    state.mark_executed(session)
    store.save(state)
    if any(f.status.terminal for f in report.unfilled) and s.mode == "live":
        result.code = EXIT_FAILED
        result.status = "live_order_rejected"
    return result


def closed_trades(report: ExecutionReport, meta: dict[str, PositionMeta]) -> list[ClosedTrade]:
    out = []
    for f in report.fills:
        st = f.status
        if st.side is not Side.SELL or f.realized_pnl is None or st.filled_avg_price is None:
            continue
        m = meta.get(st.symbol)
        if m is None:
            continue
        out.append(
            ClosedTrade(
                symbol=st.symbol,
                entry_session=m.entry_session,
                exit_session=report.session,
                qty=st.filled_qty,
                entry_price=m.entry_price,
                exit_price=st.filled_avg_price,
                fees=st.fees,
                pnl=f.realized_pnl,
                reason=f.order.reason,
            )
        )
    return out


def _open_order(
    symbol: str,
    st: OrderStatus,
    session: date,
    before: dict[str, BrokerPosition],
    atr: Decimal | None,
) -> OpenOrder:
    basis = before[symbol].avg_entry_price if symbol in before else None
    return OpenOrder(
        client_order_id=st.client_order_id,
        symbol=symbol,
        side=st.side.value,
        session=session,
        basis=basis if st.side is Side.SELL else None,
        entry_atr=atr if st.side is Side.BUY else None,
    )


# ---- guard wrapper ----------------------------------------------------------------


def _guarded(rt: Runtime, job: JobName, body: Callable[[], JobResult]) -> JobResult:
    try:
        return body()
    except Refusal as exc:
        log.error("%s refused: %s", job, exc)
        alerts.safe_send(rt.notifier, alerts.refusal_alert(job, str(exc)))
        return JobResult(EXIT_REFUSED, "refused", alerts_sent=[f"Stoa {job} refused to run"])
    except DataError as exc:
        log.error("%s data failure: %s", job, exc)
        alerts.safe_send(rt.notifier, alerts.data_failure_alert(job, str(exc)))
        return JobResult(EXIT_FAILED, "data_failure", alerts_sent=[f"Stoa {job}: bad data"])
    except Exception as exc:
        log.error("%s failed:\n%s", job, traceback.format_exc())
        alerts.safe_send(rt.notifier, alerts.job_failure_alert(job, f"{type(exc).__name__}: {exc}"))
        return JobResult(EXIT_FAILED, "failed", alerts_sent=[f"Stoa {job} failed"])
