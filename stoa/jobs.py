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
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Literal

import pandas as pd

from stoa import adjust, alerts, approvals, clock, exam, money, risk, tracking
from stoa.approvals import ApprovalClient
from stoa.broker.alpaca import AlpacaBroker, make_trading_client
from stoa.broker.base import Broker, BrokerPosition, OrderStatus
from stoa.broker.sim import SimBroker
from stoa.config import Secrets, Settings
from stoa.crosscheck import check_prices
from stoa.data.base import BarProvider, Bars, DataError, fresh_subset
from stoa.data.external import EarningsSource, PriceSource, pick_source
from stoa.data.factory import alpaca_data_client, make_provider
from stoa.features import build_features, cell, slice_features, snapshot
from stoa.healthcheck import HealthcheckPinger, NullPinger, Pinger, safe_ping
from stoa.journal import Journal, hash_obj
from stoa.live import build_live_model
from stoa.membership import sector_of
from stoa.models import (
    ZERO,
    ClosedTrade,
    Holding,
    PlannedOrder,
    PositionMeta,
    Rejection,
    Side,
    dec,
)
from stoa.planner import Decision, DecisionInput, Strategy, make_plan
from stoa.portfolio import Lot, Portfolio
from stoa.roles.llm import Budget
from stoa.roles.scout import Scout
from stoa.state import Inception, KillSwitch, OpenOrder, Plan, State, StateStore
from stoa.strategy import AgentStrategy, build_scout
from stoa.trader import ExecutionReport, execute_plan, scaled_atr
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
    strategy: Strategy | None = None  # injected for tests; otherwise built from config
    pinger: Pinger | None = None
    approvals: ApprovalClient | None = None
    price_source: PriceSource | None = None
    earnings_source: EarningsSource | None = None
    sleep: Callable[[float], None] = time.sleep
    use_external: bool = True  # build real HTTP clients for missing sources

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
        if settings.exam.enforce_live:
            exam_problems = exam.live_problems(settings)
            if exam_problems:
                raise Refusal(
                    "Live trading is blocked by the exam order: "
                    + "; ".join(exam_problems)
                    + ". Run `stoa exam status`. Nothing was traded."
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


def make_pinger(rt: Runtime) -> Pinger:
    if rt.pinger is not None:
        return rt.pinger
    url = rt.secrets.healthcheck_url
    if url is None or not rt.settings.healthcheck.enabled or not rt.use_external:
        rt.pinger = NullPinger()
    else:
        h = rt.settings.healthcheck
        rt.pinger = HealthcheckPinger(
            url.get_secret_value(), timeout=h.timeout_seconds, retries=h.retries
        )
    return rt.pinger


def make_approvals(rt: Runtime) -> ApprovalClient | None:
    if rt.approvals is not None:
        return rt.approvals
    topic = rt.secrets.ntfy_approval_topic
    if topic is None or not rt.use_external:
        return None
    tok = rt.secrets.ntfy_token
    rt.approvals = ApprovalClient(
        rt.settings.alerts.server,
        topic.get_secret_value(),
        token=tok.get_secret_value() if tok else None,
    )
    return rt.approvals


def make_price_source(rt: Runtime) -> PriceSource | None:
    if rt.price_source is not None or not rt.use_external:
        return rt.price_source
    key = rt.secrets.finnhub_key
    rt.price_source = pick_source(
        rt.settings.crosscheck.source, key.get_secret_value() if key else None
    )
    return rt.price_source


def make_earnings_source(rt: Runtime) -> EarningsSource | None:
    if rt.earnings_source is not None or not rt.use_external:
        return rt.earnings_source
    kind = rt.settings.earnings.upcoming_source
    if kind == "none":
        return None
    key = rt.secrets.finnhub_key
    rt.earnings_source = pick_source(kind, key.get_secret_value() if key else None)
    return rt.earnings_source


def build_strategy_for(
    rt: Runtime, state: State, bars: Bars, session: date, notes: list[str]
) -> Strategy:
    """The model strategy when configured and a model is promoted; otherwise v1's pipeline."""
    if rt.strategy is not None:
        return rt.strategy
    s = rt.settings
    if s.strategy.kind == "model":
        upcoming = None
        src = make_earnings_source(rt)
        if src is not None:
            ahead = session + timedelta(days=30)
            try:
                upcoming = src.upcoming(sorted(set(s.watchlist) - {s.benchmark}), session, ahead)
            except Exception as exc:
                notes.append(f"upcoming earnings unavailable ({type(exc).__name__})")
        provider = rt.provider or make_provider(s, rt.secrets)
        lm = build_live_model(s, provider, bars, session, upcoming)
        if lm is not None:
            notes.extend(lm.notes)
            return lm.strategy
        notes.append("No promoted model yet (`stoa model train`); the rule pipeline ran instead.")
    return AgentStrategy(s, make_scout(rt, state))


# ---- shared helpers -------------------------------------------------------------


def send(rt: Runtime, result: JobResult, alert: alerts.Alert) -> None:
    alerts.safe_send(rt.notifier, alert)
    result.alerts_sent.append(alert.title)


def fetch_bars(rt: Runtime, session: date, held: set[str]) -> tuple[Bars, list[str]]:
    """Fresh bars for the watchlist. The benchmark and held symbols must be fresh."""
    s = rt.settings
    provider = rt.provider or make_provider(s, rt.secrets)
    start = session - timedelta(days=s.data.live_lookback_days)
    symbols = list(s.watchlist)
    bars = provider.fetch(symbols, start, session)
    required = {s.benchmark} | (held & set(symbols))
    return fresh_subset(bars, symbols, session, required, s.data.min_fresh_fraction)


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
                        entry_atr=scaled_atr(o.entry_atr, o.ref_price, status.filled_avg_price),
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
    Deposits and withdrawals seen since that close are added, so moving money
    never looks like a down day.
    """
    base = broker_value
    if state.last_close_equity is not None and state.last_close_session == clock.previous_session(
        session
    ):
        base = state.last_close_equity
    if base is None:
        return None
    return base + state.flows_since_close


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
    prev = prev_close_equity(state, session, state.last_close_equity)
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
    sync_flows(rt, state, broker, session)
    book = Book(
        equity=close_equity(account.cash, positions, closes),
        cash=account.cash,
        positions=holdings,
        prev_equity=prev_close_equity(state, session, account.last_equity),
        universe=universe,
        unmanaged=sorted(set(positions) - set(managed)),
    )
    return book, notes


def record_day(rt: Runtime, state: State, book: Book, snap: pd.DataFrame, session: date) -> None:
    """One honest daily record: broker equity and equity after Stoa's own cost model.

    In paper and live the broker's fills carry no simulated slippage, so the
    cost drag accumulated at EXECUTE is subtracted. In signal mode the shadow
    book already pays the backtest costs, so the two numbers match.
    """
    s = rt.settings
    t = state.track
    last = t.days[-1] if t.days else None
    flow = state.flows_since_close if last is not None else ZERO
    adjusted = book.equity - t.cost_drag
    t.record(
        tracking.DailyRecord(
            session=session,
            mode=s.mode,
            broker_equity=book.equity,
            adjusted_equity=adjusted if adjusted > 0 else book.equity,
            benchmark_close=dec(cell(snap, s.benchmark, "close")),
            trades_closed=t.pending_trades,
            flow=flow,
        )
    )
    t.pending_trades = 0


def update_fallback(rt: Runtime, state: State, session: date) -> alerts.Alert | None:
    """Check the drift fallback and the quit rule; activate and alert once on a trigger."""
    s = rt.settings
    t = state.track
    prior = None
    if s.mode == "live" and s.exam.paper_state_dir is not None:
        prior = exam.load_track(s.exam.paper_state_dir).days
    expectation = tracking.load_expectation(s.shared_dir)
    reason = tracking.check_fallback(t, s.fallback, expectation, prior)
    if reason is None:
        return None
    tracking.activate(t, reason, session)
    return alerts.Alert(
        title="FALLBACK: switching to SPY",
        message=(
            f"{reason}. Stoa stops opening new positions, lets current ones exit by their "
            "normal rules, and holds SPY instead (still under the risk caps and the kill "
            "switch). Only `stoa fallback reset` (or the fallback-reset workflow) undoes this."
        ),
        priority=alerts.URGENT,
        tags=("rotating_light",),
    )


def sync_flows(rt: Runtime, state: State, broker: Broker, session: date) -> None:
    """Apply new deposits and withdrawals from Alpaca to the high-water mark.

    Runs in DECIDE and again in EXECUTE before the kill switch check, so a
    withdrawal made after the close can never trip the switch at the open.
    """
    if not isinstance(broker, AlpacaBroker):
        return
    since = state.inception.session if state.inception else session
    flows = money.apply_flows(
        state, money.parse_flows(broker.transfer_activities(after=since)), since
    )
    flow_alert = money.flows_alert(flows, state.principal_withdrawn)
    if flow_alert is not None:
        alerts.safe_send(rt.notifier, flow_alert)


def decide(rt: Runtime, force: bool = False) -> JobResult:
    started = False

    def body() -> JobResult:
        nonlocal started
        res = _decide(rt, force, on_start=lambda: safe_ping(make_pinger(rt), "start"))
        started = res.status not in ("outside_window", "already_done")
        return res

    result = _guarded(rt, "decide", body)
    if result.code == EXIT_OK and result.status == "decided":
        safe_ping(make_pinger(rt), "success", f"decided {result.session}")
    elif result.code != EXIT_OK:
        safe_ping(make_pinger(rt), "fail", result.status)
    return result


def _decide(rt: Runtime, force: bool, on_start: Callable[[], object] = lambda: None) -> JobResult:
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

    on_start()
    result = JobResult(EXIT_OK, "decided", session)
    journal = rt.journal
    held = set(state.positions_meta) | set(state.sim_positions)
    bars, dropped = fetch_bars(rt, session, held)
    features = build_features(bars, s.benchmark)
    snap = snapshot(features, session)
    atr = {sym: cell(snap, sym, "atr_14") for sym in snap.index}
    journal.write(
        now, "decide", session, "data", symbols=sorted(bars), features_hash=hash_obj(snap.to_dict())
    )

    notes: list[str] = adjust.apply_adjustments(state, bars, shadow=s.mode == "signal")
    if dropped:
        notes.append(f"No fresh bar today, skipped: {', '.join(dropped)}")
    adjust.adjust_benchmark(state, bars, s.benchmark)
    if s.mode == "signal":
        book = _book_signal(rt, state, bars, session)
    else:
        book, broker_notes = _book_broker(rt, state, session, atr, _prices(bars, session, "close"))
        notes += broker_notes

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

    record_day(rt, state, book, snap, session)
    fallback_alert = update_fallback(rt, state, session)
    if fallback_alert is not None:
        send(rt, result, fallback_alert)
    track = state.track
    halt = track.fallback_reason if track.fallback_active else None

    strategy = build_strategy_for(rt, state, bars, session, notes)
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
            halt_reason=halt,
            fallback_active=track.fallback_active,
        )
    )
    notes.extend(decision.notes)
    journal.write(
        now,
        "decide",
        session,
        "decision",
        strategy=strategy.name,
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
    needs_ok = approvals.required(
        s.approval, s.mode, state.track.first_live_session, session
    ) and bool(plan.orders)
    state.approval_required = needs_ok
    state.approval_nonce = approvals.new_nonce() if needs_ok else None
    if plan.orders:
        store.archive_plan(plan)
    journal.write(
        now,
        "decide",
        session,
        "plan",
        plan=plan.model_dump(mode="json"),
        approval_required=needs_ok,
    )
    topic = rt.secrets.ntfy_approval_topic
    for order in plan.orders:
        alert = alerts.signal_alert(order, book.equity, s.mode)
        if needs_ok and order.reason not in s.approval.exempt_reasons:
            if topic is None:
                alert = alerts.with_actions(
                    alert,
                    [],
                    "Approval is required but STOA_NTFY_APPROVAL_TOPIC is not set, "
                    "so this order will be skipped.",
                )
            else:
                assert state.approval_nonce is not None
                acts = approvals.actions(
                    s.alerts.server,
                    topic.get_secret_value(),
                    plan.plan_id,
                    order.symbol,
                    state.approval_nonce,
                )
                alert = alerts.with_actions(
                    alert, acts, "Tap Approve or Skip. Nothing is sent without Approve."
                )
        send(rt, result, alert)

    milestone = money.check_milestone(state, book.equity, s.money)
    if milestone is not None:
        send(rt, result, milestone)

    send(rt, result, alerts.daily_summary_alert(_summary(rt, state, book, snap, plan, notes, now)))
    state.last_close_session = session
    state.last_close_equity = book.equity
    state.flows_since_close = ZERO
    adjust.anchor(state, bars, session, s.benchmark)
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
    sync_flows(rt, state, broker, session)
    # The high-water mark only moves on closes, as in backtests. execute_plan checks it.
    hwm = state.high_water_mark or account.equity

    buy_symbols = sorted({o.symbol for o in plan.orders if o.side is Side.BUY})
    assets = {sym: broker.asset(sym) for sym in buy_symbols}
    universe = Universe(watchlist=s.watchlist, benchmark=s.benchmark, assets=assets)
    before = broker.positions()
    meta_before = dict(state.positions_meta)
    prev = prev_close_equity(state, session, account.last_equity)
    track = state.track
    gate = make_gate(rt, state, plan, session, result)
    sectors = {sym: sector_of(sym) for sym in set(s.watchlist) | set(before)}
    report = execute_plan(
        plan,
        broker,
        state,
        s,
        universe,
        session,
        prev,
        gate=gate,
        sectors=sectors,
        halt_reason=track.fallback_reason if track.fallback_active else None,
        fallback_active=track.fallback_active,
    )
    closed = closed_trades(report, meta_before)
    state.add_trades(closed)
    track.pending_trades += len(closed)
    for f in report.fills:
        st = f.status
        if st.filled_avg_price is not None:
            track.cost_drag += tracking.fill_cost(
                st.side, st.filled_qty, st.filled_avg_price, st.fees, s.costs
            )
    if s.mode == "live" and report.fills and track.first_live_session is None:
        track.first_live_session = session
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
                _open_order(f.order, st, session, before),
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
    state.approval_nonce = None
    state.approval_required = False
    state.mark_executed(session)
    store.save(state)
    if any(f.status.terminal for f in report.unfilled) and s.mode == "live":
        result.code = EXIT_FAILED
        result.status = "live_order_rejected"
    return result


def make_gate(
    rt: Runtime, state: State, plan: Plan, session: date, result: JobResult
) -> Callable[
    [list[PlannedOrder], Mapping[str, Decimal]], tuple[list[PlannedOrder], list[Rejection]]
]:
    """The last checks before an order is sent: the hand approval and the price cross-check."""
    s = rt.settings
    w = s.job_windows
    deadline = clock.session(session).open + timedelta(
        minutes=w.execute_after_open_minutes + w.execute_window_minutes
    )
    verdicts: dict[str, approvals.Verdict] | None = None

    def need_approval(o: PlannedOrder) -> bool:
        return state.approval_required and o.reason not in s.approval.exempt_reasons

    def gate(
        orders: list[PlannedOrder], prices: Mapping[str, Decimal]
    ) -> tuple[list[PlannedOrder], list[Rejection]]:
        nonlocal verdicts
        ok: list[PlannedOrder] = []
        rej: list[Rejection] = []
        pending = [o for o in plan.orders if need_approval(o)]
        if pending and verdicts is None:
            verdicts = approvals.wait_and_decide(
                make_approvals(rt),
                pending,
                plan_id=plan.plan_id,
                nonce=state.approval_nonce or "",
                created_at=plan.created_at,
                deadline=deadline,
                wait_minutes=s.approval.wait_minutes,
                poll_seconds=s.approval.poll_interval_seconds,
                now=rt.now,
                sleep=rt.sleep,
            )
            rt.journal.write(
                rt.now(),
                "execute",
                session,
                "approvals",
                verdicts={k: v.reason for k, v in verdicts.items()},
            )
        for o in orders:
            if need_approval(o):
                v = (verdicts or {}).get(o.symbol)
                if v is None or not v.approved:
                    reason = v.reason if v is not None else "no_answer"
                    rej.append(
                        Rejection(
                            symbol=o.symbol,
                            side=o.side,
                            reason="not_approved",
                            detail=f"approval {reason.replace('_', ' ')}",
                        )
                    )
                    continue
            ok.append(o)
        buys = [o for o in ok if o.side is Side.BUY]
        if buys and s.crosscheck.enabled:
            checks = check_prices(
                [o.symbol for o in buys], prices, make_price_source(rt), s.crosscheck
            )
            failed = {sym for sym, c in checks.items() if not c.ok}
            for sym in sorted(failed):
                c = checks[sym]
                rej.append(
                    Rejection(symbol=sym, side=Side.BUY, reason="price_check", detail=c.detail)
                )
                send(rt, result, alerts.price_check_alert(sym, c.detail))
            ok = [o for o in ok if not (o.side is Side.BUY and o.symbol in failed)]
        return ok, rej

    return gate


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
    order: PlannedOrder,
    st: OrderStatus,
    session: date,
    before: dict[str, BrokerPosition],
) -> OpenOrder:
    symbol = order.symbol
    basis = before[symbol].avg_entry_price if symbol in before else None
    buy = st.side is Side.BUY
    return OpenOrder(
        client_order_id=st.client_order_id,
        symbol=symbol,
        side=st.side.value,
        session=session,
        basis=None if buy else basis,
        entry_atr=order.entry_atr if buy else None,
        ref_price=order.ref_price if buy else None,
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
