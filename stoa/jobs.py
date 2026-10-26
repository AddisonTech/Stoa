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

import json
import logging
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

import httpx
import pandas as pd

from stoa import adjust, alerts, approvals, clock, exam, money, risk, stops, sweep, tracking
from stoa.approvals import ApprovalClient
from stoa.broker.alpaca import AlpacaBroker, make_trading_client
from stoa.broker.base import Broker, BrokerPosition, OrderStatus
from stoa.broker.sim import SimBroker
from stoa.config import Secrets, Settings
from stoa.crosscheck import check_prices
from stoa.data.base import BarProvider, Bars, DataError, fresh_subset
from stoa.data.external import EarningsSource, PriceSource, pick_source
from stoa.data.factory import alpaca_data_client, make_provider
from stoa.exits import initial_stop
from stoa.features import build_features, cell, slice_features, snapshot
from stoa.files import replace_file
from stoa.healthcheck import HealthcheckPinger, NullPinger, Pinger, Signal, safe_ping
from stoa.journal import Journal, hash_obj
from stoa.live import build_live_model, live_calendar
from stoa.membership import sector_of
from stoa.model_strategy import ModelStrategy, PanelPredictor
from stoa.models import (
    ZERO,
    ClosedTrade,
    Holding,
    PlannedOrder,
    PositionMeta,
    Rejection,
    Side,
    cents_down,
    dec,
)
from stoa.planner import Decision, DecisionInput, Strategy, client_order_id, make_plan
from stoa.portfolio import Lot, Portfolio
from stoa.reader import Reader
from stoa.roles.llm import Budget
from stoa.roles.scout import Scout
from stoa.state import (
    HeldAtClose,
    Inception,
    KillSwitch,
    OpenOrder,
    Plan,
    State,
    StateStore,
)
from stoa.strategy import AgentStrategy, build_scout
from stoa.trader import ExecutionReport, execute_plan, scaled_atr, traded_sides
from stoa.universe import AssetInfo, Universe

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2

JobName = Literal["decide", "execute", "sweep", "override"]


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
    news_client: Any = None  # stoa.data.news.AlpacaNewsClient, injectable
    analyst_client: Any = None  # anthropic client, injectable
    reader: Any = None  # stoa.reader.Reader for the morning sweep, injectable

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
    # The scout's own cap, never above the overall monthly cap; both share one ledger.
    cap = min(s.llm.monthly_cost_cap_usd, s.budget.monthly_cap_usd)
    budget = Budget(cap, state.llm_spend_by_month, clock=rt.now)
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


def score_todays_text(rt: Runtime, state: State, session: date, notes: list[str]) -> None:
    """When text features are on: fetch the last two days of watchlist news and score it.

    Runs one call per new item with the cached system prompt, charges the
    shared monthly ledger, and never fails DECIDE (text is optional). At the
    month's cap the analyst pauses (see spendcap.py) and DECIDE goes on exactly
    as it does with the analyst off.
    """
    from stoa import analyst as an
    from stoa import spendcap
    from stoa.data.news import AlpacaNewsClient, NewsStore

    s = rt.settings
    key = rt.secrets.llm_api_key
    if not (s.model.use_text and s.analyst.enabled and key is not None):
        return

    def send_alert(a: alerts.Alert) -> None:
        alerts.safe_send(rt.notifier, a)

    try:
        topic = rt.secrets.ntfy_approval_topic
        g = spendcap.gate(
            state,
            s.budget,
            s.approval,
            now=rt.now,
            sleep=rt.sleep,
            send=send_alert,
            client=make_approvals(rt),
            server=s.alerts.server,
            approval_topic=topic.get_secret_value() if topic is not None else None,
        )
        if g.paused:
            notes.append(
                f"Analyst paused at its cap ({alerts.money(g.spent)} of {alerts.money(g.cap)} "
                f"in {g.month}); the model ran without new news scores"
            )
            return
        news = rt.news_client
        if news is None:
            news = AlpacaNewsClient(alpaca_data_client(s, rt.secrets))
        symbols = sorted(set(s.watchlist) - {s.benchmark})
        df = news.fetch(symbols, session - timedelta(days=2), session)
        NewsStore(s.news_path).merge(df)
        items = an.items_from_news(df, s.analyst.text_chars, s.analyst.max_items_per_symbol_per_day)
        client = rt.analyst_client or an.make_client(key.get_secret_value())
        ledger = an.Ledger(float(g.cap), state.llm_spend_by_month, clock=rt.now)
        a = an.Analyst(
            s.analyst, s.budget, s.llm.prices, client, an.ScoreCache(s.analyst.cache_dir), ledger
        )
        rep = a.score_now(items)
        an.merge_scores(s.analyst_scores_path, a.rows(items))
        if rep.scored or rep.failed:
            notes.append(
                f"Analyst scored {rep.scored} new items (${rep.spent_usd:.3f}), {rep.failed} failed"
            )
        heads_up = spendcap.warn_if_due(state, s.budget, rt.now())
        if heads_up is not None:
            send_alert(heads_up)
    except Exception as exc:
        notes.append(f"Analyst skipped today ({type(exc).__name__})")


def build_strategy_for(
    rt: Runtime, state: State, bars: Bars, session: date, notes: list[str]
) -> Strategy:
    """The model strategy when configured and a model is promoted; otherwise v1's pipeline."""
    if rt.strategy is not None:
        return rt.strategy
    s = rt.settings
    if state.track.fallback_active:
        # SPY-hold: no model view needed. The model strategy's fallback path exits by the
        # normal rules and buys only SPY, whichever strategy was trading before.
        empty = pd.DataFrame(
            columns=["p"], index=pd.MultiIndex.from_tuples([], names=["date", "symbol"])
        )
        return ModelStrategy(
            s,
            PanelPredictor(empty, {}, name="fallback"),
            live_calendar(s, bars, None),
            name="fallback",  # fallback days never count toward the model's paper exam
        )
    if s.strategy.kind == "model":
        score_todays_text(rt, state, session, notes)
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


def fetch_references(rt: Runtime, session: date, bars: Bars) -> Bars:
    """Bars for the reference funds (MTUM), for the record only. Never traded, never fatal."""
    s = rt.settings
    out: Bars = {sym: bars[sym] for sym in s.sources.reference_symbols if sym in bars}
    missing = [sym for sym in s.sources.reference_symbols if sym not in out]
    if not missing:
        return out
    provider = rt.provider or make_provider(s, rt.secrets)
    start = session - timedelta(days=s.data.live_lookback_days)
    try:
        out.update(provider.fetch(missing, start, session))
    except Exception as exc:
        log.warning("reference bars for %s unavailable: %s", missing, type(exc).__name__)
    return out


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
    alerts: list[alerts.Alert] = field(default_factory=list)


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
        state.track.pending_trades += len(broker.closed_trades)
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
    stop_alerts: list[alerts.Alert] = []
    if isinstance(broker, stops.StopBroker):
        rec = stops.reconcile(broker, state, positions, session, s.costs, s.mode)
        stop_alerts = rec.alerts
        state.add_trades(rec.trades)
        state.track.pending_trades += len(rec.trades)
        state.track.cost_drag += rec.cost
        for t in rec.trades:
            state.add_realized(t.exit_session.year, t.pnl)
        if rec.trades:
            rt.journal.write(
                rt.now(),
                "decide",
                session,
                "stop_outs",
                closed=[t.model_dump(mode="json") for t in rec.trades],
            )
        notes += [
            f"Stopped out of {o.symbol} at {alerts.money(o.fill_price)}" for o in rec.stop_outs
        ]
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
        alerts=stop_alerts,
    )
    return book, notes


def record_day(
    rt: Runtime,
    state: State,
    book: Book,
    snap: pd.DataFrame,
    session: date,
    references: Mapping[str, Decimal] | None = None,
) -> None:
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
            strategy=t.last_strategy,
            reference_closes=dict(references or {}),
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
    """DECIDE with the dead man's switch: /start when work begins, then success or /fail.

    A run outside the window or for a session already decided pings nothing,
    so the second cron slot never resets the check.
    """

    def body() -> JobResult:
        return _decide(rt, force, on_start=lambda: safe_ping(make_pinger(rt), "start"))

    result = _guarded(rt, "decide", body)
    if result.code == EXIT_OK and result.status == "decided":
        ok = safe_ping(make_pinger(rt), "success", f"decided {result.session}")
        record_checkin(rt, "success", ok, result.session)
    elif result.code != EXIT_OK:
        ok = safe_ping(make_pinger(rt), "fail", result.status)
        record_checkin(rt, "fail", ok, result.session)
    return result


CHECKIN_FILE = "healthcheck.json"


def record_checkin(rt: Runtime, signal: Signal, sent: bool, session: date | None) -> None:
    """Whether the dead man's switch ping went out, for the dashboard. Never the URL."""
    record = {
        "at": rt.now().isoformat(),
        "session": session.isoformat() if session else None,
        "signal": signal,
        "configured": rt.secrets.healthcheck_url is not None and rt.settings.healthcheck.enabled,
        "sent": sent,
    }
    root = rt.settings.state.dir
    try:
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / f"{CHECKIN_FILE}.tmp"
        tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        replace_file(tmp, root / CHECKIN_FILE)
    except OSError as exc:
        log.warning("could not record the healthcheck check-in: %s", type(exc).__name__)


def _decide(
    rt: Runtime,
    force: bool,
    on_start: Callable[[], object] = lambda: None,
    summary: bool = True,
) -> JobResult:
    """The evening run: record the day, send the day summary, plan the next open quietly.

    The plan is not announced. The 9:00 am sweep reads the overnight news,
    settles the picks and sends them with the Approve and Skip buttons.
    ``summary=False`` is the sweep catching up on a missed evening run.
    """
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
    refs = fetch_references(rt, session, bars)
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
    adjust.adjust_references(state, refs)
    ref_closes = adjust.reference_closes(refs, session)
    if s.mode == "signal":
        book = _book_signal(rt, state, bars, session)
    else:
        book, broker_notes = _book_broker(rt, state, session, atr, _prices(bars, session, "close"))
        notes += broker_notes
    for a in book.alerts:
        send(rt, result, a)

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

    record_day(rt, state, book, snap, session, ref_closes)
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
    stops.raise_stops(state, book.positions, features, s.exits)
    closes = _prices(bars, session, "close")
    state.holdings_at_close = {
        sym: HeldAtClose(qty=h.qty, close=closes.get(sym, h.avg_entry_price))
        for sym, h in book.positions.items()
    }
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
    plan = plan.model_copy(update={"reserves": reserves_for(decision, plan, snap, book.equity, s)})
    state.track.last_strategy = strategy.name
    stale = state.pending_plan
    if stale is not None and stale.orders and stale.execute_session <= session:
        notes.append(f"plan for {stale.execute_session} was never executed and is dropped")
    # An empty plan is kept too, so EXECUTE still checks the kill switch at the open.
    state.pending_plan = plan
    # Quiet: no pick alerts and no approval code yet. The 9:00 am sweep sends the
    # picks and draws the code, so a tap can only answer the final picks.
    state.approval_required = False
    state.approval_nonce = None
    if plan.orders:
        store.archive_plan(plan)
    journal.write(now, "decide", session, "plan", plan=plan.model_dump(mode="json"))

    milestone = money.check_milestone(state, book.equity, s.money)
    if milestone is not None:
        send(rt, result, milestone)

    if summary:
        day = _summary(rt, state, book, snap, plan, notes, now, bars=bars, session=session)
        send(rt, result, alerts.daily_summary_alert(day))
    state.last_close_session = session
    state.last_close_equity = book.equity
    state.flows_since_close = ZERO
    adjust.anchor(state, bars, session, s.benchmark)
    state.last_reference_closes = ref_closes
    state.mark_decided(session)
    store.save(state)
    return result


def reserves_for(
    decision: Decision, plan: Plan, snap: pd.DataFrame, equity: Decimal, s: Settings
) -> tuple[PlannedOrder, ...]:
    """Buys the rules passed but had no slot or cash for, in the rules' order.

    Only candidates the quant gates passed and the risk engine turned down for
    capacity (positions, orders or cash). Each is sized at the position cap;
    EXECUTE resizes it to real cash and re-runs every check if it is ever added.
    """
    capacity = {"max_positions", "max_orders", "insufficient_cash"}
    killed = {k.symbol for k in decision.kills}
    planned = {o.symbol for o in plan.orders}
    turned_down = {r.symbol for r in decision.rejections if r.reason in capacity}
    out: list[PlannedOrder] = []
    for c in decision.candidates:
        sym = c.symbol
        if sym in killed or sym in planned or sym not in turned_down or sym not in snap.index:
            continue
        close = cell(snap, sym, "close")
        atr = cell(snap, sym, "atr_14")
        out.append(
            PlannedOrder(
                symbol=sym,
                side=Side.BUY,
                notional=cents_down(equity * s.risk.max_position_pct),
                ref_price=dec(close),
                entry_atr=dec(atr),
                stop_price=dec(round(initial_stop(close, atr, s.exits), 4)),
                reason="entry",
                conviction=c.conviction,
                thesis=c.thesis,
                client_order_id=client_order_id(plan.plan_id, plan.execute_session, sym, Side.BUY),
            )
        )
    return tuple(out)


def _day_change(bars: Bars, sym: str, session: date) -> float | None:
    df = bars.get(sym)
    if df is None:
        return None
    upto = df.loc[: pd.Timestamp(session), "close"]
    if len(upto) < 2 or upto.index[-1] != pd.Timestamp(session) or upto.iloc[-2] <= 0:
        return None
    return float(upto.iloc[-1] / upto.iloc[-2] - 1)


def day_lines(state: State, book: Book, snap: pd.DataFrame, bars: Bars, session: date) -> list[str]:
    """How each position and each of today's buys did today, and what was sold."""
    parts = []
    for sym, h in sorted(book.positions.items()):
        close = dec(cell(snap, sym, "close")) if sym in snap.index else h.avg_entry_price
        value = alerts.money(h.qty * close)
        meta = state.positions_meta.get(sym)
        if meta is not None and meta.entry_session == session and meta.entry_price > 0:
            move = float(close / meta.entry_price - 1)
            bought = alerts.money(meta.entry_price)
            parts.append(f"{sym} {move:+.2%} {value} (bought today at {bought})")
            continue
        change = _day_change(bars, sym, session)
        parts.append(f"{sym} {change:+.2%} {value}" if change is not None else f"{sym} {value}")
    lines = ["Today: " + ("; ".join(parts) if parts else "no positions")]
    sold = [t for t in state.recent_trades if t.exit_session == session]
    if sold:
        lines.append(
            "Sold today: "
            + "; ".join(
                f"{t.symbol} {alerts.money(t.pnl)} ({t.reason.replace('_', ' ')})" for t in sold
            )
        )
    return lines


def _summary(
    rt: Runtime,
    state: State,
    book: Book,
    snap: pd.DataFrame,
    plan: Plan,
    notes: list[str],
    now: datetime,
    *,
    bars: Bars | None = None,
    session: date | None = None,
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
    from stoa import shadow

    extra.append(shadow.summary_line(s, state, now))
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
        api_cap=s.budget.monthly_cap_usd,
        realized_ytd=realized,
        tax_reserve=money.tax_reserve(realized, s.money.tax_rate),
        kill_switch=state.kill_switch.active,
        notes=extra,
        spy_day=_day_change(bars, s.benchmark, session) if bars and session else None,
        day_lines=day_lines(state, book, snap, bars, session) if bars and session else (),
        next_line=(
            "Tomorrow: the picks come at 9:00 am, after the overnight news sweep."
            if bars and session
            else None
        ),
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
    sectors = {sym: sector_of(sym) for sym in set(s.watchlist) | set(before) | set(assets)}

    def cancel_before_sells(symbols: list[str]) -> None:
        if isinstance(broker, stops.StopBroker):
            for a in stops.cancel_stops(broker, symbols):
                send(rt, result, a)

    def cancel_everything() -> None:
        if isinstance(broker, stops.StopBroker):
            for a in stops.cancel_all_orders(broker):
                send(rt, result, a)

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
        before_sells=cancel_before_sells,
        before_flatten=cancel_everything,
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
        closed=[t.model_dump(mode="json") for t in closed],
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
        if r.reason == "price_check":
            continue  # already alerted at a higher priority by the cross-check
        send(rt, result, alerts.skipped_order_alert(r.symbol, r.side, r.reason, r.detail))
    if isinstance(broker, stops.StopBroker):
        protect(rt, broker, state, report, session, result)
    state.pending_plan = None
    state.approval_nonce = None
    state.approval_required = False
    state.mark_executed(session)
    store.save(state)
    if any(f.status.terminal for f in report.unfilled) and s.mode == "live":
        result.code = EXIT_FAILED
        result.status = "live_order_rejected"
    return result


OVERRIDE_REASON = "manual_override"


def manual_override(rt: Runtime, symbols: list[str] | None = None) -> JobResult:
    return _guarded(rt, "override", lambda: _manual_override(rt, symbols))


def _manual_override(rt: Runtime, symbols: list[str] | None) -> JobResult:
    """Paper only, once a session: send today's planned buys after EXECUTE turned them down.

    For a test of the order path when the taps came after the cutoff. The buys are
    the ones Approve was tapped on (at any time) for today's plan, or ``symbols``,
    or every planned buy if neither names one. The sector cap is lifted for this
    run only; every other check runs as usual. Recorded as a manual override.
    """
    s = rt.settings
    if s.mode != "paper":
        raise Refusal("a manual override is for the paper account only")
    check_mode(s, rt.secrets)
    now = rt.now()
    session = _market_open(now)
    if session is None:
        raise Refusal("the market is closed; a manual override runs only while it is open")
    store = rt.store
    state = store.load()
    if session not in state.executed_sessions:
        raise Refusal(f"EXECUTE has not run for {session}; a manual override comes after it")
    if session in state.override_sessions:
        raise Refusal(f"a manual override already ran for {session}; it is one time only")
    plans = [
        Plan.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(store.plans_dir.glob("*.json"))
    ]
    todays = [p for p in plans if p.execute_session == session]
    if not todays:
        raise Refusal(f"no plan was made for {session}")
    plan = todays[-1]
    buys = [o for o in plan.orders if o.side is Side.BUY]
    chosen, source = _override_symbols(rt, plan, buys, symbols)
    orders = [o.model_copy(update={"reason": OVERRIDE_REASON}) for o in buys if o.symbol in chosen]
    if not orders:
        raise Refusal("none of the named symbols is a planned buy for today")
    result = JobResult(EXIT_OK, "override", session)
    # The sector cap is lifted for this run only, in memory; nothing is saved.
    risk_cfg = s.risk.model_copy(update={"max_sector_pct": Decimal(1)})
    s_run = s.model_copy(update={"risk": risk_cfg})
    run_plan = plan.model_copy(update={"orders": tuple(orders)})

    broker = make_broker(rt)
    account = broker.account()
    if account.trading_blocked:
        raise Refusal(f"Alpaca says trading is blocked on this account (status {account.status})")
    if isinstance(broker, AlpacaBroker) and not broker.market_is_open():
        raise Refusal("Alpaca says the market is closed right now; orders would queue blind")
    assets = {o.symbol: broker.asset(o.symbol) for o in orders}
    universe = Universe(watchlist=s.watchlist, benchmark=s.benchmark, assets=assets)
    before = broker.positions()
    track = state.track
    sectors = {sym: sector_of(sym) for sym in set(s.watchlist) | set(before) | set(assets)}
    report = execute_plan(
        run_plan,
        broker,
        state,
        s_run,
        universe,
        session,
        prev_close_equity(state, session, account.last_equity),
        gate=lambda os, _prices: (list(os), []),
        sectors=sectors,
        halt_reason=track.fallback_reason if track.fallback_active else None,
        fallback_active=track.fallback_active,
    )
    for f in report.fills:
        st = f.status
        if st.filled_avg_price is not None:
            track.cost_drag += tracking.fill_cost(
                st.side, st.filled_qty, st.filled_avg_price, st.fees, s.costs
            )
    note = (
        f"Manual override, paper only: buys sent after the {s.job_windows.approval_deadline} "
        f"cutoff ({source}); the sector cap was lifted for this run only."
    )
    rt.journal.write(
        now,
        "execute",
        session,
        "override",
        plan_id=plan.plan_id,
        note=note,
        symbols=sorted(chosen),
        fills=[f.status.model_dump(mode="json") for f in report.fills],
        unfilled=[f.status.model_dump(mode="json") for f in report.unfilled],
        rejections=[r.model_dump(mode="json") for r in report.rejections],
    )
    send(
        rt,
        result,
        alerts.Alert(title="Manual override (paper)", message=note, priority=alerts.HIGH),
    )
    for f in report.fills:
        st = f.status
        assert st.filled_avg_price is not None
        send(
            rt,
            result,
            alerts.fill_alert(st.symbol, st.side, st.filled_qty, st.filled_avg_price, s.mode),
        )
    for f in report.unfilled:
        st = f.status
        if not st.terminal:
            state.open_orders = [*state.open_orders, _open_order(f.order, st, session, before)]
        send(
            rt,
            result,
            alerts.rejected_order_alert(st.symbol, st.side, st.detail or st.status, False),
        )
    for r in report.rejections:
        send(rt, result, alerts.skipped_order_alert(r.symbol, r.side, r.reason, r.detail))
    state.mark_override(session)
    store.save(state)
    return result


def _override_symbols(
    rt: Runtime, plan: Plan, buys: list[PlannedOrder], symbols: list[str] | None
) -> tuple[set[str], str]:
    planned = {o.symbol for o in buys}
    if symbols:
        return {x.upper() for x in symbols} & planned, "symbols named by hand"
    client = make_approvals(rt)
    tapped: dict[str, approvals.Decision] = {}
    if client is not None:
        try:
            msgs = client.poll(plan.created_at - timedelta(minutes=1))
        except (httpx.HTTPError, RuntimeError) as exc:
            log.warning("override: could not read the approval topic: %s", exc)
            msgs = []
        for m in msgs:
            d = approvals.parse(m)
            if d is None or d.plan_id != plan.plan_id or d.symbol not in planned:
                continue
            if d.symbol not in tapped or d.at >= tapped[d.symbol].at:
                tapped[d.symbol] = d
    approved = {sym for sym, d in tapped.items() if d.verb == approvals.APPROVE}
    if approved:
        return approved, "the symbols tapped Approve"
    return planned, "no Approve taps found, so every planned buy"


def protect(
    rt: Runtime,
    broker: stops.StopBroker,
    state: State,
    report: ExecutionReport,
    session: date,
    result: JobResult,
) -> None:
    """Place the day's protective stops. A failure alerts urgently and never fails EXECUTE.

    Runs after the plan, so a stop is never in the way of a planned sell, and
    whether or not the kill switch is on: protective stops always stay in place.
    """
    s = rt.settings
    try:
        positions = broker.positions()
        bought = {f.status.symbol for f in report.fills if f.status.side is Side.BUY}
        bought |= {f.order.symbol for f in report.unfilled if f.order.side is Side.BUY}
        for sym, sides in traded_sides(broker.orders_for_session(session)).items():
            if Side.BUY in sides:
                bought.add(sym)
        held = sorted((set(positions) & set(s.watchlist)) - bought)
        try:
            prices = broker.latest_prices(held) if held else {}
        except Exception as exc:
            log.warning("latest prices for stops unavailable: %s", type(exc).__name__)
            prices = {}
        run = stops.place_stops(
            broker,
            state,
            positions,
            session,
            managed=s.watchlist,
            bought_today=bought,
            prices=prices,
        )
    except Exception as exc:
        send(rt, result, stops.failure_alert("place", "positions", f"{type(exc).__name__}: {exc}"))
        return
    for a in run.alerts:
        send(rt, result, a)
    rt.journal.write(
        rt.now(),
        "execute",
        session,
        "stops",
        placed=[p.model_dump(mode="json") for p in run.placed],
        kept=[p.model_dump(mode="json") for p in run.kept],
        problems=[a.title for a in run.alerts],
    )


def make_gate(
    rt: Runtime, state: State, plan: Plan, session: date, result: JobResult
) -> Callable[
    [list[PlannedOrder], Mapping[str, Decimal]], tuple[list[PlannedOrder], list[Rejection]]
]:
    """The last checks before an order is sent: the hand approval and the price cross-check."""
    s = rt.settings
    w = s.job_windows
    deadline = min(
        clock.session(session).open
        + timedelta(minutes=w.execute_after_open_minutes + w.execute_window_minutes),
        clock.at_ny(session, w.approval_deadline),
    )
    verdicts: dict[str, approvals.Verdict] | None = None

    def need_approval(o: PlannedOrder) -> bool:
        return needs_tap(o, s, state.approval_required)

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


def needs_tap(o: PlannedOrder, s: Settings, sells_need: bool) -> bool:
    """Whether an order waits for Approve. Buys always do in paper and live; sells per mode."""
    if o.reason in s.approval.exempt_reasons:
        return False
    if o.side is Side.BUY and s.approval.buys == "always" and s.mode in ("paper", "live"):
        return True
    return sells_need


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


# ---- SWEEP (9:00 am) -----------------------------------------------------------------


def sweep_job(rt: Runtime, force: bool = False) -> JobResult:
    """The morning sweep: overnight news, then the final picks with Approve and Skip."""
    return _guarded(rt, "sweep", lambda: _sweep(rt, force))


def sweep_session(now: datetime, s: Settings) -> date | None:
    """The session the sweep may act for at ``now``: from sweep_start until EXECUTE starts."""
    day = clock.to_ny(now).date()
    if not clock.is_session(day):
        return None
    w = s.job_windows
    start = clock.at_ny(day, w.sweep_start)
    end = clock.session(day).open + timedelta(minutes=w.execute_after_open_minutes)
    return day if start <= now < end else None


def make_reader(rt: Runtime) -> Reader:
    if rt.reader is None:
        from stoa.reader import FinBert

        c = rt.settings.sweep
        rt.reader = FinBert(
            c.model, c.revision, device="cpu", batch_size=c.batch_size, max_length=c.max_length
        )
    reader: Reader = rt.reader
    return reader


def read_overnight(
    rt: Runtime, plan: Plan, session: date, symbols: list[str]
) -> dict[str, sweep.Tone]:
    """FinBERT's read of every article published between the plan's close and the cutoff."""
    from stoa.data.news import AlpacaNewsClient

    s = rt.settings
    since = clock.session(plan.decided_session).close
    until = min(rt.now(), clock.at_ny(session, s.job_windows.news_cutoff))
    client = rt.news_client
    if client is None:
        client = AlpacaNewsClient(alpaca_data_client(s, rt.secrets), include_content=False)
    df = client.fetch(symbols, since.date(), session)
    if not df.empty:
        created = pd.to_datetime(df["created_at"], utc=True)
        known = pd.to_datetime(df["known_at"], utc=True)
        df = df[(created > pd.Timestamp(since)) & (known <= pd.Timestamp(until))]
    if df.empty:
        return {sym: sweep.quiet(sym) for sym in symbols}  # a quiet night needs no model
    return sweep.read_tones(df, symbols, make_reader(rt), s.sweep)


def _clock_text(ts: datetime) -> str:
    ny = clock.to_ny(ts)
    return f"{ny.hour % 12 or 12}:{ny.minute:02d} {'am' if ny.hour < 12 else 'pm'}"


def _sweep(rt: Runtime, force: bool) -> JobResult:
    s = rt.settings
    w = s.job_windows
    now = rt.now()
    session = sweep_session(now, s)
    if session is None:
        if not force:
            log.info(
                "SWEEP: outside the window at %s ET; nothing to do",
                clock.to_ny(now).strftime("%Y-%m-%d %H:%M"),
            )
            return JobResult(EXIT_OK, "outside_window")
        day = clock.to_ny(now).date()
        open_today = clock.is_session(day) and now < clock.session(day).close
        session = day if open_today else clock.next_session(day)
    check_mode(s, rt.secrets)
    store = rt.store
    state = store.load()
    if session in state.swept_sessions and not force:
        log.info("SWEEP: session %s already swept", session)
        return JobResult(EXIT_OK, "already_done", session)
    result = JobResult(EXIT_OK, "swept", session)

    # News up to the cutoff counts; a run that starts a few minutes early waits for it.
    wait = (clock.at_ny(session, w.news_cutoff) - rt.now()).total_seconds()
    if 0 < wait <= 20 * 60:
        rt.sleep(wait)

    notes: list[str] = []
    plan = state.pending_plan
    prev = clock.previous_session(session)
    if (plan is None or plan.execute_session != session) and prev not in state.decided_sessions:
        caught = _decide(rt, force=True, summary=False)
        result.alerts_sent.extend(caught.alerts_sent)
        state = store.load()
        plan = state.pending_plan
        notes.append("The evening run did not happen, so the plan was made this morning.")
    if plan is None or plan.execute_session != session:
        send(
            rt,
            result,
            alerts.Alert(
                title="No picks today",
                message="There is no plan for this morning's open. Nothing will be traded.",
                priority=alerts.LOW,
                tags=("zzz",),
            ),
        )
        state.mark_swept(session)
        store.save(state)
        result.status = "no_plan"
        return result

    tones: dict[str, sweep.Tone] = {}
    outcome = sweep.Outcome(plan)
    watch = sorted(
        {o.symbol for o in plan.orders if o.side is Side.BUY} | {r.symbol for r in plan.reserves}
    )
    if s.sweep.enabled and watch and not plan.kill_switch:
        try:
            tones = read_overnight(rt, plan, session, watch)
            held = set(state.sim_positions) if s.mode == "signal" else set(state.positions_meta)
            outcome = sweep.apply(
                plan, tones, s.sweep, max_positions=s.risk.max_positions, held=held
            )
        except Exception as exc:
            log.warning("news sweep failed: %s", traceback.format_exc())
            notes.append(
                f"The news sweep could not run ({type(exc).__name__}); "
                "these are the rules' picks, unchanged."
            )
    plan = outcome.plan
    state.pending_plan = plan

    sells_need = approvals.required(s.approval, s.mode, state.track.first_live_session, session)
    state.approval_required = sells_need and any(o.side is Side.SELL for o in plan.orders)
    tapped = [o for o in plan.orders if needs_tap(o, s, state.approval_required)]
    state.approval_nonce = approvals.new_nonce() if tapped else None

    journal = rt.journal
    journal.write(
        now,
        "sweep",
        session,
        "sweep",
        tones={k: v.__dict__ for k, v in tones.items()},
        dropped=[o.symbol for o, _ in outcome.dropped],
        added=[o.symbol for o, _ in outcome.added],
        plan=plan.model_dump(mode="json"),
        approval_required=bool(tapped),
        notes=notes,
    )

    for o, tone in outcome.dropped:
        send(
            rt,
            result,
            alerts.Alert(
                title=f"Dropped BUY {o.symbol}",
                message=(
                    f"Bad overnight news: {tone.worst} (FinBERT negative {tone.max_negative:.2f}, "
                    f"mean tone {tone.mean_net:+.2f} over {tone.articles}). "
                    f"Not buying {o.symbol} today."
                ),
                priority=alerts.DEFAULT,
                tags=("newspaper",),
            ),
        )

    deadline = clock.at_ny(session, w.approval_deadline)
    when = "at " + _clock_text(
        clock.session(session).open + timedelta(minutes=w.execute_after_open_minutes)
    )
    topic = rt.secrets.ntfy_approval_topic
    late = rt.now() > deadline
    for order in plan.orders:
        alert = alerts.signal_alert(order, plan.equity_at_decide, s.mode, when=when)
        seen = tones.get(order.symbol)
        lines = [seen.line()] if seen is not None and order.side is Side.BUY else []
        acts: list[dict[str, Any]] = []
        if order in tapped:
            if late:
                lines.append(f"Approvals closed at {_clock_text(deadline)}; this will be skipped.")
            elif topic is None:
                lines.append(
                    "Approval is required but STOA_NTFY_APPROVAL_TOPIC is not set, "
                    "so this order will be skipped."
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
                lines.append(
                    f"Tap Approve or Skip by {_clock_text(deadline)}. "
                    "Nothing is sent without Approve."
                )
        if lines:
            alert = alerts.with_actions(alert, acts, "\n".join(lines))
        send(rt, result, alert)
    if not plan.orders or notes:
        head = "No trades this morning." if not plan.orders else "Morning sweep notes."
        send(
            rt,
            result,
            alerts.Alert(
                title="No trades today" if not plan.orders else "Morning sweep",
                message=" ".join([head, *notes]),
                priority=alerts.LOW,
                tags=("newspaper",),
            ),
        )
    state.mark_swept(session)
    store.save(state)
    return result


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
