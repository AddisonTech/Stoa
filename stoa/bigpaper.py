"""The big paper account: trial 13 on its own ~$100,000 Alpaca paper account.

It is kept apart from the $100 account in every way:

- its own keys, ALPACA_BIG_PAPER_KEY_ID and ALPACA_BIG_PAPER_SECRET, read only here and
  refused when they match the $100 account's keys; the $100 account never reads them;
- its own state folder (``big_paper.state_dir``), journal, equity record and workflow;
- every alert's title starts with ``big_paper.tag`` ("BIG PAPER").

It trades only when ``big_paper.enabled`` is true and both keys are set. Otherwise every
job logs why and stops, sending nothing.

The day, on the $100 account's clock:

- 9:00 am PICKS: trial 13's picks for the session (the news brain, ``newsbrain.live``),
  run through the event strategy with trial 13's overlay. Paper with hundreds of trades
  to come, so the picks are auto-approved: the alert says what will be bought and sold.
- 9:45 am EXECUTE: yesterday's picks sold, today's bought, idle money in SPY under the
  overlay's exposure cap, broker-side stops on anything held from a prior session. The
  kill switch sells every position in the account, SPY and any stock bought by hand
  included, and cancels every working order first.
- 5:15 pm SUMMARY: the day's equity against SPY, the high-water mark, the kill switch,
  the drawdown brake for the next morning, stop-outs, and the stops for tomorrow.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from stoa import alerts, clock, event_signals, risk, stops
from stoa.broker.alpaca import AlpacaBroker, make_trading_client
from stoa.broker.base import Broker, BrokerPosition
from stoa.broker.core import CoreBroker
from stoa.config import Secrets, Settings
from stoa.data.base import Bars, DataError
from stoa.data.factory import make_provider
from stoa.earnings import EarningsCalendar
from stoa.event_strategy import EventStrategy
from stoa.features import build_features, cell, slice_features
from stoa.journal import Journal
from stoa.live import live_calendar
from stoa.models import ClosedTrade, Holding, PlannedOrder, PositionMeta, Side, dec
from stoa.newsbrain import trial as t9
from stoa.newsbrain import trial13
from stoa.overlay import Overlay, OverlayState, account_settings
from stoa.planner import DecisionInput, make_plan
from stoa.state import KillSwitch, State, StateStore
from stoa.trader import ExecutionReport, execute_plan
from stoa.universe import AssetInfo, Universe

log = logging.getLogger(__name__)

EXIT_OK, EXIT_FAILED = 0, 1
KIND = t9.kind_for(trial13.GRADED)
SignalSource = Callable[[date], list[event_signals.Signal]]


class Tagged:
    """Every alert from the big account, titled with its tag so it never reads as the $100's."""

    def __init__(self, inner: alerts.Notifier, tag: str) -> None:
        self.inner, self.tag = inner, tag

    def send(self, alert: alerts.Alert) -> bool:
        title = alert.title if alert.title.startswith(self.tag) else f"{self.tag}: {alert.title}"
        return self.inner.send(
            alerts.Alert(
                title=title,
                message=alert.message,
                priority=alert.priority,
                tags=alert.tags,
                click=alert.click,
                markdown=alert.markdown,
            )
        )


@dataclass
class Result:
    code: int
    status: str
    session: date | None = None
    sent: list[str] = field(default_factory=list)


@dataclass
class BigRuntime:
    """Everything the big account touches from outside, injectable for tests."""

    base: Settings  # the loaded config; ``settings`` is the account's view of it
    secrets: Secrets
    notifier: alerts.Notifier
    now: Callable[[], datetime] = clock.now_utc
    broker: Broker | None = None
    bars: Callable[[list[str], date], Bars] | None = None
    signals: SignalSource | None = None
    calendar: Callable[[Bars, list[str], date], EarningsCalendar] | None = None
    sectors: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        self.notifier = Tagged(self.notifier, self.base.big_paper.tag)

    @property
    def settings(self) -> Settings:
        """Trial 13's account: its size cap, its starting money and its own state folder."""
        bp = self.base.big_paper
        s = account_settings(self.base, trial13.OVERLAY, bp.starting_equity)
        return s.model_copy(update={"state": s.state.model_copy(update={"dir": bp.state_dir})})

    @property
    def root(self) -> Path:
        return self.base.big_paper.state_dir

    @property
    def store(self) -> StateStore:
        return StateStore(self.root)

    @property
    def journal(self) -> Journal:
        return Journal(self.root / "journal")


# ---- the separation ------------------------------------------------------------------------


def off_reason(settings: Settings, secrets: Secrets) -> str | None:
    """Why the big account must not trade now, or None."""
    if not settings.big_paper.enabled:
        return "big_paper.enabled is false in the config"
    if not secrets.has_big_paper_keys():
        return "ALPACA_BIG_PAPER_KEY_ID and ALPACA_BIG_PAPER_SECRET are not both set"
    assert secrets.alpaca_big_paper_key_id is not None
    small = secrets.alpaca_paper_key_id
    if small is not None and (
        small.get_secret_value() == secrets.alpaca_big_paper_key_id.get_secret_value()
    ):
        return "the big paper keys are the $100 account's keys; the accounts must be different"
    return None


def make_broker(rt: BigRuntime) -> Broker:
    """The big account at Alpaca's paper endpoint, on its own keys only. Never the live URL."""
    if rt.broker is not None:
        return rt.broker
    s, sec = rt.base, rt.secrets
    assert sec.alpaca_big_paper_key_id and sec.alpaca_big_paper_secret
    key = sec.alpaca_big_paper_key_id.get_secret_value()
    secret = sec.alpaca_big_paper_secret.get_secret_value()
    data = httpx.Client(
        base_url=s.alpaca.data_url,
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        timeout=s.data.timeout_seconds,
    )
    rt.broker = AlpacaBroker(
        make_trading_client(s.alpaca.paper_url, key, secret),
        data,
        live=False,
        quote_feed=s.data.quote_feed,
        max_retries=s.data.max_retries,
        fill_timeout=s.alpaca.fill_timeout_seconds,
        poll_interval=s.alpaca.poll_interval_seconds,
    )
    return rt.broker


def data_secrets(secrets: Secrets) -> Secrets:
    """The big account's keys in the slots the bar provider reads, and nothing else."""
    return Secrets(
        alpaca_paper_key_id=secrets.alpaca_big_paper_key_id,
        alpaca_paper_secret=secrets.alpaca_big_paper_secret,
    )


def fetch_bars(rt: BigRuntime, symbols: list[str], through: date) -> Bars:
    if rt.bars is not None:
        return rt.bars(symbols, through)
    s = rt.settings
    provider = make_provider(s, data_secrets(rt.secrets))
    start = through - timedelta(days=s.data.live_lookback_days)
    return provider.fetch(sorted(set(symbols) | {s.benchmark}), start, through)


def send(rt: BigRuntime, result: Result, alert: alerts.Alert) -> None:
    if alerts.safe_send(rt.notifier, alert):
        result.sent.append(alert.title)


# ---- the overlay's memory and the equity record ----------------------------------------------


def load_overlay(rt: BigRuntime) -> Overlay:
    path = rt.root / "overlay.json"
    ov = Overlay(trial13.OVERLAY)
    if path.exists():
        body = json.loads(path.read_text(encoding="utf-8"))
        ov.halted = bool(body.get("halted"))
        cur = body.get("current")
        if cur:
            ov.current = OverlayState(
                session=date.fromisoformat(cur["session"]),
                drawdown=Decimal(cur["drawdown"]),
                size_multiplier=Decimal(cur["size_multiplier"]),
                halted=bool(cur["halted"]),
                exposure_cap=Decimal(cur["exposure_cap"]),
            )
    return ov


def save_overlay(rt: BigRuntime, ov: Overlay) -> None:
    cur = ov.current
    body: dict[str, Any] = {"halted": ov.halted, "current": None}
    if cur is not None:
        body["current"] = {
            "session": cur.session.isoformat(),
            "drawdown": str(cur.drawdown),
            "size_multiplier": str(cur.size_multiplier),
            "halted": cur.halted,
            "exposure_cap": str(cur.exposure_cap),
        }
    rt.root.mkdir(parents=True, exist_ok=True)
    (rt.root / "overlay.json").write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")


def read_curve(root: Path) -> list[dict[str, Any]]:
    path = root / "curve.json"
    rows: list[dict[str, Any]] = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    )
    return rows


def record_close(root: Path, session: date, equity: Decimal, spy_close: float | None) -> None:
    rows = [r for r in read_curve(root) if r["session"] != session.isoformat()]
    rows.append({"session": session.isoformat(), "equity": float(equity), "spy_close": spy_close})
    rows.sort(key=lambda r: str(r["session"]))
    root.mkdir(parents=True, exist_ok=True)
    (root / "curve.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


# ---- helpers -------------------------------------------------------------------------------


def picks_held(positions: Mapping[str, BrokerPosition], s: Settings) -> dict[str, BrokerPosition]:
    """Every position but the SPY core: the picks, plus anything bought by hand."""
    return {k: p for k, p in positions.items() if k != s.benchmark and p.qty > 0}


def meta_for(state: State, sym: str, p: BrokerPosition, session: date) -> PositionMeta:
    m = state.positions_meta.get(sym)
    if m is not None:
        return m
    # A position Stoa has no record of (bought by hand): treat it as held from before, so
    # the 1 session exit sells it at the next 9:45 am.
    return PositionMeta(
        entry_session=clock.previous_session(session),
        entry_price=p.avg_entry_price,
        entry_atr=Decimal(0),
    )


def sectors(rt: BigRuntime) -> Mapping[str, str]:
    if rt.sectors is not None:
        return rt.sectors
    path = Path("data/events/sic.json")
    if not path.exists():
        return {}
    from stoa.event_strategy import sector_of_sic

    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return {k: sector_of_sic(v) for k, v in raw.items()}


def default_signals(session: date) -> list[event_signals.Signal]:
    from stoa.newsbrain import live

    return live.signals(session)


def calendar(rt: BigRuntime, bars: Bars, symbols: list[str], session: date) -> EarningsCalendar:
    if rt.calendar is not None:
        return rt.calendar(bars, symbols, session)
    from stoa.data.external import pick_source

    s = rt.settings
    key = rt.secrets.finnhub_key
    src = pick_source(s.earnings.upcoming_source, key.get_secret_value() if key else None)
    upcoming = None
    if src is not None and symbols:
        try:
            upcoming = src.upcoming(symbols, session, session + timedelta(days=30))
        except Exception as exc:
            log.warning("upcoming earnings unavailable: %s", type(exc).__name__)
    return live_calendar(s, bars, upcoming)


def _guarded(rt: BigRuntime, job: str, body: Callable[[], Result]) -> Result:
    try:
        return body()
    except Exception as exc:
        log.error("big paper %s failed:\n%s", job, traceback.format_exc())
        r = Result(EXIT_FAILED, "failed")
        send(rt, r, alerts.job_failure_alert(job, f"{type(exc).__name__}: {exc}"))
        return r


def _off(rt: BigRuntime, job: str) -> Result | None:
    why = off_reason(rt.base, rt.secrets)
    if why is None:
        return None
    log.info("big paper %s: off (%s); nothing done, nothing sent", job, why)
    return Result(EXIT_OK, f"off: {why}")


# ---- 9:00 am: PICKS ---------------------------------------------------------------------------


def picks(rt: BigRuntime, force: bool = False) -> Result:
    return _off(rt, "picks") or _guarded(rt, "picks", lambda: _picks(rt, force))


def picks_session(now: datetime, s: Settings) -> date | None:
    day = clock.to_ny(now).date()
    if not clock.is_session(day):
        return None
    w = s.job_windows
    start = clock.at_ny(day, w.sweep_start)
    end = clock.session(day).open + timedelta(minutes=w.execute_after_open_minutes)
    return day if start <= now < end else None


def _picks(rt: BigRuntime, force: bool) -> Result:
    s = rt.settings
    now = rt.now()
    session = picks_session(now, s)
    if session is None:
        if not force:
            return Result(EXIT_OK, "outside_window")
        session = clock.next_session(clock.to_ny(now).date() - timedelta(days=1))
    store = rt.store
    state = store.load()
    if session in state.swept_sessions and not force:
        return Result(EXIT_OK, "already_done", session)
    result = Result(EXIT_OK, "picked", session)
    decided = clock.previous_session(session)  # the engine decides as of the close before
    broker = make_broker(rt)
    account = broker.account()
    raw = broker.positions()
    held = picks_held(raw, s)
    core = raw.get(s.benchmark)
    notes: list[str] = []
    try:
        sigs = (rt.signals or default_signals)(session)
    except Exception as exc:
        sigs = []
        notes.append(f"No new picks today: the news brain could not run ({exc}).")
    t9.register_kind(trial13.GRADED)
    wanted = sorted({x.symbol for x in sigs} | set(held))
    bars = fetch_bars(rt, wanted, decided)
    if s.benchmark not in bars:
        raise DataError(f"no {s.benchmark} bars through {decided}")
    feats = build_features(bars, s.benchmark)
    assets = {sym: broker.asset(sym) for sym in wanted}
    assets[s.benchmark] = AssetInfo(s.benchmark, tradable=True, fractionable=True)
    universe = Universe(
        watchlist=tuple(sorted(set(wanted) | {s.benchmark})),
        benchmark=s.benchmark,
        assets=assets,
        whole_shares=frozenset(k for k, a in assets.items() if not a.fractionable),
    )
    ov = load_overlay(rt)
    strategy = EventStrategy(
        settings=s,
        signals={decided: event_signals.ranked(sigs)},
        earnings=calendar(rt, bars, wanted, decided),
        sectors=sectors(rt),
        name="trial13",
        kind_of=dict.fromkeys(held, KIND),
        overlay=ov,
    )
    equity = state.last_close_equity or account.equity
    holdings = {
        sym: Holding(
            symbol=sym,
            qty=p.qty,
            avg_entry_price=p.avg_entry_price,
            meta=meta_for(state, sym, p, session),
        )
        for sym, p in held.items()
        if sym in feats
    }
    cash = account.cash + (core.market_value if core is not None else Decimal(0))
    decision = strategy.decide(
        DecisionInput(
            session=decided,
            execute_session=session,
            features=slice_features(feats, decided),
            holdings=holdings,
            equity=equity,
            cash=cash,
            prev_equity=account.last_equity,
            high_water_mark=state.high_water_mark or equity,
            kill_switch_active=state.kill_switch.active,
            universe=universe,
        )
    )
    # A held stock with no bars to judge it is still sold on the 1 session rule.
    blind = [
        PlannedOrder(
            symbol=sym,
            side=Side.SELL,
            qty=p.qty,
            ref_price=p.current_price,
            reason="hold_period",
            thesis="no daily bars to judge it; sold on the 1 session hold",
        )
        for sym, p in sorted(held.items())
        if sym not in holdings
    ]
    if blind:
        decision = dataclasses.replace(decision, orders=(*decision.orders, *blind))
    plan = make_plan(
        decision,
        strategy="trial13",
        mode="paper",
        session=decided,
        execute_session=session,
        created_at=now,
        equity=equity,
    )
    state.pending_plan = plan
    state.swept_sessions = [*state.swept_sessions, session][-60:]
    store.archive_plan(plan)
    save_overlay(rt, ov)
    store.save(state)
    rt.journal.write(
        now,
        "picks",
        session,
        "plan",
        plan=plan.model_dump(mode="json"),
        signals=[{"symbol": x.symbol, "p": x.score, "detail": x.detail} for x in sigs],
        rejections=[r.model_dump(mode="json") for r in decision.rejections],
        notes=notes,
    )
    send(rt, result, picks_alert(plan.orders, sigs, ov.current, notes, session))
    return result


def picks_alert(
    orders: Any,
    sigs: list[event_signals.Signal],
    st: OverlayState | None,
    notes: list[str],
    session: date,
) -> alerts.Alert:
    p_of = {x.symbol: x.score for x in sigs}
    buys = [o for o in orders if o.side is Side.BUY]
    sells = [o for o in orders if o.side is Side.SELL]
    lines = [f"Picks for {session}, auto-approved (paper). Orders go in at 9:45 am."]
    for o in buys:
        lines.append(
            f"Buy {o.symbol} {alerts.money(o.notional or 0)}"
            + (f" (p {p_of[o.symbol]:.2f})" if o.symbol in p_of else "")
        )
    if sells:
        lines.append(
            "Sell: " + ", ".join(f"{o.symbol} ({o.reason.replace('_', ' ')})" for o in sells)
        )
    if not buys:
        lines.append("No buys today.")
    if st is not None and st.in_drawdown:
        lines.append(
            f"Drawdown {st.drawdown:.1%}: "
            + ("no new positions (the brake)." if st.halted else "new positions at half size.")
        )
    lines += notes
    return alerts.Alert(title=f"Picks {session}", message="\n".join(lines), priority=alerts.DEFAULT)


# ---- 9:45 am: EXECUTE --------------------------------------------------------------------------


def execute(rt: BigRuntime, force: bool = False) -> Result:
    return _off(rt, "execute") or _guarded(rt, "execute", lambda: _execute(rt, force))


def _execute(rt: BigRuntime, force: bool) -> Result:
    s = rt.settings
    now = rt.now()
    w = s.job_windows
    session = clock.execute_session(now, w.execute_after_open_minutes, w.execute_window_minutes)
    if session is None:
        if not force:
            return Result(EXIT_OK, "outside_window")
        session = clock.to_ny(now).date()
    store = rt.store
    state = store.load()
    if session in state.executed_sessions and not force:
        return Result(EXIT_OK, "already_done", session)
    result = Result(EXIT_OK, "executed", session)
    plan = state.pending_plan
    if plan is None or plan.execute_session != session:
        if plan is not None and plan.execute_session < session:
            state.pending_plan = None
        state.mark_executed(session)
        store.save(state)
        send(
            rt,
            result,
            alerts.Alert(
                title=f"No plan for {session}",
                message="The 9:00 am picks did not run for today, so nothing was bought or sold. "
                "Yesterday's positions stay until the next plan.",
                priority=alerts.HIGH,
            ),
        )
        result.status = "no_plan"
        return result
    raw = make_broker(rt)
    account = raw.account()
    if account.trading_blocked:
        raise RuntimeError(f"Alpaca says trading is blocked on this account ({account.status})")
    if isinstance(raw, AlpacaBroker) and not raw.market_is_open():
        raise RuntimeError("Alpaca says the market is closed; orders would queue blind")
    hwm = state.high_water_mark or account.equity
    flatten = state.kill_switch.active or risk.kill_switch_breached(account.equity, hwm, s.risk)
    # The kill switch runs on the raw account so every position goes, SPY included.
    broker: Any = (
        raw
        if flatten
        else CoreBroker(
            raw, symbol=s.benchmark, reserve_pct=s.risk.cash_buffer_pct, session=session
        )
    )
    buys = sorted({o.symbol for o in plan.orders if o.side is Side.BUY})
    assets = {sym: raw.asset(sym) for sym in buys}
    universe = Universe(
        watchlist=tuple(
            sorted({o.symbol for o in plan.orders} | set(raw.positions()) | {s.benchmark})
        ),
        benchmark=s.benchmark,
        assets=assets,
        whole_shares=frozenset(k for k, a in assets.items() if not a.fractionable),
    )
    meta_before = dict(state.positions_meta)

    def cancel_before_sells(symbols: list[str]) -> None:
        if isinstance(raw, stops.StopBroker):
            for a in stops.cancel_stops(raw, symbols):
                send(rt, result, a)

    def cancel_everything() -> None:
        if isinstance(raw, stops.StopBroker):
            for a in stops.cancel_all_orders(raw):
                send(rt, result, a)

    report = execute_plan(
        plan,
        broker,
        state,
        s,
        universe,
        session,
        state.last_close_equity,
        gate=None,  # auto-approved: paper, and hundreds of trades to come
        sectors={o.symbol: o.sector or "Unknown" for o in plan.orders},
        before_sells=cancel_before_sells,
        before_flatten=cancel_everything,
    )
    if report.kill_switch_tripped:
        dd = float(risk.drawdown(account.equity, hwm))
        send(rt, result, alerts.kill_switch_alert(account.equity, hwm, dd))
    if isinstance(broker, CoreBroker):
        ov = load_overlay(rt)
        broker.sweep(state.kill_switch.active, ov.current.exposure_cap if ov.current else None)
    closed = closed_trades(report, meta_before)
    state.add_trades(closed)
    for t in closed:
        state.add_realized(session.year, t.pnl)
    if isinstance(raw, stops.StopBroker):
        protect(rt, raw, state, report, session, result)
    rt.journal.write(
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
        core=[o.model_dump(mode="json") for o in getattr(broker, "swaps", [])],
    )
    send(rt, result, fills_alert(report, closed, broker, session))
    state.pending_plan = None
    state.mark_executed(session)
    store.save(state)
    return result


def closed_trades(report: ExecutionReport, meta: dict[str, PositionMeta]) -> list[ClosedTrade]:
    out = []
    for f in report.fills:
        st = f.status
        m = meta.get(st.symbol)
        if st.side is not Side.SELL or f.realized_pnl is None or st.filled_avg_price is None:
            continue
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


def protect(
    rt: BigRuntime,
    broker: stops.StopBroker,
    state: State,
    report: ExecutionReport,
    session: date,
    result: Result,
) -> None:
    """Broker-side ATR stops on every pick held from a prior session (the existing stop code)."""
    s = rt.settings
    try:
        positions = broker.positions()
        bought = {f.status.symbol for f in report.fills if f.status.side is Side.BUY}
        bought |= {f.order.symbol for f in report.unfilled if f.order.side is Side.BUY}
        managed = sorted(picks_held(positions, s))
        held = sorted(set(managed) - bought)
        prices = broker.latest_prices(held) if held else {}
        run = stops.place_stops(
            broker, state, positions, session, managed=managed, bought_today=bought, prices=prices
        )
    except Exception as exc:
        send(rt, result, stops.failure_alert("place", "positions", f"{type(exc).__name__}: {exc}"))
        return
    for a in run.alerts:
        send(rt, result, a)


def fills_alert(
    report: ExecutionReport, closed: list[ClosedTrade], broker: Any, session: date
) -> alerts.Alert:
    lines = []
    for f in report.fills:
        st = f.status
        if st.filled_avg_price is None:
            continue
        lines.append(
            f"{st.side.value.title()} {st.symbol} {alerts.shares(st.filled_qty)} at "
            f"{alerts.money(st.filled_avg_price)}"
        )
    for f in report.unfilled:
        lines.append(f"Not filled: {f.status.side.value} {f.status.symbol} ({f.status.status})")
    for r in report.rejections:
        lines.append(f"Skipped {r.side.value} {r.symbol}: {r.reason.replace('_', ' ')}")
    if closed:
        pnl = sum((t.pnl for t in closed), Decimal(0))
        lines.append(f"Closed {len(closed)} trades, {alerts.money(pnl)} in all.")
    swaps = [o for o in getattr(broker, "swaps", []) if o.filled]
    if swaps:
        lines.append(
            "SPY core: " + ", ".join(f"{o.side.value} {alerts.shares(o.filled_qty)}" for o in swaps)
        )
    if report.flattened:
        lines.insert(0, "KILL SWITCH: every position sold.")
    return alerts.Alert(
        title=f"Fills {session}",
        message="\n".join(lines) or "Nothing to trade today.",
        priority=alerts.URGENT if report.flattened else alerts.DEFAULT,
    )


# ---- 5:15 pm: SUMMARY ----------------------------------------------------------------------------


def summary(rt: BigRuntime, force: bool = False) -> Result:
    return _off(rt, "summary") or _guarded(rt, "summary", lambda: _summary(rt, force))


def _summary(rt: BigRuntime, force: bool) -> Result:
    s = rt.settings
    now = rt.now()
    session = clock.decide_session(
        now, s.job_windows.decide_after_close_minutes, s.job_windows.decide_end
    )
    if session is None:
        if not force:
            return Result(EXIT_OK, "outside_window")
        session = clock.last_closed_session(now)
    store = rt.store
    state = store.load()
    if session in state.decided_sessions and not force:
        return Result(EXIT_OK, "already_done", session)
    result = Result(EXIT_OK, "summarized", session)
    broker = make_broker(rt)
    positions = broker.positions()
    if isinstance(broker, stops.StopBroker):
        rec = stops.reconcile(broker, state, positions, session, s.costs, "paper")
        state.add_trades(rec.trades)
        for t in rec.trades:
            state.add_realized(t.exit_session.year, t.pnl)
        for a in rec.alerts:
            send(rt, result, a)
        if rec.trades:
            rt.journal.write(
                now,
                "summary",
                session,
                "stop_outs",
                closed=[t.model_dump(mode="json") for t in rec.trades],
            )
        positions = broker.positions()
    account = broker.account()
    equity = account.equity
    held = picks_held(positions, s)
    bars = fetch_bars(rt, sorted(held), session)
    spy = bars.get(s.benchmark)
    spy_close = cell(spy, spy.index[-1], "close") if spy is not None and len(spy) else None
    prev = state.last_close_equity
    prev_spy = float(state.last_benchmark_close) if state.last_benchmark_close else None
    hwm = risk.next_high_water_mark(state.high_water_mark, equity)
    state.high_water_mark = hwm
    if not state.kill_switch.active and risk.kill_switch_breached(equity, hwm, s.risk):
        state.kill_switch = KillSwitch(
            active=True, tripped_session=session, reason="drawdown limit"
        )
        dd = float(risk.drawdown(equity, hwm))
        send(rt, result, alerts.kill_switch_alert(equity, hwm, dd))
    ov = load_overlay(rt)
    st = ov.step(session, equity, hwm, s.risk.max_gross_exposure_pct)
    save_overlay(rt, ov)
    feats = build_features(bars, s.benchmark)
    stops.raise_stops(state, sorted(held), feats, s.exits)
    state.last_close_session = session
    state.last_close_equity = equity
    if spy_close is not None:
        state.last_benchmark_close = dec(spy_close)
    record_close(rt.root, session, equity, spy_close)
    state.mark_decided(session)
    store.save(state)
    curve = read_curve(rt.root)
    lines = [f"Equity {alerts.money(equity)}"]
    if prev:
        day = float(equity / prev - 1)
        spy_day = spy_close / prev_spy - 1 if spy_close and prev_spy else None
        lines[0] += f", today {alerts.pct(day)}" + (
            f" (SPY {alerts.pct(spy_day)})" if spy_day is not None else ""
        )
    if len(curve) > 1 and curve[0].get("spy_close") and spy_close:
        since = float(equity) / float(curve[0]["equity"]) - 1
        spy_since = spy_close / float(curve[0]["spy_close"]) - 1
        lines.append(
            f"Since {curve[0]['session']}: {alerts.pct(since)}, SPY {alerts.pct(spy_since)}."
        )
    core = positions.get(s.benchmark)
    lines.append(
        f"{len(held)} picks held, {alerts.money(core.market_value if core else 0)} in SPY, "
        f"{alerts.money(account.cash)} cash."
    )
    lines.append(
        f"Drawdown {st.drawdown:.1%} from the high."
        + (
            " Brake: no new positions."
            if st.halted
            else " Brake: half size."
            if st.in_drawdown
            else ""
        )
    )
    if state.kill_switch.active:
        lines.append("Kill switch ON: tomorrow's 9:45 am run sells everything.")
    send(rt, result, alerts.Alert(title=f"Day summary {session}", message="\n".join(lines)))
    return result


# ---- by hand ---------------------------------------------------------------------------------


def reset_kill_switch(rt: BigRuntime) -> Result:
    """Turn the big account's kill switch off by hand; the high-water mark restarts at today's."""
    off = _off(rt, "kill-switch-reset")
    if off is not None:
        return off
    store = rt.store
    state = store.load()
    equity = make_broker(rt).account().equity
    state.kill_switch = KillSwitch()
    state.high_water_mark = equity
    ov = load_overlay(rt)
    ov.halted = False
    ov.current = None
    save_overlay(rt, ov)
    store.save(state)
    result = Result(EXIT_OK, "reset")
    send(
        rt,
        result,
        alerts.Alert(
            title="Kill switch reset",
            message=f"Reset by hand. The high-water mark starts again at {alerts.money(equity)}.",
            priority=alerts.HIGH,
        ),
    )
    return result
