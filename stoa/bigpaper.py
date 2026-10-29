"""The big paper account: a news brain trial on its own ~$100,000 Alpaca paper account.

``big_paper.strategy`` names the trial. ``trial12`` runs trial 12 exactly as
registered (spec e75fa8775469de92): its sizing (up to 5 picks at 20% of the account each),
its rails, its SPY core, no overlay and no broker-side stops, as its backtest ran.
``trial13`` adds trial 13's overlay (``stoa/overlay.py``): ATR sizing under a 5% cap, the
drawdown brake, the exposure cap and broker-side ATR stops. ``trial14`` (spec
f385bc9d5040ef7b) is trial 12 exactly with idle money held as cash (``big_paper.idle_money``):
the 9:00 am plan sells any SPY still held, alongside that day's picks, and the 9:45 am run
never buys SPY. SPY stays the benchmark line.

It is kept apart from the $100 account in every way:

- its own keys, ALPACA_BIG_PAPER_KEY_ID and ALPACA_BIG_PAPER_SECRET, read only here and
  refused when they match the $100 account's keys; the $100 account never reads them;
- its own state folder (``big_paper.state_dir``), journal, equity record and workflow;
- every alert's title starts with ``big_paper.tag`` ("BIG PAPER").

It trades only when ``big_paper.enabled`` is true and both keys are set. Otherwise every
job logs why and stops, sending nothing.

A dry run (``--dry``, feed and picks only) needs the keys but not ``enabled``: it reads the
account, never sends an order, saves nothing to the account's state and logs its alerts
instead of sending them. The feed still pulls, advances and uploads the news brain's state,
which places no order. ``--at`` replays a past morning; each dry run also keeps what it
pulled and the rows it scored in the feed's dataset (raw/ and checks/), so the Dell can
rebuild the same sessions from scratch and compare.

The day, on the $100 account's clock:

- 9:00 am PICKS: the news brain's picks for the session, read from the rolling state
  the 8:15 am feed builds (``newsbrain.feed``), run through the event strategy. Paper with
  hundreds of trades to come, so the picks are auto-approved: the alert says what will be
  bought and sold. Stale or missing data means no buys that day, and the alert says why.
- 9:45 am EXECUTE: yesterday's picks sold, today's bought, idle money in SPY (in cash under
  trial 14). ``--dry`` makes today's plan as the picks would and runs it against a copy of
  the account in the simulator: no order is sent and nothing is saved. The kill
  switch sells every position in the account, SPY and any stock bought by hand included,
  and cancels every working order first. Under trial 13: the overlay's exposure cap on
  SPY and broker-side stops on anything held from a prior session.
- 5:15 pm SUMMARY: the day's equity against SPY, the high-water mark and the kill switch;
  under trial 13 also the drawdown brake for the next morning, stop-outs and stops.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from stoa import alerts, clock, event_signals, risk, stops
from stoa.broker.alpaca import AlpacaBroker, make_trading_client
from stoa.broker.base import Broker, BrokerPosition
from stoa.broker.core import CoreBroker
from stoa.broker.sim import SimBroker
from stoa.config import Secrets, Settings
from stoa.data.base import Bars, DataError
from stoa.data.factory import make_provider
from stoa.earnings import EarningsCalendar
from stoa.event_strategy import EventStrategy
from stoa.features import build_features, cell, slice_features
from stoa.journal import Journal
from stoa.live import live_calendar
from stoa.models import ClosedTrade, Holding, PlannedOrder, PositionMeta, Side, dec
from stoa.newsbrain import learn15, review15, trial13, trial15
from stoa.newsbrain import spycore as sc
from stoa.newsbrain import trial as t9
from stoa.overlay import Overlay, OverlayState, account_settings
from stoa.planner import Decision, DecisionInput, make_plan
from stoa.portfolio import Lot, Portfolio
from stoa.state import KillSwitch, State, StateStore
from stoa.trader import ExecutionReport, execute_plan
from stoa.universe import AssetInfo, Universe

log = logging.getLogger(__name__)

EXIT_OK, EXIT_FAILED = 0, 1
KIND = t9.kind_for(sc.GRADED)  # trial 12's 1 session signal; trials 13 and 14 use the same
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
    plan: Any = None  # the picks' plan, for a dry execute


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
    feed_sources: Any = None  # newsbrain.feed.Sources; Alpaca, EDGAR and FinBERT when None
    calendar: Callable[[Bars, list[str], date], EarningsCalendar] | None = None
    sectors: Mapping[str, str] | None = None
    dry: bool = False  # no orders, no saved state, alerts logged instead of sent
    # A dry replay of past mornings: the dataset's state as of this commit (id or message),
    # carried between the replay's steps in a local copy; the live state is never touched.
    replay_from: str | None = None
    # A dry run on a model version other than the configured one (big_paper.model).
    model_override: str | None = None
    hub: Any = None  # huggingface_hub.HfApi for the shadow scoreboard; from HF_TOKEN when None
    prices_945: Any = None  # shadowboard.Prices for grading; Alpaca's minute bars when None
    chosen: list[dict[str, Any]] | None = None  # trial 15: what the morning's picks saw

    def __post_init__(self) -> None:
        self.notifier = Tagged(self.notifier, self.base.big_paper.tag)

    @property
    def strategy(self) -> str:
        return self.base.big_paper.strategy

    @property
    def overlaid(self) -> bool:
        """Trial 13's overlay is on (``big_paper.strategy: trial13``)."""
        return self.strategy == "trial13"

    @property
    def top5(self) -> bool:
        """Trial 15: the top 5 by conviction every day, equal weight, idle money in cash."""
        return self.strategy == "trial15"

    @property
    def core(self) -> bool:
        """Idle money sits in SPY (trials 12 and 13); under trial 14 it stays in cash."""
        return self.base.big_paper.idle_money == "spy"

    @property
    def stopped(self) -> bool:
        """The owner's stop (``stop.big_paper``): sell everything at 9:45 am, then nothing."""
        return self.base.stop.big_paper

    @property
    def label(self) -> str:
        """The trial and where idle money sits, for the alerts."""
        if self.stopped:
            return "Stopped by the owner"
        if self.top5:
            return "Trial 15, top 5 a day, idle money in cash"
        return f"Trial {self.strategy.removeprefix('trial')}, idle money in " + (
            "SPY" if self.core else "cash"
        )

    @property
    def settings(self) -> Settings:
        """The account's settings: its trial's size cap, its starting money, its state folder.

        Trial 12's are the config's own (20% a position, 5 positions), as its backtest ran;
        trial 14's are trial 12's.
        """
        bp = self.base.big_paper
        if self.overlaid:
            s = account_settings(self.base, trial13.OVERLAY, bp.starting_equity)
        else:
            bt = self.base.backtest.model_copy(update={"starting_equity": bp.starting_equity})
            s = self.base.model_copy(update={"backtest": bt})
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


def off_reason(settings: Settings, secrets: Secrets, dry: bool = False) -> str | None:
    """Why the big account must not run now, or None. A dry run needs only the keys."""
    if not settings.big_paper.enabled and not dry:
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
    if rt.dry:
        log.info("dry run, not sent: %s\n%s", alert.title, alert.message)
        return
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


def feed_sources(rt: BigRuntime, static: Any, read_seconds: float = 0.0) -> Any:
    """Alpaca on the big account's own keys, EDGAR, FinBERT on the CPU, and under trial 15
    the local event reader with ``read_seconds`` to spend (``big_paper.reader``)."""
    from stoa.newsbrain import feed

    if rt.feed_sources is not None:
        return rt.feed_sources
    sec = rt.secrets
    assert sec.alpaca_big_paper_key_id and sec.alpaca_big_paper_secret
    reads = rt.top5 and rt.base.big_paper.reader and read_seconds > 0
    return feed.Sources(
        news=feed.alpaca_news(
            sec.alpaca_big_paper_key_id.get_secret_value(),
            sec.alpaca_big_paper_secret.get_secret_value(),
            set(static.universe),
        ),
        filings=feed.edgar_filings({int(c) for c in static.ciks["cik"]}),
        bars=feed.alpaca_bars(make_provider(rt.settings, data_secrets(sec))),
        scorer=feed.finbert(),
        reader=feed.local_reader(set(static.ciks["symbol"].astype(str))) if reads else None,
        read_seconds=read_seconds,
    )


def hf_token(rt: BigRuntime) -> str | None:
    """The token for the feed's private dataset, from this process's environment only."""
    if not rt.base.big_paper.feed_repo:
        return None
    tok = rt.secrets.hf_token
    if tok is None:
        from stoa.newsbrain.feed import FeedError

        raise FeedError("HF_TOKEN is not set, so the news brain's state cannot be read")
    return tok.get_secret_value()


def feed_signals(rt: BigRuntime, session: date, now: datetime) -> list[event_signals.Signal]:
    """The 9:00 am step: download the state the 8:15 am feed uploaded, pull what arrived
    since, bring it to the session's 9:00 am cutoff, and read the picks. Stale data raises."""
    from stoa.newsbrain import feed

    fd = feed.Feed(rt.base.big_paper.feed_dir, raw_tag(rt, "picks", now))
    token = hf_token(rt)
    load_feed(rt, fd, token)
    feed.run(fd, feed_sources(rt, fd.state().static, READ_AT_PICKS), now)
    stale = feed.problems(fd, session, now)
    if stale:
        raise feed.FeedError("; ".join(stale))
    version, sha = model_choice(rt)
    path = fd.root / "models" / version / "model_h1.pkl" if version else None
    if version and sha is None:  # a dry run's override: the hash its own meta.json gives
        meta = json.loads((fd.root / "models" / version / "meta.json").read_text(encoding="utf-8"))
        sha = str(meta["sha256"])
    rows, sigs = feed.scored(fd, session, path, sha)
    if rt.top5:
        sic_path = fd.state().root / "static" / "sic.json"
        sic = json.loads(sic_path.read_text(encoding="utf-8"))
        rules = active_rules15(rt)
        if rules:
            log.info("trial 15 rules in use: %s", rules)
        chosen = trial15.choose(review15.apply_rules(rows, rules), sic)
        rt.chosen = trial15.records(chosen)
        sigs = trial15.signals(chosen, session)
        if not rt.dry:
            keep_rows15(rt, fd, token, session, rows)
            shadow15(rt, fd, token, session, rows, rules, sic, now)
    log.info("picks model: %s", version or FROZEN_VERSION)
    if not rt.dry and not rt.base.stop.shadow:
        lock_scoreboard(rt, session, rows, version or FROZEN_VERSION, now)
    if rt.dry:
        feed.save_check(fd, session, rows, sigs)
        log.info("dry picks for %s: %s", session, [(x.symbol, round(x.score, 4)) for x in sigs])
        if token is not None:
            feed.push_state(
                fd,
                rt.base.big_paper.feed_repo,
                token,
                f"dry picks {fd.raw_tag}",
                only=[f"raw/{fd.raw_tag}/**", f"checks/{fd.raw_tag}/**"],
            )
    return sigs


FROZEN_VERSION = "t9-frozen"  # trial 9's frozen model, in the feed state's static/
READ_AT_FEED = 20 * 60.0  # the local event reader's budget at 8:15 am, in seconds
READ_AT_PICKS = 4 * 60.0  # and at 9:00 am, so the plan is made well before 9:30


def model_choice(rt: BigRuntime) -> tuple[str | None, str | None]:
    """The picks' model version and its pinned sha256: a dry run's override (its hash is read
    from its meta.json), else big_paper.model and model_sha256; (None, None) is the frozen
    model in the feed state."""
    if rt.model_override:
        return rt.model_override, None
    if rt.top5:
        took = review15.load_json(dir15(rt) / "promoted.json")
        if took.get("version") and took.get("sha256"):
            return str(took["version"]), str(took["sha256"])
    bp = rt.base.big_paper
    if not bp.model:
        return None, None
    if not bp.model_sha256:
        from stoa.newsbrain.feed import FeedError

        raise FeedError(f"big_paper.model {bp.model} has no model_sha256 to check it against")
    return bp.model, bp.model_sha256


def hub(rt: BigRuntime, token: str) -> Any:
    from stoa.newsbrain.feed import private_or_refuse

    if rt.hub is not None:
        return rt.hub
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    private_or_refuse(api, rt.base.big_paper.feed_repo)
    return api


def lock_scoreboard(
    rt: BigRuntime, session: date, rows: Any, version: str, now: datetime
) -> dict[str, Any] | None:
    """Lock the morning's full list before the open (stoa/shadowboard.py). Never stops the
    picks: a failure is logged and the day goes ungraded."""
    from stoa import shadowboard

    token = hf_token(rt)
    if token is None or rows is None or len(rows) == 0:
        return None
    try:
        work = Path(f"{rt.base.big_paper.feed_dir}-scoreboard")
        meta = shadowboard.lock(
            hub(rt, token),
            rt.base.big_paper.feed_repo,
            rt.root,
            work,
            session,
            shadowboard.predictions(rows, version),
            now,
        )
    except Exception as exc:
        log.warning("shadow scoreboard not locked for %s: %s", session, exc)
        return None
    log.info(
        "shadow scoreboard locked %s: %s rows, sha256 %s", session, meta["rows"], meta["sha256"]
    )
    return meta


def grade_scoreboard(rt: BigRuntime, now: datetime) -> list[dict[str, Any]]:
    """Grade every locked or backfilled morning whose exit price is in."""
    from stoa import shadowboard
    from stoa.data.factory import alpaca_data_client

    token = hf_token(rt)
    if token is None:
        return []
    prices = rt.prices_945
    if prices is None:
        prices = shadowboard.alpaca_945(alpaca_data_client(rt.settings, data_secrets(rt.secrets)))
    return shadowboard.grade_pending(
        hub(rt, token),
        rt.base.big_paper.feed_repo,
        rt.root,
        Path(f"{rt.base.big_paper.feed_dir}-scoreboard"),
        prices,
        now,
        lineup(rt),
    )


def lineup(rt: BigRuntime) -> list[tuple[str, str]]:
    """The scoreboard's columns: the picks' model, then each shadow model."""
    bp = rt.base.big_paper
    return [(bp.model or FROZEN_VERSION, "picks"), *((v, "shadow") for v in bp.shadow_models)]


def shadow_summary_line(rt: BigRuntime, now: datetime) -> str:
    """The 5:15 pm summary's shadow lines, the picks' model's and each shadow model's;
    grading never stops the summary."""
    from stoa import shadowboard

    try:
        grade_scoreboard(rt, now)
    except Exception as exc:
        log.warning("shadow scoreboard not graded: %s", exc)
        return f"Shadow: not graded today ({type(exc).__name__})."
    out = []
    for shadow in [None, *rt.base.big_paper.shadow_models]:
        day = shadowboard.latest_locked(rt.root, shadow)
        waiting = None
        if day is None:
            locks = shadowboard.locked_sessions(rt.root, shadow)
            if locks:
                nxt = clock.next_session(locks[0])
                waiting = f"first locked day {locks[0]} is graded after 9:45 am on {nxt}."
        out.append(shadowboard.shadow_line(day, waiting, shadow))
    return "\n".join(out)


def shadow_lock(rt: BigRuntime) -> Result:
    """Right after the 9:00 am picks, in the same run: score the morning's rows with each
    shadow model (big_paper.shadow_models) and lock each list beside the picks' own, before
    9:30 am. The rows are the ones the picks scored, read again from the feed folder the
    picks left; the picks' model must give them the picks' locked p to the last bit, or
    nothing is locked. It never changes the picks, their state or the account, and a dry
    run scores and logs without locking."""
    import pickle

    import pandas as pd

    from stoa import shadowboard
    from stoa.newsbrain import feed

    bp = rt.base.big_paper
    if off_reason(rt.base, rt.secrets, dry=rt.dry):
        return Result(EXIT_OK, "off")
    if rt.base.stop.shadow:
        log.info("shadow scoreboard: stopped by the owner (stop.shadow); nothing scored")
        return Result(EXIT_OK, "stopped")
    token = hf_token(rt)
    if not bp.shadow_models or token is None:
        return Result(EXIT_OK, "none")
    now = rt.now()
    fd = feed.Feed(bp.feed_dir)
    if rt.dry:
        session = fd.state().session
        checks = sorted(
            (fd.root / "checks").glob("*/rows.parquet"), key=lambda p: p.stat().st_mtime
        )
        if not checks:
            raise feed.FeedError("no dry picks rows in the feed folder")
        locked = pd.read_parquet(checks[-1])
    else:
        session = clock.to_ny(now).date()
        todo = [
            v
            for v in bp.shadow_models
            if shadowboard.folder("locked", session, v) not in shadowboard.logged_hashes(rt.root)
        ]
        if not todo:
            return Result(EXIT_OK, "already_locked", session)
        path = shadowboard.folder("locked", session)
        logged = shadowboard.logged_hashes(rt.root).get(path)
        if logged is None:
            log.info("shadow models: the picks locked nothing for %s, so neither do they", session)
            return Result(EXIT_OK, "no_picks_lock", session)
        blob = (Path(f"{bp.feed_dir}-scoreboard") / path / "predictions.parquet").read_bytes()
        if shadowboard.sha256(blob) != logged:
            raise feed.FeedError(f"the picks' locked file for {session} does not match its hash")
        locked = pd.read_parquet(io.BytesIO(blob))
    version, sha = model_choice(rt)
    model = fd.root / "models" / version / "model_h1.pkl" if version else None
    if version and sha is None:
        meta = json.loads((fd.root / "models" / version / "meta.json").read_text(encoding="utf-8"))
        sha = str(meta["sha256"])
    rows, _ = feed.scored(fd, session, model, sha)
    ours = rows.set_index("symbol")["p"].sort_index()
    theirs = locked.set_index("symbol")["p"].sort_index()
    if not (ours.index.equals(theirs.index) and (ours.to_numpy() == theirs.to_numpy()).all()):
        raise feed.FeedError(f"the rows read again for {session} do not give the picks' own p")
    rows = rows.drop(columns="p")
    store = feed.Feed(Path(f"{bp.feed_dir}-shadow"))
    failed = []
    for v, pin in bp.shadow_models.items():
        if not rt.dry and shadowboard.folder("locked", session, v) in shadowboard.logged_hashes(
            rt.root
        ):
            continue
        try:
            feed.pull_state(store, bp.feed_repo, token, only=[f"models/{v}/**"])
            blob = (store.root / "models" / v / "model_h1.pkl").read_bytes()
            if shadowboard.sha256(blob) != pin:
                raise feed.FeedError(f"{v} does not match its pinned sha256")
            brain = pickle.loads(blob)
            t9.register_kind(sc.GRADED)
            df = shadowboard.predictions(rows.assign(p=brain.predict(rows)), v)
            if rt.dry:
                log.info(
                    "dry shadow %s for %s: %s rows, %s above the line, top %s",
                    v,
                    session,
                    len(df),
                    int(df["above"].sum()),
                    [(s, round(p, 4)) for s, p in zip(df["symbol"][:5], df["p"][:5], strict=True)],
                )
                continue
            meta = shadowboard.lock(
                hub(rt, token),
                bp.feed_repo,
                rt.root,
                Path(f"{bp.feed_dir}-scoreboard"),
                session,
                df,
                rt.now(),
                shadow=v,
            )
            log.info(
                "shadow %s locked %s: %s rows, %s above the line, sha256 %s",
                v,
                session,
                meta["rows"],
                meta["above"],
                meta["sha256"],
            )
        except Exception as exc:
            log.warning("shadow %s not locked for %s: %s", v, session, exc)
            failed.append(v)
    if failed:
        return Result(EXIT_FAILED, "failed", session)
    return Result(EXIT_OK, "dry" if rt.dry else "locked", session)


def replay_store(rt: BigRuntime) -> Path:
    """The replay's own copy of the dataset, beside the feed folder."""
    return Path(f"{rt.base.big_paper.feed_dir}-replay")


def load_feed(rt: BigRuntime, fd: Any, token: str | None) -> None:
    """Download the state, or in a replay copy it from the replay's own copy of the dataset
    (filled once from the ``replay_from`` commit)."""
    from stoa.newsbrain import feed

    if token is None:
        return
    repo = rt.base.big_paper.feed_repo
    if rt.replay_from is None:
        version, _ = model_choice(rt)
        only = [*feed.STATE_FILES, f"models/{version}/**"] if version else None
        feed.pull_state(fd, repo, token, only=only)
        return
    store = replay_store(rt)
    if not (store / "feed.json").exists():
        rev = feed.commit_of(repo, token, rt.replay_from)
        log.info("replay from %s (%s)", rt.replay_from, rev)
        feed.pull_state(feed.Feed(store), repo, token, revision=rev)
    feed.copy_state(store, fd.root)


def save_feed(rt: BigRuntime, fd: Any, token: str | None, message: str) -> None:
    """Upload the state; a replay moves only its own copy on and uploads only its pulls."""
    from stoa.newsbrain import feed

    if token is None:
        return
    repo = rt.base.big_paper.feed_repo
    if rt.replay_from is None:
        feed.push_state(fd, repo, token, message)
        return
    feed.copy_state(fd.root, replay_store(rt))
    feed.push_state(fd, repo, token, f"replay {fd.raw_tag}", only=[f"raw/{fd.raw_tag}/**"])


def raw_tag(rt: BigRuntime, job: str, now: datetime) -> str | None:
    """Where a dry run keeps its pulls and checks in the feed's dataset; None for real runs."""
    return f"{now.astimezone(UTC):%Y%m%dT%H%MZ}-{job}" if rt.dry else None


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
    why = off_reason(rt.base, rt.secrets, rt.dry)
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
    if rt.top5:
        return _picks15(rt, force)
    s = rt.settings
    now = rt.now()
    session = picks_session(now, s)
    force = force or rt.dry  # a dry run saves nothing, so it may run at any hour, again
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
    sigs: list[event_signals.Signal] = []
    if rt.stopped:
        notes.append("No picks. Everything held is sold at 9:45 am and nothing is bought.")
    else:
        try:
            sigs = rt.signals(session) if rt.signals else feed_signals(rt, session, now)
        except Exception as exc:
            where = "SPY" if rt.core else "cash"
            notes.append(
                f"No buys today; the account stays in {where}. The news brain's data is not "
                f"ready: {exc}."
            )
    t9.register_kind(sc.GRADED)
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
    ov = load_overlay(rt) if rt.overlaid else None
    strategy = EventStrategy(
        settings=s,
        signals={decided: event_signals.ranked(sigs)},
        earnings=calendar(rt, bars, wanted, decided),
        sectors=sectors(rt),
        name=rt.strategy,
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
    if not rt.core and core is not None and core.qty > 0:
        # Trial 14: idle money is cash, so SPY left from the SPY core is sold with the picks.
        blind.append(
            PlannedOrder(
                symbol=s.benchmark,
                side=Side.SELL,
                qty=core.qty,
                ref_price=core.current_price,
                reason="idle_cash",
                thesis="idle money is held as cash under trial 14; SPY is never bought again",
            )
        )
    if blind:
        decision = dataclasses.replace(decision, orders=(*decision.orders, *blind))
    if rt.stopped:
        # The owner's stop: every position in the account is sold, SPY and anything bought
        # by hand included, and nothing is bought.
        everything = {k: risk.HeldPosition(k, p.qty, p.current_price) for k, p in raw.items()}
        decision = dataclasses.replace(
            decision, orders=tuple(risk.flatten_orders(everything, "stopped"))
        )
    plan = make_plan(
        decision,
        strategy=rt.strategy,
        mode="paper",
        session=decided,
        execute_session=session,
        created_at=now,
        equity=equity,
    )
    if rt.dry:
        result.status = "dry"
        log.info(
            "dry plan for %s: %s",
            session,
            [(o.side.value, o.symbol, str(o.notional or o.qty)) for o in plan.orders],
        )
    else:
        state.pending_plan = plan
        state.swept_sessions = [*state.swept_sessions, session][-60:]
        store.archive_plan(plan)
        if ov is not None:
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
    current = ov.current if ov is not None else None
    if plan.orders or not rt.stopped:  # once stopped and empty, the mornings are quiet
        send(rt, result, picks_alert(plan.orders, sigs, current, notes, session, rt.label))
    result.plan = plan
    return result


def _picks15(rt: BigRuntime, force: bool) -> Result:
    """Trial 15's 9:00 am plan: sell what is held and not picked, buy the top 5 by conviction
    at (equity x 0.98) / 5 each, resize a pick already held. No SPY, no 0.55 line, no regime
    filter, no sector cap; the 9:45 am run keeps the kill switch and the price sanity check.

    Each order's reference price is the latest trade at 9:00 am (pre-market when there is
    one), so the sanity check at 9:45 am catches a move after the plan, not the overnight
    news the pick is about."""
    s = rt.settings
    now = rt.now()
    session = picks_session(now, s)
    force = force or rt.dry
    if session is None:
        if not force:
            return Result(EXIT_OK, "outside_window")
        session = clock.next_session(clock.to_ny(now).date() - timedelta(days=1))
    store = rt.store
    state = store.load()
    if session in state.swept_sessions and not force:
        return Result(EXIT_OK, "already_done", session)
    result = Result(EXIT_OK, "picked", session)
    decided = clock.previous_session(session)
    broker = make_broker(rt)
    account = broker.account()
    held = {k: p for k, p in broker.positions().items() if p.qty > 0}
    notes: list[str] = []
    sigs: list[event_signals.Signal] = []
    if rt.stopped:
        notes.append("No picks. Everything held is sold at 9:45 am and nothing is bought.")
    else:
        try:
            sigs = rt.signals(session) if rt.signals else feed_signals(rt, session, now)
        except Exception as exc:
            notes.append(
                "No buys today; what is held is sold and the money stays in cash. The news "
                f"brain's data is not ready: {exc}."
            )
        sigs = sigs[: trial15.PICKS]
    equity = state.last_close_equity or account.equity
    symbols = sorted({x.symbol for x in sigs} | set(held))
    try:
        ref = broker.latest_prices(symbols) if symbols else {}
    except Exception as exc:  # the plan still sells; buys need a price
        log.warning("no 9:00 am prices: %s", exc)
        ref = {}
    frac = {sym: broker.asset(sym).fractionable for sym in symbols}
    orders = trial15.plan_orders(sigs, held, equity, ref, frac)
    unpriced = [x.symbol for x in sigs if x.symbol not in held and not ref.get(x.symbol)]
    if unpriced:
        notes.append("No price at 9:00 am, so not bought: " + ", ".join(unpriced) + ".")
    if rt.stopped:
        everything = {k: risk.HeldPosition(k, p.qty, p.current_price) for k, p in held.items()}
        orders = risk.flatten_orders(everything, "stopped")
    plan = make_plan(
        Decision(orders=tuple(orders)),
        strategy=rt.strategy,
        mode="paper",
        session=decided,
        execute_session=session,
        created_at=now,
        equity=equity,
    )
    if rt.dry:
        result.status = "dry"
        log.info(
            "dry plan for %s: %s",
            session,
            [(o.side.value, o.symbol, str(o.notional or o.qty)) for o in plan.orders],
        )
    else:
        state.pending_plan = plan
        state.swept_sessions = [*state.swept_sessions, session][-60:]
        store.archive_plan(plan)
        store.save(state)
        rt.journal.write(
            now,
            "picks",
            session,
            "plan",
            plan=plan.model_dump(mode="json"),
            signals=[
                {"symbol": x.symbol, "conviction": x.score, "rank": i + 1, "detail": x.detail}
                for i, x in enumerate(sigs)
            ],
            notes=notes,
        )
        save_picks15(rt, session, sigs)
    if plan.orders or not rt.stopped:
        send(rt, result, picks_alert(plan.orders, sigs, None, notes, session, rt.label))
    result.plan = plan
    return result


def picks_alert(
    orders: Any,
    sigs: list[event_signals.Signal],
    st: OverlayState | None,
    notes: list[str],
    session: date,
    label: str = "",
) -> alerts.Alert:
    p_of = {
        x.symbol: f"conviction {x.score:.3f}" if x.kind == trial15.KIND else f"p {x.score:.2f}"
        for x in sigs
    }
    buys = [o for o in orders if o.side is Side.BUY]
    sells = [o for o in orders if o.side is Side.SELL]
    lines = [f"Picks for {session}, auto-approved (paper). Orders go in at 9:45 am."]
    if label:
        lines[0] = f"{label}. {lines[0]}"
    for o in buys:
        lines.append(
            f"Buy {o.symbol} {alerts.money(o.notional or 0)}"
            + (f" ({p_of[o.symbol]})" if o.symbol in p_of else "")
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
    return alerts.Alert(
        title=f"Picks {session}",
        message="\n".join(lines),
        priority=alerts.HIGH if notes else alerts.DEFAULT,
    )


# ---- 8:15 am: FEED -----------------------------------------------------------------------------


def feed(rt: BigRuntime, force: bool = False) -> Result:
    """The news brain's morning pulls: news, filings and bars, waiting for the 9:00 am step."""
    return _off(rt, "feed") or _guarded(rt, "feed", lambda: _feed(rt, force))


def _feed(rt: BigRuntime, force: bool) -> Result:
    from stoa.newsbrain import feed as nf

    now = rt.now()
    day = clock.to_ny(now).date()
    if not clock.is_session(day) and not (force or rt.dry):
        return Result(EXIT_OK, "not_a_session")
    fd = nf.Feed(rt.base.big_paper.feed_dir, raw_tag(rt, "feed", now))
    token = hf_token(rt)
    load_feed(rt, fd, token)
    out = nf.run(fd, feed_sources(rt, fd.state().static, READ_AT_FEED), now)
    log.info("big paper feed: %s", out)
    save_feed(rt, fd, token, f"feed {now.isoformat()}")
    return Result(EXIT_OK, "dry" if rt.dry else "fed", day)


# ---- 9:45 am: EXECUTE --------------------------------------------------------------------------


def execute(rt: BigRuntime, force: bool = False) -> Result:
    return _off(rt, "execute") or _guarded(rt, "execute", lambda: _execute(rt, force))


def _execute(rt: BigRuntime, force: bool) -> Result:
    if rt.dry:
        return dry_execute(rt)
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
    # The kill switch and the owner's stop run on the raw account so every position goes,
    # SPY included. Under trial 14 there is no SPY core: idle money stays in cash.
    broker: Any = (
        raw
        if flatten or rt.stopped or not rt.core
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
        None if rt.top5 else state.last_close_equity,  # trial 15 has no daily loss stop
        gate=None,  # auto-approved: paper, and hundreds of trades to come
        sectors={} if rt.top5 else {o.symbol: o.sector or "Unknown" for o in plan.orders},
        before_sells=cancel_before_sells,
        before_flatten=cancel_everything,
        stopped=rt.stopped,
    )
    if report.kill_switch_tripped:
        dd = float(risk.drawdown(account.equity, hwm))
        send(rt, result, alerts.kill_switch_alert(account.equity, hwm, dd))
    if isinstance(broker, CoreBroker):
        cap = None
        if rt.overlaid:
            ov = load_overlay(rt)
            cap = ov.current.exposure_cap if ov.current else None
        broker.sweep(state.kill_switch.active, cap)
    closed = closed_trades(report, meta_before)
    state.add_trades(closed)
    for t in closed:
        state.add_realized(session.year, t.pnl)
    if rt.overlaid and isinstance(raw, stops.StopBroker):
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
    if report.fills or report.unfilled or report.rejections or not rt.stopped:
        send(rt, result, fills_alert(report, closed, broker, session))
    state.pending_plan = None
    state.mark_executed(session)
    store.save(state)
    return result


def dry_execute(rt: BigRuntime) -> Result:
    """Today's plan as the 9:00 am picks would make it, then the 9:45 am run on a copy of the
    account in the simulator, at the latest prices: the same broker wrapper, rails and SPY
    rule as the real run. No order is sent and nothing is saved."""
    s = rt.settings
    picked = _picks(rt, force=True)
    plan, session = picked.plan, picked.session
    assert plan is not None and session is not None
    raw = make_broker(rt)
    account = raw.account()
    held = {k: p for k, p in raw.positions().items() if p.qty > 0}
    symbols = sorted({o.symbol for o in plan.orders} | set(held) | {s.benchmark})
    prices = {**{k: p.current_price for k, p in held.items()}, **raw.latest_prices(symbols)}
    buys = sorted({o.symbol for o in plan.orders if o.side is Side.BUY})
    assets = {sym: raw.asset(sym) for sym in buys}
    for k in held:
        assets.setdefault(k, AssetInfo(k, tradable=True, fractionable=True))
    assets[s.benchmark] = AssetInfo(s.benchmark, tradable=True, fractionable=True)
    universe = Universe(
        watchlist=tuple(symbols),
        benchmark=s.benchmark,
        assets=assets,
        whole_shares=frozenset(k for k, a in assets.items() if not a.fractionable),
    )
    before = clock.previous_session(session)
    sim = SimBroker(
        Portfolio(
            cash=max(account.cash, Decimal(0)),
            positions={
                k: Lot(qty=p.qty, avg_cost=p.avg_entry_price, entry_session=before)
                for k, p in held.items()
            },
        ),
        s.costs,
        universe,
    )
    sim.begin_session(session, prices)
    state = rt.store.load()  # read only: never saved
    hwm = state.high_water_mark or account.equity
    flatten = state.kill_switch.active or risk.kill_switch_breached(account.equity, hwm, s.risk)
    broker: Any = (
        sim
        if flatten or rt.stopped or not rt.core
        else CoreBroker(
            sim, symbol=s.benchmark, reserve_pct=s.risk.cash_buffer_pct, session=session
        )
    )
    report = execute_plan(
        plan,
        broker,
        state,
        s,
        universe,
        session,
        None if rt.top5 else state.last_close_equity,
        gate=None,
        sectors={} if rt.top5 else {o.symbol: o.sector or "Unknown" for o in plan.orders},
        stopped=rt.stopped,
    )
    if isinstance(broker, CoreBroker):
        cap = None
        if rt.overlaid:
            ov = load_overlay(rt)
            cap = ov.current.exposure_cap if ov.current else None
        broker.sweep(state.kill_switch.active, cap)
    filled = [o for o in sim.orders.values() if o.filled]
    spy = {
        side: sum(
            (o.filled_qty for o in filled if o.symbol == s.benchmark and o.side is side),
            Decimal(0),
        )
        for side in Side
    }
    log.info(
        "dry execute for %s (idle money in %s): %s",
        session,
        "SPY" if rt.core else "cash",
        [(o.side.value, o.symbol, str(o.filled_qty)) for o in filled],
    )
    log.info(
        "dry execute for %s: SPY sold %s shares, SPY bought %s shares, cash after %s",
        session,
        spy[Side.SELL],
        spy[Side.BUY],
        sim.portfolio.cash,
    )
    result = Result(EXIT_OK, "dry", session, plan=plan)
    send(rt, result, fills_alert(report, [], broker, session))
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
    """Trial 13's broker-side ATR stops on every pick held from a prior session."""
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
    if report.stopped:
        lines.insert(0, "STOPPED by the owner: every position sold, nothing bought.")
    elif report.flattened:
        lines.insert(0, "KILL SWITCH: every position sold.")
    kill = report.flattened and not report.stopped
    return alerts.Alert(
        title=f"Fills {session}",
        message="\n".join(lines) or "Nothing to trade today.",
        priority=alerts.URGENT if kill else alerts.HIGH if report.stopped else alerts.DEFAULT,
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
    if rt.overlaid and isinstance(broker, stops.StopBroker):
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
    st: OverlayState | None = None
    if rt.overlaid:
        ov = load_overlay(rt)
        st = ov.step(session, equity, hwm, s.risk.max_gross_exposure_pct)
        save_overlay(rt, ov)
        stops.raise_stops(state, sorted(held), build_features(bars, s.benchmark), s.exits)
    state.last_close_session = session
    state.last_close_equity = equity
    if spy_close is not None:
        state.last_benchmark_close = dec(spy_close)
    record_close(rt.root, session, equity, spy_close)
    state.mark_decided(session)
    store.save(state)
    curve = read_curve(rt.root)
    lines = [f"{rt.label}. Equity {alerts.money(equity)}"]
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
    in_spy = (
        f"{alerts.money(core.market_value if core else 0)} in SPY, "
        if rt.core or (core is not None and core.qty > 0)
        else ""
    )
    lines.append(f"{len(held)} picks held, {in_spy}{alerts.money(account.cash)} cash.")
    off_high = st.drawdown if st is not None else risk.drawdown(equity, hwm)
    brake = ""
    if st is not None and st.halted:
        brake = " Brake: no new positions."
    elif st is not None and st.in_drawdown:
        brake = " Brake: half size."
    lines.append(f"Drawdown {off_high:.1%} from the high.{brake}")
    if state.kill_switch.active:
        lines.append("Kill switch ON: tomorrow's 9:45 am run sells everything.")
    if rt.stopped:
        lines.append("Stopped: the next 9:45 am run sells anything held, and nothing is bought.")
    if not rt.base.stop.shadow:
        lines.append(shadow_summary_line(rt, now))
    if rt.top5:
        lines += summary15_lines(rt, session)
    send(rt, result, alerts.Alert(title=f"Day summary {session}", message="\n".join(lines)))
    return result


# ---- trial 15's record: each morning's 5 picks, graded against SPY ----------------------------


def dir15(rt: BigRuntime) -> Path:
    return rt.root / "trial15"


def save_picks15(rt: BigRuntime, session: date, sigs: list[event_signals.Signal]) -> None:
    """The morning's picks and what each saw, for the summary's grading and the nightly
    review (trial15/picks/<session>.json, committed with the account's state)."""
    seen = {str(r.get("symbol")): r for r in rt.chosen or []}
    body = {
        "session": session.isoformat(),
        "model": rt.base.big_paper.model or FROZEN_VERSION,
        "picks": [
            {
                "rank": i + 1,
                "symbol": x.symbol,
                "conviction": x.score,
                "detail": x.detail,
                "saw": seen.get(x.symbol, {}),
            }
            for i, x in enumerate(sigs)
        ],
    }
    out = dir15(rt) / "picks" / f"{session.isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")


def picks15(rt: BigRuntime, session: date) -> list[dict[str, Any]]:
    path = dir15(rt) / "picks" / f"{session.isoformat()}.json"
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8"))["picks"])


def graded15(rt: BigRuntime) -> list[dict[str, Any]]:
    path = dir15(rt) / "days.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def prices945(rt: BigRuntime) -> Any:
    if rt.prices_945 is not None:
        return rt.prices_945
    from stoa import shadowboard
    from stoa.data.factory import alpaca_data_client

    return shadowboard.alpaca_945(alpaca_data_client(rt.settings, data_secrets(rt.secrets)))


def grade15(rt: BigRuntime, session: date) -> dict[str, Any] | None:
    """Grade the picks bought at 9:45 am on the session before ``session`` and sold at
    9:45 am on ``session``, once, into trial15/days.jsonl."""
    before = clock.previous_session(session)
    done = {d["session"] for d in graded15(rt)}
    picks = picks15(rt, before)
    if not picks or before.isoformat() in done:
        return None
    syms = [str(p["symbol"]) for p in picks]
    at = prices945(rt)
    entry, exit_ = at(before, {*syms, "SPY"}), at(session, {*syms, "SPY"})
    adv = {str(p["symbol"]): p.get("saw", {}).get("adv") for p in picks}
    day = {
        "session": before.isoformat(),
        "sold": session.isoformat(),
        "picks": trial15.grade(syms, entry, exit_, adv),
    }
    path = dir15(rt) / "days.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(day) + "\n")
    return day


def keep_rows15(rt: BigRuntime, fd: Any, token: str | None, session: date, rows: Any) -> None:
    """The morning's scored rows, kept in the feed's dataset for the nightly refit
    (learn/rows/<session>.parquet). Never stops the picks."""
    try:
        from stoa.newsbrain import feed as nf
        from stoa.newsbrain import panel as pn

        cols = ["symbol", "sidx", *pn.FEATURES, "adv"]
        path = fd.root / learn15.ROWS / f"{session.isoformat()}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows[[c for c in cols if c in rows.columns]].to_parquet(path, index=False)
        if token is not None:
            nf.push_state(
                fd,
                rt.base.big_paper.feed_repo,
                token,
                f"learn rows {session}",
                only=[f"{learn15.ROWS}/{session.isoformat()}.parquet"],
            )
    except Exception as exc:
        log.warning("the morning's rows were not kept for learning: %s", exc)


def shadow15(
    rt: BigRuntime,
    fd: Any,
    token: str | None,
    session: date,
    rows: Any,
    rules: list[str],
    sic: Mapping[str, Any],
    now: datetime,
) -> None:
    """The nightly refit's top 5 for this morning, locked before 9:30 am beside the picks.
    It never trades and never stops the picks."""
    took = review15.load_json(dir15(rt) / "learner.json")
    version, sha = took.get("version"), took.get("sha256")
    if not version or not sha:
        return
    try:
        import hashlib
        import pickle

        from stoa.newsbrain import feed as nf

        if token is not None:
            nf.pull_state(fd, rt.base.big_paper.feed_repo, token, only=[f"models/{version}/**"])
        blob = (fd.root / "models" / str(version) / "model_h1.pkl").read_bytes()
        if hashlib.sha256(blob).hexdigest() != sha:
            raise ValueError(f"the learner {version} does not match its sha256")
        brain = pickle.loads(blob)
        scored = rows.assign(conviction=trial15.conviction(brain, rows))
        picks = trial15.choose(review15.apply_rules(scored, rules), sic)
        learn15.lock_shadow(dir15(rt), session, str(version), list(picks["symbol"]), now)
    except Exception as exc:
        log.warning("the learner's shadow picks were not locked: %s", exc)


def learn(rt: BigRuntime) -> Result:
    """The nightly refit: the decade's rows plus every morning kept since, labelled, fit with
    recent rows weighted more, saved to the feed's dataset as the next shadow candidate."""
    from stoa.newsbrain import feed as nf

    if not rt.top5:
        return Result(EXIT_OK, "not_trial15")
    token = hf_token(rt)
    if token is None:
        return Result(EXIT_OK, "no_dataset")
    fd = nf.Feed(rt.base.big_paper.feed_dir)
    repo = rt.base.big_paper.feed_repo
    nf.pull_state(fd, repo, token, only=[learn15.BASE, f"{learn15.ROWS}/**"])
    base_path = fd.root / learn15.BASE
    if not base_path.exists():
        log.info("nightly learning waits for learn/base.parquet (scripts/push-learn-base.ps1)")
        return Result(EXIT_OK, "waiting_for_base")
    import pandas as pd

    base = pd.read_parquet(base_path)
    mornings = {
        date.fromisoformat(p.stem): pd.read_parquet(p)
        for p in sorted((fd.root / learn15.ROWS).glob("*.parquet"))
    }

    def opens(symbols: list[str], start: date, end: date) -> Any:
        bars = fetch_bars(rt, symbols, end)
        cols = {s: b["open"] for s, b in bars.items() if len(b)}
        df = pd.DataFrame(cols)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    train = learn15.training_set(base, mornings, opens)
    meta = learn15.save(learn15.fit(train), train, fd.root / "models")
    model = fd.root / "models" / meta["version"]
    (model / "model.pkl").replace(model / "model_h1.pkl")
    nf.push_state(
        fd, repo, token, f"learner {meta['version']}", only=[f"models/{meta['version']}/**"]
    )
    d = dir15(rt)
    d.mkdir(parents=True, exist_ok=True)
    (d / "learner.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    log.info(
        "nightly learner %s: %d rows through %s",
        meta["version"],
        meta["rows"],
        meta["trained_through"],
    )
    return Result(EXIT_OK, "learned")


def shadow15_lines(rt: BigRuntime, session: date) -> list[str]:
    """Grade the learner's shadow top 5 of the session before, as the picks are graded, and
    say whether it takes over (``learn15.takeover``)."""
    before = clock.previous_session(session)
    d = dir15(rt)
    path = d / "shadow" / f"{before.isoformat()}.json"
    days_path = d / "shadow_days.jsonl"
    days = (
        [json.loads(x) for x in days_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if days_path.exists()
        else []
    )
    if path.exists() and before.isoformat() not in {x["session"] for x in days}:
        body = json.loads(path.read_text(encoding="utf-8"))
        if not learn15.verify(body):
            return [f"Learner's shadow picks of {before} fail their hash; not graded."]
        syms = [str(s) for s in body["picks"]]
        at = prices945(rt)
        day = {
            "session": before.isoformat(),
            "version": body["version"],
            "picks": trial15.grade(
                syms, at(before, {*syms, "SPY"}), at(session, {*syms, "SPY"}), {}
            ),
        }
        with days_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(day) + "\n")
        days.append(day)
    if not days:
        return []
    t = trial15.tally(days)
    verdict = learn15.takeover(graded15(rt), days)
    line = (
        f"Nightly learner (shadow, never trades): {t['hits']} of {t['picks']} picks beat SPY"
        + (f" ({t['hit_rate']:.0%})" if t["hit_rate"] is not None else "")
        + f" over {t['days']} days."
    )
    if verdict["takes_over"]:
        latest = review15.load_json(d / "learner.json")
        (d / "promoted.json").write_text(
            json.dumps(
                {**latest, "promoted_on": session.isoformat(), "evidence": verdict}, indent=1
            )
            + "\n",
            encoding="utf-8",
        )
        line += (
            f" It graded better over the last {trial15.WINDOW} sessions and takes over tomorrow."
        )
    else:
        line += (
            f" Takes over only after {trial15.MIN_SHADOW} graded sessions that beat the live picks."
        )
    return [line]


def active_rules15(rt: BigRuntime) -> list[str]:
    """Rules from the lessons in use: due by their count and passed on the decade."""
    d = dir15(rt)
    return review15.active_rules(
        review15.load_json(d / "lessons.json"), review15.load_json(d / "rule_checks.json")
    )


def later_news15(
    rt: BigRuntime, symbols: list[str], start: datetime, end: datetime
) -> dict[str, float]:
    """The reader's mean direction for news about each symbol known between ``start`` and
    ``end``, from the feed's labels; empty when they cannot be read."""
    try:
        token = hf_token(rt)
        if token is None:
            return {}
        from huggingface_hub import hf_hub_download

        from stoa.newsbrain import feed as nf

        path = hf_hub_download(
            rt.base.big_paper.feed_repo,
            f"{nf.EVENTS}/labels.parquet",
            repo_type="dataset",
            token=token,
        )
        import pandas as pd

        lab = pd.read_parquet(path)
        known = pd.to_datetime(lab["known_at"], utc=True)
        lab = lab[(known >= start) & (known < end) & lab["symbol"].isin(symbols)]
        return {str(k): float(v) for k, v in lab.groupby("symbol")["direction"].mean().items()}
    except Exception as exc:
        log.warning("no reader labels for the review: %s", exc)
        return {}


def review15_lines(rt: BigRuntime, day: dict[str, Any]) -> list[str]:
    """The nightly self-review of a graded day: each pick's inputs, result and, when it lost,
    the likely reason; the lessons file keeps the count."""
    before, sold = date.fromisoformat(day["session"]), date.fromisoformat(day["sold"])
    picks = picks15(rt, before)
    start = clock.at_ny(before, "09:45").astimezone(UTC)
    end = clock.at_ny(sold, "09:45").astimezone(UTC)
    news = later_news15(rt, [str(p["symbol"]) for p in picks], start, end)
    reviews = review15.review_day(picks, day["picks"], news)
    d = dir15(rt)
    out = d / "reviews" / f"{day['session']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reviews, indent=1) + "\n", encoding="utf-8")
    lessons = review15.update_lessons(
        review15.load_json(d / "lessons.json"), day["session"], reviews
    )
    (d / "lessons.json").write_text(json.dumps(lessons, indent=1) + "\n", encoding="utf-8")
    lost = [r for r in reviews if not r["hit"]]
    if not lost:
        return []
    return ["Why they lost: " + "; ".join(f"{r['symbol']}, {r['why']}" for r in lost) + "."]


def summary15_lines(rt: BigRuntime, session: date) -> list[str]:
    """The 5:15 pm summary's trial 15 lines: today's 5 picks so far against SPY, the picks
    of the session before graded 9:45 am to 9:45 am after costs, that day's hit rate and the
    running hit rate against the 75% target. A grading failure never stops the summary."""
    lines: list[str] = []
    try:
        day = grade15(rt, session)
        today = picks15(rt, session)
        if today:
            syms = [str(p["symbol"]) for p in today]
            at = prices945(rt)(session, {*syms, "SPY"})
            bars = fetch_bars(rt, syms, session)

            def so_far(sym: str) -> float | None:
                b = bars.get(sym)
                if b is None or not len(b) or b.index[-1].date() != session or not at.get(sym):
                    return None
                return float(cell(b, b.index[-1], "close")) / float(at[sym]) - 1

            spy = so_far("SPY")
            parts = [
                f"{sym} {alerts.pct(r)}" if (r := so_far(sym)) is not None else f"{sym} n/a"
                for sym in syms
            ]
            lines.append(
                "Today's 5 picks, 9:45 am to the close: "
                + ", ".join(parts)
                + (f" (SPY {alerts.pct(spy)})." if spy is not None else ".")
            )
        if day is not None:
            graded = [g for g in day["picks"] if g["hit"] is not None]
            hits = sum(1 for g in graded if g["hit"])
            words = []
            for g in day["picks"]:
                if g["hit"] is None:
                    words.append(f"{g['symbol']} no price")
                else:
                    verb = "beat" if g["hit"] else "lost to"
                    words.append(f"{g['symbol']} {verb} SPY by {abs(g['net']):.1%}")
            share = f"{hits / len(graded):.0%}" if graded else "n/a"
            lines.append(
                f"Picks of {day['session']}, 9:45 am to 9:45 am after costs: "
                + "; ".join(words)
                + f". Day hit rate {hits} of {len(graded)} ({share})."
            )
            lines += review15_lines(rt, day)
        lines += shadow15_lines(rt, session)
        t = trial15.tally(graded15(rt))
        if t["picks"]:
            lines.append(
                f"Running: {t['hits']} of {t['picks']} picks beat SPY ({t['hit_rate']:.0%}) "
                f"over {t['days']} days, target 75%. Average edge "
                f"{alerts.pct(t['avg_edge'])} a pick."
            )
        else:
            lines.append("Running hit rate: none graded yet (target 75%).")
    except Exception as exc:
        log.warning("trial 15 grading failed: %s", exc)
        lines.append(f"Trial 15 grading failed tonight ({exc}); it retries tomorrow.")
    return lines


def scoreboard(rt: BigRuntime) -> Result:
    """Grade the shadow scoreboard now, with no alert (the summary does it each evening)."""
    if off_reason(rt.base, rt.secrets, dry=True):
        return Result(EXIT_OK, "off")
    if rt.base.stop.shadow:
        return Result(EXIT_OK, "stopped")
    graded = grade_scoreboard(rt, rt.now())
    log.info("shadow scoreboard: %s", [(d["kind"], d["session"], d.get("refused")) for d in graded])
    return Result(EXIT_OK, "graded")


def model_check(rt: BigRuntime, version: str) -> Result:
    """A model version against the Dell: download it and the backfilled days, score each day's
    rows (the Dell's, which match the feed's) again here, and require the same p to the last
    bit and picks without errors. Raises on any difference."""
    import hashlib
    import pickle

    import pandas as pd

    from stoa import shadowboard
    from stoa.newsbrain import feed

    token = hf_token(rt)
    if token is None:
        raise feed.FeedError("HF_TOKEN is not set")
    fd = feed.Feed(Path(f"{rt.base.big_paper.feed_dir}-check"))
    feed.pull_state(
        fd,
        rt.base.big_paper.feed_repo,
        token,
        only=[f"models/{version}/**", f"{shadowboard.PREFIX}/backfill/**"],
    )
    meta = json.loads((fd.root / "models" / version / "meta.json").read_text(encoding="utf-8"))
    blob = (fd.root / "models" / version / "model_h1.pkl").read_bytes()
    if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
        raise feed.FeedError(f"{version}: the model does not match its meta.json")
    brain = pickle.loads(blob)
    t9.register_kind(sc.GRADED)
    days = sorted((fd.root / shadowboard.PREFIX / "backfill").glob("*/predictions.parquet"))
    checked = 0
    for path in days:
        rows = pd.read_parquet(path)
        if set(rows["model_version"]) != {version}:
            continue
        p = brain.predict(rows)
        same = bool((p == rows["p"].to_numpy()).all())
        sigs = sc.signals(rows.assign(sidx=0), p, sc.GRADED)
        log.info(
            "model check %s %s: %s rows, p %s, %s picks %s",
            version,
            path.parent.name,
            len(rows),
            "equal to the Dell's" if same else "DIFFERENT from the Dell's",
            len(sigs),
            [(x.symbol, round(x.score, 4)) for x in sigs],
        )
        if not same:
            raise feed.FeedError(f"{version} on {path.parent.name}: p differs from the Dell's")
        checked += 1
    if not checked:
        raise feed.FeedError(f"no backfilled days scored by {version} to check against")
    return Result(EXIT_OK, "checked")


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
    if rt.overlaid:
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
