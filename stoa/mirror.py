"""The $100 account as a paper mirror of trial 15 (``usd100.strategy: mirror_trial15``).

Decided 2026-11-03 by the owner (docs/DECISIONS.md, "The $100 account mirrors trial 15"):
every trading day at 9:45 am the account sells what it still holds from the session before
and buys the same 5 stocks the big paper account picked that morning, at (equity x 0.98) / 5
each, as fractional dollar orders sized from its own equity. It holds them one session.

The picks come from ``big-paper/trial15/picks/<session>.json`` on the state branch, fetched
fresh right before they are read. Each pick's 9:00 am price, which the 8% price check
compares against, comes from the big account's 9:00 am plan in ``big-paper/plans/``, written
in the same commit. Rails match trial 15: the kill switch and the price check stay; the sector
cap, correlation limit, daily loss stop, regime filter, ATR exits, broker stops, max
positions gate, earnings blackout and SPY fallback are off. It never buys SPY, never waits
for a tap, and refuses to run in live mode.

A paper case study, not a trial: no pass or fail line, nothing tuned from it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any, Protocol

from stoa import clock
from stoa.broker.base import Broker, BrokerPosition
from stoa.config import Settings
from stoa.experiments import Registry
from stoa.models import PlannedOrder, Side, cents_down
from stoa.planner import Decision, make_plan
from stoa.state import Plan, State
from stoa.trader import BeforeFlatten, BeforeSells, ExecutionReport, execute_plan
from stoa.universe import Universe

log = logging.getLogger(__name__)

STRATEGY = "mirror_trial15"
NAME = "usd100:mirror_trial15"
PICKS = 5  # trial 15's 5 a day
RESERVE = Decimal("0.02")  # trial 15's 2% kept in cash
TRIAL15_SPEC = "557c95176a137379"
FIRST_SESSION = date(2026, 11, 4)


class LiveRefused(RuntimeError):
    """Mirror mode is paper only."""


LIVE_REFUSAL = (
    "the $100 account is in mirror mode (usd100.strategy: mirror_trial15), which is paper "
    "only by the owner's decision of 2026-11-03; going live is a separate decision. "
    "Nothing was traded."
)


def check_paper(settings: Settings) -> None:
    """Refuse live money in mirror mode, whatever else is unlocked."""
    if settings.usd100.mirror and settings.mode == "live":
        raise LiveRefused(LIVE_REFUSAL)


# ---- the registry ------------------------------------------------------------------------------


def spec() -> dict[str, Any]:
    """What the case study is, as registered. Not a trial: it holds no returns."""
    return {
        "name": NAME,
        "decided": "2026-11-03 by the owner: the $100 account buys every day so it has a "
        "case study",
        "kind": "paper case study; no pass or fail line; nothing is tuned from it",
        "account": "the $100 Alpaca paper account (daily workflow)",
        "follows": f"trial 15 (spec {TRIAL15_SPEC}): the big paper account's 5 picks a "
        "session, read from big-paper/trial15/picks/<session>.json on the state branch",
        "orders": "9:45 am: sells first (everything not picked, and trims), then buys at "
        "(equity x 0.98) / 5 each as fractional dollar orders; a repeat pick is resized to "
        "that weight when $1 or more off",
        "hold": "one session, 9:45 am to 9:45 am",
        "kept": [
            "kill switch at 25% below the high",
            "8% price check against the big account's 9:00 am price",
        ],
        "off": [
            "sector cap",
            "correlation limit",
            "3% daily loss stop",
            "regime filter",
            "ATR exits",
            "broker stops",
            "max positions gate",
            "earnings blackout",
            "SPY fallback",
        ],
        "never": ["SPY", "live money", "Approve or Skip"],
        "missing_picks": "buy nothing, keep what is held, log the reason, one alert",
        "grading": "trial 15's rule (9:45 am to 9:45 am, against SPY, after costs) on the "
        "picks this account held",
        "first_orders": f"{FIRST_SESSION.isoformat()} 9:45 am New York time",
    }


def spec_hash() -> str:
    return hashlib.sha256(json.dumps(spec(), sort_keys=True).encode()).hexdigest()[:16]


def register(registry: Registry) -> bool:
    """Log the case study once (kind ``case_study``: never counted as a trial). True if new."""
    if any(e.kind == "case_study" and e.name == NAME for e in registry.load()):
        return False
    registry.append(
        kind="case_study",
        name=NAME,
        config_hash=spec_hash(),
        data_hash="none",
        params=spec(),
        metrics={},
        tags=["paper", "no_pass_fail", "no_tuning"],
    )
    return True


# ---- reading the picks from the state branch -----------------------------------------------------


class Files(Protocol):
    """The state branch, read only."""

    def refresh(self) -> str: ...  # fetch a fresh copy; returns where the files now come from

    def read(self, path: str) -> str | None: ...

    def listdir(self, path: str) -> list[str]: ...


Run = Callable[..., subprocess.CompletedProcess[str]]


class BranchFiles:
    """Files on the stoa-state branch, fetched from origin right before they are read.

    Never touches the checked-out files: it reads the fetched commit with ``git show``, so
    the run's own state is saved exactly as before. If the fetch fails, it reads the copy
    checked out at the start of the run and says so."""

    def __init__(self, root: Path, branch: str = "stoa-state", run: Run = subprocess.run) -> None:
        self.root, self.branch, self._run = root, branch, run
        self.ref: str | None = None

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return self._run(
            ["git", "-C", str(self.root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=90,
            check=False,
        )

    def refresh(self) -> str:
        self.ref = None
        try:
            got = self._git("fetch", "--quiet", "origin", self.branch)
            if got.returncode == 0:
                sha = self._git("rev-parse", "FETCH_HEAD")
                if sha.returncode == 0 and sha.stdout.strip():
                    self.ref = sha.stdout.strip()
                    return f"a fresh copy of {self.branch} at {self.ref[:7]}"
            why = (got.stderr or got.stdout or "git fetch failed").strip().splitlines()
            reason = why[-1] if why else "git fetch failed"
        except (OSError, subprocess.SubprocessError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
        log.warning("mirror: could not fetch %s: %s", self.branch, reason)
        return f"the copy checked out at the start of the run (fetch failed: {reason})"

    def read(self, path: str) -> str | None:
        if self.ref is not None:
            got = self._git("show", f"{self.ref}:{path}")
            return got.stdout if got.returncode == 0 else None
        local = self.root / path
        return local.read_text(encoding="utf-8") if local.is_file() else None

    def listdir(self, path: str) -> list[str]:
        if self.ref is not None:
            got = self._git("ls-tree", "--name-only", f"{self.ref}:{path}")
            return got.stdout.split() if got.returncode == 0 else []
        local = self.root / path
        return sorted(p.name for p in local.iterdir()) if local.is_dir() else []


@dataclass(frozen=True)
class Pick:
    rank: int
    symbol: str
    conviction: float | None
    detail: str
    ref_price: Decimal | None  # the big account's 9:00 am price, for the price check


@dataclass(frozen=True)
class Picks:
    session: date
    picks: tuple[Pick, ...]
    source: str
    notes: tuple[str, ...] = ()

    @property
    def symbols(self) -> list[str]:
        return [p.symbol for p in self.picks]


@dataclass(frozen=True)
class Missing:
    """Why the picks cannot be used today. Nothing is bought; what is held is kept."""

    session: date
    reason: str
    source: str


def _ref_prices(files: Files, settings: Settings, session: date) -> tuple[dict[str, Decimal], str]:
    """The big account's 9:00 am price for each symbol in its plan for ``session``."""
    folder = settings.usd100.plans_path.rstrip("/")
    decided = clock.previous_session(session).isoformat()
    best: dict[str, Any] | None = None
    for name in files.listdir(folder):
        if not (name.startswith(f"{decided}-") and name.endswith(".json")):
            continue
        text = files.read(f"{folder}/{name}")
        try:
            plan = json.loads(text) if text else None
        except ValueError:
            plan = None
        if not isinstance(plan, dict) or plan.get("execute_session") != session.isoformat():
            continue
        if best is None or str(plan.get("created_at", "")) > str(best.get("created_at", "")):
            best = plan
    if best is None:
        return {}, f"no 9:00 am plan for {session} in {folder}/"
    out: dict[str, Decimal] = {}
    for o in best.get("orders") or []:
        try:
            px = Decimal(str(o.get("ref_price")))
        except (ArithmeticError, ValueError):
            continue
        if px > 0:
            out[str(o.get("symbol"))] = px
    return out, ""


def load_picks(settings: Settings, session: date, files: Files) -> Picks | Missing:
    """That session's 5 picks, from a fresh copy of the state branch."""
    source = files.refresh()
    path = f"{settings.usd100.picks_path.rstrip('/')}/{session.isoformat()}.json"
    text = files.read(path)
    if text is None:
        return Missing(
            session,
            f"the big account's picks for {session} are not on the state "
            f"branch ({path} is missing; read {source})",
            source,
        )
    try:
        body = json.loads(text)
    except ValueError as exc:
        return Missing(session, f"{path} is not valid JSON ({exc})", source)
    if not isinstance(body, dict) or not isinstance(body.get("picks"), list):
        return Missing(session, f"{path} has no list of picks", source)
    if body.get("session") != session.isoformat():
        return Missing(session, f"{path} is for {body.get('session')!r}, not {session}", source)
    refs, ref_note = _ref_prices(files, settings, session)
    notes = [ref_note] if ref_note else []
    rows = sorted(
        (p for p in body["picks"] if isinstance(p, dict) and p.get("symbol")),
        key=lambda p: int(p.get("rank") or 0),
    )
    picks: list[Pick] = []
    for p in rows:
        sym = str(p["symbol"]).strip().upper()
        if sym == settings.benchmark.upper():
            notes.append(f"{sym} is never bought; skipped")
            continue
        if any(x.symbol == sym for x in picks):
            continue
        conv = p.get("conviction")
        picks.append(
            Pick(
                rank=len(picks) + 1,
                symbol=sym,
                conviction=float(conv) if isinstance(conv, int | float) else None,
                detail=str(p.get("detail") or ""),
                ref_price=refs.get(sym),
            )
        )
        if len(picks) == PICKS:
            break
    if not picks:
        notes.append(
            "the big account picked nothing this morning, so what is held is sold "
            "and the money stays in cash, as the big account does"
        )
    return Picks(session, tuple(picks), source, tuple(notes))


# ---- the 9:45 am orders -------------------------------------------------------------------------


def target_notional(equity: Decimal, n: int = PICKS) -> Decimal:
    """Each pick's money: (equity x 0.98) / 5, cents down."""
    return cents_down(equity * (Decimal(1) - RESERVE) / Decimal(n))


def plan_orders(
    picks: Picks,
    held: Mapping[str, BrokerPosition],
    equity: Decimal,
    live: Mapping[str, Decimal],
    fractionable: Mapping[str, bool],
    benchmark: str,
    min_order: Decimal = Decimal("1.00"),
) -> tuple[list[PlannedOrder], list[str]]:
    """Sell everything held that is not picked; buy each new pick at (equity x 0.98) / 5;
    resize a pick already held to that same weight when $1 or more off. Sells come first.

    Buys carry the big account's 9:00 am price as their reference, so the 8% price check at
    9:45 am measures the same move trial 15's does. A buy with no 9:00 am price is not made.
    """
    target = target_notional(equity)
    want = {p.symbol: p for p in picks.picks if p.symbol != benchmark}
    notes: list[str] = []
    sells: list[PlannedOrder] = []
    buys: list[PlannedOrder] = []
    for sym, pos in sorted(held.items()):
        if pos.qty <= 0:
            continue
        if sym not in want:
            sells.append(
                PlannedOrder(
                    symbol=sym,
                    side=Side.SELL,
                    qty=pos.qty,
                    ref_price=live.get(sym) or pos.current_price,
                    reason="hold_period",
                    thesis="held one session; not in today's 5",
                )
            )
            continue
        price = live.get(sym) or pos.current_price
        value = pos.qty * price
        diff = target - value
        if abs(diff) < min_order:
            continue
        if diff < 0:
            step = Decimal("0.000001") if fractionable.get(sym, True) else Decimal(1)
            qty = min((-diff / price).quantize(step, rounding=ROUND_DOWN), pos.qty)
            if qty > 0:
                sells.append(
                    PlannedOrder(
                        symbol=sym,
                        side=Side.SELL,
                        qty=qty,
                        ref_price=price,
                        reason="resize",
                        thesis="picked again; trimmed to its weight",
                    )
                )
            continue
        ref = want[sym].ref_price
        if ref is None:
            notes.append(f"{sym} picked again but not topped up: no 9:00 am price to check")
            continue
        buys.append(
            PlannedOrder(
                symbol=sym,
                side=Side.BUY,
                notional=cents_down(diff),
                ref_price=ref,
                reason="resize",
                probability=want[sym].conviction,
                thesis="picked again; topped up to its weight. " + want[sym].detail,
            )
        )
    for p in want.values():
        if p.symbol in held and held[p.symbol].qty > 0:
            continue
        if p.ref_price is None:
            notes.append(
                f"{p.symbol} not bought: no 9:00 am price in the big account's plan "
                "to check the 9:45 am price against"
            )
            continue
        buys.append(
            PlannedOrder(
                symbol=p.symbol,
                side=Side.BUY,
                notional=target,
                ref_price=p.ref_price,
                reason="trial15_pick",
                probability=p.conviction,
                thesis=p.detail,
            )
        )
    return sells + buys, notes


def mirror_settings(s: Settings) -> Settings:
    """The rails trial 15 keeps, for this account: the max positions gate raised to the order
    cap so it never decides which picks are bought. Everything else in ``risk`` stays; the
    daily loss stop and sector cap are off by how ``run`` calls the trader."""
    risk = s.risk.model_copy(update={"max_positions": s.risk.max_orders_per_day})
    return s.model_copy(update={"risk": risk})


def assert_no_benchmark(plan: Plan, benchmark: str) -> None:
    bad = [o.symbol for o in plan.orders if o.side is Side.BUY and o.symbol == benchmark]
    if bad:
        raise ValueError(f"the mirror never buys {benchmark}; refusing the plan")


@dataclass
class MirrorRun:
    session: date
    picks: Picks | None
    missing: Missing | None
    plan: Plan
    report: ExecutionReport
    equity: Decimal
    notes: list[str] = field(default_factory=list)
    reason: str | None = None  # why nothing was sent, when nothing was

    @property
    def sent(self) -> int:
        return len(self.report.fills) + len(self.report.unfilled)


def _zero_reason(run: MirrorRun, kill_switch: bool) -> str | None:
    if run.sent:
        return None
    if run.report.stopped:
        return "stopped by the owner (stop.usd100); nothing held to sell"
    if kill_switch:
        return "the kill switch is on: nothing is bought until it is reset by hand"
    if run.missing is not None:
        return f"no picks: {run.missing.reason}. Nothing bought, what is held is kept."
    if run.report.rejections:
        return "every order was refused: " + "; ".join(
            f"{r.side.value} {r.symbol} {r.reason.replace('_', ' ')} ({r.detail})"
            for r in run.report.rejections
        )
    if run.notes:
        return "no orders: " + "; ".join(run.notes)
    return "no orders needed: already holding today's 5 at their weight"


def run(
    picks: Picks | Missing,
    broker: Broker,
    state: State,
    settings: Settings,
    session: date,
    now: datetime,
    *,
    stopped: bool = False,
    before_sells: BeforeSells | None = None,
    before_flatten: BeforeFlatten | None = None,
) -> MirrorRun:
    """The 9:45 am run. The trader still checks the kill switch first, so it flattens even
    on a day with no picks; with no picks and no kill switch nothing is sent."""
    check_paper(settings)
    ms = mirror_settings(settings)
    bench = settings.benchmark
    account = broker.account()
    equity = account.equity
    held = {k: p for k, p in broker.positions().items() if p.qty > 0}
    notes: list[str] = []
    orders: list[PlannedOrder] = []
    good = picks if isinstance(picks, Picks) else None
    missing = picks if isinstance(picks, Missing) else None
    if good is not None and not stopped:
        notes += list(good.notes)
        symbols = sorted(set(good.symbols) | set(held))
        live = broker.latest_prices(symbols) if symbols else {}
        frac = {sym: broker.asset(sym).fractionable for sym in good.symbols}
        orders, more = plan_orders(good, held, equity, live, frac, bench, ms.risk.min_order_usd)
        notes += more
    plan = make_plan(
        Decision(orders=tuple(orders)),
        strategy=STRATEGY,
        mode=settings.mode,
        session=clock.previous_session(session),
        execute_session=session,
        created_at=now,
        equity=equity,
    )
    assert_no_benchmark(plan, bench)
    buys = sorted({o.symbol for o in plan.orders if o.side is Side.BUY})
    assets = {sym: broker.asset(sym) for sym in buys}
    universe = Universe(
        watchlist=tuple(sorted(({o.symbol for o in plan.orders} | set(held)) - {bench})),
        benchmark=bench,
        assets=assets,
        whole_shares=frozenset(k for k, a in assets.items() if not a.fractionable),
    )
    report = execute_plan(
        plan,
        broker,
        state,
        ms,
        universe,
        session,
        None,  # no daily loss stop in trial 15
        gate=None,  # paper: no Approve or Skip, no second price source
        sectors={},  # no sector cap
        halt_reason=None,  # no SPY fallback
        fallback_active=False,
        before_sells=before_sells,
        before_flatten=before_flatten,
        stopped=stopped,
    )
    out = MirrorRun(session, good, missing, plan, report, equity, notes)
    out.reason = _zero_reason(out, state.kill_switch.active)
    return out


# ---- the record ---------------------------------------------------------------------------------


def days_path(state_dir: Path) -> Path:
    return state_dir / "mirror" / "days.jsonl"


def _num(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def record(
    state_dir: Path,
    out: MirrorRun,
    held_after: Sequence[str],
    sales: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """One line a session in ``mirror/days.jsonl``: the picks, orders, fills, sales and, on a
    day with no orders, why."""
    rep = out.report
    row: dict[str, Any] = {
        "session": out.session.isoformat(),
        "source": out.picks.source if out.picks else (out.missing.source if out.missing else ""),
        "picks": [
            {"rank": p.rank, "symbol": p.symbol, "conviction": p.conviction}
            for p in (out.picks.picks if out.picks else ())
        ],
        "missing": out.missing.reason if out.missing else None,
        "equity": float(out.equity),
        "orders": [
            {
                "side": o.side.value,
                "symbol": o.symbol,
                "notional": _num(o.notional),
                "qty": _num(o.qty),
                "reason": o.reason,
            }
            for o in out.plan.orders
        ],
        "fills": [
            {
                "side": f.status.side.value,
                "symbol": f.status.symbol,
                "qty": _num(f.status.filled_qty),
                "price": _num(f.status.filled_avg_price),
                "reason": f.order.reason,
            }
            for f in rep.fills
        ],
        "sales": [dict(x) for x in sales],
        "rejected": [
            {"side": r.side.value, "symbol": r.symbol, "reason": r.reason, "detail": r.detail}
            for r in rep.rejections
        ],
        "held_after": sorted(held_after),
        "flattened": rep.flattened,
        "reason": out.reason,
        "notes": list(out.notes),
    }
    path = days_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def read_days(state_dir: Path) -> list[dict[str, Any]]:
    """Every recorded session, the last run of each, oldest first."""
    path = days_path(state_dir)
    if not path.exists():
        return []
    by: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by[str(row.get("session"))] = row
    return [by[k] for k in sorted(by)]


def graded(
    days: Sequence[Mapping[str, Any]], big_days: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Trial 15's grades (big-paper/trial15/days.jsonl) for the picks this account held after
    each 9:45 am run: same picks, same 9:45 am to 9:45 am rule, against SPY, after costs."""
    big = {str(d.get("session")): d for d in big_days}
    out: list[dict[str, Any]] = []
    for d in days:
        g = big.get(str(d.get("session")))
        mine = set(d.get("held_after") or []) & {p["symbol"] for p in d.get("picks") or []}
        if g is None or not mine:
            continue
        out.append({**g, "picks": [p for p in g.get("picks", []) if p.get("symbol") in mine]})
    return out


def summary_lines(settings: Settings, state_dir: Path, session: date) -> list[str]:
    """The 5:15 pm summary's mirror lines: today's 9:45 am run in a sentence."""
    every = read_days(state_dir)
    if not every:
        return ["Trial 15 mirror: no run yet; the first orders go in at the next 9:45 am run."]
    days = [d for d in every if d.get("session") == session.isoformat()]
    if not days:
        return [f"Trial 15 mirror: no 9:45 am run recorded for {session}."]
    d = days[-1]
    bought = [f["symbol"] for f in d.get("fills") or [] if f.get("side") == "buy"]
    sold = [f["symbol"] for f in d.get("fills") or [] if f.get("side") == "sell"]
    lines = [
        "Trial 15 mirror: picks "
        + (", ".join(p["symbol"] for p in d.get("picks") or []) or "none")
        + "; bought "
        + (", ".join(bought) or "nothing")
        + "; sold "
        + (", ".join(sold) or "nothing")
        + "."
    ]
    if d.get("reason"):
        lines.append(f"No orders today: {d['reason']}")
    return lines


# ---- a dry run in the simulator ------------------------------------------------------------------


def _big_fill_prices(files: Files, settings: Settings, session: date) -> dict[str, Decimal]:
    """What the big account actually paid or got at 9:45 am that session, from its journal."""
    folder = settings.usd100.plans_path.rsplit("/", 1)[0] + "/journal"
    text = files.read(f"{folder}/{session:%Y-%m}.jsonl") or ""
    out: dict[str, Decimal] = {}
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("session") != session.isoformat() or rec.get("step") != "execution":
            continue
        for f in (rec.get("data") or {}).get("fills") or []:
            if f.get("filled_avg_price") is not None:
                out[str(f["symbol"])] = Decimal(str(f["filled_avg_price"]))
    return out


def dry_run(
    settings: Settings,
    session: date,
    files: Files,
    now: datetime,
    equity: Decimal | None = None,
) -> tuple[MirrorRun, dict[str, Decimal], str]:
    """The 9:45 am run for ``session`` on a copy of the account in the simulator: the picks
    from the state branch, the account as recorded at its last close, priced at what the big
    account filled at that morning (else its 9:00 am prices). No order is sent, nothing saved."""
    from stoa.broker.sim import SimBroker
    from stoa.portfolio import Lot, Portfolio
    from stoa.universe import AssetInfo

    picks = load_picks(settings, session, files)
    state_text = files.read("state/state.json")
    state = State.model_validate_json(state_text) if state_text else State()
    eq = equity or state.last_close_equity or Decimal(100)
    held = {k: h for k, h in state.holdings_at_close.items() if h.qty > 0}
    prices = {k: h.close for k, h in held.items()}
    where = "the big account's 9:45 am fills"
    if isinstance(picks, Picks):
        prices.update({p.symbol: p.ref_price for p in picks.picks if p.ref_price})
        fills = _big_fill_prices(files, settings, session)
        if not fills:
            where = "the big account's 9:00 am prices (no 9:45 am fills recorded)"
        prices.update(fills)
    invested = sum((h.qty * prices[k] for k, h in held.items()), Decimal(0))
    syms = sorted(set(prices) - {settings.benchmark})
    universe = Universe(
        watchlist=tuple(syms),
        benchmark=settings.benchmark,
        assets={s: AssetInfo(s, tradable=True, fractionable=True) for s in syms},
    )
    sim = SimBroker(
        Portfolio(
            cash=eq - invested,
            positions={
                k: Lot(qty=h.qty, avg_cost=h.close, entry_session=clock.previous_session(session))
                for k, h in held.items()
            },
        ),
        settings.costs,
        universe,
    )
    sim.begin_session(session, prices)
    copy = state.model_copy(deep=True)
    copy.high_water_mark = max(state.high_water_mark or eq, eq)
    out = run(picks, sim, copy, settings, session, now, stopped=settings.stop.usd100)
    return out, prices, where
