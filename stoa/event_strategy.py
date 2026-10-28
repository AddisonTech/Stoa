"""Trial 8: the event-driven strategy, as pre-registered in docs/DECISIONS.md.

Every session it looks at the signals that became known at that close
(``event_signals``), keeps those that are liquid, fit a small account and pass
the earnings rule, ranks them (insider clusters first, then earnings drift by
size) and asks the unchanged risk engine for the open slots, 20% each. Exits:
the hold period (20 or 40 sessions), the ATR stop and trailing stop, the
earnings rules and the kill switch. No fitted parameters anywhere.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from stoa import risk, wide
from stoa.clock import sessions_after
from stoa.config import Settings
from stoa.data.base import Bars
from stoa.earnings import EarningsCalendar, entry_block, exit_signals
from stoa.event_signals import HOLD, Signal, ranked
from stoa.exits import initial_stop, trailing_stop
from stoa.features import cell, regime_is_bull, snapshot
from stoa.model_strategy import daily_returns
from stoa.models import Candidate, ClosedTrade, ExitSignal, Holding, Side, dec
from stoa.overlay import BRAKE_NOTE, HALT_NOTE, Overlay
from stoa.planner import Decision, DecisionInput, ScoutRun

UNKNOWN_SECTOR = "Unknown"


@dataclass
class Entry:
    session: date  # decision session; the buy is at the next open
    symbol: str
    kind: str
    score: float


@dataclass
class EventStrategy:
    settings: Settings
    signals: Mapping[date, list[Signal]]  # by the session at whose close each became known
    earnings: EarningsCalendar
    sectors: Mapping[str, str] = field(default_factory=dict)
    name: str = "event"
    risk_managed: bool = True
    kind_of: dict[str, str] = field(default_factory=dict)  # held symbol -> signal kind
    entries: list[Entry] = field(default_factory=list)  # every planned entry, for diagnostics
    overlay: Overlay | None = None  # trial 13's risk overlay; None leaves every rule as it was

    def decide(self, inp: DecisionInput) -> Decision:
        s = self.settings
        bench = s.benchmark
        todays = ranked(list(self.signals.get(inp.session, [])))
        wanted = set(inp.holdings) | {sig.symbol for sig in todays} | {bench}
        snap = snapshot({k: v for k, v in inp.features.items() if k in wanted}, inp.session)
        bull = regime_is_bull(inp.features, bench, inp.session)
        positions = {
            sym: risk.HeldPosition(sym, h.qty, dec(cell(snap, sym, "close")))
            for sym, h in inp.holdings.items()
            if sym in snap.index
        }
        missing = sorted(set(inp.holdings) - set(positions))
        if missing:
            raise ValueError(f"no close on {inp.session} for held symbols {missing}")
        exits = self._exits(inp)
        candidates = [c for c in todays if self._eligible(c, inp, snap)]
        held_syms = set(inp.holdings) | {e.symbol for e in exits}
        syms = held_syms | {c.symbol for c in candidates}
        returns = daily_returns(
            inp.features, sorted(syms & set(snap.index)), s.risk.correlation_lookback, inp.session
        )
        ctx = risk.RiskContext(
            session=inp.session,
            equity=inp.equity,
            cash=inp.cash,
            positions=positions,
            prev_equity=inp.prev_equity,
            high_water_mark=inp.high_water_mark,
            kill_switch_active=inp.kill_switch_active,
            regime_bull=bull,
            universe=inp.universe,
            limits=s.risk,
            sectors={x: self.sectors.get(x, UNKNOWN_SECTOR) for x in syms | {bench}},
            returns=returns,
            halt_reason=inp.halt_reason,
            fallback_active=False,
        )
        if self.overlay is not None:
            st = self.overlay.step(
                inp.session, inp.equity, inp.high_water_mark, s.risk.max_gross_exposure_pct
            )
            ctx = dataclasses.replace(
                ctx,
                limits=self.overlay.limits(s.risk, st),
                size_multiplier=st.size_multiplier,
                size_note=BRAKE_NOTE,
                halt_reason=inp.halt_reason or (HALT_NOTE if st.halted else None),
            )
        entries_possible = (
            not inp.kill_switch_active
            and not risk.kill_switch_breached(inp.equity, inp.high_water_mark, s.risk)
            and risk.entry_block_reason(ctx) is None
        )
        ideas: list[risk.EntryIdea] = []
        blocks: dict[str, tuple[str, str]] = {}
        picked: list[Signal] = []
        if entries_possible:
            for sig in candidates:
                b = entry_block(self.earnings, sig.symbol, inp.session, s.earnings)
                if b is not None:
                    blocks[sig.symbol] = b
                close = cell(snap, sig.symbol, "close")
                atr = cell(snap, sig.symbol, "atr_14")
                if not (math.isfinite(atr) and atr > 0):
                    continue
                ideas.append(
                    risk.EntryIdea(
                        candidate=Candidate(
                            symbol=sig.symbol,
                            conviction=3,
                            thesis=f"{sig.kind.replace('_', ' ')}: {sig.detail}",
                            score=sig.score,
                            source=sig.kind,
                        ),
                        ref_price=dec(close),
                        atr=dec(atr),
                        stop_price=dec(round(initial_stop(close, atr, s.exits), 4)),
                        target_weight=(
                            self.overlay.weight(close, atr, s.exits)
                            if self.overlay is not None
                            else None
                        ),
                    )
                )
                picked.append(sig)
        ctx = dataclasses.replace(ctx, entry_blocks=blocks)
        result = risk.evaluate(ctx, exits, ideas)
        bought = {o.symbol for o in result.orders if o.side is Side.BUY}
        for sig in picked:
            if sig.symbol in bought:
                self.kind_of[sig.symbol] = sig.kind
                self.entries.append(Entry(inp.session, sig.symbol, sig.kind, sig.score))
        return Decision(
            orders=result.orders,
            exits=tuple(exits),
            candidates=tuple(i.candidate for i in ideas),
            rejections=result.rejections,
            kill_switch_tripped=result.kill_switch_tripped,
            regime_bull=bull,
            scout=ScoutRun(source="event"),
        )

    # ---- filters --------------------------------------------------------------------

    def _eligible(self, sig: Signal, inp: DecisionInput, snap: pd.DataFrame) -> bool:
        """Has today's bar, is liquid today, is not held, and fits the account."""
        sym = sig.symbol
        if sym == self.settings.benchmark or sym in inp.holdings or sym not in snap.index:
            return False
        close = cell(snap, sym, "close")
        dv = cell(snap, sym, "dollar_vol_20")
        if not (close >= wide.MIN_PRICE and dv >= wide.MIN_DOLLAR_VOLUME):
            return False
        if not inp.universe.can_buy(sym):
            return False
        return wide.fits_account(
            inp.universe.is_fractionable(sym),
            dec(close),
            inp.equity,
            self.settings.risk.max_position_pct,
        )

    # ---- exits ----------------------------------------------------------------------

    def _exits(self, inp: DecisionInput) -> list[ExitSignal]:
        s = self.settings
        out: list[ExitSignal] = []
        seen: set[str] = set()
        for sym, h in sorted(inp.holdings.items()):
            sig = stop_exit(h, inp.features.get(sym), inp.session, s)
            if sig is None:
                held = sessions_after(h.meta.entry_session, inp.session)
                hold = HOLD.get(self.kind_of.get(sym, ""), min(HOLD.values()))
                if held >= hold:
                    sig = ExitSignal(
                        symbol=sym, reason="hold_period", detail=f"held {held} of {hold} sessions"
                    )
            if sig is not None:
                out.append(sig)
                seen.add(sym)
        for e in exit_signals(self.earnings, sorted(inp.holdings), inp.session, s.earnings):
            if e.symbol not in seen:
                out.append(e)
                seen.add(e.symbol)
        return out


def stop_exit(
    holding: Holding, history: pd.DataFrame | None, session: date, s: Settings
) -> ExitSignal | None:
    """The ATR stop and the trailing stop on the close, as in ``exits``; nothing else."""
    if history is None or history.empty or history.index[-1] != pd.Timestamp(session):
        return None
    cfg = s.exits
    meta = holding.meta
    close = float(history["close"].iloc[-1])
    entry, atr = float(meta.entry_price), float(meta.entry_atr)
    stop = initial_stop(entry, atr, cfg)
    if close <= stop:
        return ExitSignal(
            symbol=holding.symbol, reason="stop_loss", detail=f"close {close:.2f} <= {stop:.2f}"
        )
    if cfg.trailing_enabled:
        since = history.loc[pd.Timestamp(meta.entry_session) :]
        highest = float(since["high"].max()) if not since.empty else entry
        trail = trailing_stop(highest, entry, atr, cfg)
        if trail > stop and close <= trail:
            return ExitSignal(
                symbol=holding.symbol,
                reason="trailing_stop",
                detail=f"close {close:.2f} <= trailing {trail:.2f}",
            )
    return None


# ---- costs ----------------------------------------------------------------------------


def liquidity_slippage(bars: Bars) -> Callable[[str, date | None], float]:
    """Registered cost model: bps by the 20 session dollar volume up to the decision day."""
    dv = {sym: wide.dollar_volume(df) for sym, df in bars.items()}

    def slip(symbol: str, session: date | None) -> float:
        series = dv.get(symbol)
        if series is None or session is None:
            return wide.COST_MAX_BPS
        before = series.loc[: pd.Timestamp(session) - pd.Timedelta(days=1)].dropna()
        return wide.slippage_bps(float(before.iloc[-1]) if len(before) else None)

    return slip


# ---- diagnostics --------------------------------------------------------------------


@dataclass(frozen=True)
class SignalStats:
    kind: str
    trades: int
    hit_rate: float  # share of trades that beat SPY over the same holding period
    avg_excess: float  # average trade return minus SPY's over the same holding period


def diagnostics(
    entries: list[Entry], trades: list[ClosedTrade], spy: pd.DataFrame, kinds: tuple[str, ...]
) -> list[SignalStats]:
    """Per-signal results from closed trades, matched to the entry decision that opened them.

    A trade buys at an open and sells at an open, so SPY is measured open to open
    over the same two sessions.
    """
    decided: dict[str, list[Entry]] = {}
    for e in entries:
        decided.setdefault(e.symbol, []).append(e)
    rows: dict[str, list[float]] = {k: [] for k in kinds}
    for t in trades:
        before = [e for e in decided.get(t.symbol, []) if e.session < t.entry_session]
        a, b = pd.Timestamp(t.entry_session), pd.Timestamp(t.exit_session)
        if not before or a not in spy.index or b not in spy.index:
            continue
        kind = max(before, key=lambda e: e.session).kind
        ret = float(t.exit_price / t.entry_price) - 1.0
        bench = cell(spy, b, "open") / cell(spy, a, "open") - 1.0
        rows.setdefault(kind, []).append(ret - bench)
    out = []
    for k in kinds:
        xs = rows.get(k, [])
        out.append(
            SignalStats(
                kind=k,
                trades=len(xs),
                hit_rate=sum(x > 0 for x in xs) / len(xs) if xs else math.nan,
                avg_excess=sum(xs) / len(xs) if xs else math.nan,
            )
        )
    return out


def sector_of_sic(sic: str | None) -> str:
    """The registered sector for the cap: the SIC major group (first two digits)."""
    if not sic or not str(sic)[:2].isdigit():
        return UNKNOWN_SECTOR
    return f"SIC {str(sic)[:2]}"
