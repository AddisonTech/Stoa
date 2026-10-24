"""The v0.2 strategy: the calibrated model ranks, the rules decide, the risk engine sizes.

Each session:
1. Exits. Stops, trailing stops, trend breaks and the hard time limit from v1,
   plus: sell at the next open before a confirmed earnings release, and sell
   a position once it has been held for the model horizon and the model no
   longer favors it.
2. Ideas. Every tradable symbol with a calibrated probability of beating SPY
   above ``model.threshold``, highest first, up to ``model.max_candidates``.
3. Gates. The v1 quant gates, the earnings no-entry rule, the regime filter
   (no new entries while SPY is under its 200 day average), the sector cap and
   the correlation check.
4. Sizing. Edge over volatility, scaled to the portfolio volatility target,
   halved on FOMC and CPI days, then clipped by every hard cap.

With the SPY-hold fallback active (drift or quit rule), the model is not used:
positions exit by their normal rules, nothing new is bought except SPY, and
SPY is topped up toward ``risk.max_benchmark_pct`` as cash frees up.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Protocol

import pandas as pd

from stoa import events, risk, sizing
from stoa.clock import sessions_after
from stoa.config import Settings
from stoa.earnings import EarningsCalendar, entry_block, exit_signals
from stoa.exits import evaluate_exits, initial_stop
from stoa.features import cell, regime_is_bull, snapshot
from stoa.membership import sector_of
from stoa.models import Candidate, ExitSignal, QuantKill, dec
from stoa.planner import Decision, DecisionInput, ScoutRun
from stoa.roles.quant import apply_gates

TRADING_DAYS = 252
FALLBACK_MIN_TOP_UP = Decimal("0.02")


class Predictor(Protocol):
    """Calibrated probabilities for one decision session."""

    name: str

    def probabilities(self, session: date) -> pd.Series: ...

    def payoff_ratio(self, session: date) -> float: ...


@dataclass
class PanelPredictor:
    """Probabilities read from a precomputed (date, symbol) panel.

    Backtests use the pooled walk-forward predictions: each session's numbers
    came from a model trained only on data before that session's fold.
    Sessions outside every test fold have no predictions, so no entries.
    """

    panel: pd.DataFrame  # index (date, symbol), column p
    ratios: Mapping[date, float]
    name: str = "model"

    def probabilities(self, session: date) -> pd.Series:
        ts = pd.Timestamp(session)
        if ts not in self.panel.index.get_level_values("date"):
            return pd.Series(dtype=float)
        return self.panel.xs(ts, level="date")["p"].astype(float)

    def payoff_ratio(self, session: date) -> float:
        return float(self.ratios.get(session, 1.0))


def is_rebalance(rebalance: str, inp: DecisionInput) -> bool:
    """Whether new positions may open on this decision.

    ``monthly``: only when the orders execute on the first session of a month,
    that is when the decision session and the execute session fall in different
    months. Exits, stops and the kill switch are not affected.
    """
    if rebalance == "daily":
        return True
    d, x = inp.session, inp.execute_session
    return (d.year, d.month) != (x.year, x.month)


def conviction_from(p: float) -> int:
    """1 to 5, for the alert text only. Size comes from the sizing module."""
    return max(1, min(5, int((p - 0.5) * 50) + 1))


def daily_returns(
    features: Mapping[str, pd.DataFrame], symbols: list[str], n: int, session: date
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    ts = pd.Timestamp(session)
    for s in symbols:
        f = features.get(s)
        if f is None or ts not in f.index:
            continue
        close = f["close"].loc[:ts].tail(n + 1)
        r = close.pct_change().dropna()
        if len(r) >= max(20, n // 2) and r.notna().all():
            out[s] = [float(x) for x in r.tail(n)]
    return out


class ModelStrategy:
    risk_managed = True

    def __init__(
        self,
        settings: Settings,
        predictor: Predictor,
        earnings: EarningsCalendar,
        *,
        name: str = "model",
        sectors: Mapping[str, str] | None = None,
        tradable: set[str] | None = None,
    ) -> None:
        self.settings = settings
        self.predictor = predictor
        self.earnings = earnings
        self.name = name
        self.sectors = dict(sectors) if sectors is not None else None
        self.tradable = tradable

    def _sector_map(self, symbols: set[str]) -> dict[str, str]:
        if self.sectors is not None:
            return {s: self.sectors.get(s, "Unknown") for s in symbols}
        return {s: sector_of(s) for s in symbols}

    def decide(self, inp: DecisionInput) -> Decision:
        s = self.settings
        bench = s.benchmark
        snap = snapshot(inp.features, inp.session)
        bull = regime_is_bull(inp.features, bench, inp.session)
        positions = {
            sym: risk.HeldPosition(sym, h.qty, dec(cell(snap, sym, "close")))
            for sym, h in inp.holdings.items()
            if sym in snap.index
        }
        missing = sorted(set(inp.holdings) - set(positions))
        if missing:
            raise ValueError(f"no close on {inp.session} for held symbols {missing}")

        probs = (
            pd.Series(dtype=float)
            if inp.fallback_active
            else self.predictor.probabilities(inp.session)
        )
        exits = self._exits(inp, probs)
        held_syms = set(inp.holdings) | {e.symbol for e in exits}
        likely = set(probs[probs > s.model.threshold].index) if len(probs) else set()
        returns = daily_returns(
            inp.features,
            sorted((held_syms | likely) & set(snap.index)),
            s.risk.correlation_lookback,
            inp.session,
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
            sectors=self._sector_map(held_syms | likely | {bench}),
            returns=returns,
            halt_reason=inp.halt_reason,
            fallback_active=inp.fallback_active,
        )
        entries_possible = (
            not inp.kill_switch_active
            and not risk.kill_switch_breached(inp.equity, inp.high_water_mark, s.risk)
            and risk.entry_block_reason(ctx) is None
        )
        candidates: list[Candidate] = []
        kills: list[QuantKill] = []
        ideas: list[risk.EntryIdea] = []
        blocks: dict[str, tuple[str, str]] = {}
        multiplier = Decimal(1)
        note = ""
        waiting = ""
        if entries_possible and inp.fallback_active:
            ideas = self._fallback_ideas(inp, snap, positions)
        elif entries_possible and not is_rebalance(s.model.rebalance, inp):
            waiting = "no new entries until the monthly rebalance"
        elif entries_possible:
            candidates = self._candidates(inp, snap, probs)
            passed, kills = apply_gates(candidates, snap, set(inp.holdings), s.quant)
            for c in passed:
                b = entry_block(self.earnings, c.symbol, inp.session, s.earnings)
                if b is not None:
                    blocks[c.symbol] = b
            reason = events.light_reason(inp.session) if s.events.enabled else None
            if reason:
                multiplier = dec(s.events.size_multiplier)
                note = reason
            selling = {e.symbol for e in exits}
            keep = {k: v for k, v in positions.items() if k not in selling}
            ideas = self._size(inp, snap, keep, passed, returns)
        ctx = dataclasses.replace(
            ctx, entry_blocks=blocks, size_multiplier=multiplier, size_note=note
        )
        result = risk.evaluate(ctx, exits, ideas)
        notes = tuple(n for n in (f"go light: {note}" if note else "", waiting) if n)
        return Decision(
            orders=result.orders,
            exits=tuple(exits),
            candidates=tuple(candidates),
            kills=tuple(kills),
            rejections=result.rejections,
            kill_switch_tripped=result.kill_switch_tripped,
            regime_bull=bull,
            scout=ScoutRun(source=self.predictor.name if not inp.fallback_active else "fallback"),
            notes=notes,
        )

    # ---- exits ------------------------------------------------------------------------

    def _exits(self, inp: DecisionInput, probs: pd.Series) -> list[ExitSignal]:
        s = self.settings
        out = evaluate_exits(inp.holdings, inp.features, inp.session, s.exits)
        seen = {e.symbol for e in out}
        if inp.fallback_active:
            out = [e for e in out if e.symbol != s.benchmark]
            seen = {e.symbol for e in out}
        for e in exit_signals(self.earnings, sorted(inp.holdings), inp.session, s.earnings):
            if e.symbol not in seen and not (inp.fallback_active and e.symbol == s.benchmark):
                out.append(e)
                seen.add(e.symbol)
        if not inp.fallback_active:
            hold = s.model.hold
            for sym, h in sorted(inp.holdings.items()):
                if sym in seen:
                    continue
                held = sessions_after(h.meta.entry_session, inp.session)
                p = float(probs.get(sym, math.nan))
                if held >= hold and not (p > s.model.threshold):
                    detail = (
                        f"held {held} sessions; model p {p:.2f}"
                        if math.isfinite(p)
                        else f"held {held} sessions; no model view"
                    )
                    out.append(ExitSignal(symbol=sym, reason="horizon_exit", detail=detail))
                    seen.add(sym)
        return out

    # ---- entries ----------------------------------------------------------------------

    def _candidates(
        self, inp: DecisionInput, snap: pd.DataFrame, probs: pd.Series
    ) -> list[Candidate]:
        s = self.settings
        allowed = {
            sym
            for sym in snap.index
            if inp.universe.is_listed(sym)
            and sym != s.benchmark
            and sym not in inp.holdings
            and (self.tradable is None or sym in self.tradable)
        }
        p = probs[[x for x in probs.index if x in allowed]]
        p = p[p > s.model.threshold].sort_values(ascending=False, kind="stable")
        out = []
        for sym, prob in p.head(s.model.max_candidates).items():
            vol = cell(snap, sym, "vol_20")
            vol_txt = f"{vol:.0%}" if math.isfinite(vol) else "n/a"
            out.append(
                Candidate(
                    symbol=str(sym),
                    conviction=conviction_from(float(prob)),
                    thesis=(
                        f"Model: {prob:.0%} chance of beating SPY over the next "
                        f"{s.model.horizon} sessions. 20 day volatility {vol_txt}."
                    ),
                    score=float(prob),
                    source=self.predictor.name,
                )
            )
        return out

    def _vol(self, snap: pd.DataFrame, sym: str) -> float:
        if self.settings.sizing.vol_measure == "atr":
            v = cell(snap, sym, "atr_pct") * math.sqrt(TRADING_DAYS)
        else:
            v = cell(snap, sym, "vol_20")
        return v if math.isfinite(v) else self.settings.sizing.vol_floor

    def _size(
        self,
        inp: DecisionInput,
        snap: pd.DataFrame,
        positions: Mapping[str, risk.HeldPosition],
        passed: list[Candidate],
        returns: Mapping[str, list[float]],
    ) -> list[risk.EntryIdea]:
        s = self.settings
        sz = s.sizing
        existing = {
            sym: float(p.value / inp.equity) for sym, p in positions.items() if inp.equity > 0
        }
        weights: dict[str, float] = {}
        if sz.method == "confidence_vol" and passed:
            items = [
                sizing.SizingInput(c.symbol, c.score, self._vol(snap, c.symbol)) for c in passed
            ]
            cov_syms = [x for x in {*existing, *(c.symbol for c in passed)} if x in returns]
            n = min((len(returns[x]) for x in cov_syms), default=0)
            cov = sizing.covariance({x: returns[x][-n:] for x in cov_syms}) if n >= 20 else {}
            if not cov:  # no history: fall back to each name's own volatility
                vols = {i.symbol: i.volatility for i in items}
                cov = {a: {b: (vols[a] ** 2 if a == b else 0.0) for b in vols} for a in vols}
            weights = sizing.size_positions(
                items,
                existing,
                cov,
                edge=sz.edge,
                payoff_ratio=self.predictor.payoff_ratio(inp.session),
                kelly_fraction=sz.kelly_fraction,
                vol_floor=sz.vol_floor,
                vol_target=sz.portfolio_vol_target,
                min_edge=sz.min_edge,
            )
        ideas = []
        for c in passed:
            close = cell(snap, c.symbol, "close")
            atr = cell(snap, c.symbol, "atr_14")
            w = weights.get(c.symbol) if sz.method == "confidence_vol" else None
            ideas.append(
                risk.EntryIdea(
                    candidate=c,
                    ref_price=dec(close),
                    atr=dec(atr),
                    stop_price=dec(round(initial_stop(close, atr, s.exits), 4)),
                    target_weight=dec(round(w, 6)) if w is not None else None,
                    probability=c.score,
                )
            )
        return ideas

    def _fallback_ideas(
        self,
        inp: DecisionInput,
        snap: pd.DataFrame,
        positions: Mapping[str, risk.HeldPosition],
    ) -> list[risk.EntryIdea]:
        s = self.settings
        bench = s.benchmark
        if bench not in snap.index or inp.equity <= 0:
            return []
        current = positions[bench].value / inp.equity if bench in positions else Decimal(0)
        target = s.risk.max_benchmark_pct
        if target - current < FALLBACK_MIN_TOP_UP:
            return []
        close = cell(snap, bench, "close")
        atr = cell(snap, bench, "atr_14")
        c = Candidate(
            symbol=bench,
            conviction=1,
            thesis="SPY-hold fallback: hold the index until a person resets it.",
            score=0.0,
            source="fallback",
        )
        return [
            risk.EntryIdea(
                candidate=c,
                ref_price=dec(close),
                atr=dec(atr if math.isfinite(atr) else 0.0),
                stop_price=dec(0.01),
                target_weight=target,
                max_weight=target,
            )
        ]


@dataclass
class ModelPredictor:
    """Probabilities from one fixed, already trained model (live DECIDE and the holdout).

    ``dataset`` holds the standardized features. Each session is predicted on
    its own cross-section, using only that session's feature rows, which were
    built from data up to that close.
    """

    model: Any
    dataset: Any
    name: str = "model"

    def probabilities(self, session: date) -> pd.Series:
        rows = self.dataset.on(session)
        if rows.empty:
            return pd.Series(dtype=float)
        p = self.model.predict(rows)
        return pd.Series(p, index=rows.index.get_level_values("symbol"), dtype=float)

    def payoff_ratio(self, session: date) -> float:
        return float(self.model.payoff_ratio)
