"""The agent strategy: exits, scout, quant gates, then the risk engine.

This is the one pipeline that runs in backtest, signal, paper and live.
"""

from __future__ import annotations

from stoa import risk
from stoa.config import Settings
from stoa.exits import evaluate_exits, initial_stop
from stoa.features import cell, regime_is_bull, snapshot
from stoa.models import Candidate, QuantKill, dec
from stoa.planner import Decision, DecisionInput, ScoutRun
from stoa.roles.quant import apply_gates
from stoa.roles.scout import Scout, ScoutInput


class AgentStrategy:
    risk_managed = True

    def __init__(self, settings: Settings, scout: Scout, name: str = "agents") -> None:
        self.settings = settings
        self.scout = scout
        self.name = name

    def decide(self, inp: DecisionInput) -> Decision:
        s = self.settings
        snap = snapshot(inp.features, inp.session)
        bull = regime_is_bull(inp.features, s.benchmark, inp.session)
        positions = {
            sym: risk.HeldPosition(sym, h.qty, dec(cell(snap, sym, "close")))
            for sym, h in inp.holdings.items()
            if sym in snap.index
        }
        missing = sorted(set(inp.holdings) - set(positions))
        if missing:
            raise ValueError(f"no close on {inp.session} for held symbols {missing}")
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
        )
        exits = evaluate_exits(inp.holdings, inp.features, inp.session, s.exits)

        candidates: list[Candidate] = []
        scout_run: ScoutRun | None = None
        kills: list[QuantKill] = []
        entries_possible = (
            not inp.kill_switch_active
            and not risk.kill_switch_breached(inp.equity, inp.high_water_mark, s.risk)
            and risk.entry_block_reason(ctx) is None
        )
        if entries_possible:
            candidates, scout_run = self.scout.propose(
                ScoutInput(
                    session=inp.session,
                    snapshot=snap.loc[[x for x in snap.index if inp.universe.is_listed(x)]],
                    benchmark=s.benchmark,
                    regime_bull=bull,
                    holdings=inp.holdings,
                    recent_trades=inp.recent_trades,
                )
            )
            passed, kills = apply_gates(candidates, snap, set(inp.holdings), s.quant)
        else:
            passed = []

        ideas = []
        for c in passed:
            close = cell(snap, c.symbol, "close")
            atr = cell(snap, c.symbol, "atr_14")
            ideas.append(
                risk.EntryIdea(
                    candidate=c,
                    ref_price=dec(close),
                    atr=dec(atr),
                    stop_price=dec(round(initial_stop(close, atr, s.exits), 4)),
                )
            )
        result = risk.evaluate(ctx, exits, ideas)
        return Decision(
            orders=result.orders,
            exits=tuple(exits),
            candidates=tuple(candidates),
            kills=tuple(kills),
            rejections=result.rejections,
            kill_switch_tripped=result.kill_switch_tripped,
            regime_bull=bull,
            scout=scout_run,
        )
