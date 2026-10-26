"""The agent strategy: exits, scout, quant gates, then the risk engine.

This is the one pipeline that runs in backtest, signal, paper and live.
"""

from __future__ import annotations

from stoa import risk
from stoa.baselines import BuyAndHold, SMACrossover
from stoa.config import Secrets, Settings
from stoa.exits import evaluate_exits, initial_stop
from stoa.features import cell, regime_is_bull, snapshot
from stoa.membership import sector_of
from stoa.models import Candidate, QuantKill, dec
from stoa.planner import Decision, DecisionInput, ScoutRun, Strategy
from stoa.roles.llm import Budget, LLMClient, ReplayCache, make_llm_client
from stoa.roles.quant import apply_gates
from stoa.roles.scout import LLMScout, RuleScout, Scout, ScoutInput


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
            # Without a map every pick was "Unknown" and the plan had no sector cap.
            sectors={sym: sector_of(sym) for sym in set(snap.index) | set(inp.holdings)},
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


def build_scout(
    settings: Settings, secrets: Secrets, *, llm: bool, budget: Budget | None = None
) -> Scout:
    """The rule scout, or the LLM scout (replay cache first) when ``llm`` is set."""
    if not llm:
        return RuleScout(settings.scout)
    key = secrets.llm_api_key
    client = make_llm_client(key.get_secret_value(), settings.llm) if key is not None else None
    llm_client = LLMClient(
        settings.llm,
        client,
        ReplayCache(settings.llm.cache_dir),
        budget or Budget(settings.llm.monthly_cost_cap_usd),
    )
    return LLMScout(settings.scout, settings.llm, llm_client)


def build_strategy(
    settings: Settings,
    secrets: Secrets,
    name: str,
    *,
    llm: bool = False,
    budget: Budget | None = None,
) -> Strategy:
    if name == "spy":
        return BuyAndHold(settings.benchmark)
    if name == "sma":
        return SMACrossover(settings.benchmark)
    if name == "agents":
        return AgentStrategy(settings, build_scout(settings, secrets, llm=llm, budget=budget))
    raise ValueError(f"unknown strategy {name!r}")
