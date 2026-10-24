"""Turns approved decisions into an executable plan with deterministic order ids.

The decision input carries everything a strategy may look at. Features are
already cut at the decision session, so nothing downstream can see the future.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Protocol

import pandas as pd

from stoa.models import (
    Candidate,
    ClosedTrade,
    ExitSignal,
    Holding,
    PlannedOrder,
    QuantKill,
    Rejection,
    Side,
)
from stoa.state import Plan
from stoa.universe import Universe


@dataclass(frozen=True)
class ScoutRun:
    """What the scout did on one decision, for the journal and the cost ledger."""

    source: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0  # what this call costs at list price, cached or not
    charged_usd: float = 0.0  # what this run actually paid (zero on a replay cache hit)
    cache_hit: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True)
class DecisionInput:
    session: date
    execute_session: date
    features: Mapping[str, pd.DataFrame]  # rows <= session only
    holdings: Mapping[str, Holding]
    equity: Decimal
    cash: Decimal
    prev_equity: Decimal | None
    high_water_mark: Decimal
    kill_switch_active: bool
    universe: Universe
    recent_trades: tuple[ClosedTrade, ...] = ()


@dataclass(frozen=True)
class Decision:
    orders: tuple[PlannedOrder, ...]
    exits: tuple[ExitSignal, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    kills: tuple[QuantKill, ...] = ()
    rejections: tuple[Rejection, ...] = ()
    kill_switch_tripped: bool = False
    regime_bull: bool | None = None
    scout: ScoutRun | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


class Strategy(Protocol):
    name: str
    risk_managed: bool  # False only for the baselines, which must stay fully invested

    def decide(self, inp: DecisionInput) -> Decision: ...


def plan_id_for(strategy: str, mode: str, session: date) -> str:
    """Same strategy, mode and session always give the same plan id."""
    return hashlib.sha256(f"{strategy}|{mode}|{session.isoformat()}".encode()).hexdigest()[:10]


def client_order_id(plan_id: str, execute_session: date, symbol: str, side: Side) -> str:
    """Deterministic per (trade date, symbol, side, plan id). A retry can never double-submit."""
    return f"stoa-{execute_session:%Y%m%d}-{symbol}-{side.value}-{plan_id}"


def make_plan(
    decision: Decision,
    *,
    strategy: str,
    mode: str,
    session: date,
    execute_session: date,
    created_at: datetime,
    equity: Decimal,
) -> Plan:
    pid = plan_id_for(strategy, mode, session)
    sells = [o for o in decision.orders if o.side is Side.SELL]
    buys = [o for o in decision.orders if o.side is Side.BUY]
    orders = tuple(
        o.model_copy(
            update={"client_order_id": client_order_id(pid, execute_session, o.symbol, o.side)}
        )
        for o in sells + buys
    )
    symbols = [o.symbol for o in orders]
    if len(set(symbols)) != len(symbols):
        raise ValueError("a plan may hold at most one order per symbol")
    return Plan(
        plan_id=pid,
        strategy=strategy,
        mode=mode,
        decided_session=session,
        execute_session=execute_session,
        created_at=created_at,
        orders=orders,
        equity_at_decide=equity,
        kill_switch=decision.kill_switch_tripped,
    )
