"""Durable state: kill switch, high-water mark, executions, pending plan.

In paper and live modes the broker owns positions and cash. Local state only
holds what the broker cannot: entry metadata for exits, the high-water mark,
the kill switch, the money plan flags, which sessions already ran, and the plan
waiting for the next open. Backtests use the same model in memory.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from stoa.models import ZERO, PlannedOrder, PositionMeta

STATE_VERSION = 1
SESSION_HISTORY = 60


class KillSwitch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active: bool = False
    tripped_session: date | None = None
    reason: str = ""


class Plan(BaseModel):
    """Orders decided after one close, to be executed after the next open."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str
    strategy: str
    mode: str
    decided_session: date
    execute_session: date
    created_at: datetime
    orders: tuple[PlannedOrder, ...]
    equity_at_decide: Decimal
    kill_switch: bool = False


class Inception(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: date
    equity: Decimal
    benchmark_close: Decimal


class State(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    version: int = STATE_VERSION
    high_water_mark: Decimal | None = None
    kill_switch: KillSwitch = Field(default_factory=KillSwitch)
    principal_withdrawn: bool = False
    milestone_alerted: bool = False
    inception: Inception | None = None
    last_close_session: date | None = None
    last_close_equity: Decimal | None = None
    decided_sessions: list[date] = Field(default_factory=list)
    executed_sessions: list[date] = Field(default_factory=list)
    pending_plan: Plan | None = None
    positions_meta: dict[str, PositionMeta] = Field(default_factory=dict)
    realized_by_year: dict[str, Decimal] = Field(default_factory=dict)
    llm_spend_by_month: dict[str, Decimal] = Field(default_factory=dict)
    processed_activity_ids: list[str] = Field(default_factory=list)
    sim_cash: Decimal | None = None
    sim_positions: dict[str, Decimal] = Field(default_factory=dict)
    sim_avg_cost: dict[str, Decimal] = Field(default_factory=dict)

    def mark_decided(self, session: date) -> None:
        self.decided_sessions = _push(self.decided_sessions, session)

    def mark_executed(self, session: date) -> None:
        self.executed_sessions = _push(self.executed_sessions, session)

    def realized_for_year(self, year: int) -> Decimal:
        return self.realized_by_year.get(str(year), ZERO)

    def add_realized(self, year: int, amount: Decimal) -> None:
        key = str(year)
        self.realized_by_year = {
            **self.realized_by_year,
            key: self.realized_for_year(year) + amount,
        }


def _push(items: list[date], day: date) -> list[date]:
    out = [d for d in items if d != day] + [day]
    return sorted(out)[-SESSION_HISTORY:]


class StateStore:
    """state.json on disk, written atomically. Plans are also archived under plans/."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "state.json"
        self.plans_dir = root / "plans"

    def load(self) -> State:
        if not self.path.exists():
            return State()
        return State.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, state: State) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.path, state.model_dump_json(indent=2) + "\n")

    def archive_plan(self, plan: Plan) -> Path:
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        path = self.plans_dir / f"{plan.decided_session.isoformat()}-{plan.plan_id}.json"
        _atomic_write(path, plan.model_dump_json(indent=2) + "\n")
        return path


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def dump_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, default=str)
