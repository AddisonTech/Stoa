"""The money plan: grow the deposit to a multiple, then take the deposit back out.

API keys cannot move money, so the withdrawal itself happens on Alpaca's site.
Stoa sends the alert, the owner withdraws and runs `stoa money confirm-withdrawal`,
and the high-water mark (and with it the kill switch) rebases to what is left.

Withdrawals and deposits are also read from Alpaca account activities. Any
cash flow found there moves the high-water mark by the same amount, so taking
money out can never look like a trading loss and trip the kill switch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from stoa.alerts import HIGH, Alert, money
from stoa.config import MoneyConfig
from stoa.models import ZERO, cents_up
from stoa.state import KillSwitch, State

MAX_ACTIVITY_IDS = 500


def milestone(cfg: MoneyConfig) -> Decimal:
    return cfg.deposit * cfg.milestone_multiple


def milestone_reached(equity: Decimal, cfg: MoneyConfig) -> bool:
    return equity >= milestone(cfg)


def whole_dollars(value: Decimal) -> str:
    return f"${value:,.0f}" if value == value.to_integral_value() else money(value)


def withdraw_alert(equity: Decimal, cfg: MoneyConfig) -> Alert:
    amount = whole_dollars(cfg.deposit)
    steps = (
        f"Equity is {money(equity)}, at least {cfg.milestone_multiple}x your {amount} deposit.\n"
        f"1. Open the Alpaca app or site and go to Banking, then Withdraw.\n"
        f"2. Withdraw {amount} to your bank. If cash is short, wait for the next sell.\n"
        f"3. Run: stoa money confirm-withdrawal (or the money-confirm workflow).\n"
        f"From then on Stoa trades only the profit."
    )
    return Alert(title=f"Withdraw {amount} now", message=steps, priority=HIGH, tags=("moneybag",))


def plan_active(state: State, cfg: MoneyConfig) -> bool:
    """The plan only makes sense when the account started near the configured deposit."""
    if state.inception is None:
        return False
    return deposit_mismatch_note(state.inception.equity, cfg) is None


def check_milestone(state: State, equity: Decimal, cfg: MoneyConfig) -> Alert | None:
    """Return the withdraw alert the first time the milestone is reached, else None.

    Outside deposits are taken out first, so adding money never counts as growth.
    """
    if state.principal_withdrawn or state.milestone_alerted or not plan_active(state, cfg):
        return None
    if not milestone_reached(equity - max(ZERO, state.net_flows), cfg):
        return None
    state.milestone_alerted = True
    return withdraw_alert(equity, cfg)


def confirm_withdrawal(state: State, equity_now: Decimal, deposit: Decimal) -> Decimal:
    """Mark the deposit as withdrawn and rebase the high-water mark. Returns the old mark.

    If the withdrawal has not shown up in account activities yet, the next one
    that does is matched to it, so the high-water mark is not lowered twice.
    """
    if state.principal_withdrawn:
        raise ValueError("the withdrawal is already confirmed")
    if equity_now <= 0:
        raise ValueError("equity must be positive to rebase the high-water mark")
    old = state.high_water_mark or equity_now
    state.principal_withdrawn = True
    state.milestone_alerted = True
    state.high_water_mark = equity_now
    state.principal_flow_pending = state.withdrawn_since_milestone < deposit
    return old


def withdraw_shadow(state: State, amount: Decimal) -> Decimal:
    """Signal mode: take the deposit out of the paper portfolio. Returns equity after."""
    if state.sim_cash is None or state.last_close_equity is None:
        raise ValueError("signal mode has not run yet")
    if state.sim_cash < amount:
        raise ValueError(f"paper cash {state.sim_cash} is below {amount}; wait for a sell")
    state.sim_cash = state.sim_cash - amount
    state.net_flows = state.net_flows - amount
    state.last_close_equity = state.last_close_equity - amount
    return state.last_close_equity


def reset_kill_switch(state: State, equity_now: Decimal) -> Decimal:
    """Clear the kill switch and rebase the high-water mark so it does not trip again at once."""
    if not state.kill_switch.active:
        raise ValueError("the kill switch is not active")
    if equity_now <= 0:
        raise ValueError("equity must be positive")
    old = state.high_water_mark or equity_now
    state.kill_switch = KillSwitch()
    state.high_water_mark = equity_now
    return old


def tax_reserve(realized: Decimal, rate: Decimal) -> Decimal:
    """Suggested reserve on net realized short-term gains. Losses reserve nothing."""
    if realized <= 0:
        return ZERO
    return cents_up(realized * rate)


@dataclass(frozen=True)
class CashFlow:
    activity_id: str
    day: date | None
    amount: Decimal  # positive deposit, negative withdrawal
    kind: str


def parse_flows(activities: Iterable[Mapping[str, Any]]) -> list[CashFlow]:
    """Cash flows from Alpaca activities. A TRANS entry that mirrors a CSD or CSW
    entry (same date and amount) is dropped so one transfer never counts twice."""
    flows = []
    for a in activities:
        aid = str(a.get("id") or "")
        raw = a.get("net_amount")
        if not aid or raw in (None, ""):
            continue
        day_raw = a.get("date")
        day = date.fromisoformat(str(day_raw)[:10]) if day_raw else None
        flows.append(CashFlow(aid, day, Decimal(str(raw)), str(a.get("activity_type", ""))))
    specific = {(f.day, f.amount) for f in flows if f.kind in ("CSD", "CSW")}
    return [f for f in flows if not (f.kind == "TRANS" and (f.day, f.amount) in specific)]


def apply_flows(state: State, flows: Iterable[CashFlow], since: date | None) -> list[CashFlow]:
    """Apply new cash flows to the high-water mark and net flow total.

    Flows dated on or before ``since`` (the inception session) are marked seen
    without effect, because they are already part of the starting equity.
    Returns the flows applied.
    """
    seen = set(state.processed_activity_ids)
    applied = []
    hwm = state.high_water_mark
    for f in flows:
        if f.activity_id in seen:
            continue
        seen.add(f.activity_id)
        state.processed_activity_ids = [*state.processed_activity_ids, f.activity_id][
            -MAX_ACTIVITY_IDS:
        ]
        if since is not None and f.day is not None and f.day <= since:
            continue
        state.net_flows = state.net_flows + f.amount
        applied.append(f)
        if f.amount < 0 and state.milestone_alerted:
            state.withdrawn_since_milestone = state.withdrawn_since_milestone - f.amount
        if f.amount < 0 and state.principal_flow_pending:
            state.principal_flow_pending = False  # already covered by the confirm rebase
            continue
        if hwm is not None:
            hwm = max(ZERO, hwm + f.amount)
    state.high_water_mark = hwm
    return applied


def flows_alert(flows: list[CashFlow], principal_withdrawn: bool) -> Alert | None:
    if not flows:
        return None
    total = sum((f.amount for f in flows), ZERO)
    word = "withdrawal" if total < 0 else "deposit"
    amount = money(abs(total))
    msg = f"Stoa saw a {word} of {amount} on Alpaca and moved its high-water mark to match."
    if total < 0 and not principal_withdrawn:
        msg += " If this was the planned withdrawal, run: stoa money confirm-withdrawal"
    return Alert(
        title=f"Detected {word} {money(abs(total))}", message=msg, priority=HIGH, tags=("bank",)
    )


def deposit_mismatch_note(start_equity: Decimal, cfg: MoneyConfig) -> str | None:
    """Alpaca paper accounts start at $100,000 unless reset. Say so when it does not match."""
    if start_equity > cfg.deposit * Decimal("1.5") or start_equity < cfg.deposit * Decimal("0.5"):
        return (
            f"Note: the account started at {money(start_equity)} "
            f"but money.deposit is {money(cfg.deposit)}. "
            "Reset the Alpaca paper balance or change money.deposit."
        )
    return None
