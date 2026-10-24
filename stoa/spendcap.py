"""The analyst's monthly spend cap: a heads-up near it, a pause at it, and a tap to raise it.

- At ``budget.warn_fraction`` (80%) of this month's cap, one ntfy heads-up,
  once per month.
- At the cap, the analyst pauses and asks once, with Approve and Skip buttons
  that post to the approval topic, exactly like a trade approval: a fresh
  nonce, and DECIDE waits ``approval.wait_minutes`` for the tap. Approve raises
  this month's cap by ``budget.raise_step_usd``. Skip, no answer, a wrong nonce
  or a late tap leave it paused for the rest of the month at that level.
- Raises belong to their month. On the 1st the cap is ``budget.monthly_cap_usd``
  again. No raise ever takes a month past ``budget.hard_ceiling_usd``.

Trading never waits on any of this. With the analyst paused, DECIDE runs the
model without new news scores, the same as with the analyst turned off.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from stoa import alerts, approvals
from stoa.approvals import ApprovalClient
from stoa.config import ApprovalConfig, BudgetConfig
from stoa.models import ZERO, dec
from stoa.state import CapAsk, CapRaise, State

KEY = "ANALYST"  # the approval key, where a trade would carry its symbol


def month_of(ts: datetime) -> str:
    """The ledger's month: the UTC calendar month, the same key the spend ledger uses."""
    return (ts.astimezone(UTC) if ts.tzinfo is not None else ts).strftime("%Y-%m")


def cap_for(state: State, cfg: BudgetConfig, month: str) -> Decimal:
    """This month's cap: the base plus this month's approved raises, never past the ceiling."""
    raised = sum((r.amount for r in state.cap_raises if r.month == month), ZERO)
    return min(dec(cfg.monthly_cap_usd) + raised, dec(cfg.hard_ceiling_usd))


def spent(state: State, month: str) -> Decimal:
    return state.llm_spend_by_month.get(month, ZERO)


@dataclass
class Gate:
    month: str
    cap: Decimal
    spent: Decimal
    paused: bool
    sent: list[str] = field(default_factory=list)


def warn_if_due(state: State, cfg: BudgetConfig, now: datetime) -> alerts.Alert | None:
    """The 80% heads-up, at most once per month. Marks the month as warned."""
    month = month_of(now)
    cap = cap_for(state, cfg, month)
    used = spent(state, month)
    if month in state.cap_warned_months or cap <= 0 or used < cap * dec(cfg.warn_fraction):
        return None
    state.cap_warned_months = [*state.cap_warned_months, month][-24:]
    return alerts.Alert(
        title="Analyst spend at 80% of the cap",
        message=(
            f"Claude analyst spend this month is {alerts.money(used)} of the "
            f"{alerts.money(cap)} cap. At the cap it pauses and asks before spending more. "
            "Trading is not affected."
        ),
        priority=alerts.DEFAULT,
        tags=("moneybag",),
    )


def gate(
    state: State,
    cfg: BudgetConfig,
    approval: ApprovalConfig,
    *,
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
    send: Callable[[alerts.Alert], object],
    client: ApprovalClient | None,
    server: str,
    approval_topic: str | None,
) -> Gate:
    """Decide whether the analyst may spend today. Asks for a raise once at each cap level."""
    at = now()
    month = month_of(at)
    out = Gate(month, cap_for(state, cfg, month), spent(state, month), paused=False)
    heads_up = warn_if_due(state, cfg, at)
    if heads_up is not None:
        send(heads_up)
        out.sent.append(heads_up.title)
    if out.spent < out.cap:
        return out
    out.paused = True
    if any(a.month == month and a.cap == out.cap for a in state.cap_asks):
        return out  # already asked at this level this month: stay paused, no repeat alert
    ceiling = dec(cfg.hard_ceiling_usd)
    nonce = approvals.new_nonce()
    if out.cap >= ceiling:
        _record_ask(state, month, out.cap, nonce, at, "ceiling")
        _send(out, send, _paused_alert(out, f"the hard ceiling of {alerts.money(ceiling)}"))
        return out
    if client is None or approval_topic is None:
        _record_ask(state, month, out.cap, nonce, at, "unavailable")
        _send(out, send, _paused_alert(out, "no approval topic is set, so it cannot ask"))
        return out
    step = min(dec(cfg.raise_step_usd), ceiling - out.cap)
    plan_id = f"cap{month.replace('-', '')}x{int(out.cap * 100)}"
    ask = alerts.with_actions(
        alerts.Alert(
            title="Analyst paused at its cap",
            message=(
                f"Claude analyst spend this month is {alerts.money(out.spent)} of the "
                f"{alerts.money(out.cap)} cap, so it has paused. Approve raises this month's "
                f"cap by {alerts.money(step)} (never past {alerts.money(ceiling)}). "
                "Trading carries on either way, without new news scores."
            ),
            priority=alerts.HIGH,
            tags=("moneybag",),
        ),
        approvals.actions(server, approval_topic, plan_id, KEY, nonce),
        "Tap Approve or Skip. Nothing more is spent without Approve.",
    )
    _send(out, send, ask)
    deadline = at + timedelta(minutes=approval.wait_minutes)
    verdict = approvals.wait_for(
        client,
        [KEY],
        plan_id=plan_id,
        nonce=nonce,
        created_at=at,
        deadline=deadline,
        wait_minutes=approval.wait_minutes,
        poll_seconds=approval.poll_interval_seconds,
        now=now,
        sleep=sleep,
    )[KEY]
    _record_ask(state, month, out.cap, nonce, at, verdict.reason)
    if not verdict.approved:
        why = verdict.reason.replace("_", " ")
        _send(out, send, _paused_alert(out, f"the raise was not approved ({why})"))
        return out
    state.cap_raises = [
        *state.cap_raises,
        CapRaise(month=month, amount=step, at=now(), nonce=nonce),
    ]
    out.cap = cap_for(state, cfg, month)
    out.paused = out.spent >= out.cap
    _send(
        out,
        send,
        alerts.Alert(
            title="Analyst cap raised",
            message=(
                f"This month's analyst cap is now {alerts.money(out.cap)} "
                f"(spent {alerts.money(out.spent)}). It goes back to "
                f"{alerts.money(dec(cfg.monthly_cap_usd))} on the 1st."
            ),
            priority=alerts.DEFAULT,
            tags=("moneybag",),
        ),
    )
    return out


def _record_ask(
    state: State, month: str, cap: Decimal, nonce: str, at: datetime, outcome: str
) -> None:
    state.cap_asks = [
        *state.cap_asks,
        CapAsk(month=month, cap=cap, nonce=nonce, asked_at=at, outcome=outcome),
    ][-60:]


def _send(out: Gate, send: Callable[[alerts.Alert], object], alert: alerts.Alert) -> None:
    send(alert)
    out.sent.append(alert.title)


def _paused_alert(g: Gate, why: str) -> alerts.Alert:
    return alerts.Alert(
        title="Analyst stays paused",
        message=(
            f"Claude analyst spend this month is {alerts.money(g.spent)} of the "
            f"{alerts.money(g.cap)} cap and it stays paused: {why}. It resumes on the 1st. "
            "Trading carries on without new news scores."
        ),
        priority=alerts.DEFAULT,
        tags=("moneybag",),
    )
