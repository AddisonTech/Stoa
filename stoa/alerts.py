"""Phone alerts through ntfy, plus the alert texts themselves.

Alerts must never crash a job. A failed send is logged and retried, and the
job's final status still reflects the trading outcome. The ntfy topic is the
password, so it is never logged and never committed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol

import httpx

from stoa.models import PlannedOrder, Side
from stoa.redact import redact

log = logging.getLogger(__name__)

URGENT = 5
HIGH = 4
DEFAULT = 3
LOW = 2
MIN = 1


@dataclass(frozen=True)
class Alert:
    title: str
    message: str
    priority: int = DEFAULT
    tags: tuple[str, ...] = ()
    click: str | None = None
    markdown: bool = False
    actions: tuple[dict[str, object], ...] = ()  # ntfy action buttons (at most 3)

    def __post_init__(self) -> None:
        if not MIN <= self.priority <= URGENT:
            raise ValueError(f"priority {self.priority} outside 1..5")


class Notifier(Protocol):
    def send(self, alert: Alert) -> bool: ...


def truncate_utf8(text: str, max_bytes: int) -> str:
    """Keep messages under ntfy's attachment threshold without splitting a character."""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    suffix = "\n[truncated]"
    cut = raw[: max_bytes - len(suffix.encode())]
    return cut.decode("utf-8", errors="ignore") + suffix


class NtfyNotifier:
    """Publishes JSON to the server root, which handles UTF-8 titles cleanly."""

    def __init__(
        self,
        server: str,
        topic: str,
        *,
        token: str | None = None,
        default_click: str | None = None,
        retries: int = 2,
        max_message_bytes: int = 3500,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not topic:
            raise ValueError("ntfy topic is empty")
        self.server = server.rstrip("/")
        self._topic = topic
        self._token = token
        self.default_click = default_click
        self.retries = retries
        self.max_message_bytes = max_message_bytes
        self.client = client or httpx.Client(timeout=timeout)
        self.sleep = sleep

    def __repr__(self) -> str:
        return f"NtfyNotifier(server={self.server!r}, topic=<hidden>)"

    def payload(self, alert: Alert) -> dict[str, object]:
        body: dict[str, object] = {
            "topic": self._topic,
            "title": redact(alert.title),
            "message": truncate_utf8(redact(alert.message), self.max_message_bytes),
            "priority": alert.priority,
        }
        if alert.tags:
            body["tags"] = list(alert.tags)
        click = alert.click or self.default_click
        if click:
            body["click"] = click
        if alert.markdown:
            body["markdown"] = True
        if alert.actions:
            body["actions"] = [dict(a) for a in alert.actions[:3]]
        return body

    def send(self, alert: Alert) -> bool:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        for attempt in range(self.retries + 1):
            try:
                resp = self.client.post(
                    self.server + "/", json=self.payload(alert), headers=headers
                )
                if resp.status_code < 300:
                    return True
                log.warning("ntfy returned HTTP %s for %r", resp.status_code, alert.title)
            except httpx.HTTPError as exc:
                log.warning("ntfy send failed for %r: %s", alert.title, type(exc).__name__)
            if attempt < self.retries:
                self.sleep(1.0 + attempt)
        log.error("alert not delivered after %d attempts: %r", self.retries + 1, alert.title)
        return False


class ConsoleNotifier:
    """Default when no topic is set: prints alerts to the log."""

    def send(self, alert: Alert) -> bool:
        print(f"[alert p{alert.priority}] {redact(alert.title)}\n{redact(alert.message)}\n")
        return True


@dataclass
class RecordingNotifier:
    """Test double that keeps every alert."""

    alerts: list[Alert] = field(default_factory=list)
    fail: bool = False

    def send(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return not self.fail

    def titles(self) -> list[str]:
        return [a.title for a in self.alerts]


def safe_send(notifier: Notifier, alert: Alert) -> bool:
    """Send without ever raising."""
    try:
        return notifier.send(alert)
    except Exception as exc:
        log.error("notifier raised %s for %r", type(exc).__name__, alert.title)
        return False


# ---- alert texts --------------------------------------------------------------


def money(value: Decimal | float) -> str:
    v = float(value)
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"


def pct(value: float, signed: bool = True) -> str:
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def shares(qty: Decimal) -> str:
    text = f"{qty:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _hhmm(hhmm: str) -> str:
    """A HH:MM time without the leading zero: 09:40 becomes 9:40."""
    return hhmm.lstrip("0") if not hhmm.startswith("00") else hhmm


def approvals_closed_alert(symbols: Sequence[str], approved: set[str], deadline: str) -> Alert:
    """The 9:40 am note: which picks were approved, and that the rest are skipped."""
    yes = [x for x in symbols if x in approved]
    rest = [x for x in symbols if x not in approved]
    head = f"Approved {', '.join(yes)}" if yes else "Approved none"
    tail = f"; skipped the rest ({', '.join(rest)})" if rest else ""
    after = (
        "Approved buys go in at 9:45 am."
        if yes
        else "Nothing will be bought today. Sells and stops are not affected."
    )
    return Alert(
        title=f"Approvals closed at {_hhmm(deadline)}",
        message=f"{head}{tail}. {after}",
        priority=HIGH,
        tags=("lock",),
    )


def too_late_alert(verb: str, symbol: str, deadline: str) -> Alert:
    """The reply to a tap after the deadline. The tap is not acted on."""
    return Alert(
        title=f"Too late: {verb} {symbol}",
        message=f"Too late, approvals closed at {_hhmm(deadline)}.",
        priority=HIGH,
        tags=("hourglass",),
    )


def signal_alert(
    order: PlannedOrder, equity: Decimal, mode: str, when: str = "at the open"
) -> Alert:
    if mode == "signal":
        when = f"{when} (you place it)"
    if order.side is Side.BUY:
        assert order.notional is not None
        share = float(order.notional / equity) if equity > 0 else 0.0
        stop = f" Stop {money(order.stop_price)}." if order.stop_price is not None else ""
        why = order.thesis or order.reason
        msg = (
            f"BUY {order.symbol} {money(order.notional)} ({share:.0%} of account) {when}."
            f"{stop} Why: {why}"
        )
        return Alert(
            title=f"BUY {order.symbol}",
            message=msg,
            priority=HIGH,
            tags=("chart_with_upwards_trend",),
        )
    qty = shares(order.qty) if order.qty is not None else "all"
    why = order.reason.replace("_", " ")
    detail = f" ({order.thesis})" if order.thesis else ""
    msg = f"SELL {order.symbol} all {qty} sh {when}. Why: {why}{detail}."
    return Alert(
        title=f"SELL {order.symbol}",
        message=msg,
        priority=HIGH,
        tags=("chart_with_downwards_trend",),
    )


def with_actions(alert: Alert, actions: list[dict[str, object]], note: str) -> Alert:
    """The same alert with action buttons and a line telling the reader to tap one."""
    return Alert(
        title=alert.title,
        message=f"{alert.message}\n{note}",
        priority=alert.priority,
        tags=alert.tags,
        click=alert.click,
        markdown=alert.markdown,
        actions=tuple(actions),
    )


def fill_alert(symbol: str, side: Side, qty: Decimal, price: Decimal, mode: str) -> Alert:
    gross = qty * price
    action = f"{side.value.upper()} {symbol}"
    msg = f"{action}: {shares(qty)} sh at {money(price)} = {money(gross)} ({mode})."
    return Alert(
        title=f"Filled {side.value} {symbol}",
        message=msg,
        priority=HIGH,
        tags=("white_check_mark",),
    )


def rejected_order_alert(symbol: str, side: Side, detail: str, live: bool) -> Alert:
    return Alert(
        title=f"Order rejected: {side.value} {symbol}",
        message=(
            f"The broker did not fill {side.value} {symbol}: {detail}. "
            f"Check the {'live' if live else 'paper'} account in the Alpaca app."
        ),
        priority=URGENT,  # a failure, in paper too
        tags=("x",),
    )


def skipped_order_alert(symbol: str, side: Side, reason: str, detail: str) -> Alert:
    if reason == "not_approved":
        msg = f"Not approved ({detail.removeprefix('approval ')}). Nothing was sent for {symbol}."
    else:
        msg = f"Risk check {reason.replace('_', ' ')}: {detail}. Nothing was sent for {symbol}."
    return Alert(
        title=f"Skipped {side.value} {symbol}",
        message=msg,
        priority=HIGH if side is Side.SELL else LOW,
        tags=("no_entry_sign",),
    )


def price_check_alert(symbol: str, detail: str) -> Alert:
    return Alert(
        title=f"Skipped buy {symbol}: price check",
        message=f"The second price source did not confirm {symbol}: {detail}. Nothing was sent.",
        priority=HIGH,
        tags=("warning",),
    )


def kill_switch_alert(equity: Decimal, hwm: Decimal, drawdown: float) -> Alert:
    msg = (
        f"Equity {money(equity)} is {drawdown:.1%} below the high of {money(hwm)}. "
        "Stoa is selling everything at the next open and will not buy again until you reset it. "
        "Read docs/RUNBOOK.md, section Kill switch, before resetting."
    )
    return Alert(
        title="KILL SWITCH TRIPPED", message=msg, priority=URGENT, tags=("rotating_light",)
    )


def job_failure_alert(job: str, error: str) -> Alert:
    return Alert(
        title=f"Stoa {job} failed",
        message=f"{error[:500]}\nNothing else happens until this is fixed. Check the Actions log.",
        priority=URGENT,
        tags=("warning",),
    )


def data_failure_alert(job: str, error: str) -> Alert:
    return Alert(
        title=f"Stoa {job}: bad data",
        message=f"{error[:500]}\nStoa did not trade on partial data. It retries at the next slot.",
        priority=URGENT,
        tags=("warning",),
    )


def refusal_alert(job: str, reason: str) -> Alert:
    return Alert(
        title=f"Stoa {job} refused to run",
        message=reason,
        priority=URGENT,
        tags=("lock",),
    )


def llm_fallback_alert(reason: str, spent: float, cap: float) -> Alert:
    msg = (
        f"The AI scout was not used today ({reason.replace('_', ' ')}). "
        f"The rule-based scout ran instead. API spend this month {money(spent)} of {money(cap)}."
    )
    return Alert(title="Scout fell back to rules", message=msg, priority=HIGH, tags=("robot",))


@dataclass(frozen=True)
class PositionLine:
    symbol: str
    value: Decimal
    gain: float


@dataclass(frozen=True)
class Summary:
    session: str
    mode: str
    equity: Decimal
    day_change: float | None
    since_start: float | None
    spy_since_start: float | None
    positions: Sequence[PositionLine]
    cash: Decimal
    next_actions: Sequence[str]
    api_spend: float
    api_cap: float
    realized_ytd: Decimal
    tax_reserve: Decimal
    kill_switch: bool
    notes: Sequence[str] = ()
    spy_day: float | None = None  # SPY's change today, shown beside the account's
    day_lines: Sequence[str] = ()  # how each position and today's buys did today
    next_line: str | None = None  # replaces the list of planned orders


def daily_summary_alert(s: Summary) -> Alert:
    today = ""
    if s.day_change is not None and s.spy_day is not None:
        today = f" ({pct(s.day_change)} today, SPY {pct(s.spy_day)})"
    elif s.day_change is not None:
        today = f" ({pct(s.day_change)} today)"
    lines = [f"Equity {money(s.equity)}{today}"]
    if s.since_start is not None and s.spy_since_start is not None:
        edge = s.since_start - s.spy_since_start
        lines.append(
            f"Since start {pct(s.since_start)} vs SPY {pct(s.spy_since_start)} ({pct(edge)})"
        )
    if s.day_lines:
        lines.extend(s.day_lines)
    elif s.positions:
        held = ", ".join(f"{p.symbol} {money(p.value)} {pct(p.gain)}" for p in s.positions)
        lines.append(f"Holding: {held}")
    else:
        lines.append("Holding: nothing")
    lines.append(f"Cash {money(s.cash)}")
    if s.next_line is not None:
        lines.append(s.next_line)
    else:
        lines.append(
            "Next: " + ("; ".join(s.next_actions) if s.next_actions else "no trades at the open")
        )
    lines.append(f"API spend {money(s.api_spend)} of {money(s.api_cap)} this month")
    lines.append(
        f"Realized short-term gains this year {money(s.realized_ytd)}, "
        f"set aside {money(s.tax_reserve)} for tax"
    )
    if s.kill_switch:
        lines.append("KILL SWITCH ACTIVE: no new buys until reset")
    lines.extend(s.notes)
    return Alert(
        title=f"Stoa {s.session} ({s.mode})",
        message="\n".join(lines),
        priority=DEFAULT,
        tags=("bar_chart",),
    )


def hello_alert() -> Alert:
    return Alert(
        title="Stoa test alert",
        message="If you can read this on your phone, alerts work.",
        priority=DEFAULT,
        tags=("tada",),
    )
