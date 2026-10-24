"""Approve trades by hand with ntfy buttons. Nothing executes without the tap.

How it works (checked against the ntfy docs):
1. DECIDE draws a random nonce for the plan and stores it in state.
2. Each trade alert gets two ``http`` action buttons. Tapping one makes the
   phone POST a plain body like ``approve 3f2a9c01aa NVDA k8f...`` to a
   second, secret topic (``STOA_NTFY_APPROVAL_TOPIC``). The trade alert
   itself goes to the normal alert topic. ntfy allows up to three actions per
   notification; Stoa uses two.
3. EXECUTE polls the approval topic once (``GET /<topic>/json?poll=1&since=``
   the plan's creation time) and reads the bodies. An order runs only if the
   newest message for it says approve, carries the right plan id and nonce,
   and was published after the plan was created and before EXECUTE's window
   closed. Anything else is skipped and reported.

Security: no token, key or password ever goes into a notification. The only
secrets involved are the two topic names (which are passwords) and the nonce.
A wrong or replayed nonce, a different plan id, or a late tap does nothing.
ntfy keeps messages for about 12 hours by default, which covers the night
between DECIDE and EXECUTE.
"""

from __future__ import annotations

import json
import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from stoa.config import ApprovalConfig, Mode
from stoa.models import PlannedOrder

log = logging.getLogger(__name__)

APPROVE = "approve"
SKIP = "skip"


def new_nonce() -> str:
    return secrets.token_urlsafe(12)


def body(verb: str, plan_id: str, symbol: str, nonce: str) -> str:
    return f"{verb} {plan_id} {symbol} {nonce}"


def actions(
    server: str, approval_topic: str, plan_id: str, symbol: str, nonce: str
) -> list[dict[str, Any]]:
    """Two ntfy http actions that post the decision to the secret approval topic."""
    url = f"{server.rstrip('/')}/{approval_topic}"
    return [
        {
            "action": "http",
            "label": "Approve",
            "url": url,
            "method": "POST",
            "body": body(APPROVE, plan_id, symbol, nonce),
            "clear": True,
        },
        {
            "action": "http",
            "label": "Skip",
            "url": url,
            "method": "POST",
            "body": body(SKIP, plan_id, symbol, nonce),
            "clear": True,
        },
    ]


def required(
    cfg: ApprovalConfig,
    mode: Mode,
    first_live_session: date | None,
    today: date,
) -> bool:
    """Whether this plan needs a human tap before EXECUTE may send it.

    ``auto``: on in live mode until ``live_days`` calendar days after the
    first live trade (and always before that first trade). ``always`` forces
    it on in any mode, which is how to test it on paper. ``never`` turns it off.
    """
    if cfg.mode == "never":
        return False
    if cfg.mode == "always":
        return True
    if mode != "live":
        return False
    if first_live_session is None:
        return True
    return (today - first_live_session).days < cfg.live_days


@dataclass(frozen=True)
class Decision:
    verb: str
    plan_id: str
    symbol: str
    nonce: str
    at: datetime


def parse(message: dict[str, Any]) -> Decision | None:
    if message.get("event") != "message":
        return None
    parts = str(message.get("message", "")).split()
    if len(parts) != 4 or parts[0] not in (APPROVE, SKIP):
        return None
    try:
        at = datetime.fromtimestamp(int(message["time"]), UTC)
    except (KeyError, TypeError, ValueError):
        return None
    return Decision(parts[0], parts[1], parts[2].upper(), parts[3], at)


class ApprovalClient:
    """Reads the approval topic with one poll request."""

    def __init__(
        self,
        server: str,
        topic: str,
        *,
        token: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 15.0,
    ) -> None:
        if not topic:
            raise ValueError("approval topic is empty")
        self.server = server.rstrip("/")
        self._topic = topic
        self._token = token
        self.client = client or httpx.Client(timeout=timeout)

    def __repr__(self) -> str:
        return f"ApprovalClient(server={self.server!r}, topic=<hidden>)"

    def poll(self, since: datetime) -> list[dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        params = {"poll": "1", "since": str(int(since.timestamp()))}
        resp = self.client.get(f"{self.server}/{self._topic}/json", params=params, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"approval poll failed: HTTP {resp.status_code}")
        out = []
        for line in resp.text.splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        return out


@dataclass(frozen=True)
class Verdict:
    symbol: str
    approved: bool
    reason: str  # approved | skipped | wrong_nonce | late | no_answer | unavailable


def decide(
    orders: list[PlannedOrder],
    messages: list[dict[str, Any]],
    *,
    plan_id: str,
    nonce: str,
    created_at: datetime,
    deadline: datetime,
) -> dict[str, Verdict]:
    """The verdict for each order from the messages read off the approval topic."""
    return decide_keys(
        [o.symbol for o in orders],
        messages,
        plan_id=plan_id,
        nonce=nonce,
        created_at=created_at,
        deadline=deadline,
    )


def decide_keys(
    keys: list[str],
    messages: list[dict[str, Any]],
    *,
    plan_id: str,
    nonce: str,
    created_at: datetime,
    deadline: datetime,
) -> dict[str, Verdict]:
    """The verdict for each key (a symbol, or another name such as ANALYST)."""
    latest: dict[str, Decision] = {}
    bad_nonce: set[str] = set()
    late: set[str] = set()
    for m in messages:
        d = parse(m)
        if d is None or d.plan_id != plan_id:
            continue
        if not secrets.compare_digest(d.nonce, nonce):
            bad_nonce.add(d.symbol)
            continue
        if d.at < created_at - timedelta(seconds=5):
            continue
        if d.at > deadline:
            late.add(d.symbol)
            continue
        cur = latest.get(d.symbol)
        if cur is None or d.at >= cur.at:
            latest[d.symbol] = d
    out = {}
    for k in keys:
        d = latest.get(k)
        if d is not None and d.verb == APPROVE:
            out[k] = Verdict(k, True, "approved")
        elif d is not None:
            out[k] = Verdict(k, False, "skipped")
        elif k in late:
            out[k] = Verdict(k, False, "late")
        elif k in bad_nonce:
            out[k] = Verdict(k, False, "wrong_nonce")
        else:
            out[k] = Verdict(k, False, "no_answer")
    return out


def wait_and_decide(
    client: ApprovalClient | None,
    orders: list[PlannedOrder],
    *,
    plan_id: str,
    nonce: str,
    created_at: datetime,
    deadline: datetime,
    wait_minutes: float,
    poll_seconds: float,
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
) -> dict[str, Verdict]:
    """Poll until every order has an answer, ``wait_minutes`` pass, or the deadline."""
    return wait_for(
        client,
        [o.symbol for o in orders],
        plan_id=plan_id,
        nonce=nonce,
        created_at=created_at,
        deadline=deadline,
        wait_minutes=wait_minutes,
        poll_seconds=poll_seconds,
        now=now,
        sleep=sleep,
    )


def wait_for(
    client: ApprovalClient | None,
    keys: list[str],
    *,
    plan_id: str,
    nonce: str,
    created_at: datetime,
    deadline: datetime,
    wait_minutes: float,
    poll_seconds: float,
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
) -> dict[str, Verdict]:
    """Poll until every key has an answer, ``wait_minutes`` pass, or the deadline."""
    if client is None:
        return {k: Verdict(k, False, "unavailable") for k in keys}
    stop = min(deadline, now() + timedelta(minutes=wait_minutes))
    # Bounded by count as well as by the clock, so a stalled clock can never hang EXECUTE.
    max_polls = int(wait_minutes * 60 / poll_seconds) + 1
    polls = 0
    while True:
        polls += 1
        try:
            msgs = client.poll(created_at - timedelta(minutes=1))
            verdicts = decide_keys(
                keys, msgs, plan_id=plan_id, nonce=nonce, created_at=created_at, deadline=deadline
            )
        except (httpx.HTTPError, RuntimeError) as exc:
            log.warning("approval poll failed: %s", type(exc).__name__)
            verdicts = {k: Verdict(k, False, "unavailable") for k in keys}
        answered = all(v.reason not in ("no_answer", "unavailable") for v in verdicts.values())
        if answered or now() >= stop or polls >= max_polls:
            return verdicts
        sleep(poll_seconds)
