from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from stoa import alerts, jobs, spendcap
from stoa.config import ApprovalConfig, BudgetConfig, Secrets
from stoa.data.base import Bars
from stoa.state import CapRaise, State, StateStore
from tests.conftest import SMALL_WATCHLIST, make_settings
from tests.test_jobs import DECIDE_AT, FixedProvider

D = Decimal
NOW = datetime(2026, 10, 20, 22, 0, tzinfo=UTC)
BUDGET = BudgetConfig()
APPROVAL = ApprovalConfig(wait_minutes=5, poll_interval_seconds=30)
TOPIC = "approval-topic-for-tests"
Reply = Callable[[alerts.Alert, datetime], list[dict[str, Any]]]


class Clock:
    def __init__(self, start: datetime) -> None:
        self.t = start

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)


class FakeApprovals:
    """Answers from the buttons of the alert that was really sent."""

    def __init__(self, sent: list[alerts.Alert], reply: Reply | None) -> None:
        self.sent = sent
        self.reply = reply
        self.polls = 0

    def poll(self, since: datetime) -> list[dict[str, Any]]:
        self.polls += 1
        asks = [a for a in self.sent if a.actions]
        if not asks or self.reply is None:
            return []
        return self.reply(asks[-1], NOW)


def tap(index: int, *, delay: timedelta = timedelta(minutes=1), nonce: str | None = None) -> Reply:
    """Tap button ``index`` (0 Approve, 1 Skip) ``delay`` after the ask."""

    def reply(ask: alerts.Alert, asked: datetime) -> list[dict[str, Any]]:
        body = str(ask.actions[index]["body"])
        if nonce is not None:
            verb, plan, key, _ = body.split()
            body = f"{verb} {plan} {key} {nonce}"
        return [{"event": "message", "message": body, "time": int((asked + delay).timestamp())}]

    return reply


def run_gate(
    state: State,
    reply: Reply | None = None,
    *,
    budget: BudgetConfig = BUDGET,
    at: datetime = NOW,
    topic: str | None = TOPIC,
) -> tuple[spendcap.Gate, list[alerts.Alert]]:
    sent: list[alerts.Alert] = []
    clock = Clock(at)
    client = FakeApprovals(sent, reply) if topic else None
    g = spendcap.gate(
        state,
        budget,
        APPROVAL,
        now=clock.now,
        sleep=clock.sleep,
        send=sent.append,
        client=client,  # type: ignore[arg-type]
        server="https://ntfy.example",
        approval_topic=topic,
    )
    return g, sent


def spent(amount: str, month: str = "2026-10") -> State:
    return State(llm_spend_by_month={month: D(amount)})


def test_heads_up_at_80_percent_fires_once_a_month() -> None:
    state = spent("20.00")
    g, sent = run_gate(state)
    assert not g.paused and [a.title for a in sent] == ["Analyst spend at 80% of the cap"]
    assert sent[0].priority == alerts.DEFAULT and "$20.00" in sent[0].message
    _, again = run_gate(state)
    assert again == []
    assert run_gate(spent("19.99"))[1] == []  # under 80%: nothing


def test_cap_pauses_and_asks_with_buttons_on_the_approval_topic() -> None:
    state = spent("25.00")
    g, sent = run_gate(state, reply=None)
    ask = next(a for a in sent if a.actions)
    assert ask.title == "Analyst paused at its cap" and ask.priority == alerts.HIGH
    assert [b["label"] for b in ask.actions] == ["Approve", "Skip"]
    assert all(str(b["url"]).endswith(f"/{TOPIC}") for b in ask.actions)
    assert " ANALYST " in str(ask.actions[0]["body"])
    assert g.paused and state.cap_asks[-1].outcome == "no_answer"


def test_approve_raises_this_month_by_ten() -> None:
    state = spent("25.00")
    g, sent = run_gate(state, tap(0))
    assert not g.paused and g.cap == D(35)
    ask = next(a for a in sent if a.actions)
    nonce = str(ask.actions[0]["body"]).split()[-1]
    r = state.cap_raises[0]
    assert (r.month, r.amount, r.nonce) == ("2026-10", D(10), nonce) and r.at >= NOW
    assert state.cap_asks[-1].outcome == "approved" and sent[-1].title == "Analyst cap raised"
    assert spendcap.cap_for(state, BUDGET, "2026-10") == D(35)


@pytest.mark.parametrize(
    ("reply", "outcome"),
    [
        (tap(1), "skipped"),
        (None, "no_answer"),
        (tap(0, nonce="not-the-nonce"), "wrong_nonce"),
        (tap(0, delay=timedelta(minutes=6)), "late"),
    ],
)
def test_skip_silence_wrong_nonce_and_late_all_stay_paused(
    reply: Reply | None, outcome: str
) -> None:
    state = spent("25.00")
    g, sent = run_gate(state, reply)
    assert g.paused and state.cap_raises == []
    assert state.cap_asks[-1].outcome == outcome
    assert sent[-1].title == "Analyst stays paused"
    # It asked once at this level; the next run stays paused without asking again.
    g2, again = run_gate(state, tap(0))
    assert g2.paused and again == [] and state.cap_raises == []


def test_raises_end_with_the_month() -> None:
    state = spent("33.00")
    state.cap_raises = [CapRaise(month="2026-10", amount=D(10), at=NOW, nonce="n")]
    assert spendcap.cap_for(state, BUDGET, "2026-10") == D(35)
    assert spendcap.cap_for(state, BUDGET, "2026-11") == D(25)
    nov = datetime(2026, 11, 2, 22, 0, tzinfo=UTC)
    g, _ = run_gate(state, at=nov)
    assert g.month == "2026-11" and g.cap == D(25) and g.spent == 0 and not g.paused


def test_hard_ceiling_holds() -> None:
    tight = BudgetConfig(monthly_cap_usd=25, hard_ceiling_usd=30)
    state = spent("25.00")
    g, _ = run_gate(state, tap(0), budget=tight)
    assert g.cap == D(30) and state.cap_raises[0].amount == D(5)  # only up to the ceiling
    state.llm_spend_by_month["2026-10"] = D("30.00")
    g2, sent = run_gate(state, tap(0), budget=tight)
    assert g2.paused and g2.cap == D(30) and not any(a.actions for a in sent)
    assert state.cap_asks[-1].outcome == "ceiling" and "hard ceiling" in sent[-1].message
    # Even raises recorded past it cannot lift the cap.
    state.cap_raises.append(CapRaise(month="2026-10", amount=D(50), at=NOW, nonce="x"))
    assert spendcap.cap_for(state, tight, "2026-10") == D(30)


def test_no_approval_topic_stays_paused_without_buttons() -> None:
    state = spent("25.00")
    g, sent = run_gate(state, topic=None)
    assert g.paused and state.cap_asks[-1].outcome == "unavailable"
    assert not any(a.actions for a in sent)


def test_ceiling_below_the_cap_is_rejected() -> None:
    with pytest.raises(ValueError, match="hard_ceiling"):
        BudgetConfig(monthly_cap_usd=25, hard_ceiling_usd=20)


class Untouchable:
    """Fails the test if DECIDE calls the analyst or the news feed while paused."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the paused analyst was used ({name})")


def test_trading_continues_while_the_analyst_is_paused(tmp_path: Path, bars: Bars) -> None:
    s = make_settings(
        watchlist=",".join(SMALL_WATCHLIST),
        mode="paper",
        state__dir=str(tmp_path / "state"),
        scout__kind="rule",
        strategy__kind="model",
        model__use_text="true",
        crosscheck__enabled="false",
        data__provider="csv",
    )
    StateStore(s.state.dir).save(State(llm_spend_by_month={"2021-10": D("25.00")}))
    rec = alerts.RecordingNotifier()
    rt = jobs.Runtime(
        settings=s,
        secrets=Secrets.from_env(
            {"ALPACA_PAPER_KEY_ID": "pk", "ALPACA_PAPER_SECRET": "ps", "ANTHROPIC_API_KEY": "k"}
        ),
        notifier=rec,
        now=lambda: DECIDE_AT,
        provider=FixedProvider(bars),
        broker=None,
        use_external=False,
        sleep=lambda _s: None,
        news_client=Untouchable(),
        analyst_client=Untouchable(),
    )
    from tests.test_jobs import sim

    rt.broker = sim(s, bars)
    res = jobs.decide(rt)
    assert (res.code, res.status) == (0, "decided")
    state = StateStore(s.state.dir).load()
    assert state.pending_plan is not None and state.cap_asks[-1].outcome == "unavailable"
    titles = [a.title for a in rec.alerts]
    assert "Analyst stays paused" in titles and titles[-1].startswith("Stoa 2021-10-12")
    summary = rec.alerts[-1].message
    assert "Analyst paused at its cap" in summary
