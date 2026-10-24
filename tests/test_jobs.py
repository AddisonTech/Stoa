from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from stoa import alerts, jobs
from stoa.broker.sim import SimBroker
from stoa.config import Secrets, Settings
from stoa.data.base import Bars
from stoa.features import cell
from stoa.models import dec
from stoa.portfolio import Portfolio
from stoa.state import StateStore
from stoa.universe import Universe
from tests.conftest import SMALL_WATCHLIST, make_settings

PAPER = {"ALPACA_PAPER_KEY_ID": "pk", "ALPACA_PAPER_SECRET": "ps"}
LIVE = {"ALPACA_LIVE_KEY_ID": "lk", "ALPACA_LIVE_SECRET": "ls", "STOA_LIVE_CONFIRM": "I_UNDERSTAND"}
DECIDE_DAY = date(2021, 10, 12)
EXEC_DAY = date(2021, 10, 13)
DECIDE_AT = datetime(2021, 10, 12, 22, 20, tzinfo=UTC)
EXEC_AT = datetime(2021, 10, 13, 13, 50, tzinfo=UTC)


class FixedProvider:
    name = "fixed"

    def __init__(self, bars: Bars) -> None:
        self.bars = bars
        self.calls = 0

    def fetch(self, symbols: list[str], start: date, end: date) -> Bars:
        self.calls += 1
        return {s: self.bars[s].loc[pd.Timestamp(start) : pd.Timestamp(end)] for s in symbols}


def settings_for(tmp: Path, mode: str = "paper", **extra: str) -> Settings:
    return make_settings(
        watchlist=",".join(SMALL_WATCHLIST),
        mode=mode,
        state__dir=str(tmp / "state"),
        scout__kind="rule",
        strategy__kind="rule",
        crosscheck__enabled="false",
        data__provider="csv",
        **extra,
    )


def prices(bars: Bars, day: date, col: str) -> dict[str, Decimal]:
    return {s: dec(cell(df, pd.Timestamp(day), col)) for s, df in bars.items()}


def sim(settings: Settings, bars: Bars, cash: str = "100") -> SimBroker:
    b = SimBroker(
        Portfolio(cash=Decimal(cash)),
        settings.costs,
        Universe.assume_all_fractionable(settings.watchlist, "SPY"),
    )
    b.begin_session(DECIDE_DAY, prices(bars, DECIDE_DAY, "close"))
    return b


def runtime(
    settings: Settings,
    bars: Bars,
    env: dict[str, str],
    now: datetime,
    broker: SimBroker | None = None,
) -> tuple[jobs.Runtime, alerts.RecordingNotifier]:
    rec = alerts.RecordingNotifier()
    rt = jobs.Runtime(
        settings=settings,
        secrets=Secrets.from_env(env),
        notifier=rec,
        now=lambda: now,
        provider=FixedProvider(bars),
        broker=broker,
        use_external=False,
        sleep=lambda _s: None,
    )
    return rt, rec


def test_outside_window_exits_zero_quietly(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path)
    rt, rec = runtime(s, bars, PAPER, datetime(2021, 10, 12, 15, 0, tzinfo=UTC))
    res = jobs.decide(rt)
    assert (res.code, res.status) == (0, "outside_window") and rec.alerts == []
    rt2, rec2 = runtime(s, bars, PAPER, DECIDE_AT)
    assert jobs.execute(rt2).status == "outside_window" and rec2.alerts == []


def test_decide_paper_builds_plan_and_summary_then_is_idempotent(
    tmp_path: Path, bars: Bars
) -> None:
    s = settings_for(tmp_path)
    broker = sim(s, bars)
    rt, rec = runtime(s, bars, PAPER, DECIDE_AT, broker)
    res = jobs.decide(rt)
    assert res.code == 0 and res.status == "decided" and res.session == DECIDE_DAY
    assert rec.alerts[-1].title == f"Stoa {DECIDE_DAY} (paper)"
    state = StateStore(s.state.dir).load()
    assert DECIDE_DAY in state.decided_sessions
    assert state.inception is not None and state.high_water_mark == Decimal(100)
    plan = state.pending_plan
    assert plan is not None and plan.execute_session == EXEC_DAY
    buy_titles = [a.title for a in rec.alerts if a.title.startswith("BUY ")]
    assert len(buy_titles) == len(plan.orders)
    for o in plan.orders:
        assert o.client_order_id.startswith(f"stoa-{EXEC_DAY:%Y%m%d}-{o.symbol}-buy-")
    assert (s.state.dir / "journal").exists() and any((s.state.dir / "plans").iterdir())
    again, rec2 = runtime(s, bars, PAPER, DECIDE_AT, broker)
    assert jobs.decide(again).status == "already_done" and rec2.alerts == []


def test_execute_after_decide_fills_and_never_double_submits(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path)
    broker = sim(s, bars)
    rt, _ = runtime(s, bars, PAPER, DECIDE_AT, broker)
    jobs.decide(rt)
    plan = StateStore(s.state.dir).load().pending_plan
    assert plan is not None and plan.orders
    broker.begin_session(EXEC_DAY, prices(bars, EXEC_DAY, "open"))
    rt2, rec = runtime(s, bars, PAPER, EXEC_AT, broker)
    res = jobs.execute(rt2)
    assert res.code == 0 and res.status == "executed"
    fills = [a for a in rec.alerts if a.title.startswith("Filled buy")]
    assert len(fills) == len(plan.orders) == len(broker.fills)
    state = StateStore(s.state.dir).load()
    assert state.pending_plan is None and EXEC_DAY in state.executed_sessions
    assert set(state.positions_meta) == {o.symbol for o in plan.orders}
    assert broker.portfolio.cash >= 0
    # A forced retry of the same session cannot submit again.
    state.pending_plan = plan
    StateStore(s.state.dir).save(state)
    rt3, _ = runtime(s, bars, PAPER, EXEC_AT, broker)
    jobs.execute(rt3, force=True)
    assert len(broker.fills) == len(plan.orders)


def test_stale_data_sends_urgent_alert_and_fails(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path)
    stale = {k: v.loc[: pd.Timestamp("2021-10-11")] for k, v in bars.items()}
    rt, rec = runtime(s, stale, PAPER, DECIDE_AT, sim(s, bars))
    res = jobs.decide(rt)
    assert res.code == 1 and res.status == "data_failure"
    assert rec.alerts[0].priority == alerts.URGENT and "bad data" in rec.alerts[0].title
    assert StateStore(s.state.dir).load().decided_sessions == []


def test_unexpected_exception_alerts_and_fails(tmp_path: Path, bars: Bars) -> None:
    class Boom(FixedProvider):
        def fetch(self, symbols: list[str], start: date, end: date) -> Bars:
            raise RuntimeError("network on fire")

    s = settings_for(tmp_path)
    rt, rec = runtime(s, bars, PAPER, DECIDE_AT, sim(s, bars))
    rt.provider = Boom(bars)
    res = jobs.decide(rt)
    assert res.code == 1 and rec.alerts[0].priority == alerts.URGENT
    assert "network on fire" in rec.alerts[0].message


def test_alert_failure_does_not_change_outcome(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path)
    rt, rec = runtime(s, bars, PAPER, DECIDE_AT, sim(s, bars))
    rec.fail = True
    assert jobs.decide(rt).code == 0


@pytest.mark.parametrize(
    "drop", ["mode", "STOA_LIVE_CONFIRM", "ALPACA_LIVE_KEY_ID", "ALPACA_LIVE_SECRET"]
)
def test_live_triple_lock_refuses_with_any_one_missing(
    tmp_path: Path, bars: Bars, drop: str
) -> None:
    mode = "paper" if drop == "mode" else "live"
    env = {k: v for k, v in LIVE.items() if k != drop}
    s = settings_for(tmp_path, mode=mode)
    if drop == "mode":
        problems = jobs.live_lock_problems(s, Secrets.from_env(env))
        assert problems == ["config mode is not live"]
        return
    rt, rec = runtime(s, bars, env, DECIDE_AT, sim(s, bars))
    res = jobs.decide(rt)
    assert res.code == jobs.EXIT_REFUSED
    assert rec.alerts[0].priority == alerts.URGENT and "locked" in rec.alerts[0].message
    rt2, _ = runtime(s, bars, env, EXEC_AT, sim(s, bars))
    assert jobs.execute(rt2).code == jobs.EXIT_REFUSED


def test_live_lock_wrong_phrase_refuses(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path, mode="live")
    env = {**LIVE, "STOA_LIVE_CONFIRM": "yes please"}
    rt, _ = runtime(s, bars, env, DECIDE_AT, sim(s, bars))
    assert jobs.decide(rt).code == jobs.EXIT_REFUSED


def test_live_unlocked_runs(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path, mode="live", exam__enforce_live="false")
    assert jobs.live_lock_problems(s, Secrets.from_env(LIVE)) == []
    rt, _ = runtime(s, bars, LIVE, DECIDE_AT, sim(s, bars))
    assert jobs.decide(rt).code == 0


def test_paper_needs_paper_keys(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path)
    rt, rec = runtime(s, bars, {}, DECIDE_AT, sim(s, bars))
    assert jobs.decide(rt).code == jobs.EXIT_REFUSED
    assert "ALPACA_PAPER_KEY_ID" in rec.alerts[0].message


def test_paper_keys_never_used_for_live(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path, mode="live")
    env = {**PAPER, "STOA_LIVE_CONFIRM": "I_UNDERSTAND"}
    rt, _ = runtime(s, bars, env, DECIDE_AT, sim(s, bars))
    assert jobs.decide(rt).code == jobs.EXIT_REFUSED


def test_signal_mode_never_touches_broker_and_settles_shadow(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path, mode="signal")
    rt, _ = runtime(s, bars, {}, DECIDE_AT)
    assert jobs.decide(rt).code == 0
    assert rt.broker is None
    plan = StateStore(s.state.dir).load().pending_plan
    assert plan is not None and plan.orders
    rt_exec, _ = runtime(s, bars, {}, EXEC_AT)
    assert jobs.execute(rt_exec).status == "signal_mode"
    rt2, rec2 = runtime(s, bars, {}, datetime(2021, 10, 13, 22, 20, tzinfo=UTC))
    assert jobs.decide(rt2).code == 0
    state = StateStore(s.state.dir).load()
    assert set(state.sim_positions) == {o.symbol for o in plan.orders}
    assert state.sim_cash is not None and state.sim_cash >= 0
    assert EXEC_DAY in state.executed_sessions
    assert "Settled" in rec2.alerts[-1].message


def test_kill_switch_trip_in_decide_alerts_urgently(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path)
    store = StateStore(s.state.dir)
    state = store.load()
    state.high_water_mark = Decimal(200)
    store.save(state)
    rt, rec = runtime(s, bars, PAPER, DECIDE_AT, sim(s, bars))
    jobs.decide(rt)
    assert "KILL SWITCH TRIPPED" in rec.titles()
    after = store.load()
    assert after.kill_switch.active
    assert after.pending_plan is None or all(
        o.side.value == "sell" for o in after.pending_plan.orders
    )


def test_execute_without_plan_marks_session(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path)
    broker = sim(s, bars)
    broker.begin_session(EXEC_DAY, prices(bars, EXEC_DAY, "open"))
    rt, rec = runtime(s, bars, PAPER, EXEC_AT, broker)
    res = jobs.execute(rt)
    assert res.status == "no_plan" and rec.alerts == []
    assert EXEC_DAY in StateStore(s.state.dir).load().executed_sessions


def test_execute_refuses_when_market_closed_even_forced(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path)
    rt, _ = runtime(s, bars, PAPER, datetime(2021, 10, 13, 22, 0, tzinfo=UTC), sim(s, bars))
    assert jobs.execute(rt, force=True).code == jobs.EXIT_REFUSED


def test_force_decide_outside_window_uses_last_closed_session(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path)
    rt, _ = runtime(s, bars, PAPER, datetime(2021, 10, 13, 12, 0, tzinfo=UTC), sim(s, bars))
    res = jobs.decide(rt, force=True)
    assert res.code == 0 and res.session == DECIDE_DAY


def test_milestone_alert_sent_from_decide_once(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path)
    store = StateStore(s.state.dir)
    broker = sim(s, bars)
    rt, rec = runtime(s, bars, PAPER, DECIDE_AT, broker)
    jobs.decide(rt)
    assert "Withdraw $100 now" not in rec.titles()
    # Equity doubles; the next session's DECIDE sends the withdraw alert once.
    broker.portfolio.cash = Decimal(210)
    broker.portfolio.positions.clear()
    nxt = datetime(2021, 10, 13, 22, 20, tzinfo=UTC)
    rt2, rec2 = runtime(s, bars, PAPER, nxt, broker)
    state = store.load()
    state.pending_plan = None
    state.positions_meta = {}
    store.save(state)
    jobs.decide(rt2)
    assert rec2.titles().count("Withdraw $100 now") == 1
    rt3, rec3 = runtime(s, bars, PAPER, datetime(2021, 10, 14, 22, 20, tzinfo=UTC), broker)
    jobs.decide(rt3)
    assert "Withdraw $100 now" not in rec3.titles()


def test_llm_budget_ledger_is_the_state_ledger(tmp_path: Path, bars: Bars) -> None:
    """Spend recorded by the LLM scout must land in the state that DECIDE saves."""
    from stoa.roles.scout import LLMScout

    s = make_settings(
        watchlist=",".join(SMALL_WATCHLIST), state__dir=str(tmp_path / "s"), scout__kind="llm"
    )
    state = StateStore(s.state.dir).load()
    rt, _ = runtime(s, bars, {**PAPER, "ANTHROPIC_API_KEY": "sk-test"}, DECIDE_AT)
    scout = jobs.make_scout(rt, state)
    assert isinstance(scout, LLMScout)
    scout.client.budget.record(0.25)
    assert state.llm_spend_by_month["2021-10"] == Decimal("0.25")


def test_rule_scout_when_no_key(tmp_path: Path, bars: Bars) -> None:
    from stoa.roles.scout import RuleScout

    s = make_settings(
        watchlist=",".join(SMALL_WATCHLIST), state__dir=str(tmp_path / "s"), scout__kind="auto"
    )
    rt, _ = runtime(s, bars, PAPER, DECIDE_AT)
    assert isinstance(jobs.make_scout(rt, StateStore(s.state.dir).load()), RuleScout)
