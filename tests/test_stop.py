"""The owner's stop (config `stop`, 2026-10-29): a stopped account sells everything it holds at
its next 9:45 am run, buys nothing and then trades nothing; a stopped shadow scoreboard scores,
locks and grades nothing. Never the kill switch."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from stoa import alerts, bigpaper, clock, dashboard, jobs
from stoa.config import Secrets, Settings, load_settings
from stoa.data.base import Bars
from stoa.models import Side
from stoa.state import StateStore
from tests.conftest import ROOT, make_settings
from tests.test_bigpaper import (
    BIG,
    SESSION,
    SMALL,
    FakeBroker,
    at,
    bars_for,
    no_earnings,
)
from tests.test_jobs import DECIDE_AT, EXEC_AT, EXEC_DAY, PAPER, prices, runtime, settings_for, sim

D = Decimal
DAY3 = date(2021, 10, 14)


def test_the_shipped_config_stops_the_100_dollar_account_and_shadow_not_trial_15() -> None:
    s = load_settings(ROOT / "config" / "default.yaml")
    assert s.stop.usd100 and s.stop.shadow
    assert not s.stop.big_paper and s.big_paper.strategy == "trial15"  # back on, 2026-10-29
    assert s.risk.kill_switch_drawdown_pct == D("0.25")
    assert s.big_paper.enabled  # on, so the 9:45 am run can sell; the stop keeps it from buying


# ---- the $100 account -----------------------------------------------------------------------


def test_the_100_dollar_account_sells_everything_then_buys_nothing(
    tmp_path: Path, bars: Bars
) -> None:
    s = settings_for(tmp_path)
    stopped = settings_for(tmp_path, stop__usd100="true")
    broker = sim(s, bars)
    jobs.decide(runtime(s, bars, PAPER, DECIDE_AT, broker)[0])
    broker.begin_session(EXEC_DAY, prices(bars, EXEC_DAY, "open"))
    jobs.execute(runtime(s, bars, PAPER, EXEC_AT, broker)[0])
    held = {k for k, lot in broker.portfolio.positions.items() if lot.qty > 0}
    assert held  # bought on the first day
    # The evening plan is made before the stop; the stop's 9:45 am run sells everything anyway.
    broker.begin_session(EXEC_DAY, prices(bars, EXEC_DAY, "close"))
    jobs.decide(runtime(s, bars, PAPER, datetime(2021, 10, 13, 22, 20, tzinfo=UTC), broker)[0])
    rt, rec = runtime(stopped, bars, PAPER, datetime(2021, 10, 14, 13, 0, tzinfo=UTC), broker)
    assert jobs.sweep_job(rt).status == "stopped"
    assert [a.title for a in rec.alerts] == ["Stopped by the owner"]
    assert all(sym in rec.alerts[0].message for sym in held)
    broker.begin_session(DAY3, prices(bars, DAY3, "open"))
    rt, rec = runtime(stopped, bars, PAPER, datetime(2021, 10, 14, 13, 50, tzinfo=UTC), broker)
    assert jobs.execute(rt).status == "executed"
    assert not any(lot.qty > 0 for lot in broker.portfolio.positions.values())
    today = [f for f in broker.fills if f.session == DAY3]
    assert {f.symbol for f in today} == held and all(f.side is Side.SELL for f in today)
    state = StateStore(stopped.state.dir).load()
    assert not state.kill_switch.active
    # Then nothing: the evening plan buys nothing.
    broker.begin_session(DAY3, prices(bars, DAY3, "close"))
    rt, rec = runtime(stopped, bars, PAPER, datetime(2021, 10, 14, 22, 20, tzinfo=UTC), broker)
    jobs.decide(rt)
    plan = StateStore(stopped.state.dir).load().pending_plan
    assert plan is not None and not [o for o in plan.orders if o.side is Side.BUY]
    assert "Stopped by the owner: the next 9:45 am run sells" in rec.alerts[-1].message


# ---- the big paper account and the shadow scoreboard --------------------------------------------


def big_settings(tmp_path: Path) -> Settings:
    return make_settings(
        big_paper__state_dir=str(tmp_path / "big"),
        big_paper__strategy="trial14",
        big_paper__idle_money="cash",
        stop__big_paper="true",
        stop__shadow="true",
    )


def test_the_big_account_sells_everything_then_stays_quiet(tmp_path: Path) -> None:
    fb = FakeBroker(
        D(1000),
        {"SPY": D(400), "AAPL": D(100), "ZYXI": D(5)},
        held={"SPY": D(10), "AAPL": D(50), "ZYXI": D(400)},  # ZYXI: bought by hand
    )

    def no_brain(day: date) -> list[object]:
        raise AssertionError("the news brain is not read while stopped")

    rec = alerts.RecordingNotifier()
    rt = bigpaper.BigRuntime(
        base=big_settings(tmp_path),
        secrets=Secrets.from_env({**SMALL, **BIG}),
        notifier=rec,
        broker=fb,  # type: ignore[arg-type]
        now=lambda: at(SESSION, "09:00"),
        bars=bars_for,
        signals=no_brain,  # type: ignore[arg-type]
        calendar=no_earnings,
        sectors={},
    )
    assert bigpaper.picks(rt).status == "picked"
    plan = StateStore(tmp_path / "big").load().pending_plan
    assert plan is not None
    assert {(o.symbol, o.side, o.reason) for o in plan.orders} == {
        (sym, Side.SELL, "stopped") for sym in ("SPY", "AAPL", "ZYXI")
    }
    msg = rec.alerts[-1].message
    assert msg.startswith("Stopped by the owner.") and "No buys today." in msg
    assert "Sell: AAPL (stopped), SPY (stopped), ZYXI (stopped)" in msg

    rt.now = lambda: at(SESSION, "09:46")
    assert bigpaper.execute(rt).status == "executed"
    assert fb.positions() == {}
    assert not any(o.side is Side.BUY for o in fb.orders.values())
    assert not StateStore(tmp_path / "big").load().kill_switch.active
    fills = rec.alerts[-1]
    assert fills.message.startswith("STOPPED by the owner") and fills.priority == alerts.HIGH

    rt.now = lambda: at(SESSION, "17:16")
    assert bigpaper.summary(rt).status == "summarized"
    day = rec.alerts[-1].message
    assert "Stopped: the next 9:45 am run sells anything held" in day
    assert "Shadow" not in day  # the scoreboard is stopped too

    sent = len(rec.alerts)
    nxt = clock.next_session(SESSION)
    rt.now = lambda: at(nxt, "09:00")
    assert bigpaper.picks(rt).status == "picked"
    rt.now = lambda: at(nxt, "09:46")
    assert bigpaper.execute(rt).status == "executed"
    assert len(rec.alerts) == sent and fb.orders_for_session(nxt) == []


def test_the_stopped_scoreboard_scores_and_grades_nothing(tmp_path: Path) -> None:
    rt = bigpaper.BigRuntime(
        base=big_settings(tmp_path),
        secrets=Secrets.from_env({**SMALL, **BIG, "HF_TOKEN": "hf-test"}),
        notifier=alerts.RecordingNotifier(),
        hub=object(),  # touched only if the scoreboard ran
    )
    assert bigpaper.shadow_lock(rt).status == "stopped"
    assert bigpaper.scoreboard(rt).status == "stopped"


def test_the_dashboard_says_the_big_account_is_stopped() -> None:
    data = {"big_paper": {"strategy": "trial14", "stopped": True}}
    assert dashboard.big_paper_title(data) == "Big paper account (stopped)"
    assert "Stopped by the owner" in dashboard.render_big_paper(data)
