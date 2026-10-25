"""The new day: a quiet DECIDE at 5:15 pm, the 9:00 am sweep, EXECUTE at 9:45 am."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stoa import alerts, clock, jobs, sweep
from stoa.broker.sim import SimBroker
from stoa.config import Secrets, Settings, SweepConfig
from stoa.data.base import Bars
from stoa.data.news import NEWS_COLUMNS
from stoa.models import PlannedOrder, Side
from stoa.portfolio import Portfolio
from stoa.state import Plan, StateStore
from stoa.universe import Universe
from tests.conftest import SMALL_WATCHLIST, make_settings
from tests.test_jobs import FixedProvider, prices

D = Decimal
DECIDE_DAY = date(2021, 10, 12)
EXEC_DAY = date(2021, 10, 13)
DECIDE_AT = datetime(2021, 10, 12, 21, 15, tzinfo=UTC)  # 5:15 pm New York
SWEEP_AT = datetime(2021, 10, 13, 13, 0, tzinfo=UTC)  # 9:00 am
EXEC_AT = datetime(2021, 10, 13, 13, 45, tzinfo=UTC)  # 9:45 am
ENV = {
    "ALPACA_PAPER_KEY_ID": "pk",
    "ALPACA_PAPER_SECRET": "ps",
    "STOA_NTFY_APPROVAL_TOPIC": "approval-topic-xyz",
}
CFG = SweepConfig()


# ---- fakes ----------------------------------------------------------------------------


class FakeReader:
    """Tone by keyword: 'plunge' reads negative, 'soars' positive, anything else neutral."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def score(self, texts: Sequence[str]) -> np.ndarray:
        self.seen.extend(texts)
        rows = []
        for t in texts:
            if "plunge" in t:
                rows.append([0.02, 0.95, 0.03])
            elif "soars" in t:
                rows.append([0.96, 0.01, 0.03])
            else:
                rows.append([0.1, 0.1, 0.8])
        return np.array(rows, dtype=np.float32)


class BrokenReader:
    def score(self, texts: Sequence[str]) -> np.ndarray:
        raise RuntimeError("model download failed")


class FakeNews:
    def __init__(self, items: list[tuple[str, str, datetime]]) -> None:
        self.items = items
        self.calls: list[tuple[list[str], date, date]] = []

    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        self.calls.append((symbols, start, end))
        rows = [
            {
                "id": f"alpaca:{i}",
                "symbol": sym,
                "known_at": at,
                "created_at": at,
                "headline": head,
                "summary": "",
                "content": "",
                "source": "benzinga",
                "kind": "news",
                "n_symbols": 2 if "roundup" in head else 1,
            }
            for i, (sym, head, at) in enumerate(self.items)
            if sym in symbols
        ]
        return pd.DataFrame(rows, columns=list(NEWS_COLUMNS))


class Topic:
    """The approval topic: whatever taps the test puts on it."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def tap(self, verb: str, plan: Plan, symbol: str, nonce: str, at: datetime) -> None:
        self.messages.append(
            {
                "event": "message",
                "time": int(at.timestamp()),
                "message": f"{verb} {plan.plan_id} {symbol} {nonce}",
            }
        )

    def poll(self, since: datetime) -> list[dict[str, Any]]:
        return list(self.messages)


# ---- harness ----------------------------------------------------------------------------


def settings_for(tmp: Path, **extra: str) -> Settings:
    return make_settings(
        watchlist=",".join(SMALL_WATCHLIST),
        mode="paper",
        state__dir=str(tmp / "state"),
        scout__kind="rule",
        strategy__kind="rule",
        crosscheck__enabled="false",
        data__provider="csv",
        **extra,
    )


def sim(s: Settings, bars: Bars) -> SimBroker:
    b = SimBroker(
        Portfolio(cash=D(100)), s.costs, Universe.assume_all_fractionable(s.watchlist, "SPY")
    )
    b.begin_session(DECIDE_DAY, prices(bars, DECIDE_DAY, "close"))
    return b


def rt_for(
    s: Settings, bars: Bars, broker: SimBroker, now: datetime, **kw: Any
) -> tuple[jobs.Runtime, alerts.RecordingNotifier]:
    rec = alerts.RecordingNotifier()
    rt = jobs.Runtime(
        settings=s,
        secrets=Secrets.from_env(ENV),
        notifier=rec,
        now=lambda: now,
        provider=FixedProvider(bars),
        broker=broker,
        use_external=False,
        sleep=lambda _s: None,
        **kw,
    )
    return rt, rec


def decided(tmp: Path, bars: Bars, **extra: str) -> tuple[Settings, SimBroker, Plan]:
    s = settings_for(tmp, **extra)
    broker = sim(s, bars)
    rt, rec = rt_for(s, bars, broker, DECIDE_AT)
    assert jobs.decide(rt).status == "decided"
    assert not [a for a in rec.alerts if a.title.startswith(("BUY ", "SELL "))]
    plan = StateStore(s.state.dir).load().pending_plan
    assert plan is not None and plan.orders
    return s, broker, plan


def overnight(hours: float) -> datetime:
    return clock.session(DECIDE_DAY).close + timedelta(hours=hours)


# ---- pure rules -------------------------------------------------------------------------


def buy(sym: str) -> PlannedOrder:
    return PlannedOrder(symbol=sym, side=Side.BUY, notional=D(20), ref_price=D(100), reason="entry")


def plan_of(buys: list[str], reserves: list[str] = ()) -> Plan:  # type: ignore[assignment]
    return Plan(
        plan_id="p",
        strategy="agents",
        mode="paper",
        decided_session=DECIDE_DAY,
        execute_session=EXEC_DAY,
        created_at=DECIDE_AT,
        orders=tuple(buy(b) for b in buys),
        equity_at_decide=D(100),
        reserves=tuple(buy(r) for r in reserves),
    )


def tone(sym: str, mean: float, pos: float = 0.0, neg: float = 0.0) -> sweep.Tone:
    return sweep.Tone(sym, 2, mean, pos, neg, f"{sym} good", f"{sym} bad")


def test_bad_news_drops_a_planned_buy_and_good_news_fills_the_slot() -> None:
    plan = plan_of(["AAPL", "MSFT"], ["NVDA", "JPM"])
    tones = {
        "AAPL": tone("AAPL", -0.6, neg=0.93),
        "MSFT": tone("MSFT", 0.1, pos=0.5),
        "NVDA": tone("NVDA", 0.2, pos=0.7),  # positive, but not strong enough
        "JPM": tone("JPM", 0.7, pos=0.95),
    }
    out = sweep.apply(plan, tones, CFG, max_positions=2, held=set())
    assert [o.symbol for o, _ in out.dropped] == ["AAPL"]
    assert [o.symbol for o, _ in out.added] == ["JPM"]
    assert [o.symbol for o in out.plan.orders] == ["MSFT", "JPM"]
    added = out.plan.orders[-1]
    assert added.reason == "news_add" and "Added on overnight news: JPM good" in added.thesis


def test_mild_news_changes_nothing_and_a_full_book_takes_no_adds() -> None:
    plan = plan_of(["AAPL"], ["JPM"])
    mild = {"AAPL": tone("AAPL", -0.2, neg=0.9), "JPM": tone("JPM", 0.8, pos=0.97)}
    out = sweep.apply(plan, mild, CFG, max_positions=2, held={"KO"})
    assert out.dropped == [] and out.added == []  # AAPL's mean tone is not bad enough
    assert [o.symbol for o in out.plan.orders] == ["AAPL"]


def test_adds_stop_at_max_adds_and_skip_held_names() -> None:
    plan = plan_of([], ["KO", "JPM", "XOM"])
    strong = {s: tone(s, 0.8, pos=0.97) for s in ("KO", "JPM", "XOM")}
    out = sweep.apply(plan, strong, CFG, max_positions=5, held={"KO"})
    assert [o.symbol for o, _ in out.added] == ["JPM"]


def test_read_tones_skips_articles_that_tag_many_tickers() -> None:
    rows = FakeNews([("AAPL", "Market roundup: stocks plunge", overnight(1))]).fetch(
        ["AAPL"], DECIDE_DAY, EXEC_DAY
    )
    rows["n_symbols"] = 12
    reader = FakeReader()
    tones = sweep.read_tones(rows, ["AAPL"], reader, CFG)
    assert tones["AAPL"].articles == 0 and reader.seen == []


def test_read_tones_averages_each_symbols_articles() -> None:
    news = FakeNews(
        [
            ("AAPL", "AAPL shares plunge on recall", overnight(1)),
            ("AAPL", "AAPL plunge deepens", overnight(2)),
            ("MSFT", "MSFT soars after upgrade", overnight(3)),
        ]
    ).fetch(["AAPL", "MSFT", "KO"], DECIDE_DAY, EXEC_DAY)
    tones = sweep.read_tones(news, ["AAPL", "MSFT", "KO"], FakeReader(), CFG)
    assert tones["AAPL"].articles == 2 and tones["AAPL"].mean_net == pytest.approx(-0.93)
    assert tones["MSFT"].best == "MSFT soars after upgrade"
    assert tones["KO"].articles == 0 and tones["KO"].line() == "No overnight news."


# ---- the day, end to end ------------------------------------------------------------------


def test_sweep_sends_picks_with_buttons_and_execute_places_only_approved(
    tmp_path: Path, bars: Bars
) -> None:
    s, broker, plan = decided(tmp_path, bars)
    rt, rec = rt_for(s, bars, broker, SWEEP_AT, news_client=FakeNews([]))
    assert jobs.sweep_job(rt).status == "swept"
    state = StateStore(s.state.dir).load()
    assert state.approval_nonce and EXEC_DAY in state.swept_sessions
    picks = [a for a in rec.alerts if a.title.startswith("BUY ")]
    assert len(picks) == len(plan.orders)
    assert all(len(a.actions) == 2 and "by 9:40 am" in a.message for a in picks)
    assert all("at 9:45 am" in a.message for a in picks)

    topic = Topic()
    first, second = plan.orders[0].symbol, plan.orders[1].symbol
    topic.tap("approve", plan, first, state.approval_nonce, SWEEP_AT + timedelta(minutes=5))
    # A tap after 9:40 does not count.
    topic.tap("approve", plan, second, state.approval_nonce, SWEEP_AT + timedelta(minutes=41))
    broker.begin_session(EXEC_DAY, prices(bars, EXEC_DAY, "open"))
    rt2, rec2 = rt_for(s, bars, broker, EXEC_AT, approvals=topic)
    assert jobs.execute(rt2).status == "executed"
    assert [f.symbol for f in broker.fills] == [first]
    assert any(a.title == f"Filled buy {first}" for a in rec2.alerts)
    late = [a for a in rec2.alerts if a.title == f"Skipped buy {second}"]
    assert late and "late" in late[0].message


def test_sweep_drops_a_pick_on_bad_overnight_news(tmp_path: Path, bars: Bars) -> None:
    s, broker, plan = decided(tmp_path, bars)
    victim = plan.orders[0].symbol
    news = FakeNews(
        [
            (victim, f"{victim} shares plunge after guidance cut", overnight(3)),
            # Published before the close it was planned on: not overnight news.
            (victim, f"{victim} soars", overnight(-1)),
            # Published after 9:00 am: too late for this morning.
            (victim, f"{victim} soars again", overnight(17.5)),
        ]
    )
    reader = FakeReader()
    rt, rec = rt_for(s, bars, broker, SWEEP_AT, news_client=news, reader=reader)
    jobs.sweep_job(rt)
    assert reader.seen == [f"{victim} shares plunge after guidance cut"]
    assert any(a.title == f"Dropped BUY {victim}" for a in rec.alerts)
    after = StateStore(s.state.dir).load().pending_plan
    assert after is not None and victim not in {o.symbol for o in after.orders}
    assert not [a for a in rec.alerts if a.title == f"BUY {victim}"]


def test_a_broken_reader_leaves_the_picks_unchanged_and_says_so(tmp_path: Path, bars: Bars) -> None:
    s, broker, plan = decided(tmp_path, bars)
    sym = plan.orders[0].symbol
    news = FakeNews([(sym, f"{sym} shares plunge", overnight(2))])
    rt, rec = rt_for(s, bars, broker, SWEEP_AT, news_client=news, reader=BrokenReader())
    assert jobs.sweep_job(rt).status == "swept"
    after = StateStore(s.state.dir).load().pending_plan
    assert after is not None and after.orders == plan.orders
    note = [a for a in rec.alerts if a.title == "Morning sweep"]
    assert note and "could not run (RuntimeError)" in note[0].message


def test_sweep_runs_once_and_only_in_its_window(tmp_path: Path, bars: Bars) -> None:
    s, broker, _ = decided(tmp_path, bars)
    early, rec = rt_for(s, bars, broker, datetime(2021, 10, 13, 12, 30, tzinfo=UTC))
    assert jobs.sweep_job(early).status == "outside_window" and rec.alerts == []
    rt, _ = rt_for(s, bars, broker, SWEEP_AT, news_client=FakeNews([]))
    assert jobs.sweep_job(rt).status == "swept"
    again, rec2 = rt_for(s, bars, broker, SWEEP_AT, news_client=FakeNews([]))
    assert jobs.sweep_job(again).status == "already_done" and rec2.alerts == []


def test_sweep_catches_up_when_the_evening_run_was_missed(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path)
    broker = sim(s, bars)
    rt, rec = rt_for(s, bars, broker, SWEEP_AT, news_client=FakeNews([]))
    assert jobs.sweep_job(rt).status == "swept"
    state = StateStore(s.state.dir).load()
    assert DECIDE_DAY in state.decided_sessions
    plan = state.pending_plan
    assert plan is not None and plan.execute_session == EXEC_DAY and plan.orders
    titles = [a.title for a in rec.alerts]
    assert f"Stoa {DECIDE_DAY} (paper)" not in titles  # no evening summary in the morning
    assert len([t for t in titles if t.startswith("BUY ")]) == len(plan.orders)
    assert any("evening run did not happen" in a.message for a in rec.alerts)


def test_execute_without_a_sweep_sends_no_buys(tmp_path: Path, bars: Bars) -> None:
    s, broker, _ = decided(tmp_path, bars)
    broker.begin_session(EXEC_DAY, prices(bars, EXEC_DAY, "open"))
    rt, rec = rt_for(s, bars, broker, EXEC_AT, approvals=Topic())
    assert jobs.execute(rt).status == "executed"
    assert broker.fills == []
    assert any(a.title.startswith("Skipped buy") for a in rec.alerts)


def test_day_summary_shows_today_per_position_and_spy(tmp_path: Path, bars: Bars) -> None:
    s, broker, _ = decided(tmp_path, bars, approval__buys="mode")
    broker.begin_session(EXEC_DAY, prices(bars, EXEC_DAY, "open"))
    rt, _ = rt_for(s, bars, broker, EXEC_AT)
    jobs.execute(rt)
    bought = {f.symbol for f in broker.fills}
    assert bought
    broker.begin_session(EXEC_DAY, prices(bars, EXEC_DAY, "close"))
    rt2, rec = rt_for(s, bars, broker, datetime(2021, 10, 13, 21, 15, tzinfo=UTC))
    assert jobs.decide(rt2).status == "decided"
    summary = rec.alerts[-1]
    assert summary.title == f"Stoa {EXEC_DAY} (paper)"
    assert "today, SPY " in summary.message.splitlines()[0]
    today = next(line for line in summary.message.splitlines() if line.startswith("Today: "))
    for sym in bought:
        assert f"{sym} " in today and "bought today at $" in today
    assert not [a for a in rec.alerts if a.title.startswith(("BUY ", "SELL "))]


# ---- the public trade log -------------------------------------------------------------------


def test_trade_log_lists_picks_fills_and_skips(tmp_path: Path, bars: Bars) -> None:
    from stoa import dashboard
    from stoa.journal import read_journal

    s, broker, plan = decided(tmp_path, bars)
    rt, _ = rt_for(s, bars, broker, SWEEP_AT, news_client=FakeNews([]))
    jobs.sweep_job(rt)
    state = StateStore(s.state.dir).load()
    assert state.approval_nonce is not None
    topic = Topic()
    first = plan.orders[0].symbol
    topic.tap("approve", plan, first, state.approval_nonce, SWEEP_AT + timedelta(minutes=5))
    broker.begin_session(EXEC_DAY, prices(bars, EXEC_DAY, "open"))
    rt2, _ = rt_for(s, bars, broker, EXEC_AT, approvals=topic)
    jobs.execute(rt2)

    rows = dashboard.trade_log(
        read_journal(s.state.dir / "journal"), StateStore(s.state.dir).load()
    )
    kinds = {(r["kind"], r["symbol"]) for r in rows}
    assert {("pick", o.symbol) for o in plan.orders} <= kinds
    assert ("filled", first) in kinds
    skipped = [r for r in rows if r["kind"] == "skipped"]
    assert len(skipped) == len(plan.orders) - 1
    assert all("not approved" in r["reason"] for r in skipped)
    filled = next(r for r in rows if r["kind"] == "filled")
    assert filled["side"] == "buy" and " sh at $" in filled["size"]
    assert all(r["date"] == EXEC_DAY.isoformat() for r in rows)


def test_trade_log_empty_state_renders(tmp_path: Path) -> None:
    from stoa import dashboard

    html_text = dashboard.render_trades({"trades": []})
    assert "No trades yet" in html_text
