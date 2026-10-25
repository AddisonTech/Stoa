"""Approvals, healthcheck pings, the price cross-check, honest paper costs, the drift
fallback, the quit rule, exam gating, and a scan for secrets in a full simulated run."""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

from stoa import alerts, approvals, jobs, redact, tracking
from stoa.broker.sim import SimBroker
from stoa.config import CostConfig, CrossCheckConfig, ExamConfig, FallbackConfig, Secrets
from stoa.crosscheck import check_prices, compare
from stoa.data.base import Bars
from stoa.data.news import NEWS_COLUMNS
from stoa.features import cell
from stoa.healthcheck import HealthcheckPinger, safe_ping
from stoa.models import PlannedOrder, Side, dec
from stoa.portfolio import Portfolio
from stoa.state import StateStore
from stoa.universe import Universe
from tests.conftest import SMALL_WATCHLIST, make_settings

D = Decimal
PLAN = "abc123"
NONCE = "n0nce-XYZ"
CREATED = datetime(2021, 10, 12, 22, 20, tzinfo=UTC)
DEADLINE = datetime(2021, 10, 13, 15, 45, tzinfo=UTC)


def order(sym: str, side: Side = Side.BUY) -> PlannedOrder:
    return PlannedOrder(
        symbol=sym,
        side=side,
        notional=D(10) if side is Side.BUY else None,
        qty=D(1) if side is Side.SELL else None,
        ref_price=D(100),
        reason="entry",
    )


def msg(text: str, at: datetime, event: str = "message") -> dict[str, Any]:
    return {"id": "x", "event": event, "time": int(at.timestamp()), "message": text}


# ---- approvals ------------------------------------------------------------------------


def test_approval_actions_post_to_the_secret_topic_without_any_token() -> None:
    acts = approvals.actions("https://ntfy.sh/", "approve-topic-secret", PLAN, "NVDA", NONCE)
    assert [a["label"] for a in acts] == ["Approve", "Skip"]
    assert all(a["action"] == "http" and a["method"] == "POST" for a in acts)
    assert acts[0]["url"] == "https://ntfy.sh/approve-topic-secret"
    assert acts[0]["body"] == f"approve {PLAN} NVDA {NONCE}"
    assert all("headers" not in a for a in acts)  # never a token in a notification
    n = alerts.NtfyNotifier("https://ntfy.sh", "alerts-topic")
    body = n.payload(alerts.with_actions(alerts.Alert("BUY NVDA", "x"), acts, "Tap one."))
    assert body["actions"] == acts and "Tap one." in str(body["message"])


@pytest.mark.parametrize(
    ("messages", "expect"),
    [
        ([msg(f"approve {PLAN} NVDA {NONCE}", CREATED + timedelta(hours=1))], "approved"),
        ([msg(f"skip {PLAN} NVDA {NONCE}", CREATED + timedelta(hours=1))], "skipped"),
        ([msg(f"approve {PLAN} NVDA wrong", CREATED + timedelta(hours=1))], "wrong_nonce"),
        ([msg(f"approve {PLAN} NVDA {NONCE}", DEADLINE + timedelta(minutes=1))], "late"),
        ([], "no_answer"),
        ([msg(f"approve other NVDA {NONCE}", CREATED + timedelta(hours=1))], "no_answer"),
        (
            [
                msg(f"approve {PLAN} NVDA {NONCE}", CREATED + timedelta(hours=1)),
                msg(f"skip {PLAN} NVDA {NONCE}", CREATED + timedelta(hours=2)),
            ],
            "skipped",
        ),
        ([msg(f"approve {PLAN} NVDA {NONCE}", CREATED - timedelta(days=1))], "no_answer"),
    ],
)
def test_approval_verdicts(messages: list[dict[str, Any]], expect: str) -> None:
    v = approvals.decide(
        [order("NVDA")],
        messages,
        plan_id=PLAN,
        nonce=NONCE,
        created_at=CREATED,
        deadline=DEADLINE,
    )
    assert v["NVDA"].reason == expect
    assert v["NVDA"].approved == (expect == "approved")


def test_approval_required_rules() -> None:
    from stoa.config import ApprovalConfig

    auto = ApprovalConfig()
    assert not approvals.required(auto, "paper", None, date(2026, 1, 5))
    assert approvals.required(auto, "live", None, date(2026, 1, 5))
    assert approvals.required(auto, "live", date(2026, 1, 5), date(2026, 2, 3))
    assert not approvals.required(auto, "live", date(2026, 1, 5), date(2026, 2, 4))
    assert approvals.required(ApprovalConfig(mode="always"), "paper", None, date(2026, 1, 5))
    assert not approvals.required(ApprovalConfig(mode="never"), "live", None, date(2026, 1, 5))


def test_approval_client_polls_with_since_and_parses_json_lines() -> None:
    seen: list[httpx.Request] = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        lines = [
            json.dumps({"event": "open", "time": 1}),
            json.dumps(msg(f"approve {PLAN} NVDA {NONCE}", CREATED + timedelta(hours=1))),
        ]
        return httpx.Response(200, text="\n".join(lines))

    c = approvals.ApprovalClient(
        "https://ntfy.sh", "appr-topic", client=httpx.Client(transport=httpx.MockTransport(h))
    )
    out = c.poll(CREATED)
    assert len(out) == 2
    assert seen[0].url.path == "/appr-topic/json"
    assert seen[0].url.params["poll"] == "1"
    assert seen[0].url.params["since"] == str(int(CREATED.timestamp()))
    assert "appr-topic" not in repr(c)


def test_wait_and_decide_unavailable_and_polling() -> None:
    v = approvals.wait_and_decide(
        None,
        [order("NVDA")],
        plan_id=PLAN,
        nonce=NONCE,
        created_at=CREATED,
        deadline=DEADLINE,
        wait_minutes=1,
        poll_seconds=1,
        now=lambda: CREATED,
        sleep=lambda _s: None,
    )
    assert v["NVDA"].reason == "unavailable"
    calls = {"n": 0}
    t = [CREATED + timedelta(hours=15)]

    class Slow:
        def poll(self, since: datetime) -> list[dict[str, Any]]:
            calls["n"] += 1
            if calls["n"] < 3:
                return []
            return [msg(f"approve {PLAN} NVDA {NONCE}", t[0])]

    def sleep(s: float) -> None:
        t[0] += timedelta(seconds=s)

    v2 = approvals.wait_and_decide(
        Slow(),
        [order("NVDA")],
        plan_id=PLAN,
        nonce=NONCE,  # type: ignore[arg-type]
        created_at=CREATED,
        deadline=DEADLINE,
        wait_minutes=5,
        poll_seconds=30,
        now=lambda: t[0],
        sleep=sleep,
    )
    assert v2["NVDA"].approved and calls["n"] == 3


# ---- healthcheck ------------------------------------------------------------------------


def test_healthcheck_pings_start_success_fail_and_never_logs_the_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    url = "https://hc-ping.com/1f2e3d4c-aaaa-bbbb-cccc-0123456789ab"
    seen: list[str] = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        return httpx.Response(200 if "fail" not in str(req.url) else 500, text="OK")

    p = HealthcheckPinger(
        url, client=httpx.Client(transport=httpx.MockTransport(h)), sleep=lambda _s: None
    )
    redact.clear()
    Secrets.from_env({"STOA_HEALTHCHECK_URL": url})
    caplog.set_level(logging.DEBUG)
    caplog.handler.addFilter(redact.RedactingFilter())
    redact.install()
    assert p.ping("start") and p.ping("success")
    assert not p.ping("fail")  # server error: retried, then gives up quietly
    assert seen[:2] == [f"{url}/start", url]
    assert all(s == f"{url}/fail" for s in seen[2:]) and len(seen) == 2 + 3
    assert url not in caplog.text and "1f2e3d4c" not in caplog.text
    assert "1f2e3d4c" not in repr(p)
    redact.clear()
    with pytest.raises(ValueError):
        HealthcheckPinger("http://insecure.example/ping")


def test_safe_ping_never_raises() -> None:
    class Boom:
        def ping(self, signal: str, body: str = "") -> bool:
            raise RuntimeError("down")

    assert safe_ping(Boom(), "start") is False  # type: ignore[arg-type]


# ---- price cross-check ----------------------------------------------------------------------


class FixedSource:
    name = "fixed"

    def __init__(self, prices: dict[str, Decimal | None]) -> None:
        self.prices = prices

    def price(self, symbol: str) -> Decimal | None:
        return self.prices.get(symbol)


def test_crosscheck_agree_disagree_unavailable() -> None:
    cfg = CrossCheckConfig()
    assert compare(D(100), D("100.9"), 0.01) == "agree"
    assert compare(D(100), D("102"), 0.01) == "disagree"
    assert compare(D(100), None, 0.01) == "unavailable"
    src = FixedSource({"A": D("100.5"), "B": D(110), "C": None})
    out = check_prices(["A", "B", "C"], {"A": D(100), "B": D(100), "C": D(100)}, src, cfg)
    assert out["A"].ok and not out["B"].ok and not out["C"].ok
    assert "apart" in out["B"].detail and "unavailable" in out["C"].detail
    lenient = CrossCheckConfig(on_unavailable="proceed")
    assert check_prices(["C"], {"C": D(100)}, src, lenient)["C"].ok
    assert not check_prices(["A"], {"A": D(100)}, None, cfg)["A"].ok


# ---- honest paper -----------------------------------------------------------------------------


def test_fill_cost_matches_the_backtest_model() -> None:
    costs = CostConfig()
    buy = tracking.fill_cost(Side.BUY, D(1), D(100), D(0), costs)
    assert buy == D("0.05")  # 5 bps of $100
    sell = tracking.fill_cost(Side.SELL, D(1), D(100), D(0), costs)
    assert sell == D("0.05") + D("0.01")  # slippage plus the $0.01 minimum fee
    already = tracking.fill_cost(Side.SELL, D(1), D(100), D("0.01"), costs)
    assert already == D("0.05")


def days(
    values: list[float], spy: list[float], mode: str = "paper", trades_every: int = 2
) -> list[tracking.DailyRecord]:
    start = date(2025, 1, 2)
    out = []
    for i, (v, sp) in enumerate(zip(values, spy, strict=True)):
        out.append(
            tracking.DailyRecord(
                session=start + timedelta(days=i),
                mode=mode,
                broker_equity=dec(v * 1.01),
                adjusted_equity=dec(v),
                benchmark_close=dec(sp),
                trades_closed=1 if i % trades_every == 0 and i else 0,
            )
        )
    return out


def walk(n: int, drift: float, vol: float, seed: int) -> list[float]:
    r = np.random.default_rng(seed).normal(drift, vol, n)
    return [100.0, *list(100 * np.cumprod(1 + r))]


def test_grade_and_paper_exam_on_cost_adjusted_record() -> None:
    good = walk(60, 0.002, 0.004, 1)
    spy = walk(60, 0.0005, 0.012, 2)
    t = tracking.Track(days=days(good, spy))
    g, ok = tracking.paper_exam(t, ExamConfig())
    assert g.sessions == 60 and g.trades >= 20 and ok
    assert g.raw_total_return != g.total_return
    short = tracking.Track(days=days(good[:30], spy[:30]))
    assert not tracking.paper_exam(short, ExamConfig())[1]
    few = tracking.Track(days=days(good, spy, trades_every=10))
    assert not tracking.paper_exam(few, ExamConfig())[1]
    bad = tracking.Track(days=days(walk(60, -0.002, 0.02, 3), spy))
    assert not tracking.paper_exam(bad, ExamConfig())[1]


def test_flows_are_not_returns() -> None:
    recs = days([100.0, 101.0, 51.0, 52.0], [100.0, 100.0, 100.0, 100.0])
    recs[2] = recs[2].model_copy(update={"flow": D(-50)})
    g = tracking.grade(recs)
    assert g.total_return == pytest.approx(1.01 * (51 + 50) / 101 * 52 / 51 - 1)
    assert g.max_drawdown < 0.01


# ---- drift fallback and the quit rule ------------------------------------------------------------


def test_fallback_triggers_on_sharpe_band_drawdown_and_quit_rule() -> None:
    cfg = FallbackConfig(sharpe_window=40, min_history_sessions=40, quit_after_sessions=126)
    spy = walk(130, 0.0004, 0.01, 9)
    exp = tracking.Expectation(sharpe_mean=1.5, sharpe_p05=0.2, drawdown_p95=0.15, window=40)
    poor = tracking.Track(days=days(walk(60, -0.002, 0.01, 4), spy[:61]))
    reason = tracking.check_fallback(poor, cfg, exp)
    assert reason is not None and ("Sharpe" in reason or "drawdown" in reason)
    crash_vals = [100.0 + 0.1 * i for i in range(30)] + [102.9 - 1.0 * i for i in range(1, 32)]
    crash = tracking.Track(days=days(crash_vals, spy[:61]))
    loose = tracking.Expectation(sharpe_mean=1, sharpe_p05=-99, drawdown_p95=0.15, window=40)
    r2 = tracking.check_fallback(crash, cfg, loose)
    assert r2 is not None and "drawdown" in r2
    flat = tracking.Track(days=days([100.0 + 0.001 * i for i in range(131)], spy))
    r3 = tracking.check_fallback(flat, cfg, None)
    assert r3 is not None and r3.startswith("quit rule")
    fine = tracking.Track(days=days(walk(60, 0.002, 0.004, 5), spy[:61]))
    assert tracking.check_fallback(fine, cfg, exp) is None


def test_quit_rule_counts_paper_and_live_together() -> None:
    cfg = FallbackConfig(quit_after_sessions=126, min_history_sessions=63)
    spy = walk(130, 0.001, 0.01, 11)
    paper = days([100 + 0.01 * i for i in range(80)], spy[:80])
    live = tracking.Track(days=days([10 + 0.001 * i for i in range(50)], spy[80:130], "live"))
    assert tracking.check_fallback(live, cfg, None) is None
    r = tracking.check_fallback(live, cfg, None, prior=paper)
    assert r is not None and "paper and live" in r


def test_fallback_activate_and_reset() -> None:
    t = tracking.Track()
    tracking.activate(t, "test", date(2026, 3, 2))
    assert t.fallback_active and tracking.check_fallback(t, FallbackConfig()) is None
    tracking.reset(t)
    assert not t.fallback_active and t.fallback_reason == ""


def test_expectation_bands_from_oos_returns() -> None:
    r = list(np.random.default_rng(1).normal(0.0005, 0.01, 500))
    e = tracking.expectation_from_returns(r, 63)
    assert e.sharpe_p05 < e.sharpe_mean and 0 < e.drawdown_p95 < 1
    with pytest.raises(ValueError):
        tracking.expectation_from_returns(r[:30], 63)


# ---- jobs integration: DECIDE and EXECUTE with every new guard ---------------------------------

DECIDE_AT = datetime(2021, 10, 12, 22, 20, tzinfo=UTC)
EXEC_AT = datetime(2021, 10, 13, 13, 50, tzinfo=UTC)
SWEEP_AT = datetime(2021, 10, 13, 13, 0, tzinfo=UTC)  # 9:00 am New York
PAPER = {"ALPACA_PAPER_KEY_ID": "pk-paper-key-1", "ALPACA_PAPER_SECRET": "ps-paper-secret-1"}


class FixedProvider:
    name = "fixed"

    def __init__(self, bars: Bars) -> None:
        self.bars = bars

    def fetch(self, symbols: list[str], start: date, end: date) -> Bars:
        return {s: self.bars[s].loc[pd.Timestamp(start) : pd.Timestamp(end)] for s in symbols}


class RecordingPinger:
    def __init__(self) -> None:
        self.signals: list[str] = []

    def ping(self, signal: str, body: str = "") -> bool:
        self.signals.append(signal)
        return True


def prices(bars: Bars, day: date, col: str) -> dict[str, Decimal]:
    return {s: dec(cell(df, pd.Timestamp(day), col)) for s, df in bars.items()}


def setup(tmp: Path, bars: Bars, **extra: str) -> tuple[Any, SimBroker]:
    extra.setdefault("approval__buys", "mode")
    s = make_settings(
        watchlist=",".join(SMALL_WATCHLIST),
        state__dir=str(tmp / "state"),
        scout__kind="rule",
        strategy__kind="rule",
        data__provider="csv",
        **extra,
    )
    broker = SimBroker(
        Portfolio(cash=D(100)), s.costs, Universe.assume_all_fractionable(s.watchlist, "SPY")
    )
    broker.begin_session(DECIDE_AT.date(), prices(bars, DECIDE_AT.date(), "close"))
    return s, broker


def rt_for(
    s: Any, bars: Bars, broker: SimBroker, now: datetime, env: dict[str, str], **kw: Any
) -> tuple[jobs.Runtime, alerts.RecordingNotifier]:
    rec = alerts.RecordingNotifier()
    rt = jobs.Runtime(
        settings=s,
        secrets=Secrets.from_env(env),
        notifier=rec,
        now=lambda: now,
        provider=FixedProvider(bars),
        broker=broker,
        use_external=False,
        sleep=lambda _s: None,
        **kw,
    )
    return rt, rec


def test_decide_pings_start_then_success_and_fail_on_error(tmp_path: Path, bars: Bars) -> None:
    s, broker = setup(tmp_path, bars, crosscheck__enabled="false")
    pinger = RecordingPinger()
    rt, _ = rt_for(s, bars, broker, DECIDE_AT, PAPER, pinger=pinger)
    assert jobs.decide(rt).status == "decided"
    assert pinger.signals == ["start", "success"]
    rt2, _ = rt_for(s, bars, broker, DECIDE_AT, PAPER, pinger=pinger)
    assert jobs.decide(rt2).status == "already_done"
    assert pinger.signals == ["start", "success"]  # a second slot does not ping again
    s3, b3 = setup(tmp_path / "b", bars)
    p3 = RecordingPinger()
    rt3, _ = rt_for(s3, bars, b3, DECIDE_AT, {}, pinger=p3)  # no paper keys: refused
    assert jobs.decide(rt3).code == jobs.EXIT_REFUSED and p3.signals == ["fail"]


class NoNews:
    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame(columns=list(NEWS_COLUMNS))


def approval_flow(tmp: Path, bars: Bars, reply: str | None) -> tuple[SimBroker, Any]:
    # The production default: every buy needs a tap on the 9:00 am pick alert.
    s, broker = setup(tmp, bars, approval__buys="always", crosscheck__enabled="false")
    env = {**PAPER, "STOA_NTFY_APPROVAL_TOPIC": "approval-topic-xyz"}
    rt, rec = rt_for(s, bars, broker, DECIDE_AT, env)
    jobs.decide(rt)
    assert not [a for a in rec.alerts if a.title.startswith("BUY ")]
    rt_s, rec_s = rt_for(s, bars, broker, SWEEP_AT, env, news_client=NoNews())
    assert jobs.sweep_job(rt_s).status == "swept"
    state = StateStore(s.state.dir).load()
    plan = state.pending_plan
    assert plan is not None and plan.orders and state.approval_nonce
    buy_alerts = [a for a in rec_s.alerts if a.title.startswith("BUY ")]
    assert buy_alerts and all(len(a.actions) == 2 for a in buy_alerts)
    nonce = state.approval_nonce or ""

    class Topic:
        def poll(self, since: datetime) -> list[dict[str, Any]]:
            if reply is None:
                return []
            sym = plan.orders[0].symbol
            text = reply.format(plan=plan.plan_id, sym=sym, nonce=nonce)
            return [msg(text, SWEEP_AT + timedelta(minutes=10))]

    broker.begin_session(EXEC_AT.date(), prices(bars, EXEC_AT.date(), "open"))
    rt2, rec2 = rt_for(s, bars, broker, EXEC_AT, env, approvals=Topic())
    jobs.execute(rt2)
    return broker, (plan, rec2)


def test_execute_only_sends_approved_orders(tmp_path: Path, bars: Bars) -> None:
    broker, (plan, rec) = approval_flow(tmp_path, bars, "approve {plan} {sym} {nonce}")
    assert [f.symbol for f in broker.fills] == [plan.orders[0].symbol]
    skipped = [a for a in rec.alerts if a.title.startswith("Skipped buy")]
    assert len(skipped) == len(plan.orders) - 1
    assert all("Not approved (no answer)" in a.message for a in skipped)


def test_execute_skip_wrong_nonce_and_silence_send_nothing(tmp_path: Path, bars: Bars) -> None:
    for i, reply in enumerate(["skip {plan} {sym} {nonce}", "approve {plan} {sym} bad", None]):
        broker, _ = approval_flow(tmp_path / str(i), bars, reply)
        assert broker.fills == []


def test_execute_price_check_skips_disagreeing_buys(tmp_path: Path, bars: Bars) -> None:
    s, broker = setup(tmp_path, bars)
    rt, _ = rt_for(s, bars, broker, DECIDE_AT, PAPER)
    jobs.decide(rt)
    plan = StateStore(s.state.dir).load().pending_plan
    assert plan is not None and len(plan.orders) >= 2
    broker.begin_session(EXEC_AT.date(), prices(bars, EXEC_AT.date(), "open"))
    live = prices(bars, EXEC_AT.date(), "open")
    a, b = plan.orders[0].symbol, plan.orders[1].symbol
    src = FixedSource({a: live[a], b: live[b] * D("1.05")})  # b disagrees by 5%
    rt2, rec = rt_for(s, bars, broker, EXEC_AT, PAPER, price_source=src)
    jobs.execute(rt2)
    filled = {f.symbol for f in broker.fills}
    assert a in filled and b not in filled
    assert any(t.startswith(f"Skipped buy {b}: price check") for t in rec.titles())


def test_execute_charges_simulated_costs_to_the_paper_record(tmp_path: Path, bars: Bars) -> None:
    s, broker = setup(tmp_path, bars, crosscheck__enabled="false")
    rt, _ = rt_for(s, bars, broker, DECIDE_AT, PAPER)
    jobs.decide(rt)
    broker.begin_session(EXEC_AT.date(), prices(bars, EXEC_AT.date(), "open"))
    rt2, _ = rt_for(s, bars, broker, EXEC_AT, PAPER)
    jobs.execute(rt2)
    track = StateStore(s.state.dir).load().track
    gross = sum((f.gross for f in broker.fills), D(0))
    assert track.cost_drag == pytest.approx(gross * D("0.0005"), abs=D("0.000001"))
    nxt = datetime(2021, 10, 13, 22, 20, tzinfo=UTC)
    rt3, _ = rt_for(s, bars, broker, nxt, PAPER)
    jobs.decide(rt3)
    rec = StateStore(s.state.dir).load().track.days[-1]
    assert rec.adjusted_equity == rec.broker_equity - track.cost_drag


def test_live_blocked_until_the_exam_passes(tmp_path: Path, bars: Bars) -> None:
    live = {
        "ALPACA_LIVE_KEY_ID": "lk-live-key",
        "ALPACA_LIVE_SECRET": "ls-live-secret",
        "STOA_LIVE_CONFIRM": "I_UNDERSTAND",
    }
    s, broker = setup(tmp_path, bars, mode="live")
    rt, rec = rt_for(s, bars, broker, DECIDE_AT, live)
    res = jobs.decide(rt)
    assert res.code == jobs.EXIT_REFUSED
    assert "exam order" in rec.alerts[0].message and "walk-forward" in rec.alerts[0].message


def test_fallback_trips_in_decide_and_buys_only_spy(tmp_path: Path, bars: Bars) -> None:
    s, broker = setup(tmp_path, bars, crosscheck__enabled="false")
    store = StateStore(s.state.dir)
    state = store.load()
    tracking.activate(state.track, "test drift", DECIDE_AT.date())
    store.save(state)
    from stoa.earnings import EarningsCalendar, empty_calendar
    from stoa.model_strategy import ModelStrategy, PanelPredictor

    empty = pd.DataFrame(
        columns=["p"], index=pd.MultiIndex.from_tuples([], names=["date", "symbol"])
    )
    strat = ModelStrategy(
        s, PanelPredictor(empty, {}), EarningsCalendar(empty_calendar()), sectors={}
    )
    rt, _ = rt_for(s, bars, broker, DECIDE_AT, PAPER, strategy=strat)
    jobs.decide(rt)
    plan = store.load().pending_plan
    assert plan is not None
    assert [o.symbol for o in plan.orders if o.side is Side.BUY] in ([], ["SPY"])


def test_no_secret_appears_in_logs_or_alerts_of_a_full_run(
    tmp_path: Path, bars: Bars, caplog: pytest.LogCaptureFixture
) -> None:
    """Run DECIDE and EXECUTE with every secret configured and scan all output for them."""
    secrets_env = {
        "ALPACA_PAPER_KEY_ID": "PKSECRETKEYID0001",
        "ALPACA_PAPER_SECRET": "paper-secret-value-0001",
        "ANTHROPIC_API_KEY": "sk-ant-secret-0001",
        "STOA_NTFY_TOPIC": "stoa-topic-secret-0001",
        "STOA_NTFY_TOKEN": "tk_ntfy_secret_0001",
        "STOA_NTFY_APPROVAL_TOPIC": "stoa-approval-secret-0001",
        "STOA_HEALTHCHECK_URL": "https://hc-ping.com/secret-uuid-0001",
        "FINNHUB_API_KEY": "finnhub-secret-0001",
    }
    redact.clear()
    caplog.set_level(logging.DEBUG)
    redact.install()  # what `stoa` does at startup
    for h in logging.getLogger().handlers:
        h.addFilter(redact.RedactingFilter())
    caplog.handler.addFilter(redact.RedactingFilter())
    s, broker = setup(tmp_path, bars, approval__mode="always", crosscheck__enabled="false")
    rt, rec = rt_for(s, bars, broker, DECIDE_AT, secrets_env, pinger=RecordingPinger())
    jobs.decide(rt)
    broker.begin_session(EXEC_AT.date(), prices(bars, EXEC_AT.date(), "open"))

    class Topic:
        def poll(self, since: datetime) -> list[dict[str, Any]]:
            raise RuntimeError(f"cannot reach {secrets_env['STOA_NTFY_APPROVAL_TOPIC']}")

    rt2, rec2 = rt_for(s, bars, broker, EXEC_AT, secrets_env, approvals=Topic())
    jobs.execute(rt2)
    logging.getLogger("stoa.test").error("simulated leak %s", secrets_env["STOA_HEALTHCHECK_URL"])

    class Boom(FixedProvider):
        def fetch(self, symbols: list[str], start: date, end: date) -> Bars:
            raise RuntimeError(f"auth failed for {secrets_env['ALPACA_PAPER_SECRET']}")

    s3, b3 = setup(tmp_path / "boom", bars)
    rt3, rec3 = rt_for(s3, bars, b3, DECIDE_AT, secrets_env)
    rt3.provider = Boom(bars)
    jobs.decide(rt3)
    n = alerts.NtfyNotifier("https://ntfy.sh", secrets_env["STOA_NTFY_TOPIC"])
    sent = [
        json.dumps({k: v for k, v in n.payload(a).items() if k != "topic"})
        for r in (rec, rec2, rec3)
        for a in r.alerts
    ]
    journal = "".join(p.read_text() for p in (s.state.dir / "journal").glob("*.jsonl"))
    state_text = (s.state.dir / "state.json").read_text()
    haystack = caplog.text + "\n".join(sent) + journal + state_text
    for name, value in secrets_env.items():
        if name == "STOA_NTFY_APPROVAL_TOPIC":
            # The approval topic is meant to be inside the Approve/Skip button URLs,
            # which only ever go to the phone. It must not appear anywhere else.
            buttons_removed = "\n".join(
                json.dumps({k: v for k, v in n.payload(a).items() if k not in ("topic", "actions")})
                for r in (rec, rec2, rec3)
                for a in r.alerts
            )
            assert value not in caplog.text + buttons_removed + journal + state_text, name
            continue
        assert value not in haystack, name
    redact.clear()


def test_decide_uses_the_promoted_model(tmp_path: Path) -> None:
    """Train and promote on synthetic data, then a live-style DECIDE ranks with that model."""
    from stoa.data.synthetic import synthetic_universe
    from stoa.pipeline import load_inputs
    from stoa.retrain import train

    s = make_settings(
        watchlist=",".join(SMALL_WATCHLIST),
        state__dir=str(tmp_path / "state"),
        scout__kind="rule",
        model__universe="watchlist",
        model__min_train_sessions="300",
        model__test_sessions="126",
        model__candidates="[logistic]",
        model__use_quality="false",
        model__max_ece="0.2",
        demo__signal="2.0",
        crosscheck__enabled="false",
        earnings__block_when_unknown="false",
        data__provider="csv",
    )
    cutoff = date(2021, 6, 30)
    out = train(
        s,
        load_inputs(s, cutoff, synthetic=True),
        cutoff=cutoff,
        results_dir=tmp_path / "r",
        force_promote=True,
    )
    assert out.promoted
    bars, _ = synthetic_universe(
        list(s.watchlist), date(2016, 1, 4), date(2021, 12, 31), seed=s.metrics.seed, signal=2.0
    )
    broker = SimBroker(
        Portfolio(cash=D(100)), s.costs, Universe.assume_all_fractionable(s.watchlist, "SPY")
    )
    broker.begin_session(
        DECIDE_AT.date(), prices({k: bars[k] for k in s.watchlist}, DECIDE_AT.date(), "close")
    )
    rt, rec = rt_for(s, bars, broker, DECIDE_AT, PAPER)
    res = jobs.decide(rt)
    assert res.status == "decided", rec.titles()
    journal = (s.state.dir / "journal").glob("*.jsonl")
    records = [json.loads(line) for p in journal for line in p.read_text().splitlines()]
    decision = next(r for r in records if r["step"] == "decision")
    assert decision["data"]["strategy"] == "model"
    assert all(c["source"].startswith("model:") for c in decision["data"]["candidates"])
    summary = rec.alerts[-1].message
    assert "No promoted model" not in summary


def test_paper_exam_counts_only_the_strategy_that_would_go_live() -> None:
    spy = walk(80, 0.0005, 0.012, 2)
    recs = days(walk(80, 0.002, 0.004, 1), spy)
    recs = [
        r.model_copy(update={"strategy": "agents" if i < 40 else "model"})
        for i, r in enumerate(recs)
    ]
    t = tracking.Track(days=recs)
    g_all, _ = tracking.paper_exam(t, ExamConfig())
    g_model, ok = tracking.paper_exam(t, ExamConfig(), "model")
    assert g_all.sessions == 80 and g_model.sessions == 41
    assert not ok or g_model.sessions >= ExamConfig().paper_min_sessions


def test_fallback_overrides_the_rule_pipeline_in_decide(tmp_path: Path, bars: Bars) -> None:
    """With the fallback active, even strategy.kind=rule must not open new non-SPY positions."""
    s, broker = setup(
        tmp_path, bars, crosscheck__enabled="false", earnings__block_when_unknown="false"
    )
    store = StateStore(s.state.dir)
    state = store.load()
    tracking.activate(state.track, "test drift", DECIDE_AT.date())
    store.save(state)
    rt, _ = rt_for(s, bars, broker, DECIDE_AT, PAPER)
    assert jobs.decide(rt).status == "decided"
    plan = store.load().pending_plan
    assert plan is not None
    buys = [o for o in plan.orders if o.side is Side.BUY]
    assert all(o.symbol == "SPY" for o in buys)
    assert store.load().track.days[-1].strategy == ""
    assert store.load().track.last_strategy == "fallback"
