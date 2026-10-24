"""Trial 7, the analyst forward test: shadow scoring and the one evaluation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from stoa import alerts, analyst, clock, jobs, shadow
from stoa.config import Settings
from stoa.data.base import Bars
from stoa.experiments import Registry
from stoa.state import State, StateStore
from tests.conftest import ROOT, SMALL_WATCHLIST, make_settings
from tests.test_analyst import FakeClient
from tests.test_jobs import DECIDE_AT, PAPER, runtime, settings_for, sim

D = Decimal
START = date(2026, 10, 26)
AFTER_FIRST = datetime(2026, 10, 26, 23, 45, tzinfo=UTC)  # the job's slot on the first session


def at_ny(day: str, hh: int, mm: int = 0) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromisoformat(f"{day}T{hh:02d}:{mm:02d}"), tz=clock.NY).tz_convert(
        UTC
    )


def news_row(i: int, symbol: str, when: pd.Timestamp) -> dict[str, Any]:
    return {
        "id": f"alpaca:{i}",
        "symbol": symbol,
        "known_at": when,
        "created_at": when,
        "headline": f"{symbol} headline {i}",
        "summary": "Revenue beat estimates and the outlook was raised.",
        "content": "Revenue beat estimates and the outlook was raised. " * 5,
        "source": "benzinga",
        "url": "",
        "kind": "news",
    }


class FakeNews:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.asked: list[tuple[list[str], date, date]] = []

    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        self.asked.append((symbols, start, end))
        df = pd.DataFrame(self.rows)
        return df.loc[df["symbol"].isin(symbols)].reset_index(drop=True)


class Untouchable:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the analyst was called while it must not be ({name})")


def shadow_settings(tmp: Path, **kw: str) -> Settings:
    return make_settings(
        watchlist=",".join(SMALL_WATCHLIST),
        state__dir=str(tmp / "state"),
        analyst__cache_dir=str(tmp / "cache"),
        analyst__batch_poll_seconds="1",
        **kw,
    )


def run(
    s: Settings,
    ledger: shadow.ShadowLedger,
    news: Any,
    *,
    client: Any = None,
    state: State | None = None,
    now: datetime = AFTER_FIRST,
) -> tuple[shadow.ShadowRun, list[alerts.Alert]]:
    sent: list[alerts.Alert] = []
    out = shadow.run_session(
        s,
        state or State(),
        ledger,
        now=lambda: now,
        sleep=lambda _s: None,
        send=sent.append,
        news=news,
        client=client if client is not None else FakeClient(),
        approvals=None,
        approval_topic=None,
        members=lambda _d: {"AAPL", "MSFT"},
    )
    return out, sent


# ---- forward only -------------------------------------------------------------------------


def test_news_before_the_forward_start_is_never_scored(tmp_path: Path) -> None:
    s = shadow_settings(tmp_path)
    rows = [
        news_row(1, "AAPL", at_ny("2026-10-23", 11)),  # before the start: holdout era
        news_row(2, "MSFT", at_ny("2026-10-25", 20)),  # the weekend before: still before
        news_row(3, "AAPL", at_ny("2026-10-26", 9, 15)),  # first session: scored
        news_row(4, "MSFT", at_ny("2026-10-26", 15, 59)),  # first session: scored
        news_row(5, "AAPL", at_ny("2026-10-26", 16, 30)),  # after the close: next session
        news_row(6, "TSLA", at_ny("2026-10-26", 12)),  # not a member that day
    ]
    ledger = shadow.ShadowLedger()
    client = FakeClient()
    out, _ = run(s, ledger, FakeNews(rows), client=client)
    assert (out.session, out.status, out.items) == (START, "scored", 2)
    scored = pd.read_parquet(shadow.scores_path(s))
    assert set(scored["item_id"]) == {"alpaca:3", "alpaca:4"}
    assert (pd.to_datetime(scored["known_at"], utc=True) >= at_ny("2026-10-26", 0)).all()
    assert ledger.sessions == [START] and ledger.items_by_session == {"2026-10-26": 2}
    assert len(client.messages.batches.created) == 1  # through the Batch API
    again, _ = run(s, ledger, FakeNews(rows), client=Untouchable())
    assert again.status == "already_scored"


def test_nothing_runs_before_the_forward_start(tmp_path: Path) -> None:
    s = shadow_settings(tmp_path)
    friday = datetime(2026, 10, 23, 23, 45, tzinfo=UTC)
    out, sent = run(s, shadow.ShadowLedger(), Untouchable(), client=Untouchable(), now=friday)
    assert out.status == "not_started" and sent == []


def test_the_window_is_close_to_close_and_never_before_the_start() -> None:
    lo, hi = shadow.window(START, START)
    assert lo == at_ny("2026-10-26", 0).to_pydatetime()  # not Friday's close
    assert hi == clock.session(START).close
    lo2, hi2 = shadow.window(date(2026, 10, 27), START)
    assert lo2 == clock.session(START).close and hi2 == clock.session(date(2026, 10, 27)).close


# ---- spend ----------------------------------------------------------------------------------


def test_cap_pause_is_respected_and_counts_every_claude_call(tmp_path: Path) -> None:
    s = shadow_settings(tmp_path)
    rows = [news_row(1, "AAPL", at_ny("2026-10-26", 10))]
    # DECIDE (the scout) spent $20 and the shadow job $5 this month: the $25 cap is reached.
    state = State(llm_spend_by_month={"2026-10": D("20.00")})
    ledger = shadow.ShadowLedger(spend_by_month={"2026-10": D("5.00")})
    out, sent = run(s, ledger, FakeNews(rows), client=Untouchable(), state=state)
    assert out.status == "paused" and ledger.sessions == []
    assert ledger.skipped[-1].reason == "paused at the cap"
    assert ledger.cap_asks[-1].outcome == "unavailable"  # recorded in the shadow ledger
    assert state.cap_asks == []  # state.json is never written by the shadow job
    assert any(a.title == "Analyst stays paused" for a in sent)


def test_spend_is_charged_to_the_shadow_ledger(tmp_path: Path) -> None:
    s = shadow_settings(tmp_path)
    ledger = shadow.ShadowLedger()
    run(s, ledger, FakeNews([news_row(1, "AAPL", at_ny("2026-10-26", 10))]))
    assert ledger.spend_by_month["2026-10"] > 0


def test_cost_estimate_math() -> None:
    cfg = make_settings().analyst
    p = make_settings().llm.prices[cfg.model]
    budget = make_settings().budget
    it = analyst.Item(
        item_id="a", symbol="AAPL", known_at=AFTER_FIRST, kind="news", title="T", body="x" * 4000
    )
    usd, n = analyst.estimate_cost([it] * 10, cfg, p, budget, batch=True)
    text_tokens = len(it.text(cfg.text_chars)) / analyst.CHARS_PER_TOKEN + 20
    sys_t = cfg.est_system_tokens
    expect = (
        p.input_per_mtok * (10 * text_tokens) / 1e6
        + p.input_per_mtok * budget.cache_write_multiplier_1h * int(10 * 0.1 * sys_t) / 1e6
        + p.input_per_mtok * budget.cache_read_multiplier * int(10 * 0.9 * sys_t) / 1e6
        + p.output_per_mtok * 10 * cfg.est_output_tokens / 1e6
    ) * budget.batch_discount
    assert n == 10 and usd == pytest.approx(expect, rel=1e-3)
    # The pre-registered scope: about 620 items a session stays far under $0.60.
    assert usd / 10 * 620 < 0.60


# ---- trading never sees the shadow --------------------------------------------------------


def write_shadow(s: Settings) -> None:
    d = shadow.shadow_dir(s)
    d.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "item_id": f"x{i}",
            "symbol": sym,
            "known_at": pd.Timestamp("2021-10-12T15:00:00Z"),
            "kind": "news",
            "sentiment": 1.0 if i % 2 else -1.0,
            "guidance": 1.0,
            "surprise": 1.0,
            "risk": 0.0,
            "materiality": 1.0,
            "confidence": 1.0,
            "contaminated": False,
            "model": "claude-haiku-5-5",
        }
        for i, sym in enumerate(SMALL_WATCHLIST)
    ]
    pd.DataFrame(rows).to_parquet(shadow.scores_path(s))
    shadow.save_ledger(s, shadow.ShadowLedger(sessions=[START], spend_by_month={"2021-10": D(1)}))


def plan_orders(tmp: Path, bars: Bars, with_shadow: bool) -> tuple[list[tuple[Any, ...]], str]:
    s = settings_for(tmp)
    if with_shadow:
        write_shadow(s)
    rt, rec = runtime(s, bars, PAPER, DECIDE_AT, sim(s, bars))
    assert jobs.decide(rt).status == "decided"
    plan = StateStore(s.state.dir).load().pending_plan
    assert plan is not None
    orders = [(o.symbol, o.side, o.notional, o.qty, o.reason) for o in plan.orders]
    return orders, rec.alerts[-1].message


def test_shadow_scores_never_change_orders(tmp_path: Path, bars: Bars) -> None:
    without, summary_a = plan_orders(tmp_path / "a", bars, with_shadow=False)
    with_, summary_b = plan_orders(tmp_path / "b", bars, with_shadow=True)
    assert without == with_ and without
    assert "Analyst shadow: 0 of 210 sessions scored" in summary_a
    assert "Analyst shadow: 1 of 210 sessions scored" in summary_b


def test_a_broken_shadow_ledger_never_breaks_decide(tmp_path: Path, bars: Bars) -> None:
    s = settings_for(tmp_path)
    shadow.shadow_dir(s).mkdir(parents=True)
    shadow.ledger_path(s).write_text("{not json", encoding="utf-8")
    rt, rec = runtime(s, bars, PAPER, DECIDE_AT, sim(s, bars))
    assert jobs.decide(rt).status == "decided"
    assert "Analyst shadow: progress unavailable" in rec.alerts[-1].message


def test_the_workflow_cannot_hold_up_trading() -> None:
    wf = ROOT / ".github" / "workflows"
    analyst_wf = yaml.safe_load((wf / "analyst.yml").read_text(encoding="utf-8"))
    daily = yaml.safe_load((wf / "daily.yml").read_text(encoding="utf-8"))
    assert analyst_wf["concurrency"]["group"] != daily["concurrency"]["group"]
    runs = " ".join(str(st.get("run", "")) for st in analyst_wf["jobs"]["shadow"]["steps"])
    assert "stoa analyst shadow" in runs
    for forbidden in ("stoa decide", "stoa execute", "kill-switch", "money confirm"):
        assert forbidden not in runs
    daily_text = (wf / "daily.yml").read_text(encoding="utf-8")
    assert "stoa analyst" not in daily_text and "shadow" not in daily_text  # trading never waits
    failure = [
        st for st in analyst_wf["jobs"]["shadow"]["steps"] if "failure()" in str(st.get("if"))
    ]
    assert len(failure) == 1 and "Trading is not affected" in failure[0]["run"]


# ---- the one evaluation -------------------------------------------------------------------


def fake_oos(text: bool) -> pd.DataFrame:
    rng = np.random.default_rng(1 if text else 2)
    days = clock.sessions_between(date(2026, 1, 2), date(2027, 8, 26))
    idx = pd.MultiIndex.from_product(
        [[pd.Timestamp(d) for d in days], [f"S{i}" for i in range(30)]], names=["date", "symbol"]
    )
    y = rng.integers(0, 2, len(idx)).astype(float)
    p = np.clip(
        0.48 + (0.08 if text else 0.0) * (y - 0.5) + rng.normal(0, 0.02, len(idx)), 0.01, 0.99
    )
    return pd.DataFrame({"p": p, "y": y, "fwd_excess": (y - 0.5) * 0.02}, index=idx)


def test_evaluate_refuses_early_then_runs_once_and_refuses_twice(tmp_path: Path) -> None:
    s = shadow_settings(tmp_path)
    sessions = clock.sessions_between(START, date(2027, 12, 31))
    shadow.save_ledger(s, shadow.ShadowLedger(sessions=sessions[:209]))
    calls: list[bool] = []

    def runner(se: Settings, _inputs: Any, text: bool) -> pd.DataFrame:
        assert (se.model.horizon, se.model.candidates, se.model.price_features) == (
            10,
            ("logistic",),
            None,
        )
        assert se.model.normalize == "zscore" and se.model.use_quality
        calls.append(text)
        return fake_oos(text)

    now = datetime(2027, 9, 15, tzinfo=UTC)
    with pytest.raises(shadow.ShadowError, match="209 of 210"):
        shadow.evaluate(s, tmp_path, now, runner=runner, inputs=object())
    assert calls == []
    shadow.save_ledger(s, shadow.ShadowLedger(sessions=sessions[:210]))
    out = shadow.evaluate(s, tmp_path, now, runner=runner, inputs=object())
    assert calls == [False, True]
    assert out["sessions"] == 210 and set(out["checks"]) == {
        "calibrated",
        "beats_base_rate",
        "beats_without_text",
        "deflated_sharpe",
    }
    recs = Registry(s.registry_path).load()
    assert [r.name for r in recs] == [shadow.EVAL_NAME]  # exactly one trial
    with pytest.raises(shadow.ShadowError, match="already evaluated"):
        shadow.evaluate(s, tmp_path, now, runner=runner, inputs=object())
    shadow.evaluation_path(s).unlink()  # even without the marker, the registry remembers
    with pytest.raises(shadow.ShadowError, match="already evaluated"):
        shadow.evaluate(s, tmp_path, now, runner=runner, inputs=object())
    assert len(Registry(s.registry_path).load()) == 1 and calls == [False, True]


def test_evaluation_rows_come_only_from_the_forward_window(tmp_path: Path) -> None:
    s = shadow_settings(tmp_path)
    sessions = clock.sessions_between(START, date(2027, 12, 31))
    shadow.save_ledger(s, shadow.ShadowLedger(sessions=sessions[:210]))
    out = shadow.evaluate(
        s,
        tmp_path,
        datetime(2027, 9, 15, tzinfo=UTC),
        runner=lambda _s, _i, text: fake_oos(text),
        inputs=object(),
    )
    forward = len(clock.sessions_between(START, date(2027, 8, 26))) * 30
    assert out["with_text"]["n"] == forward and out["without_text"]["n"] == forward
    assert out["checks"]["beats_without_text"]  # the fake text carries signal


def test_shadow_settings_match_the_pre_registration() -> None:
    s = make_settings()
    assert s.analyst.shadow_start == START and s.analyst.shadow_sessions == 210
    assert s.analyst.model == "claude-haiku-5-5" and not s.model.use_text
    assert s.analyst.shadow_start > s.backtest.holdout_end  # no overlap with the holdout
    assert START - timedelta(days=3) == s.backtest.holdout_end
