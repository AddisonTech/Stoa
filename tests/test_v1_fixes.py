"""Regression tests for the defects found in the v1 review."""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal

import httpx
import pandas as pd
import pytest

from stoa import adjust, alerts, jobs, money, redact
from stoa.broker.alpaca import AlpacaBroker
from stoa.broker.sim import SimBroker
from stoa.config import CostConfig, Secrets
from stoa.data.base import Bars, DataError, fresh_subset
from stoa.models import PositionMeta
from stoa.portfolio import Portfolio
from stoa.state import Inception, State
from stoa.trader import scaled_atr
from stoa.universe import Universe

D = Decimal


def frame(closes: dict[str, float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(k) for k in closes])
    c = list(closes.values())
    return pd.DataFrame(
        {"open": c, "high": c, "low": c, "close": c, "volume": [1e6] * len(c)}, index=idx
    )


def test_split_rescales_entry_price_atr_and_shadow_book() -> None:
    """A 4 for 1 split re-adjusts history; stored entry and stop must follow."""
    state = State(
        positions_meta={
            "NVDA": PositionMeta(
                entry_session=date(2024, 6, 3),
                entry_price=D(400),
                entry_atr=D(12),
                ref_session=date(2024, 6, 7),
                ref_close=D(420),
            )
        },
        sim_positions={"NVDA": D("0.05")},
        sim_avg_cost={"NVDA": D(400)},
    )
    # After the split the provider shows the old bar at a quarter of the price.
    bars: Bars = {"NVDA": frame({"2024-06-07": 105.0, "2024-06-10": 106.0})}
    notes = adjust.apply_adjustments(state, bars, shadow=True)
    m = state.positions_meta["NVDA"]
    assert m.entry_price == D(100) and m.entry_atr == D(3)
    assert state.sim_positions["NVDA"] == D("0.2") and state.sim_avg_cost["NVDA"] == D(100)
    assert notes and "NVDA" in notes[0]
    adjust.anchor(state, bars, date(2024, 6, 10), "SPY")
    assert state.positions_meta["NVDA"].ref_session == date(2024, 6, 10)
    assert state.positions_meta["NVDA"].ref_close == D("106.0")
    # No change in history: nothing is rescaled the next day.
    assert adjust.apply_adjustments(state, bars, shadow=True) == []


def test_dividend_adjusts_inception_benchmark_close() -> None:
    state = State(
        inception=Inception(session=date(2024, 6, 3), equity=D(100), benchmark_close=D(500)),
        last_close_session=date(2024, 6, 7),
        last_benchmark_close=D(510),
    )
    bars: Bars = {"SPY": frame({"2024-06-07": 505.0, "2024-06-10": 507.0})}
    adjust.adjust_benchmark(state, bars, "SPY")
    assert state.inception is not None
    assert state.inception.benchmark_close == D(500) * D(505) / D(510)


def test_scaled_atr_follows_fill_scale() -> None:
    assert scaled_atr(D(12), D(400), D(100)) == D(3)
    assert scaled_atr(D(2), D(100), D(101)) == D("2.02")
    assert scaled_atr(None, D(100), D(101)) == D(0)


def test_flows_since_close_feed_the_daily_loss_baseline() -> None:
    """A $50 withdrawal after the close must not look like a 50% down day."""
    s = State(
        high_water_mark=D(100),
        inception=Inception(session=date(2026, 7, 1), equity=D(100), benchmark_close=D(500)),
        last_close_session=date(2026, 9, 1),
        last_close_equity=D(100),
    )
    flows = money.parse_flows(
        [{"id": "w1", "activity_type": "CSW", "net_amount": "-50", "date": "2026-09-02"}]
    )
    money.apply_flows(s, flows, date(2026, 7, 1))
    assert s.flows_since_close == D(-50) and s.high_water_mark == D(50)
    assert jobs.prev_close_equity(s, date(2026, 9, 2), None) == D(50)


def test_execute_reads_flows_before_the_kill_switch() -> None:
    s = State(high_water_mark=D(200))
    s.inception = Inception(session=date(2026, 7, 1), equity=D(100), benchmark_close=D(500))
    acts = [{"id": "w9", "activity_type": "CSW", "net_amount": "-100", "date": "2026-09-02"}]

    def h(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v2/account/activities"
        return httpx.Response(200, json=acts)

    c = httpx.Client(base_url="https://paper.example", transport=httpx.MockTransport(h))
    broker = AlpacaBroker(c, c, live=False, sleep=lambda _s: None)
    rt = jobs.Runtime(
        settings=None,  # type: ignore[arg-type]
        secrets=Secrets(),
        notifier=alerts.RecordingNotifier(),
    )
    jobs.sync_flows(rt, s, broker, date(2026, 9, 2))
    assert s.high_water_mark == D(100)


def test_fresh_subset_drops_stale_non_held_and_fails_on_held() -> None:
    day = date(2024, 6, 10)
    good = frame({"2024-06-07": 1.0, "2024-06-10": 1.0})
    stale = frame({"2024-06-07": 1.0})
    bars: Bars = {"SPY": good, "AAPL": good, "DEAD": stale}
    fresh, dropped = fresh_subset(bars, ["SPY", "AAPL", "DEAD"], day, {"SPY"}, 0.5)
    assert set(fresh) == {"SPY", "AAPL"} and dropped == ["DEAD"]
    with pytest.raises(DataError, match="DEAD"):
        fresh_subset(bars, ["SPY", "AAPL", "DEAD"], day, {"SPY", "DEAD"}, 0.5)
    with pytest.raises(DataError, match="only 2 of 3"):
        fresh_subset(bars, ["SPY", "AAPL", "DEAD"], day, {"SPY"}, 0.9)


def test_sim_latest_prices_skips_untraded_symbols() -> None:
    u = Universe.assume_all_fractionable(("SPY", "AAPL"), "SPY")
    b = SimBroker(Portfolio(cash=D(100)), CostConfig(), u)
    b.begin_session(date(2024, 6, 10), {"SPY": D(500)})
    assert b.latest_prices(["SPY", "AAPL"]) == {"SPY": D(500)}


def test_redaction_masks_secrets_in_logs_and_alerts(caplog: pytest.LogCaptureFixture) -> None:
    redact.clear()
    Secrets.from_env({"STOA_NTFY_TOPIC": "topic-abcdef-123", "ALPACA_PAPER_SECRET": "sekret-999"})
    logger = logging.getLogger("stoa.test.redact")
    caplog.handler.addFilter(redact.RedactingFilter())
    logger.error("posting to https://ntfy.sh/%s failed", "topic-abcdef-123")
    assert "topic-abcdef-123" not in caplog.text and redact.MASK in caplog.text
    n = alerts.NtfyNotifier("https://ntfy.example", "topic-abcdef-123")
    body = n.payload(alerts.Alert("t", "error: key sekret-999 rejected"))
    assert body["message"] == f"error: key {redact.MASK} rejected"
    assert body["topic"] == "topic-abcdef-123"  # the topic itself must still be sent
    redact.clear()
