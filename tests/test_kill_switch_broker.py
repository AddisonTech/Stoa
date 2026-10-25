"""The kill switch at the broker: every position sold, every working order cancelled.

A wide-universe name (not on the watchlist) left open in the account must be
flattened too, and any order still working, Stoa's or not, must be cancelled
before the flatten sells go out.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from stoa import alerts
from stoa.state import KillSwitch, State, StateStore
from tests.conftest import SMALL_WATCHLIST
from tests.test_stops import (
    PRIOR,
    FakeAlpaca,
    meta,
    plan_selling,
    run_execute,
    settings_for,
)

D = Decimal
WIDE = "ZYXI"  # a small cap from the wide universe, never on the watchlist


def wide_book() -> FakeAlpaca:
    assert WIDE not in SMALL_WATCHLIST
    fake = FakeAlpaca(
        {"AAPL": ("1", "100"), WIDE: ("40", "6.10")},
        {"AAPL": "70", WIDE: "4.20", "TSLA": "200"},
    )
    fake.add_order(
        client_order_id="stoa-stop-20211013-AAPL-1",
        symbol="AAPL",
        side="sell",
        type="stop",
        qty="1",
        stop_price="65",
    )
    fake.add_order(
        client_order_id="stoa-stop-20211013-ZYXI-1",
        symbol=WIDE,
        side="sell",
        type="stop",
        qty="40",
        stop_price="3.90",
    )
    fake.add_order(
        client_order_id="placed-by-hand-1",
        symbol="TSLA",
        side="buy",
        type="limit",
        qty="1",
    )
    return fake


def test_active_kill_switch_flattens_a_wide_universe_position(tmp_path: Path) -> None:
    fake = wide_book()
    state = State(
        pending_plan=plan_selling([]),
        kill_switch=KillSwitch(active=True, reason="drawdown limit"),
        positions_meta={"AAPL": meta(PRIOR, "100", "65"), WIDE: meta(PRIOR, "6.10", "3.90")},
    )
    run_execute(tmp_path, fake, state)
    assert fake.positions == {}
    posts = [c for c in fake.calls if c[0] == "POST"]
    assert ("POST", "market", "sell AAPL") in posts
    assert ("POST", "market", f"sell {WIDE}") in posts
    # Nothing new is protected or bought: the account is flat.
    assert not [c for c in posts if c[1] == "stop" or c[2].startswith("buy")]
    saved = StateStore(settings_for(tmp_path).state.dir).load()
    assert saved.positions_meta == {}


def test_kill_switch_cancels_every_open_order_before_selling(tmp_path: Path) -> None:
    fake = wide_book()
    state = State(
        pending_plan=plan_selling([]),
        kill_switch=KillSwitch(active=True, reason="drawdown limit"),
        positions_meta={"AAPL": meta(PRIOR, "100", "65")},
    )
    run_execute(tmp_path, fake, state)
    assert all(o["status"] == "canceled" for o in fake.orders if o["type"] != "market")
    cancels = [i for i, c in enumerate(fake.calls) if c[0] == "DELETE"]
    sells = [i for i, c in enumerate(fake.calls) if c[:2] == ("POST", "market")]
    assert len(cancels) == 3  # both stops and the order placed by hand
    assert max(cancels) < min(sells)
    assert ("DELETE", "order", "TSLA placed-by-hand-1") in fake.calls


def test_kill_switch_tripping_at_the_open_flattens_the_wide_position(tmp_path: Path) -> None:
    fake = wide_book()
    # Equity at the open is 1000 cash + 70 + 168 = 1238, far below the 5000 high.
    state = State(
        pending_plan=plan_selling([]),
        high_water_mark=D(5000),
        positions_meta={"AAPL": meta(PRIOR, "100", "65")},
    )
    rec = run_execute(tmp_path, fake, state)
    assert fake.positions == {}
    assert "KILL SWITCH TRIPPED" in [a.title for a in rec.alerts]
    saved = StateStore(settings_for(tmp_path).state.dir).load()
    assert saved.kill_switch.active


def test_a_failed_cancel_alerts_and_the_flatten_still_runs(tmp_path: Path) -> None:
    fake = wide_book()
    fake.fail.add(("DELETE", "TSLA"))
    state = State(
        pending_plan=plan_selling([]),
        kill_switch=KillSwitch(active=True, reason="drawdown limit"),
    )
    rec = run_execute(tmp_path, fake, state)
    urgent = [a.title for a in rec.alerts if a.priority == alerts.URGENT]
    assert "Order cancel failed: buy TSLA" in urgent
    assert fake.positions == {}
