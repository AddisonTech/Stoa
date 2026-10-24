from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stoa import risk
from stoa.config import RiskLimits
from stoa.models import Candidate, ExitSignal, PlannedOrder, Side
from stoa.universe import AssetInfo, Universe

D = Decimal
DAY = date(2024, 6, 3)
LIMITS = RiskLimits()
WATCH = ("SPY", "AAPL", "MSFT", "NVDA", "KO", "XOM", "JPM", "NOFRAC")
UNIVERSE = Universe(
    watchlist=WATCH,
    benchmark="SPY",
    assets={s: AssetInfo(s, True, s != "NOFRAC") for s in WATCH},
)


def idea(sym: str, price: str = "100") -> risk.EntryIdea:
    c = Candidate(symbol=sym, conviction=4, thesis="t", score=1.0, source="rule")
    return risk.EntryIdea(candidate=c, ref_price=D(price), atr=D("2"), stop_price=D("95"))


def pos(sym: str, value: str) -> risk.HeldPosition:
    return risk.HeldPosition(sym, D(value) / D(100), D(100))


def ctx(
    equity: str = "100",
    cash: str | None = None,
    positions: dict[str, risk.HeldPosition] | None = None,
    prev: str | None = None,
    hwm: str | None = None,
    ks: bool = False,
    bull: bool = True,
    limits: RiskLimits = LIMITS,
) -> risk.RiskContext:
    positions = positions or {}
    invested = sum((p.value for p in positions.values()), D(0))
    return risk.RiskContext(
        session=DAY,
        equity=D(equity),
        cash=D(cash) if cash is not None else D(equity) - invested,
        positions=positions,
        prev_equity=D(prev) if prev else None,
        high_water_mark=D(hwm or equity),
        kill_switch_active=ks,
        regime_bull=bull,
        universe=UNIVERSE,
        limits=limits,
    )


def reasons(result: risk.RiskResult) -> dict[str, str]:
    return {r.symbol: r.reason for r in result.rejections}


def buys(result: risk.RiskResult) -> list[PlannedOrder]:
    return [o for o in result.orders if o.side is Side.BUY]


def test_entry_sized_at_max_position() -> None:
    res = risk.evaluate(ctx(), [], [idea("AAPL")])
    assert [(o.symbol, o.notional) for o in buys(res)] == [("AAPL", D("20.00"))]
    assert buys(res)[0].stop_price == D("95") and buys(res)[0].entry_atr == D("2")


def test_max_positions_pass_and_fail() -> None:
    held = {s: pos(s, "15") for s in ["MSFT", "NVDA", "KO", "XOM"]}
    ok = risk.evaluate(ctx(positions=held), [], [idea("AAPL")])
    assert len(buys(ok)) == 1
    held["JPM"] = pos("JPM", "15")
    full = risk.evaluate(ctx(positions=held), [], [idea("AAPL")])
    assert buys(full) == [] and reasons(full) == {"AAPL": "max_positions"}


def test_cash_buffer_pass_and_fail() -> None:
    # 22 cash on 100 equity: 2 is the buffer, 20 is free.
    ok = risk.evaluate(ctx(cash="22"), [], [idea("AAPL")])
    assert buys(ok)[0].notional == D("20.00")
    # 10 cash: 8 free after the buffer, sized down to it.
    small = risk.evaluate(ctx(cash="10"), [], [idea("AAPL")])
    assert buys(small)[0].notional == D("8.00")
    # 2 cash: nothing free after the buffer.
    none = risk.evaluate(ctx(cash="2"), [], [idea("AAPL")])
    assert buys(none) == [] and reasons(none) == {"AAPL": "insufficient_cash"}


def test_gross_exposure_never_exceeds_equity() -> None:
    # Cash is plentiful but positions already sit at 95% of equity.
    held = {"MSFT": pos("MSFT", "95")}
    res = risk.evaluate(ctx(equity="100", cash="50", positions=held), [], [idea("AAPL")])
    assert buys(res)[0].notional == D("5.00")


def test_max_size_pass_and_fail() -> None:
    res = risk.evaluate(ctx(equity="1000"), [], [idea("AAPL"), idea("MSFT")])
    assert all(o.notional == D("200.00") for o in buys(res))
    assert all(o.notional <= D("1000") * LIMITS.max_position_pct for o in buys(res))


def test_min_order_pass_and_fail() -> None:
    ok = risk.evaluate(ctx(equity="5.00"), [], [idea("AAPL")])
    assert buys(ok)[0].notional == D("1.00")
    tiny = risk.evaluate(ctx(equity="4.00"), [], [idea("AAPL")])
    assert buys(tiny) == [] and reasons(tiny) == {"AAPL": "min_order"}


def test_non_watchlist_and_non_fractionable() -> None:
    res = risk.evaluate(ctx(), [], [idea("TSLA"), idea("NOFRAC"), idea("AAPL")])
    assert reasons(res) == {"TSLA": "not_in_watchlist", "NOFRAC": "not_fractionable"}
    assert [o.symbol for o in buys(res)] == ["AAPL"]


def test_already_held_rejected() -> None:
    res = risk.evaluate(ctx(positions={"AAPL": pos("AAPL", "20")}), [], [idea("AAPL")])
    assert reasons(res) == {"AAPL": "already_held"}


def test_same_day_round_trip_blocked_in_plan() -> None:
    held = {"AAPL": pos("AAPL", "20")}
    exit_sig = ExitSignal(symbol="AAPL", reason="stop_loss", detail="x")
    res = risk.evaluate(ctx(positions=held), [exit_sig], [idea("AAPL"), idea("MSFT")])
    sides = {(o.symbol, o.side) for o in res.orders}
    assert ("AAPL", Side.SELL) in sides and ("AAPL", Side.BUY) not in sides
    assert reasons(res)["AAPL"] == "same_day_round_trip"


def test_daily_loss_limit_pass_and_fail() -> None:
    ok = risk.evaluate(ctx(equity="97.01", prev="100"), [], [idea("AAPL")])
    assert len(buys(ok)) == 1
    bad = risk.evaluate(ctx(equity="97", prev="100"), [], [idea("AAPL")])
    assert buys(bad) == [] and reasons(bad) == {"AAPL": "daily_loss_limit"}


def test_daily_loss_limit_still_allows_exits() -> None:
    held = {"AAPL": pos("AAPL", "20")}
    sig = ExitSignal(symbol="AAPL", reason="stop_loss", detail="x")
    res = risk.evaluate(ctx(equity="90", prev="100", positions=held), [sig], [idea("MSFT")])
    assert [(o.symbol, o.side) for o in res.orders] == [("AAPL", Side.SELL)]


def test_risk_off_blocks_entries_not_exits() -> None:
    held = {"AAPL": pos("AAPL", "20")}
    sig = ExitSignal(symbol="AAPL", reason="trend_break", detail="x")
    res = risk.evaluate(ctx(positions=held, bull=False), [sig], [idea("MSFT")])
    assert [(o.symbol, o.side) for o in res.orders] == [("AAPL", Side.SELL)]
    assert reasons(res) == {"MSFT": "risk_off"}


def test_kill_switch_trips_and_flattens() -> None:
    held = {"AAPL": pos("AAPL", "30"), "MSFT": pos("MSFT", "20")}
    res = risk.evaluate(ctx(equity="75", hwm="100", positions=held), [], [idea("NVDA")])
    assert res.kill_switch_tripped
    assert {(o.symbol, o.side, o.reason) for o in res.orders} == {
        ("AAPL", Side.SELL, "kill_switch"),
        ("MSFT", Side.SELL, "kill_switch"),
    }
    assert {o.symbol: o.qty for o in res.orders} == {"AAPL": D("0.3"), "MSFT": D("0.2")}
    assert reasons(res) == {"NVDA": "kill_switch"}


def test_kill_switch_not_tripped_just_above_limit() -> None:
    res = risk.evaluate(ctx(equity="75.01", hwm="100"), [], [idea("NVDA")])
    assert not res.kill_switch_tripped and len(buys(res)) == 1


def test_kill_switch_active_halts_entries_and_flattens_remaining() -> None:
    held = {"AAPL": pos("AAPL", "10")}
    res = risk.evaluate(ctx(positions=held, ks=True), [], [idea("NVDA")])
    assert not res.kill_switch_tripped
    assert [(o.symbol, o.side) for o in res.orders] == [("AAPL", Side.SELL)]
    assert reasons(res) == {"NVDA": "kill_switch"}


def test_max_orders_per_day() -> None:
    limits = RiskLimits(max_orders_per_day=5, max_positions=5)
    held = {s: pos(s, "10") for s in ["MSFT", "NVDA", "KO", "XOM"]}
    sigs = [ExitSignal(symbol=s, reason="time_exit", detail="x") for s in held]
    res = risk.evaluate(ctx(positions=held, limits=limits), sigs, [idea("AAPL"), idea("JPM")])
    assert len(res.orders) == 5
    assert reasons(res)["JPM"] == "max_orders"


def test_exit_for_unheld_symbol_rejected() -> None:
    res = risk.evaluate(ctx(), [ExitSignal(symbol="AAPL", reason="stop_loss", detail="x")], [])
    assert reasons(res) == {"AAPL": "no_position"}


def test_drawdown_and_hwm_helpers() -> None:
    assert risk.drawdown(D(80), D(100)) == D("0.2")
    assert risk.drawdown(D(120), D(100)) == 0
    assert risk.next_high_water_mark(None, D(5)) == D(5)
    assert risk.next_high_water_mark(D(10), D(5)) == D(10)


# ---- execution-time checks ------------------------------------------------------


def exec_ctx(**kw: object) -> risk.ExecutionContext:
    base: dict[str, object] = {
        "equity": D(100),
        "cash": D(100),
        "invested": D(0),
        "prev_equity": D(100),
        "high_water_mark": D(100),
        "kill_switch_active": False,
        "held": frozenset(),
        "open_positions": 0,
        "orders_today": 0,
        "traded_today": {},
        "universe": UNIVERSE,
        "limits": LIMITS,
    }
    base.update(kw)
    return risk.ExecutionContext(**base)  # type: ignore[arg-type]


def buy(sym: str, notional: str = "20", ref: str = "100") -> PlannedOrder:
    return PlannedOrder(
        symbol=sym, side=Side.BUY, notional=D(notional), ref_price=D(ref), reason="entry"
    )


def sell(sym: str, qty: str = "1") -> PlannedOrder:
    return PlannedOrder(
        symbol=sym, side=Side.SELL, qty=D(qty), ref_price=D(100), reason="stop_loss"
    )


@pytest.mark.parametrize(
    ("live", "ok"), [("108", True), ("92", True), ("108.01", False), ("91.99", False)]
)
def test_price_sanity(live: str, ok: bool) -> None:
    got, rej = risk.check_buys([buy("AAPL")], {"AAPL": D(live)}, exec_ctx())
    assert (len(got) == 1) is ok
    if not ok:
        assert rej[0].reason == "price_sanity"


def test_execution_resizes_to_real_cash() -> None:
    got, _ = risk.check_buys([buy("AAPL", "20")], {"AAPL": D(100)}, exec_ctx(cash=D("12")))
    assert got[0].notional == D("10.00")


def test_execution_rejects_round_trip_and_held() -> None:
    traded = {"AAPL": frozenset({Side.SELL}), "KO": frozenset({Side.BUY})}
    got, rej = risk.check_buys(
        [buy("AAPL"), buy("MSFT")],
        {"AAPL": D(100), "MSFT": D(100)},
        exec_ctx(traded_today=traded, held=frozenset({"MSFT"})),
    )
    assert got == []
    assert {r.symbol: r.reason for r in rej} == {
        "AAPL": "same_day_round_trip",
        "MSFT": "already_held",
    }
    sells_ok, srej = risk.check_sells([sell("KO")], {"KO": D(1)}, exec_ctx(traded_today=traded))
    assert sells_ok == [] and srej[0].reason == "same_day_round_trip"


def test_execution_kill_switch_and_daily_loss() -> None:
    got, rej = risk.check_buys([buy("AAPL")], {"AAPL": D(100)}, exec_ctx(kill_switch_active=True))
    assert got == [] and rej[0].reason == "kill_switch"
    got, rej = risk.check_buys([buy("AAPL")], {"AAPL": D(100)}, exec_ctx(equity=D(96), cash=D(96)))
    assert got == [] and rej[0].reason == "daily_loss_limit"


def test_execution_watchlist_and_fractionable() -> None:
    got, rej = risk.check_buys(
        [buy("TSLA"), buy("NOFRAC")], {"TSLA": D(100), "NOFRAC": D(100)}, exec_ctx()
    )
    assert got == [] and [r.reason for r in rej] == ["not_in_watchlist", "not_fractionable"]


def test_execution_max_positions_and_orders() -> None:
    got, rej = risk.check_buys([buy("AAPL")], {"AAPL": D(100)}, exec_ctx(open_positions=5))
    assert got == [] and rej[0].reason == "max_positions"
    got, rej = risk.check_buys([buy("AAPL")], {"AAPL": D(100)}, exec_ctx(orders_today=10))
    assert got == [] and rej[0].reason == "max_orders"


def test_sells_close_whatever_is_held_now() -> None:
    got, rej = risk.check_sells([sell("AAPL", "2"), sell("KO")], {"AAPL": D("0.5")}, exec_ctx())
    assert [(o.symbol, o.qty) for o in got] == [("AAPL", D("0.5"))]
    split, _ = risk.check_sells([sell("NVDA", "0.1")], {"NVDA": D("1.0")}, exec_ctx())
    assert split[0].qty == D("1.0")
    assert rej[0].reason == "no_position"
