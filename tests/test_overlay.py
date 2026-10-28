"""Trial 13's risk overlay: sizing, caps, the drawdown brake and broker-side stops."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from stoa import clock, event_signals
from stoa.backtest import _broker_stops
from stoa.broker.base import OrderRequest
from stoa.broker.sim import SimBroker
from stoa.broker.sweep import SweepBroker
from stoa.config import CostConfig, ExitConfig, RiskLimits
from stoa.data.base import Bars
from stoa.earnings import EarningsCalendar, empty_calendar
from stoa.event_signals import EARNINGS, Signal
from stoa.event_strategy import EventStrategy
from stoa.experiments import Registry
from stoa.features import build_features, slice_features
from stoa.models import PositionMeta, Side
from stoa.newsbrain import spycore as sc
from stoa.newsbrain import trial13
from stoa.overlay import HALT_NOTE, Overlay, OverlayConfig, account_settings
from stoa.planner import DecisionInput
from stoa.portfolio import Portfolio
from stoa.state import State
from stoa.universe import AssetInfo, Universe
from tests.conftest import ROOT, SMALL_WATCHLIST, make_settings

D = Decimal
DAY = date(2024, 3, 11)


def step(ov: Overlay, dd: str) -> object:
    return ov.step(DAY, D(1) - D(dd), D(1), D(1))


def test_the_brake_halves_at_10_holds_at_15_and_lets_go_back_within_10() -> None:
    ov = Overlay()
    st = ov.step(DAY, D("95"), D("100"), D(1))
    assert (st.size_multiplier, st.halted, st.exposure_cap) == (D(1), False, D(1))
    st = ov.step(DAY, D("89"), D("100"), D(1))
    assert (st.size_multiplier, st.halted, st.exposure_cap) == (D("0.5"), False, D("0.50"))
    st = ov.step(DAY, D("85"), D("100"), D(1))
    assert st.halted and st.exposure_cap == D("0.50")
    st = ov.step(DAY, D("88"), D("100"), D(1))  # 12% off: still held, not back within 10%
    assert st.halted and st.size_multiplier == D("0.5")
    st = ov.step(DAY, D("90"), D("100"), D(1))  # back to 10%: entries allowed, at half size
    assert not st.halted and st.size_multiplier == D("0.5")
    st = ov.step(DAY, D("91"), D("100"), D(1))
    assert (st.size_multiplier, st.halted, st.exposure_cap) == (D(1), False, D(1))


def test_each_position_risks_one_percent_at_the_atr_stop() -> None:
    ov = Overlay()
    w = ov.weight(100.0, 2.0, ExitConfig())  # stop 5 dollars away, 5% of the price
    assert w == pytest.approx(D("0.2"))  # 20% of equity loses 1% at the stop
    assert ov.weight(100.0, 10.0, ExitConfig()) == pytest.approx(D("0.04"))
    assert ov.weight(100.0, 0.0, ExitConfig()) is None
    lim = ov.limits(RiskLimits(), ov.step(DAY, D("80"), D("100"), D(1)))
    assert lim.max_position_pct == D("0.05") and lim.max_gross_exposure_pct == D("0.50")


def test_account_settings_set_the_cap_and_the_money() -> None:
    s = account_settings(make_settings(), OverlayConfig(), D("100000"))
    assert s.risk.max_position_pct == D("0.05")
    assert s.backtest.starting_equity == D("100000")
    assert s.risk.max_positions == make_settings().risk.max_positions  # trial 12's rails stay


# ---- the strategy with the overlay ----------------------------------------------------------


def liquid_day(bars: Bars) -> date:
    f = build_features(bars, "SPY")
    for ts in reversed(f["SPY"].index[:-60]):
        if all(
            f[s].loc[ts, "close"] >= 5 and f[s].loc[ts, "dollar_vol_20"] >= 5e6
            for s in ("AAPL", "MSFT", "NVDA", "JPM")
            if ts in f[s].index
        ):
            return ts.date()
    raise AssertionError("no liquid day")


def decide(strat: EventStrategy, bars: Bars, day: date, equity: str, hwm: str) -> object:
    feats = build_features(bars, "SPY")
    syms = tuple(sorted(set(SMALL_WATCHLIST)))
    uni = Universe(
        watchlist=syms, benchmark="SPY", assets={s: AssetInfo(s, True, True) for s in syms}
    )
    return strat.decide(
        DecisionInput(
            session=day,
            execute_session=clock.next_session(day),
            features=slice_features(feats, day),
            holdings={},
            equity=D(equity),
            cash=D(equity),
            prev_equity=D(equity),
            high_water_mark=D(hwm),
            kill_switch_active=False,
            universe=uni,
        )
    )


def strategy(day: date, overlay: Overlay | None) -> EventStrategy:
    base = make_settings(
        watchlist=",".join(SMALL_WATCHLIST),
        earnings__block_when_unknown="false",
        risk__block_entries_when_risk_off="false",
        risk__max_pair_correlation="1.0",
    )
    s = account_settings(base, OverlayConfig(), D("100000"))
    sigs = [Signal(x, day, EARNINGS, 0.1 + i / 10, "") for i, x in enumerate(("AAPL", "MSFT"))]
    return EventStrategy(
        s, {day: event_signals.ranked(sigs)}, EarningsCalendar(empty_calendar()), overlay=overlay
    )


def buys(d: object) -> dict[str, Decimal]:
    return {o.symbol: o.notional for o in d.orders if o.side is Side.BUY}  # type: ignore[attr-defined]


def test_the_overlay_sizes_by_atr_under_the_cap_and_halves_in_a_drawdown(bars: Bars) -> None:
    day = liquid_day(bars)
    snap = build_features(bars, "SPY")
    full = buys(decide(strategy(day, Overlay()), bars, day, "100000", "100000"))
    assert set(full) == {"AAPL", "MSFT"}
    for sym, notional in full.items():
        row = snap[sym].loc[pd.Timestamp(day)]
        want = min(D("0.05"), D("0.01") / D(str(float(2.5 * row["atr_14"] / row["close"]))))
        assert notional == pytest.approx(D("100000") * want, abs=D("0.01"))
    half = buys(decide(strategy(day, Overlay()), bars, day, "88000", "100000"))
    for sym in full:
        assert half[sym] == pytest.approx(full[sym] * D("0.88") / 2, rel=D("0.001"))
    held = decide(strategy(day, Overlay()), bars, day, "84000", "100000")
    assert buys(held) == {}
    # Without the overlay the same picks take the full 5% slot.
    plain = buys(decide(strategy(day, None), bars, day, "100000", "100000"))
    assert set(plain) == set(full) and all(v == D("5000.00") for v in plain.values())


def test_a_held_brake_names_itself() -> None:
    assert "15%" in HALT_NOTE and "10%" in HALT_NOTE


# ---- the SPY core under the exposure cap ---------------------------------------------------------


D1, D2 = date(2024, 3, 11), date(2024, 3, 12)


def sweeper(cash: str = "100") -> SweepBroker:
    uni = Universe.assume_all_fractionable(("AAA", "SPY"), "SPY")
    b = SweepBroker(
        Portfolio(cash=D(cash)), CostConfig(), uni, None, symbol="SPY", reserve_pct=D("0.02")
    )
    b.begin_session(D1, {"SPY": D("500"), "AAA": D("20")})
    return b


def test_the_sweep_stops_at_the_cap_and_leaves_the_rest_in_cash() -> None:
    b = sweeper()
    b.sweep(kill_switch=False, cap=D("0.50"))
    eq = b.portfolio.equity(b.prices)
    assert b.fund_value() == pytest.approx(D("50"), abs=D("0.1"))
    assert b.portfolio.cash == pytest.approx(eq - b.fund_value(), abs=D("0.01"))


def test_over_the_cap_the_fund_is_sold_down_but_never_the_day_it_was_bought() -> None:
    b = sweeper()
    b.sweep(kill_switch=False)  # all but the 2% reserve in SPY
    b.sweep(kill_switch=False, cap=D("0.50"))  # same session: no round trip, nothing sold
    assert b.fund_value() > D("90")
    b.begin_session(D2, {"SPY": D("500"), "AAA": D("20")})
    b.sweep(kill_switch=False, cap=D("0.50"))
    assert b.fund_value() == pytest.approx(D("50"), abs=D("0.2"))
    b.submit(OrderRequest(symbol="AAA", side=Side.BUY, notional=D("20"), client_order_id="a"))
    assert b.portfolio.qty("AAA") > 0  # picks still buy from cash


def test_no_cap_is_trial_12_exactly() -> None:
    a, b = sweeper(), sweeper()
    a.sweep(kill_switch=False)
    b.sweep(kill_switch=False, cap=None)
    assert a.portfolio.qty("SPY") == b.portfolio.qty("SPY") and a.portfolio.cash == b.portfolio.cash


# ---- broker-side stops in the backtest ----------------------------------------------------------


def stop_case(low: float, at_945: float) -> tuple[SimBroker, State, int]:
    entry = date(2024, 3, 8)
    uni = Universe.assume_all_fractionable(("AAA", "SPY"), "SPY")
    b = SimBroker(Portfolio(cash=D("1000")), CostConfig(), uni)
    b.begin_session(entry, {"AAA": D("100")})
    b.submit(OrderRequest(symbol="AAA", side=Side.BUY, notional=D("500"), client_order_id="b"))
    st = State()
    st.positions_meta = {
        "AAA": PositionMeta(entry_session=entry, entry_price=D("100"), entry_atr=D("2"))
    }  # stop at 95
    idx = pd.DatetimeIndex([pd.Timestamp(entry), pd.Timestamp(D1)])
    df = pd.DataFrame(
        {"open": [100, at_945], "high": [101, 101], "low": [99, low], "close": [100, 98]},
        index=idx,
    )
    b.begin_session(D1, {"AAA": D(str(at_945))})
    hit = _broker_stops(b, st, {"AAA": df}, make_settings(), D1, {"AAA": D(str(at_945))}, None)
    return b, st, hit


def test_a_stop_fills_at_the_stop_when_the_low_reaches_it() -> None:
    b, st, hit = stop_case(low=94.0, at_945=99.0)
    assert hit == 1 and b.portfolio.qty("AAA") == 0 and "AAA" not in st.positions_meta
    assert b.closed_trades[-1].reason == "broker_stop"
    assert b.fills[-1].price <= D("95")


def test_a_stop_already_crossed_at_945_fills_there_and_a_safe_day_keeps_the_position() -> None:
    b, _, hit = stop_case(low=90.0, at_945=92.0)
    assert hit == 1 and b.fills[-1].price <= D("92")
    b, _, hit = stop_case(low=96.0, at_945=99.0)
    assert hit == 0 and b.portfolio.qty("AAA") > 0


def test_a_position_bought_today_has_no_stop() -> None:
    uni = Universe.assume_all_fractionable(("AAA",), "SPY")
    b = SimBroker(Portfolio(cash=D("1000")), CostConfig(), uni)
    b.begin_session(D1, {"AAA": D("100")})
    b.submit(OrderRequest(symbol="AAA", side=Side.BUY, notional=D("500"), client_order_id="b"))
    st = State()
    st.positions_meta = {
        "AAA": PositionMeta(entry_session=D1, entry_price=D("100"), entry_atr=D("2"))
    }
    df = pd.DataFrame(
        {"open": [100], "high": [100], "low": [50], "close": [60]},
        index=pd.DatetimeIndex([pd.Timestamp(D1)]),
    )
    assert _broker_stops(b, st, {"AAA": df}, make_settings(), D1, {}, None) == 0


# ---- trial 13's registration ------------------------------------------------------------------


def test_trial_13_is_registered_before_it_can_run(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry.jsonl")
    with pytest.raises(RuntimeError, match="not pre-registered"):
        trial13.run_trial13(make_settings(), pd.DataFrame(), lambda d, s: {}, reg, tmp_path)
    trial13.preregister(reg)
    trial13.preregister(reg)  # once only
    entries = reg.load()
    assert len(entries) == 1 and entries[0].kind == "preregistration"
    assert entries[0].config_hash == trial13.spec_hash() and trial13.preregistered(reg)
    spec = entries[0].params
    for piece in ("a_volatility_size", "b_position_cap", "c_exposure_cap", "d_drawdown_brake"):
        assert spec["overlay"][piece]["reason"]
    assert "in-sample" in spec["in_sample"] and sc.spec_hash() in spec["same_as_trial_12"]


def test_trial_12_is_untouched() -> None:
    assert sc.spec_hash() == "e75fa8775469de92"


def test_decisions_md_records_trial_13() -> None:
    text = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    sec = text[text.index("## Trial 13") :]
    assert trial13.spec_hash() in sec and "in-sample" in sec


def test_trial_13_reads_nothing_after_the_test_window() -> None:
    assert trial13.t9.TEST_END < trial13.sealed.HOLDOUT_START
