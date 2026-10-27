from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from stoa.broker.base import OrderRequest
from stoa.broker.sweep import SweepBroker
from stoa.config import CostConfig
from stoa.models import Side
from stoa.newsbrain import sealed
from stoa.newsbrain import spycore as sc
from stoa.portfolio import Portfolio
from stoa.universe import Universe

D1, D2 = date(2024, 3, 11), date(2024, 3, 12)
SPY_BPS, PICK_BPS = 5.0, 40.0


def broker(cash: str = "100") -> SweepBroker:
    uni = Universe.assume_all_fractionable(("AAA", "BBB", "SPY"), "SPY")
    b = SweepBroker(
        Portfolio(cash=Decimal(cash)),
        CostConfig(),
        uni,
        lambda sym, d: SPY_BPS if sym == "SPY" else PICK_BPS,
        symbol="SPY",
        reserve_pct=Decimal("0.02"),
    )
    b.begin_session(D1, {"SPY": Decimal("500"), "AAA": Decimal("20"), "BBB": Decimal("10")})
    return b


def test_idle_cash_above_the_reserve_goes_into_spy_at_its_own_cost() -> None:
    b = broker()
    b.sweep(kill_switch=False)
    spy = b.portfolio.qty("SPY")
    assert spy > 0
    fill = b.stats.fills[-1]
    assert fill.price == Decimal("500") * (1 + Decimal(SPY_BPS) / 10_000)  # SPY's own 5 bps
    assert b.portfolio.cash == pytest.approx(Decimal("2"), abs=Decimal("0.01"))  # the 2% reserve
    assert b.positions() == {}  # SPY is idle money, never a position
    acct = b.account()
    assert acct.cash == pytest.approx(acct.equity, rel=Decimal("0.001"))
    assert b.closed_trades == [] and b.stats.full_sessions == 1


def test_a_buy_is_funded_by_selling_spy_at_the_same_session_price_with_the_sell_fee() -> None:
    b = broker()
    b.sweep(kill_switch=False)  # SPY bought at the end of the first session
    b.begin_session(D2, {"SPY": Decimal("500"), "AAA": Decimal("20"), "BBB": Decimal("10")})
    held = b.portfolio.qty("SPY")
    st = b.submit(
        OrderRequest(symbol="AAA", side=Side.BUY, notional=Decimal("20"), client_order_id="a1")
    )
    assert st.status == "filled"
    sell = b.stats.fills[-1]
    assert sell.side is Side.SELL and sell.session == D2
    assert sell.price == Decimal("500") * (1 - Decimal(SPY_BPS) / 10_000)
    assert sell.fees >= Decimal("0.01")  # the sell fee, minimum one cent
    assert b.portfolio.qty("SPY") < held and b.portfolio.qty("AAA") > 0
    assert b.closed_trades == []  # the SPY sale is a swap, not a trade
    assert b.stats.cost > 0 and set(b.positions()) == {"AAA"}


def test_a_pick_exit_goes_back_to_spy_and_counts_as_one_trade() -> None:
    b = broker()
    b.submit(
        OrderRequest(symbol="AAA", side=Side.BUY, notional=Decimal("20"), client_order_id="a1")
    )
    b.sweep(kill_switch=False)
    b.begin_session(D2, {"SPY": Decimal("505"), "AAA": Decimal("22"), "BBB": Decimal("10")})
    qty = b.portfolio.qty("AAA")
    b.submit(OrderRequest(symbol="AAA", side=Side.SELL, qty=qty, client_order_id="a2"))
    before = b.portfolio.qty("SPY")
    b.sweep(kill_switch=False)
    assert b.portfolio.qty("SPY") > before
    assert [t.symbol for t in b.closed_trades] == ["AAA"]


def test_the_kill_switch_sells_spy_and_stops_the_sweep() -> None:
    b = broker()
    b.sweep(kill_switch=False)
    b.begin_session(D2, {"SPY": Decimal("480"), "AAA": Decimal("20"), "BBB": Decimal("10")})
    b.sweep(kill_switch=True)
    assert b.portfolio.qty("SPY") == 0 and b.portfolio.cash > Decimal("95")  # SPY fell 4%
    b.sweep(kill_switch=True)
    assert b.portfolio.qty("SPY") == 0


def test_spy_is_never_a_pick() -> None:
    b = broker()
    st = b.submit(
        OrderRequest(symbol="SPY", side=Side.BUY, notional=Decimal("5"), client_order_id="s1")
    )
    assert st.status == "rejected"
    sigs = sc.signals(
        pd.DataFrame(
            {
                "symbol": ["SPY", "AAA"],
                "sidx": [1, 1],
                "session": [D2, D2],
                "sent": 0.1,
                "surprise": 0.0,
                "n": 1,
                "filing_tone": 0.0,
                "ripple": 0.0,
            }
        ),
        [0.9, 0.8],
        1,
    )
    assert [s.symbol for s in sigs] == ["AAA"]


def test_cash_fallback_and_spy_prices_every_session() -> None:
    seen: list[set[str]] = []

    def at_945(d: date, syms: set[str]) -> dict[str, Decimal]:
        seen.append(set(syms))
        return {s: Decimal("1") for s in syms}

    sc.core_fills(at_945)(D1, {"AAA"}, set())
    assert seen[-1] == {"AAA", "SPY"}
    # Without the core the account is trial 9's: a plain sim broker, idle money in cash.
    from stoa.broker.sim import SimBroker

    plain = SimBroker(
        Portfolio(cash=Decimal("100")),
        CostConfig(),
        Universe.assume_all_fractionable(("AAA",), "SPY"),
    )
    assert plain.account().cash == Decimal("100") and not isinstance(plain, SweepBroker)


def test_no_holdout_rows_are_read() -> None:
    rows = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "session": [date(2022, 6, 1), date(2025, 6, 2), date(2026, 8, 3)],
        }
    )
    got = sc.test_rows(rows)
    assert list(got["symbol"]) == ["B"]
    assert all(d < sealed.HOLDOUT_START for d in got["session"])
    with pytest.raises(sealed.HoldoutBreach):
        sealed.check(date(2026, 8, 3), "a trial 12 read")


def test_spec_is_stable_and_states_the_known_limits() -> None:
    spec = sc.spec()
    assert spec["graded_horizon"] == 1 and spec["reported_horizons"] == [1, 5, 20]
    assert "lead, not proof" in spec["known_limits"] and "STARVED" in spec["starved"]
    assert sc.spec_hash() == sc.spec_hash() and len(sc.spec_hash()) == 16


def test_no_same_session_round_trip_in_spy() -> None:
    b = broker()
    b.sweep(kill_switch=False)
    b.begin_session(D2, {"SPY": Decimal("500"), "AAA": Decimal("20"), "BBB": Decimal("10")})
    b.submit(
        OrderRequest(symbol="AAA", side=Side.BUY, notional=Decimal("30"), client_order_id="a1")
    )
    swaps = b.stats.swaps
    b.sweep(kill_switch=False)  # SPY was sold today: spare cash waits for tomorrow
    assert b.stats.swaps == swaps
    # And a buy on the session SPY was bought cannot sell it: that buy waits for cash.
    c = broker("100")
    c.sweep(kill_switch=False)
    st = c.submit(
        OrderRequest(symbol="BBB", side=Side.BUY, notional=Decimal("50"), client_order_id="b1")
    )
    assert st.status == "rejected" and c.portfolio.qty("BBB") == 0


def test_decisions_md_carries_trial_12s_spec_hash() -> None:
    from tests.conftest import ROOT

    text = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    sec = text[text.index("## Trial 12") :]
    assert sc.spec_hash() in sec and "lead, not proof" in sec
