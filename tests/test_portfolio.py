from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stoa.models import Fill, Side
from stoa.portfolio import AccountingError, Portfolio

D = Decimal
T0 = date(2024, 1, 2)
T1 = date(2024, 1, 3)


def fill(
    side: Side, qty: str, price: str, fees: str = "0", day: date = T0, sym: str = "AAPL"
) -> Fill:
    return Fill(
        symbol=sym,
        side=side,
        qty=D(qty),
        price=D(price),
        fees=D(fees),
        session=day,
        client_order_id="x",
    )


def test_buy_then_sell_exact() -> None:
    p = Portfolio(cash=D("100"))
    p.apply_fill(fill(Side.BUY, "0.123456789", "150.25"))
    assert p.cash == D("100") - D("0.123456789") * D("150.25")
    assert p.qty("AAPL") == D("0.123456789")
    closed = p.apply_fill(fill(Side.SELL, "0.123456789", "160", "0.01", day=T1))
    assert closed is not None
    assert closed.pnl == D("0.123456789") * (D("160") - D("150.25")) - D("0.01")
    assert p.realized_pnl == closed.pnl
    assert p.cash == D("100") + closed.pnl
    assert p.positions == {}


def test_equity_is_cash_plus_market_value() -> None:
    p = Portfolio(cash=D("100"))
    p.apply_fill(fill(Side.BUY, "0.5", "100"))
    assert p.equity({"AAPL": D("110")}) == p.cash + D("0.5") * D("110")
    with pytest.raises(AccountingError):
        p.equity({})


def test_cash_never_negative() -> None:
    p = Portfolio(cash=D("10"))
    with pytest.raises(AccountingError):
        p.apply_fill(fill(Side.BUY, "1", "10.01"))
    assert p.cash == D("10")
    with pytest.raises(AccountingError):
        Portfolio(cash=D("-1"))


def test_no_shorting() -> None:
    p = Portfolio(cash=D("100"))
    with pytest.raises(AccountingError):
        p.apply_fill(fill(Side.SELL, "1", "10"))
    p.apply_fill(fill(Side.BUY, "1", "10"))
    with pytest.raises(AccountingError):
        p.apply_fill(fill(Side.SELL, "1.000000001", "10", day=T1))


def test_same_day_round_trip_blocked_both_ways() -> None:
    p = Portfolio(cash=D("100"))
    p.apply_fill(fill(Side.BUY, "1", "10"))
    with pytest.raises(AccountingError, match="round trip"):
        p.apply_fill(fill(Side.SELL, "1", "10"))
    p.apply_fill(fill(Side.SELL, "1", "10", day=T1))
    with pytest.raises(AccountingError, match="round trip"):
        p.apply_fill(fill(Side.BUY, "1", "10", day=T1))


def test_average_cost_and_partial_sells() -> None:
    p = Portfolio(cash=D("1000"))
    p.apply_fill(fill(Side.BUY, "1", "100"))
    p.apply_fill(fill(Side.BUY, "1", "110", day=T0))
    assert p.positions["AAPL"].avg_cost == D("105")
    assert p.apply_fill(fill(Side.SELL, "0.5", "120", day=T1)) is None
    closed = p.apply_fill(fill(Side.SELL, "1.5", "90", day=T1))
    assert closed is not None
    assert closed.qty == D("2")
    assert closed.pnl == D("0.5") * D("15") + D("1.5") * D("-15")
    assert closed.exit_price == (D("0.5") * D("120") + D("1.5") * D("90")) / D("2")


def test_buy_fees_go_into_cost_basis() -> None:
    p = Portfolio(cash=D("100"))
    p.apply_fill(fill(Side.BUY, "1", "10", "0.10"))
    assert p.positions["AAPL"].avg_cost == D("10.10")
    assert p.fees_paid == D("0.10")


def test_invalid_fill_rejected() -> None:
    p = Portfolio(cash=D("100"))
    with pytest.raises(AccountingError):
        p.apply_fill(fill(Side.BUY, "0", "10"))
