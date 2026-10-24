from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from stoa.backtest import run_backtest
from stoa.broker.sim import buy_fill_price, sell_fees, sell_fill_price
from stoa.config import Settings
from stoa.data.base import Bars
from stoa.journal import Journal
from stoa.models import Side, dec
from stoa.roles.scout import RuleScout
from stoa.strategy import AgentStrategy

START = date(2019, 1, 1)
END = date(2020, 12, 31)


@pytest.fixture(scope="module")
def agent_result(small_settings: Settings, bars: Bars):  # type: ignore[no-untyped-def]
    strat = AgentStrategy(small_settings, RuleScout(small_settings.scout))
    return run_backtest(small_settings, bars, strat, START, END, journal=Journal(None))


def test_accounting_invariants_full_run(agent_result, small_settings: Settings, bars: Bars) -> None:  # type: ignore[no-untyped-def]
    res = agent_result
    assert res.trades, "expected the synthetic run to trade"
    cash = res.starting_equity
    by_day: dict[date, dict[str, set[Side]]] = {}
    for f in res.fills:
        sides = by_day.setdefault(f.session, {}).setdefault(f.symbol, set())
        sides.add(f.side)
        assert len(sides) == 1, f"round trip in {f.symbol} on {f.session}"
        if f.side is Side.BUY:
            assert f.fees == 0
            cash -= f.gross
        else:
            assert f.fees == sell_fees(f.gross, small_settings.costs)
            cash += f.gross - f.fees
        assert cash >= 0
    assert cash == res.final_cash
    last = res.end
    closes = {s: dec(float(bars[s].at[pd.Timestamp(last), "close"])) for s in res.final_positions}
    equity = res.final_cash + sum(q * closes[s] for s, q in res.final_positions.items())
    assert float(equity) == pytest.approx(res.equity.iloc[-1], rel=1e-12)
    assert any(f.qty != f.qty.to_integral_value() for f in res.fills)


def test_fill_prices_include_slippage_once(
    agent_result, small_settings: Settings, bars: Bars
) -> None:  # type: ignore[no-untyped-def]
    for f in agent_result.fills:
        ref = dec(float(bars[f.symbol].at[pd.Timestamp(f.session), "open"]))
        if f.side is Side.BUY:
            assert f.price == buy_fill_price(ref, small_settings.costs.slippage_bps)
        else:
            assert f.price == sell_fill_price(ref, small_settings.costs.slippage_bps)


def test_closed_trade_pnl_reconciles(agent_result, bars: Bars) -> None:  # type: ignore[no-untyped-def]
    """Equity change = realized P&L on closed trades + unrealized P&L on open lots."""
    res = agent_result
    realized = sum((t.pnl for t in res.trades), Decimal(0))
    buys: dict[str, list] = {}
    for f in res.fills:
        buys.setdefault(f.symbol, []).append(f)
    unrealized = Decimal(0)
    for sym, qty in res.final_positions.items():
        last_buys = [f for f in buys[sym] if f.side is Side.BUY]
        basis = last_buys[-1].price  # one entry per position in the agent strategy
        close = dec(float(bars[sym].at[pd.Timestamp(res.end), "close"]))
        unrealized += qty * (close - basis)
    change = dec(res.equity.iloc[-1]) - res.starting_equity
    assert float(realized + unrealized) == pytest.approx(float(change), abs=1e-9)


def test_risk_limits_hold_through_run(agent_result, small_settings: Settings) -> None:  # type: ignore[no-untyped-def]
    res = agent_result
    assert res.exposure.max() <= float(small_settings.risk.max_gross_exposure_pct) + 1e-9
    per_day: dict[date, int] = {}
    for f in res.fills:
        per_day[f.session] = per_day.get(f.session, 0) + 1
    assert max(per_day.values()) <= small_settings.risk.max_orders_per_day


def test_deterministic(small_settings: Settings, bars: Bars, agent_result) -> None:  # type: ignore[no-untyped-def]
    again = run_backtest(
        small_settings,
        bars,
        AgentStrategy(small_settings, RuleScout(small_settings.scout)),
        START,
        END,
    )
    pd.testing.assert_series_equal(again.equity, agent_result.equity)
