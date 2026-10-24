from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stoa.backtest import run_backtest
from stoa.baselines import BuyAndHold, SMACrossover
from stoa.broker.sim import buy_fill_price
from stoa.clock import next_session, sessions_between
from stoa.config import Settings
from stoa.data.base import Bars, DataError
from stoa.features import build_features
from stoa.models import Side, dec

START = date(2019, 1, 1)
END = date(2020, 12, 31)


def test_spy_buy_and_hold_equals_price_ratio_minus_costs(
    small_settings: Settings, bars: Bars
) -> None:
    res = run_backtest(small_settings, bars, BuyAndHold("SPY"), START, END)
    spy = bars["SPY"]
    first = sessions_between(START, END)[0]
    open0 = dec(float(spy.at[pd.Timestamp(first), "open"]))
    close_n = float(spy["close"].loc[: pd.Timestamp(END)].iloc[-1])
    price = buy_fill_price(open0, small_settings.costs.slippage_bps)
    qty = res.fills[0].qty
    assert len(res.fills) == 1 and res.fills[0].side is Side.BUY
    assert res.fills[0].price == price
    expected = float(res.final_cash) + float(qty) * close_n
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)
    # Before rounding, this is 100 * close/open / (1 + slippage).
    ideal = 100 * close_n / float(open0) / (1 + small_settings.costs.slippage_bps / 1e4)
    assert res.equity.iloc[-1] == pytest.approx(ideal, abs=0.02)
    assert res.trades == []


def test_sma_rule_switches_on_crossover(small_settings: Settings, bars: Bars) -> None:
    res = run_backtest(small_settings, bars, SMACrossover("SPY"), START, END)
    feats = build_features(bars, "SPY")["SPY"]
    days = sessions_between(START, END)
    signal = {
        d: feats.at[pd.Timestamp(d), "sma_50"] > feats.at[pd.Timestamp(d), "sma_200"] for d in days
    }
    prev = sessions_between(date(2018, 12, 1), START)[-1]
    signal[prev] = feats.at[pd.Timestamp(prev), "sma_50"] > feats.at[pd.Timestamp(prev), "sma_200"]
    held = False
    expected = []
    for d in [prev, *days[:-1]]:
        if signal[d] and not held:
            expected.append((next_session(d), Side.BUY))
            held = True
        elif not signal[d] and held:
            expected.append((next_session(d), Side.SELL))
            held = False
    got = [(f.session, f.side) for f in res.fills]
    assert got == expected
    assert len(got) >= 2


def test_first_fill_is_first_window_session(small_settings: Settings, bars: Bars) -> None:
    res = run_backtest(small_settings, bars, BuyAndHold("SPY"), START, END)
    assert res.fills[0].session == sessions_between(START, END)[0]
    assert res.equity.index[0] < pd.Timestamp(sessions_between(START, END)[0])


def test_needs_warmup(small_settings: Settings, bars: Bars) -> None:
    with pytest.raises(DataError, match="warmup"):
        run_backtest(small_settings, bars, BuyAndHold("SPY"), date(2016, 6, 1), date(2016, 12, 31))
