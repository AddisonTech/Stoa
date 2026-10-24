from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stoa.backtest import run_backtest
from stoa.config import Settings
from stoa.data.base import Bars, validate_bars
from stoa.models import Side
from stoa.roles.scout import RuleScout
from stoa.strategy import AgentStrategy
from tests.conftest import make_settings

SYMS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "XOM", "KO"]


def crash_bars(crash_on: date) -> Bars:
    """Steady uptrend, then every stock gaps down 6% a day for a week."""
    idx = pd.bdate_range("2017-01-02", "2019-12-31")
    idx = idx[
        ~idx.isin(
            pd.to_datetime(["2017-12-25", "2018-12-25", "2019-12-25", "2018-01-01", "2019-01-01"])
        )
    ]
    out: Bars = {}
    for k, sym in enumerate(SYMS):
        rng = np.random.default_rng(k)
        rets = 0.0012 + rng.normal(0, 0.004, len(idx))
        crash = (idx >= pd.Timestamp(crash_on)) & (
            idx < pd.Timestamp(crash_on) + pd.Timedelta(days=8)
        )
        rets[crash] = -0.06
        close = 100 * np.exp(np.cumsum(rets))
        open_ = np.concatenate(([close[0]], close[:-1])) * np.where(crash, 0.97, 1.0)
        df = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.002,
                "low": np.minimum(open_, close) * 0.998,
                "close": close,
                "volume": np.full(len(idx), 1e7),
            },
            index=idx,
        )
        out[sym] = validate_bars(sym, df)
    return out


def test_kill_switch_trips_flattens_and_halts() -> None:
    settings: Settings = make_settings(
        watchlist=",".join(SYMS),
        exits__trailing_enabled="false",
        exits__stop_atr_multiple="50",
        exits__trend_break_sessions="50",
        exits__max_hold_sessions="500",
        risk__daily_loss_limit_pct="0.5",
        quant__max_volatility_20="5",
        quant__max_rsi_14="100",
        quant__max_return_5="1",
        quant__min_dollar_volume_20="0",
    )
    bars = crash_bars(date(2019, 6, 3))
    strat = AgentStrategy(settings, RuleScout(settings.scout))
    res = run_backtest(settings, bars, strat, date(2019, 1, 2), date(2019, 12, 31))
    assert res.kill_switch_trips == 1
    trip_sells = [f for f in res.fills if f.side is Side.SELL]
    assert trip_sells, "kill switch should flatten"
    last_sell = max(f.session for f in trip_sells)
    assert not [f for f in res.fills if f.side is Side.BUY and f.session > last_sell]
    assert res.final_positions == {}
    mdd = float((res.equity / res.equity.cummax() - 1).min())
    assert mdd < -0.25
