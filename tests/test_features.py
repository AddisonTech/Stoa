from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stoa.data.base import Bars
from stoa.features import FEATURE_COLUMNS, build_features, regime_is_bull, rsi, snapshot, true_range


def test_feature_columns_present(bars: Bars) -> None:
    feats = build_features(bars, "SPY")
    assert set(FEATURE_COLUMNS) <= set(feats["AAPL"].columns)


def test_returns_and_sma_match_hand_calc(bars: Bars) -> None:
    feats = build_features(bars, "SPY")
    close = bars["AAPL"]["close"]
    f = feats["AAPL"]
    t = 300
    assert f["ret_20"].iloc[t] == pytest.approx(close.iloc[t] / close.iloc[t - 20] - 1)
    assert f["sma_50"].iloc[t] == pytest.approx(close.iloc[t - 49 : t + 1].mean())
    assert f["dist_sma_200"].iloc[t] == pytest.approx(
        close.iloc[t] / close.iloc[t - 199 : t + 1].mean() - 1
    )
    assert math.isnan(f["sma_200"].iloc[198])
    assert not math.isnan(f["sma_200"].iloc[199])


def test_atr_matches_hand_calc(bars: Bars) -> None:
    df = bars["MSFT"]
    t = 100
    prev = df["close"].iloc[t - 14 : t].to_numpy()
    hi = df["high"].iloc[t - 13 : t + 1].to_numpy()
    lo = df["low"].iloc[t - 13 : t + 1].to_numpy()
    tr = np.maximum.reduce([hi - lo, np.abs(hi - prev), np.abs(lo - prev)])
    assert build_features(bars, "SPY")["MSFT"]["atr_14"].iloc[t] == pytest.approx(tr.mean())
    assert true_range(df).iloc[0] == pytest.approx(df["high"].iloc[0] - df["low"].iloc[0])


def test_rsi_bounds_and_edge_cases() -> None:
    up = pd.Series(np.arange(1.0, 40.0))
    down = pd.Series(np.arange(40.0, 1.0, -1))
    flat = pd.Series(np.full(30, 5.0))
    assert rsi(up).iloc[-1] == 100.0
    assert rsi(down).iloc[-1] == 0.0
    assert rsi(flat).iloc[-1] == 50.0
    assert math.isnan(rsi(up).iloc[13])


def test_relative_strength_vs_spy(bars: Bars) -> None:
    f = build_features(bars, "SPY")
    spy = f["SPY"]
    assert spy["rs_spy_60"].dropna().abs().max() < 1e-12
    t = 400
    a = f["AAPL"]
    assert a["rs_spy_60"].iloc[t] == pytest.approx(
        (1 + a["ret_60"].iloc[t]) / (1 + spy["ret_60"].iloc[t]) - 1
    )


def test_regime_flag(bars: Bars) -> None:
    f = build_features(bars, "SPY")
    spy = f["SPY"]
    t = spy.index[500]
    expected = spy.at[t, "close"] > spy.at[t, "sma_200"]
    assert regime_is_bull(f, "SPY", t.date()) == expected
    assert regime_is_bull(f, "SPY", spy.index[10].date()) is False
    assert regime_is_bull(f, "SPY", date(2016, 1, 1)) is False


def test_features_independent_of_history_start(bars: Bars) -> None:
    """A live run fetches ~450 days; features must equal those from a long history."""
    full = build_features(bars, "SPY")
    cut = {s: df.iloc[-450:] for s, df in bars.items()}
    short = build_features(cut, "SPY")
    day = bars["SPY"].index[-1]
    pd.testing.assert_series_equal(full["AAPL"].loc[day], short["AAPL"].loc[day], check_names=False)


def test_snapshot_rows(bars: Bars) -> None:
    f = build_features(bars, "SPY")
    day = bars["SPY"].index[-1].date()
    snap = snapshot(f, day)
    assert list(snap.columns) == list(FEATURE_COLUMNS)
    assert set(snap.index) == set(bars)
