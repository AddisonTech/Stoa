"""Indicators computed strictly from past data.

Every feature at row t depends only on rows <= t. All smoothing uses fixed
rolling windows (simple averages, not exponential), so a value never depends on
how far back the history starts. That makes a feature computed on a 450 day live
fetch identical to the same feature computed inside a 10 year backtest.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, cast

import numpy as np
import pandas as pd

from stoa.data.base import Bars

TRADING_DAYS = 252

FEATURE_COLUMNS = (
    "close",
    "high",
    "ret_5",
    "ret_20",
    "ret_60",
    "ret_120",
    "sma_20",
    "sma_50",
    "sma_200",
    "dist_sma_20",
    "dist_sma_50",
    "dist_sma_200",
    "atr_14",
    "atr_pct",
    "vol_20",
    "rsi_14",
    "dist_52w_high",
    "dollar_vol_20",
    "rs_spy_60",
    "regime_bull",
)


def cell(df: pd.DataFrame, row: Any, column: str) -> float:
    """One numeric cell as a plain float (NaN stays NaN)."""
    return float(cast(float, df.at[row, column]))


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    tr = ranges.max(axis=1, skipna=False)
    tr.iloc[:1] = (df["high"] - df["low"]).iloc[:1]
    return tr


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI with simple moving averages of gains and losses (Cutler's RSI)."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    out = pd.Series(np.nan, index=close.index, dtype=float)
    both = (gain == 0) & (loss == 0)
    out[loss == 0] = 100.0
    rs = gain / loss.where(loss > 0)
    valid = loss > 0
    out[valid] = 100.0 - 100.0 / (1.0 + rs[valid])
    out[both] = 50.0
    out[gain.isna() | loss.isna()] = np.nan
    return out


def symbol_features(df: pd.DataFrame, benchmark_close: pd.Series) -> pd.DataFrame:
    close = df["close"]
    f = pd.DataFrame(index=df.index)
    f["close"] = close
    f["high"] = df["high"]
    for n in (5, 20, 60, 120):
        f[f"ret_{n}"] = close / close.shift(n) - 1.0
    for n in (20, 50, 200):
        sma = _sma(close, n)
        f[f"sma_{n}"] = sma
        f[f"dist_sma_{n}"] = close / sma - 1.0
    atr = _sma(true_range(df), 14)
    f["atr_14"] = atr
    f["atr_pct"] = atr / close
    log_ret = np.log(close).diff()
    f["vol_20"] = log_ret.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(TRADING_DAYS)
    f["rsi_14"] = rsi(close, 14)
    f["dist_52w_high"] = close / close.rolling(TRADING_DAYS, min_periods=TRADING_DAYS).max() - 1.0
    f["dollar_vol_20"] = _sma(close * df["volume"], 20)
    bench = benchmark_close.reindex(df.index)
    bench_ret_60 = bench / bench.shift(60) - 1.0
    f["rs_spy_60"] = (1.0 + f["ret_60"]) / (1.0 + bench_ret_60) - 1.0
    bench_sma_200 = _sma(benchmark_close, 200).reindex(df.index)
    regime = (bench > bench_sma_200).astype(float)
    regime[bench_sma_200.isna() | bench.isna()] = np.nan
    f["regime_bull"] = regime
    return f


def build_features(bars: Bars, benchmark: str) -> dict[str, pd.DataFrame]:
    """Feature frames for every symbol, indexed like its bars."""
    if benchmark not in bars:
        raise KeyError(f"benchmark {benchmark} missing from bars")
    bench_close = bars[benchmark]["close"]
    return {sym: symbol_features(df, bench_close) for sym, df in bars.items()}


def snapshot(features: Mapping[str, pd.DataFrame], day: date) -> pd.DataFrame:
    """One row per symbol with a bar on ``day``, indexed by symbol."""
    ts = pd.Timestamp(day)
    rows = {sym: f.loc[ts] for sym, f in features.items() if ts in f.index}
    if not rows:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))
    return pd.DataFrame.from_dict(rows, orient="index").loc[:, list(FEATURE_COLUMNS)]


def regime_is_bull(features: Mapping[str, pd.DataFrame], benchmark: str, day: date) -> bool:
    """True when the benchmark closed above its 200 day average. Unknown counts as bearish."""
    f = features.get(benchmark)
    ts = pd.Timestamp(day)
    if f is None or ts not in f.index:
        return False
    return cell(f, ts, "regime_bull") == 1.0


def slice_features(features: Mapping[str, pd.DataFrame], day: date) -> dict[str, pd.DataFrame]:
    cutoff = pd.Timestamp(day)
    return {s: f.loc[:cutoff] for s, f in features.items()}
