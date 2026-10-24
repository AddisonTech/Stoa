"""The training table: one row per (session, symbol), features and a label, no lookahead.

Label (``model.horizon`` = H sessions, default 10):
    y = 1 if the stock's return from the open of t+1 to the close of t+H beats
        SPY's return over exactly the same span, else 0.
    The open of t+1 is the execution price of a decision made at the close of t.
    A label is known only after the close of t+H. Rows whose label is not yet
    known on a given cutoff are never used for training at that cutoff.

Features at t use bars up to the close of t, fundamentals and text scores whose
``known_at`` is at or before the close of t (through :mod:`stoa.pit`), and
nothing else. Every feature is standardized cross-sectionally per session
(a z-score across the universe that day, clipped to +-3), so the model learns
rankings, not price levels or market regimes.

Universe on t: point-in-time index members on t (plus the watchlist when
``strategy.trade_universe`` is ``watchlist``) that have at least 260 bars, a
close above ``model.min_price`` and 20 day dollar volume above
``model.min_dollar_volume_20``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from stoa import pit
from stoa.data.base import Bars
from stoa.data.sec import QUALITY_COLUMNS
from stoa.features import build_features
from stoa.membership import Membership

TEXT_COLUMNS = (
    "t_sentiment",
    "t_guidance",
    "t_surprise",
    "t_risk",
    "t_materiality",
    "t_count",
)
PRICE_COLUMNS = (
    "f_mom_12_1",
    "f_mom_6_1",
    "f_mom_3",
    "f_rev_1m",
    "f_trend_200",
    "f_trend_50",
    "f_vol_60",
    "f_dist_52w",
    "f_rs_spy_60",
    "f_beta_120",
)
MIN_HISTORY = 260
CLIP = 3.0


@dataclass(frozen=True)
class FeatureSet:
    price: bool = True
    quality: bool = True
    text: bool = False
    price_columns: tuple[str, ...] | None = None  # a subset of PRICE_COLUMNS; None means all
    normalize: str = "zscore"  # zscore or rank, across stocks on each date

    def __post_init__(self) -> None:
        unknown = sorted(set(self.price_columns or ()) - set(PRICE_COLUMNS))
        if unknown:
            raise ValueError(f"unknown price features {unknown}; known: {list(PRICE_COLUMNS)}")
        if self.normalize not in NORMALIZERS:
            raise ValueError(f"unknown normalization {self.normalize!r}")

    @classmethod
    def from_columns(cls, columns: Iterable[str], normalize: str = "zscore") -> FeatureSet:
        """The feature set a trained model was fit on, from its column list."""
        cols = list(columns)
        price = tuple(c for c in cols if c in PRICE_COLUMNS)
        return cls(
            quality="q_roe" in cols,
            text="t_sentiment" in cols,
            price_columns=None if set(price) == set(PRICE_COLUMNS) else price,
            normalize=normalize,
        )

    def columns(self) -> list[str]:
        cols = list(self.price_columns or PRICE_COLUMNS)
        if self.quality:
            cols += list(QUALITY_COLUMNS)
        if self.text:
            cols += list(TEXT_COLUMNS)
        return cols

    @property
    def name(self) -> str:
        price = "price"
        if self.price_columns is not None:
            price += ":" + ",".join(c.removeprefix("f_") for c in self.price_columns)
        parts = [price] + (["quality"] if self.quality else []) + (["text"] if self.text else [])
        if self.normalize != "zscore":
            parts.append(self.normalize)
        return "+".join(parts)


def _wide(bars: Bars, column: str) -> pd.DataFrame:
    return pd.DataFrame({s: df[column] for s, df in bars.items()}).sort_index()


def price_features(bars: Bars, benchmark: str) -> pd.DataFrame:
    """Raw (unstandardized) price features, long format indexed by (date, symbol)."""
    close = _wide(bars, "close")
    rets = close.pct_change(fill_method=None)
    bench = rets[benchmark]
    f: dict[str, pd.DataFrame] = {}
    f["f_mom_12_1"] = close.shift(21) / close.shift(252) - 1.0
    f["f_mom_6_1"] = close.shift(21) / close.shift(126) - 1.0
    f["f_mom_3"] = close / close.shift(63) - 1.0
    f["f_rev_1m"] = -(close / close.shift(21) - 1.0)
    f["f_trend_200"] = close / close.rolling(200, min_periods=200).mean() - 1.0
    f["f_trend_50"] = close / close.rolling(50, min_periods=50).mean() - 1.0
    f["f_vol_60"] = -rets.rolling(60, min_periods=60).std(ddof=1) * np.sqrt(252)
    f["f_dist_52w"] = close / close.rolling(252, min_periods=252).max() - 1.0
    bench_60 = (1 + bench).rolling(60, min_periods=60).apply(np.prod, raw=True) - 1
    stock_60 = close / close.shift(60) - 1.0
    f["f_rs_spy_60"] = stock_60.sub(bench_60, axis=0)
    cov = rets.rolling(120, min_periods=120).cov(bench)
    var = bench.rolling(120, min_periods=120).var()
    f["f_beta_120"] = -cov.div(var, axis=0)
    long = pd.concat({k: v.stack(future_stack=True) for k, v in f.items()}, axis=1)
    long.index = long.index.set_names(["date", "symbol"])
    return long


def labels(bars: Bars, benchmark: str, horizon: int) -> pd.DataFrame:
    """Excess return from the next open to the close ``horizon`` sessions later, and y.

    Also returns ``label_end``: the session whose close completes the label.
    All symbols are aligned on the benchmark's calendar.
    """
    idx = bars[benchmark].index
    opens = _wide(bars, "open").reindex(idx)
    close = _wide(bars, "close").reindex(idx)
    # A stock that stops trading (delisted, acquired, bankrupt) inside the horizon
    # is exited at its last close. Without this, the final weeks of every dead
    # stock would have no label and drop out of training: survivorship bias.
    alive_until = close.apply(lambda c: c.last_valid_index())
    dead_fill = close.ffill()
    for sym, last in alive_until.items():
        if pd.notna(last):  # a symbol with no close on this calendar has nothing to carry
            dead_fill.loc[dead_fill.index > last, sym] = close.at[last, sym]
    entry = opens.shift(-1)
    exit_ = dead_fill.shift(-horizon)
    ret = exit_ / entry - 1.0
    excess = ret.sub(ret[benchmark], axis=0)
    end = pd.Series(idx, index=idx).shift(-horizon)
    out = pd.DataFrame(
        {
            "fwd_ret": ret.stack(future_stack=True),
            "fwd_excess": excess.stack(future_stack=True),
        }
    )
    out.index = out.index.set_names(["date", "symbol"])
    out["label_end"] = end.reindex(out.index.get_level_values("date")).to_numpy()
    out["y"] = (out["fwd_excess"] > 0).astype(float)
    out.loc[out["fwd_excess"].isna(), "y"] = np.nan
    return out


def universe_mask(
    bars: Bars,
    membership: Membership | None,
    extra: Iterable[str],
    min_price: float,
    min_dollar_volume: float,
    exclude: Iterable[str] = (),
) -> pd.Series:
    """True where (date, symbol) is tradable on that date."""
    close = _wide(bars, "close")
    dv = (close * _wide(bars, "volume")).rolling(20, min_periods=20).mean()
    count = close.notna().cumsum()
    ok = (count >= MIN_HISTORY) & (close >= min_price) & (dv >= min_dollar_volume)
    extra_set = set(extra)
    excluded = set(exclude)
    if membership is not None:
        member = pd.DataFrame(False, index=close.index, columns=close.columns)
        for iv in membership.intervals:
            if iv.symbol not in member.columns:
                continue
            lo = pd.Timestamp(iv.start)
            hi = pd.Timestamp(iv.end) if iv.end else None
            rows = (member.index >= lo) & ((member.index < hi) if hi is not None else True)
            member.loc[rows, iv.symbol] = True
        for sym in extra_set & set(member.columns):
            member[sym] = True
        ok = ok & member
    for sym in excluded & set(ok.columns):
        ok[sym] = False
    stacked = ok.stack(future_stack=True)
    assert isinstance(stacked, pd.Series)
    stacked.index = stacked.index.set_names(["date", "symbol"])
    return stacked.astype(bool)


def standardize(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cross-sectional z-score per date, clipped to +-3; missing values become 0 (the mean)."""
    out = panel.copy()
    g = out.groupby(level="date")
    for c in columns:
        mean = g[c].transform("mean")
        sd = g[c].transform("std")
        z = (out[c] - mean) / sd.replace(0, np.nan)
        out[c] = z.clip(-CLIP, CLIP).fillna(0.0)
    return out


def rank_normalize(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cross-sectional percentile rank per date, scaled to -1..1; missing values become 0.

    Ties share their average rank. Only the order of stocks on a date matters, so
    one extreme value cannot stretch the scale for everyone else.
    """
    out = panel.copy()
    g = out.groupby(level="date")
    for c in columns:
        rank = g[c].rank(method="average")
        n = g[c].transform("count")
        scaled = 2.0 * (rank - 0.5) / n - 1.0
        out[c] = scaled.where(n > 1, 0.0).fillna(0.0)
    return out


NORMALIZERS = {"zscore": standardize, "rank": rank_normalize}


def _attach_pit(
    panel: pd.DataFrame, records: pd.DataFrame | None, columns: list[str]
) -> pd.DataFrame:
    if records is None or records.empty:
        for c in columns:
            panel[c] = np.nan
        return panel
    sessions = sorted({d.date() for d in panel.index.get_level_values("date")})
    have = [c for c in columns if c in records.columns]
    asof = pit.asof_panel(records, sessions, have)
    for c in columns:
        if c in asof.columns:
            panel[c] = asof[c].reindex(panel.index)
        else:
            panel[c] = np.nan
    return panel


def text_daily(scores: pd.DataFrame | None, lookback_days: int) -> pd.DataFrame | None:
    """Analyst scores rolled into per-item records the as-of join can use.

    Each record carries the mean of the item scores published in the
    ``lookback_days`` calendar days up to and including it, so the as-of value
    at a cutoff summarizes recent text. Items are weighted by materiality
    times confidence.
    """
    if scores is None or scores.empty:
        return None
    s = scores.copy()
    s["known_at"] = pd.to_datetime(s["known_at"], utc=True)
    s = s.sort_values("known_at")
    w = (s["materiality"] * s["confidence"]).clip(lower=0.05)
    s["_w"] = w
    out = []
    window = pd.Timedelta(days=lookback_days)
    for sym, grp in s.groupby("symbol"):
        g = grp.set_index("known_at")
        roll = g.rolling(window)
        wsum = roll["_w"].sum()
        rec = pd.DataFrame(index=g.index)
        for src, dst in (
            ("sentiment", "t_sentiment"),
            ("guidance", "t_guidance"),
            ("surprise", "t_surprise"),
            ("risk", "t_risk"),
        ):
            rec[dst] = (g[src] * g["_w"]).rolling(window).sum() / wsum
        rec["t_materiality"] = roll["materiality"].max()
        rec["t_count"] = roll["_w"].count()
        rec["symbol"] = sym
        out.append(rec.reset_index())
    return pd.concat(out, ignore_index=True)


@dataclass
class Dataset:
    panel: pd.DataFrame  # indexed by (date, symbol); standardized features plus label columns
    features: list[str]
    horizon: int
    sessions: list[date]

    def rows_known_by(self, cutoff_session: date) -> pd.DataFrame:
        """Rows whose label is fully known at the close of ``cutoff_session``."""
        p = self.panel
        known = p["label_end"].notna() & (p["label_end"] <= pd.Timestamp(cutoff_session))
        return p.loc[known & p["y"].notna()]

    def on(self, session: date) -> pd.DataFrame:
        ts = pd.Timestamp(session)
        if ts not in self.panel.index.get_level_values("date"):
            return self.panel.iloc[0:0]
        return cast(pd.DataFrame, self.panel.xs(ts, level="date", drop_level=False))


def build_dataset(
    bars: Bars,
    benchmark: str,
    horizon: int,
    *,
    membership: Membership | None = None,
    extra_symbols: Iterable[str] = (),
    exclude: Iterable[str] = (),
    fundamentals: pd.DataFrame | None = None,
    text_scores: pd.DataFrame | None = None,
    features: FeatureSet | None = None,
    min_price: float = 5.0,
    min_dollar_volume: float = 20_000_000.0,
    text_lookback_days: int = 10,
    start: date | None = None,
) -> Dataset:
    """Assemble the standardized feature panel with labels for every tradable row."""
    features = features or FeatureSet()
    raw = price_features(bars, benchmark)[list(features.price_columns or PRICE_COLUMNS)]
    lab = labels(bars, benchmark, horizon)
    mask = universe_mask(
        bars,
        membership,
        extra_symbols,
        min_price,
        min_dollar_volume,
        exclude=[benchmark, *exclude],
    )
    keep = mask[mask].index
    panel = raw.reindex(keep)
    if start is not None:
        panel = panel.loc[panel.index.get_level_values("date") >= pd.Timestamp(start)]
    if features.quality:
        panel = _attach_pit(panel, fundamentals, list(QUALITY_COLUMNS))
    if features.text:
        panel = _attach_pit(panel, text_daily(text_scores, text_lookback_days), list(TEXT_COLUMNS))
    cols = features.columns()
    panel = NORMALIZERS[features.normalize](panel, cols)
    panel = panel.join(lab, how="left")
    panel = panel.sort_index()
    sessions = sorted({d.date() for d in panel.index.get_level_values("date")})
    return Dataset(panel=panel, features=cols, horizon=horizon, sessions=sessions)


def regime_series(bars: Bars, benchmark: str) -> pd.Series:
    """True on sessions where the benchmark closed above its 200 day average."""
    f = build_features({benchmark: bars[benchmark]}, benchmark)[benchmark]
    return f["regime_bull"] == 1.0
