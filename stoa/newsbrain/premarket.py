"""Trial 11: the news brain, entering in pre-market (docs/DECISIONS.md, trial 11).

A row is a liquid (symbol, session t) of trial 9's universe with an article known
between 4:00 and 9:25 am New York time on t. The decision is made at the minute
after that first article lands, with the news features as known then; the buy is a
limit order filled from pre-market minute bars at a pre-market cost. The same
signals entering at 9:45 am are replayed beside it for comparison.

Trial 9's model settings, walk-forward protocol, freezing, replay engine, risk
rails, pass bar and holdout wall are reused unchanged; only the rows, the
features' cutoff, the labels and the entry fill are new.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stoa import clock, event_signals, wide
from stoa.backtest import run_backtest
from stoa.config import Settings
from stoa.data.base import Bars
from stoa.event_run import base_bars, source
from stoa.experiments import Registry, deflate, return_moments
from stoa.newsbrain import ROOT, sealed
from stoa.newsbrain import panel as pn
from stoa.newsbrain import trial as t9
from stoa.newsbrain.alpaca_pull import MINUTE_DIR
from stoa.scoreboard import summarize

log = logging.getLogger(__name__)

# ---- the registered design (docs/DECISIONS.md, trial 11); changing any is a new trial ----

SPEC = "trial11:newsbrain:premarket:v1"
WINDOW_OPEN = "04:00"  # articles known from here
WINDOW_CLOSE = "09:25"  # ... until here (exclusive); fills may use minutes up to 9:25
LIMIT_OVER_REFERENCE = 0.01  # buy limit: the last pre-market trade before the decision plus 1%
MIN_MINUTE_DOLLARS = 1_000.0  # a minute must trade at least this much to fill
PREMARKET_MULTIPLE = 2.0
VOLUME_REF_DOLLARS = 50_000.0
VOLUME_FACTOR_MAX = 4.0
CAP_FACTOR = {"small": 1.5, "mid": 1.2, "large": 1.0}
COST_CAP_BPS = 300.0
FILING_CUTOFF = "04:00"  # filing features as known at 4:00 am on t
OUT = ROOT / "trial11"


def spec() -> dict[str, Any]:
    """The pre-registered rules, as written to the registry before any run."""
    return {
        "spec": SPEC,
        "rows": "trial 9 universe rows (liquid at the close of t-1) with an article "
        f"(at most 3 tickers) known from {WINDOW_OPEN} to before {WINDOW_CLOSE} New York time on t",
        "decision_minute": "the first such article's known time, rounded up to the whole minute",
        "features": "trial 9's 13 features; fresh-news features from fresh articles known by the "
        "decision minute, their 60 session baselines as trial 9; filing features as known at "
        f"{FILING_CUTOFF} on t; ripple and bar controls as trial 9",
        "entry": {
            "order": "buy limit",
            "reference": "close of the last minute bar on t before the decision minute, "
            "from 4:00 am; else the close of t-1",
            "limit": f"reference x (1 + {LIMIT_OVER_REFERENCE})",
            "minutes": f"from the decision minute through {WINDOW_CLOSE}",
            "qualifying_minute": f"dollar volume (v x vw) at least ${MIN_MINUTE_DOLLARS:,.0f}"
            " and n >= 1",
            "fill": "at the first qualifying minute whose vw x (1 + cost) is at most the limit, "
            "at vw x (1 + cost); no fill by 9:25 means no trade",
        },
        "cost_bps": {
            "formula": "min(cap, multiple x base x volume_factor x cap_factor)",
            "base": "trial 9: 25 x sqrt($10M / 20 session dollar volume), clipped to 5..75",
            "multiple": PREMARKET_MULTIPLE,
            "volume_factor": f"sqrt(${VOLUME_REF_DOLLARS:,.0f} / max(minute dollar volume, "
            f"${MIN_MINUTE_DOLLARS:,.0f})), clipped to 1..{VOLUME_FACTOR_MAX}",
            "cap_factor": CAP_FACTOR,
            "cap": COST_CAP_BPS,
            "size": "small under $2B market value, mid $2B to $10B, large over $10B (SEC shares "
            "outstanding x reference); else by dollar volume as trial 9",
            "exit": "sells at 9:45 am at trial 9's cost plus the sell fee",
        },
        "labels": "fill to the open of t+h (trial 9's rule if missing), minus SPY from its last "
        "minute close at or before the fill minute (else its open of t) to its open of t+h, "
        "minus the exit cost and the sell fee; 1 if above 0; only rows that filled",
        "model": {"hgb": t9.HGB, "calibration": "isotonic, last quarter, h gap"},
        "walk_forward": "2016-01-01 to 2022-12-31, trial 9's folds; frozen on labels ending by "
        "2022-12-30",
        "test": "2023-01-01 to 2026-06-30, replayed through trial 9's engine and risk rails",
        "signals": f"p >= {t9.THRESHOLD}, taken in order of decision minute (earliest first), "
        f"up to {t9.TOP_N} a session",
        "comparison": "the same signals, same order, buying at 9:45 am at trial 9's cost",
        "graded": f"the {t9.GRADED} session pre-market run, on trial 9's pass bar",
        "logged_trials": "pre-market and 9:45 am runs, each horizon (6 configurations)",
        "report": "by small, mid and large cap and horizon; gain or loss vs the 9:45 am entry",
        "holdout": "2026-07-01 to 2026-10-23 is never read",
    }


def spec_hash() -> str:
    return hashlib.sha256(json.dumps(spec(), sort_keys=True).encode()).hexdigest()[:16]


# ---- cost and fill ------------------------------------------------------------------------


def cost_bps(adv: float | None, minute_dollars: float, bucket: str) -> float:
    base = wide.slippage_bps(adv)
    vol = math.sqrt(VOLUME_REF_DOLLARS / max(minute_dollars, MIN_MINUTE_DOLLARS))
    vol = min(VOLUME_FACTOR_MAX, max(1.0, vol))
    return min(COST_CAP_BPS, PREMARKET_MULTIPLE * base * vol * CAP_FACTOR[bucket])


def decision_minute(known_at: pd.Timestamp) -> pd.Timestamp:
    return known_at.ceil("min")


@dataclass(frozen=True)
class Fill:
    minute: pd.Timestamp  # UTC
    price: float  # including the pre-market cost
    reference: float
    limit: float
    cost_bps: float


def find_fill(
    bars: pd.DataFrame,
    minute: pd.Timestamp,
    prev_close: float,
    adv: float | None,
    bucket: str,
    last_minute: pd.Timestamp,
) -> Fill | None:
    """The limit-order fill from one symbol's minute bars on t (columns t, c, v, vw, n)."""
    before = bars[(bars["t"] < minute)]
    ref = float(before["c"].iloc[-1]) if len(before) else float(prev_close)
    if not (math.isfinite(ref) and ref > 0):
        return None
    limit = ref * (1 + LIMIT_OVER_REFERENCE)
    window = bars[(bars["t"] >= minute) & (bars["t"] <= last_minute)]
    row: Any
    for row in window.itertuples(index=False):
        vw, v, n = float(row.vw), float(row.v), float(row.n)
        dollars = v * vw
        if not (math.isfinite(dollars) and dollars >= MIN_MINUTE_DOLLARS and n >= 1):
            continue
        c = cost_bps(adv, dollars, bucket)
        price = vw * (1 + c / 1e4)
        if price <= limit:
            return Fill(pd.Timestamp(row.t), price, ref, limit, c)
    return None


# ---- features as known at the decision minute ----------------------------------------------


def news_baselines(news: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, session with news, by trial 9's 9:00 am binning): that session's `sent`
    and the mean and spread of `sent` over the last 60 such sessions up to and including
    it. Taken as of the session before t, they are exactly trial 9's baselines for t."""
    news = news.assign(w=np.where(news["first"], 1.0, pn.REWRITE_WEIGHT))
    news["wnet"] = news["w"] * news["net"]
    day = news.groupby(["symbol", "sidx"], sort=True).agg(
        n=("net", "size"), wsum=("w", "sum"), wnet=("wnet", "sum")
    )
    day = day.reset_index()
    day["sent"] = day["wnet"] / day["wsum"]
    roll = day.groupby("symbol", sort=False)["sent"].rolling(
        pn.BASELINE_SESSIONS, min_periods=pn.BASELINE_MIN
    )
    day["mean"] = roll.mean().reset_index(level=0, drop=True)
    day["std"] = roll.std().reset_index(level=0, drop=True)
    return pd.DataFrame(day[["symbol", "sidx", "n", "mean", "std"]])


def triggers(news: pd.DataFrame, days: list[date], cut: np.ndarray) -> pd.DataFrame:
    """Per (symbol, session t) with a pre-market article: the decision minute and the
    fresh-news aggregates as known then (articles known after 9:00 am on t-1 and by it).

    ``news`` is panel.load_news output (sorted by symbol and time) with a ``first`` column.
    """
    pos = {pd.Timestamp(d): i for i, d in enumerate(days)}
    known = pd.to_datetime(news["known_at"], utc=True)
    known_ny = known.dt.tz_convert(clock.NY)
    local_day = known_ny.dt.tz_localize(None).dt.normalize()
    hhmm = known_ny.dt.strftime("%H:%M")
    tidx = local_day.map(pos)
    inside = tidx.notna() & (hhmm >= WINDOW_OPEN) & (hhmm < WINDOW_CLOSE)
    first = (
        news.assign(t=tidx, known=known)[inside]
        .groupby(["symbol", "t"])["known"]
        .min()
        .rename("trigger_at")
        .reset_index()
    )
    first["t"] = first["t"].astype(int)
    first["minute"] = first["trigger_at"].map(decision_minute)
    # An article known by 9:00 am on t is binned to t, one known 9:00 to 9:25 am to t+1.
    cand = pd.concat(
        [news.assign(t=news["sidx"]), news.assign(t=news["sidx"] - 1)], ignore_index=True
    )
    cand["known"] = pd.to_datetime(cand["known_at"], utc=True)
    m = cand.merge(first[["symbol", "t", "minute"]], on=["symbol", "t"], how="inner")
    after = pd.to_datetime(cut[np.maximum(m["t"].to_numpy() - 1, 0)], utc=True)
    m = m[(m["known"] <= m["minute"]) & ((m["t"] == 0) | (m["known"] > after))]
    m = m.assign(w=np.where(m["first"], 1.0, pn.REWRITE_WEIGHT))
    m["wnet"] = m["w"] * m["net"]
    agg = m.groupby(["symbol", "t"]).agg(
        n=("net", "size"), wsum=("w", "sum"), wnet=("wnet", "sum"), first_break=("first", "mean")
    )
    agg = agg.reset_index()
    agg["sent"] = agg["wnet"] / agg["wsum"]
    agg["first_break"] = agg["first_break"].astype("float32")
    out = agg.merge(first[["symbol", "t", "minute"]], on=["symbol", "t"])
    cols = ["symbol", "sidx", "minute", "n", "sent", "first_break"]
    return pd.DataFrame(out.rename(columns={"t": "sidx"})[cols])


def with_baselines(trig: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """surprise, quiet_spike and coverage for each trigger, from sessions before t only."""
    out = []
    for sym, grp in trig.groupby("symbol", sort=False):
        b = base[base["symbol"] == sym].sort_values("sidx")
        g = grp.sort_values("sidx").copy()
        s_idx = b["sidx"].to_numpy()
        j = np.searchsorted(s_idx, g["sidx"].to_numpy(), side="left") - 1  # last session < t
        has = j >= 0
        mean = np.where(has, b["mean"].to_numpy()[np.maximum(j, 0)], np.nan)
        std = np.where(has, b["std"].to_numpy()[np.maximum(j, 0)], np.nan)
        # Average daily articles over the 60 sessions before t, zero-news days included.
        cum = np.concatenate([[0.0], np.cumsum(b["n"].to_numpy(dtype=float))])
        hi = np.searchsorted(s_idx, g["sidx"].to_numpy(), side="left")
        lo = np.searchsorted(s_idx, g["sidx"].to_numpy() - pn.BASELINE_SESSIONS, side="left")
        avg = (cum[hi] - cum[lo]) / pn.BASELINE_SESSIONS
        sur = (g["sent"].to_numpy() - mean) / std
        g["surprise"] = np.where(np.isfinite(sur) & (std > 1e-6), sur, 0.0)
        g["quiet_spike"] = np.log((1 + g["n"].to_numpy()) / (1 + avg))
        g["coverage"] = np.log1p(avg)
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ---- minute bars --------------------------------------------------------------------------


def month_premarket(root: Path, stamp: str, symbols: set[str]) -> pd.DataFrame:
    """Minute bars from 4:00 to 9:30 am New York time for the given symbols in one month."""
    parts = []
    want = sorted(symbols)
    for path in sorted((root / stamp).glob("*.parquet")):
        if pq.ParquetFile(path).metadata.num_rows == 0:
            continue  # an empty group file: no bars that month, and no text type to filter
        df = pd.read_parquet(
            path, columns=["symbol", "t", "c", "v", "vw", "n"], filters=[("symbol", "in", want)]
        )
        if len(df):
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["symbol", "t", "c", "v", "vw", "n"])
    df = pd.concat(parts, ignore_index=True)
    df["symbol"] = df["symbol"].astype(str)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    hhmm = df["t"].dt.tz_convert(clock.NY).dt.strftime("%H:%M")
    df = df[(hhmm >= WINDOW_OPEN) & (hhmm < "09:30")]
    return pd.DataFrame(df.sort_values(["symbol", "t"]).reset_index(drop=True))


class SharesOutstanding:
    """SEC shares outstanding as of a date, parsed once per company."""

    def __init__(self) -> None:
        self._facts: dict[int, tuple[list[str], list[float]]] = {}

    def __call__(self, cik: int, on: date) -> float | None:
        if cik not in self._facts:
            path = Path("data/sec/companyfacts") / f"CIK{cik:010d}.json"
            pairs: list[tuple[str, float]] = []
            if path.exists():
                body = json.loads(path.read_text(encoding="utf-8"))
                facts = body.get("facts", {}).get("dei", {})
                for unit in (
                    facts.get("EntityCommonStockSharesOutstanding", {}).get("units", {}).values()
                ):
                    pairs += [(f["filed"], float(f.get("val", 0))) for f in unit if f.get("filed")]
            pairs.sort()
            self._facts[cik] = ([p[0] for p in pairs], [p[1] for p in pairs])
        filed, vals = self._facts[cik]
        i = bisect.bisect_right(filed, on.isoformat())
        return vals[i - 1] if i else None


# ---- rows, fills and labels ---------------------------------------------------------------


def exit_price(df: pd.DataFrame, i: int, h: int) -> float:
    """Trial 9's exit: the open of t+h, else the last close by then."""
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].ffill().to_numpy(dtype=float)
    j = min(i + h, len(df) - 1)
    if j <= i:
        return math.nan
    o = opens[j] if i + h < len(df) else math.nan
    return float(o) if math.isfinite(o) and o > 0 else float(closes[j])


def build_rows(
    panel: pd.DataFrame, days: list[date], minute_root: Path = MINUTE_DIR
) -> pd.DataFrame:
    """Trial 11's rows: features at the decision minute, the fill and the labels."""
    _, cut = pn.calendar(pn.PANEL_START, pn.PANEL_END)
    news = pn.load_news(cut)
    news = news.assign(first=pn.first_breaks(news))
    trig = with_baselines(triggers(news, days, cut), news_baselines(news))
    del news
    cut4 = np.array(
        [pd.Timestamp(clock.at_ny(d, FILING_CUTOFF)).value for d in days], dtype=np.int64
    )
    fil = pn.filing_features(cut4, pn.load_ciks())
    keep = [
        "symbol",
        "sidx",
        "session",
        "ret_1",
        "ret_5",
        "vol_20",
        "log_adv",
        "adv",
        "slip",
        "prev_close",
        "ripple",
    ]
    rows = trig.merge(panel[keep], on=["symbol", "sidx"], how="inner")
    rows = rows.merge(fil, on=["symbol", "sidx"], how="left")
    for c in ("filing_tone", "ripple"):
        rows[c] = rows[c].fillna(0.0)
    for c in ("guidance", "beat_miss"):
        rows[c] = rows[c].fillna(0).astype("int8")
    rows = rows[rows["session"] < sealed.HOLDOUT_START].reset_index(drop=True)
    log.info("%d pre-market trigger rows", len(rows))

    ciks = pn.load_ciks()
    cik_of = dict(zip(ciks["symbol"], ciks["cik"], strict=True))
    shares = SharesOutstanding()
    spy = pn.bar_frame("SPY", days)
    assert spy is not None
    stamps = rows["session"].map(lambda d: f"{d.year:04d}-{d.month:02d}")
    fills: dict[int, Fill | None] = {}
    spy_ref: dict[int, float] = {}
    buckets: dict[int, str] = {}
    for stamp, grp in rows.groupby(stamps, sort=True):
        if not sealed.month_allowed(str(stamp)):
            continue
        bars = month_premarket(minute_root, str(stamp), set(grp["symbol"]) | {"SPY"})
        by_sym = {str(s): g for s, g in bars.groupby("symbol", sort=False)}
        spy_bars = by_sym.get("SPY", bars.iloc[0:0])
        r: Any
        for r in grp.itertuples():
            sealed.check(r.session, "pre-market minute bars")
            mb = by_sym.get(r.symbol)
            day0 = pd.Timestamp(clock.at_ny(r.session, WINDOW_OPEN))
            last = pd.Timestamp(clock.at_ny(r.session, WINDOW_CLOSE))
            day_bars = mb[(mb["t"] >= day0) & (mb["t"] <= last)] if mb is not None else None
            sh = shares(int(cik_of[r.symbol]), r.session) if r.symbol in cik_of else None
            before = day_bars[day_bars["t"] < r.minute] if day_bars is not None else None
            ref = (
                float(before["c"].iloc[-1]) if before is not None and len(before) else r.prev_close
            )
            bucket = t9.cap_bucket(sh * ref if sh else None, r.adv)  # sized at the decision
            buckets[r.Index] = bucket
            fill = (
                find_fill(day_bars, r.minute, r.prev_close, r.adv, bucket, last)
                if day_bars is not None and len(day_bars)
                else None
            )
            fills[r.Index] = fill
            if fill is not None:
                s_before = spy_bars[(spy_bars["t"] <= fill.minute) & (spy_bars["t"] >= day0)]
                spy_ref[r.Index] = float(s_before["c"].iloc[-1]) if len(s_before) else math.nan
        log.info("pre-market fills %s: %d rows", stamp, len(grp))

    rows["bucket"] = [buckets.get(i, "small") for i in rows.index]

    def col(name: str, empty: Any) -> list[Any]:
        return [getattr(f, name) if (f := fills.get(i)) is not None else empty for i in rows.index]

    rows["fill_minute"] = col("minute", pd.NaT)
    rows["fill_price"] = col("price", math.nan)
    rows["fill_cost_bps"] = col("cost_bps", math.nan)
    rows["limit"] = col("limit", math.nan)

    # Labels, only for rows that filled.
    labels: dict[str, list[float]] = {f"{k}_{h}": [] for h in pn.HORIZONS for k in ("excess", "y")}
    frames: dict[str, pd.DataFrame | None] = {}
    spy_open = spy["open"].to_numpy(dtype=float)
    rec: Any
    for rec in rows.itertuples():
        if rec.symbol not in frames:
            frames[rec.symbol] = pn.bar_frame(rec.symbol, days)
        df = frames[rec.symbol]
        for h in pn.HORIZONS:
            ex = math.nan
            if df is not None and math.isfinite(rec.fill_price):
                px = exit_price(df, rec.sidx, h)
                s0 = spy_ref.get(rec.Index, math.nan)
                s0 = s0 if math.isfinite(s0) else spy_open[rec.sidx]
                s1 = exit_price(spy, rec.sidx, h)
                if math.isfinite(px) and math.isfinite(s1) and s0 > 0:
                    ex = (px / rec.fill_price - 1) - (s1 / s0 - 1) - rec.slip - pn.FEE
            labels[f"excess_{h}"].append(ex)
            labels[f"y_{h}"].append(float(ex > 0) if math.isfinite(ex) else math.nan)
    for k, v in labels.items():
        rows[k] = v
    return rows


# ---- the replay ---------------------------------------------------------------------------


def kind_for(h: int) -> str:
    return f"premarket_h{h}"


def register_kind(h: int) -> None:
    k = kind_for(h)
    event_signals.HOLD[k] = h - 1
    event_signals.PRIORITY[k] = 2


def signals_for(test: pd.DataFrame, p: np.ndarray, h: int) -> list[event_signals.Signal]:
    """p >= 0.55, earliest decision first, up to 10 a session. Only rows that filled."""
    df = test.assign(p=p)
    df = df[(df["p"] >= t9.THRESHOLD) & df["fill_price"].notna()]
    df = df.sort_values(["sidx", "minute", "symbol"]).groupby("sidx").head(t9.TOP_N)
    out = []
    for r in df.to_dict("records"):
        ny = pd.Timestamp(r["minute"]).tz_convert(clock.NY)
        since_open = (ny.hour - 4) * 60 + ny.minute
        session = r["session"]
        assert isinstance(session, date)
        out.append(
            event_signals.Signal(
                str(r["symbol"]),
                clock.previous_session(session),
                kind_for(h),
                1.0 - since_open / 10_000,  # earliest first; the engine ranks by score
                f"p {r['p']:.2f} at {ny:%H:%M}: news tone {r['sent']:+.2f}, "
                f"{int(r['n'])} articles, limit {r['limit']:.2f}",
            )
        )
    return out


class PremarketFills:
    """Buys at their pre-market fill; everything else at 9:45 am (trial 9's prices).

    The engine adds its own slippage to a buy, so the price handed to it is the fill
    divided by that, and the total is the fill with its pre-market cost.
    """

    def __init__(
        self,
        entries: dict[tuple[date, str], float],
        at_945: Callable[[date, set[str]], dict[str, Decimal]],
        slip: Callable[[str, date | None], float],
    ) -> None:
        self.entries = entries
        self.at_945 = at_945
        self.slip = slip

    def __call__(self, day: date, buys: set[str], others: set[str]) -> dict[str, Decimal]:
        out = self.at_945(day, others | {b for b in buys if (day, b) not in self.entries})
        for b in buys:
            fill = self.entries.get((day, b))
            if fill is not None:
                out[b] = Decimal(repr(fill / (1 + self.slip(b, day) / 1e4)))
        return out


def replay(
    settings: Settings,
    bars: Bars,
    sigs: list[event_signals.Signal],
    h: int,
    make_fills: Callable[[Any], Any],
    name: str,
) -> dict[str, t9.Replay]:
    out: dict[str, t9.Replay] = {}
    for w, (start, end) in t9.WINDOWS.items():
        sealed.check(end, f"the {w} replay")
        src = source(settings, bars, end, signals=sigs, name=name)
        assert src.window_bars is not None
        res = run_backtest(
            settings,
            src.window_bars(start, end),
            src.strategy,
            start,
            end,
            universe=src.universe,
            slippage=src.slippage,
            fill_prices=make_fills(src),
        )
        out[w] = t9.Replay(h, w, res, summarize(res, settings, w))
    return out


def log_trials(
    registry: Registry, runs: dict[str, dict[int, dict[str, t9.Replay]]]
) -> dict[str, float]:
    now = datetime.now(UTC)
    for entry, by_h in runs.items():
        for h, rep in by_h.items():
            starved = len(rep["test"].result.trades) < t9.MIN_TRADES
            rets = list(rep["validation"].run.daily_returns)
            registry.append(
                kind="backtest",
                name=f"trial11:{entry}:h{h}:validation",
                config_hash=f"{spec_hash()}-{entry}-h{h}",
                data_hash=hashlib.sha256(json.dumps(rets).encode()).hexdigest()[:16],
                params={"trial": 11, "entry": entry, "horizon": h, "window": "validation"},
                metrics={
                    "sharpe_daily": return_moments(rets)[0],
                    **rep["validation"].run.metrics.to_dict(),
                },
                returns=pd.Series(rets),
                tags=["starved"] if starved else [],
                now=now,
            )
    graded = runs["premarket"][t9.GRADED]["validation"].run.daily_returns
    return deflate(registry, list(graded), variance_kind="backtest")


def preregister(registry: Registry) -> None:
    """The rules, in the registry before any run. Not a trial: it holds no returns."""
    if any(
        e.kind == "preregistration" and e.name == "trial11:preregistration" for e in registry.load()
    ):
        return
    registry.append(
        kind="preregistration",
        name="trial11:preregistration",
        config_hash=spec_hash(),
        data_hash="none",
        params=spec(),
        metrics={},
    )


def run_trial11(
    settings: Settings,
    rows: pd.DataFrame,
    days: list[date],
    at_945: Callable[[date, set[str]], dict[str, Decimal]],
    registry: Registry,
    out: Path = OUT,
) -> dict[str, Any]:
    sealed.check(t9.TEST_END, "the test window")
    rows = rows[rows["session"] < sealed.HOLDOUT_START]
    for h in pn.HORIZONS:
        register_kind(h)
        t9.register_kind(h)
    out.mkdir(parents=True, exist_ok=True)
    filled = rows[rows["fill_price"].notna()]
    wf = {h: t9.walk_forward(filled, days, h) for h in pn.HORIZONS}
    frozen = {h: t9.freeze(filled, days, h, out) for h in pn.HORIZONS}
    (out / "walkforward.json").write_text(json.dumps(wf, indent=1, default=str), encoding="utf-8")
    log.info("frozen; replaying the test window")

    test_all = rows[(rows["session"] >= t9.TEST_START) & (rows["session"] <= t9.TEST_END)]
    bars = base_bars(settings, t9.TEST_END)
    adv = {
        (s, d): a
        for s, d, a in zip(test_all["symbol"], test_all["session"], test_all["adv"], strict=True)
    }
    entries = {
        (d, s): float(f)
        for s, d, f in zip(
            test_all["symbol"], test_all["session"], test_all["fill_price"], strict=True
        )
        if math.isfinite(f)
    }
    runs: dict[str, dict[int, dict[str, t9.Replay]]] = {"premarket": {}, "945": {}}
    trades: dict[str, dict[int, pd.DataFrame]] = {"premarket": {}, "945": {}}
    advantage: dict[int, Any] = {}
    for h in pn.HORIZONS:
        p = frozen[h].predict(test_all)
        sigs = signals_for(test_all, p, h)
        runs["premarket"][h] = replay(
            settings,
            bars,
            sigs,
            h,
            lambda src: PremarketFills(entries, at_945, src.slippage),
            f"news11pm_h{h}",
        )
        same = [
            event_signals.Signal(s.symbol, s.session, t9.kind_for(h), s.score, s.detail)
            for s in sigs
        ]
        runs["945"][h] = replay(
            settings,
            bars,
            same,
            h,
            lambda src: lambda d, buys, others: at_945(d, buys | others),
            f"news11at945_h{h}",
        )
        for entry in ("premarket", "945"):
            trades[entry][h] = t9.trade_table(runs[entry][h]["test"].result, bars["SPY"], adv)
        # Entry advantage per signal: how much cheaper the pre-market fill was than 9:45 am.
        chosen = test_all.assign(p=p)
        chosen = chosen[(chosen["p"] >= t9.THRESHOLD) & chosen["fill_price"].notna()]
        adv_rows = []
        c: Any
        for c in chosen.itertuples():
            px = at_945(c.session, {c.symbol}).get(c.symbol)
            if px is None:
                continue
            at945 = float(px) * (1 + c.slip)
            adv_rows.append({"bucket": c.bucket, "gain": at945 / c.fill_price - 1})
        a = pd.DataFrame(adv_rows)
        advantage[h] = {
            b: {
                "signals": int((a["bucket"] == b).sum()) if len(a) else 0,
                "mean_entry_gain": float(a[a["bucket"] == b]["gain"].mean())
                if len(a) and (a["bucket"] == b).any()
                else None,
            }
            for b in ("small", "mid", "large")
        }

    base = t9.baselines(settings, bars)
    deflated = log_trials(registry, runs)
    graded = runs["premarket"][t9.GRADED]
    chk = t9.checks(graded, base, deflated["dsr"])
    passed = all(c["passed"] for c in chk)

    def summary(run: Any) -> dict[str, Any]:
        m = run.metrics
        return {
            "return": m.total_return,
            "sharpe": m.sharpe,
            "sortino": m.sortino,
            "max_drawdown": m.max_drawdown,
            "trades": m.trade_count,
        }

    def cap_compare(h: int) -> list[dict[str, Any]]:
        pm = {b["bucket"]: b for b in t9.by_bucket(trades["premarket"][h])}
        at = {b["bucket"]: b for b in t9.by_bucket(trades["945"][h])}
        return [
            {
                "bucket": b,
                "premarket": pm[b],
                "at_945": at[b],
                "pnl_gain_vs_945": pm[b]["pnl"] - at[b]["pnl"],
                **advantage[h][b],
            }
            for b in ("small", "mid", "large")
        ]

    trig_by_cap = rows.groupby("bucket").agg(
        triggered=("symbol", "size"), filled=("fill_price", "count")
    )
    report = {
        "trial": 11,
        "passed": passed,
        "verdict": t9.verdict(passed, len(graded["test"].result.trades)),
        "checks": chk,
        "deflated": deflated,
        "walkforward": wf,
        "frozen": {
            h: {"rows": f.rows, "last_label_session": f.last_label_session}
            for h, f in frozen.items()
        },
        "runs": {
            entry: {str(h): {w: summary(r.run) for w, r in rep.items()} for h, rep in by_h.items()}
            for entry, by_h in runs.items()
        },
        "baselines": {k: summary(v) for k, v in base.items()},
        "by_cap": {str(h): cap_compare(h) for h in pn.HORIZONS},
        "fills": {
            str(k): {kk: int(vv) for kk, vv in v.items()}
            for k, v in trig_by_cap.to_dict("index").items()
        },
        "spec_hash": spec_hash(),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    for entry, by_h in trades.items():
        for h, tt in by_h.items():
            tt.to_csv(out / f"trades_{entry}_h{h}.csv", index=False)
    return report
