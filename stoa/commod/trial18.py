"""Trial 18: commodity funds around the weekly EIA reports, long or flat, unleveraged.

Two reports a week. The Weekly Natural Gas Storage Report (normally Thursday 10:30 am New
York time) gives the change in working gas in storage; the Weekly Petroleum Status Report
(normally Wednesday 10:30 am) the change in commercial crude oil stocks. Each report also
publishes the 5-year average change for the same week, the market's plain "expected" number.

The surprise is the actual change less the expectation. For gas the expectation is also
moved for the week's weather: the 5-year average change plus b_h x (heating degree days
less normal) + b_c x (cooling degree days less normal), with b_h and b_c fitted by least
squares (no intercept) on earlier reports only. The surprise is scaled by the standard
deviation of the earlier reports' surprises. A draw bigger than expected, or a build smaller
than expected (a scaled surprise at or below -THRESHOLD), is bullish: buy the fund (UNG for
gas, USO for oil); otherwise stay flat.

- 18a buys at the open of the first regular session after the release and sells at the
  open HOLD sessions later (Stoa's twice-a-day schedule can do that).
- 18b buys at 10:35 am on the release day (the open of the first minute bar at or after
  10:35) and sells at that day's close.

The account is split in two sleeves, half for gas and half for oil; a sleeve holds its fund
with 98% of its money while a trade is open and cash otherwise. Costs are the standard
model: slippage on each side by the fund's 20 session dollar volume (5 to 75 bps) and the
0.3 bps fee on sales.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from stoa import clock, metrics, wide
from stoa.newsbrain.panel import FEE

SPEC_A = "trial18a:eia:next_open:v1"
SPEC_B = "trial18b:eia:intraday:v1"
NAME_A = "trial18a:preregistration"
NAME_B = "trial18b:preregistration"
DECIDED = date(2026, 11, 16)
FUNDS = {"gas": "UNG", "oil": "USO"}
DIAGNOSTIC = {"gas_bull": "BOIL", "gas_bear": "KOLD"}
THRESHOLD = 1.0  # scaled surprise at or below -1 is bullish
HOLD = 5  # sessions, 18a
INTRADAY_ENTRY = "10:35"
MIN_HISTORY = 52  # earlier reports needed before a surprise can be scaled (and b fitted)
CASH_RESERVE = 0.02
EQUITY = 100_000.0
START = date(2016, 1, 4)  # the first bar on disk
HIT_BAR = 0.54  # the per-trade bar of trials 15 and 16
DSR_BAR = 0.95
DRAWDOWN_CAP = 0.25
MIN_TRADES = 30
TUNE_WINDOWS = {
    "tune_a": (date(2019, 1, 1), date(2020, 12, 31)),
    "tune_b": (date(2021, 1, 1), date(2022, 12, 31)),
    "tune_c": (date(2023, 1, 1), date(2024, 12, 31)),
}


# ---- the signal --------------------------------------------------------------------------------


def surprises(reports: pd.DataFrame) -> pd.DataFrame:
    """``reports``: one fuel's rows with released (UTC), actual, five_year, and for gas
    hdd_dev and cdd_dev. Adds expected, surprise, scale, z and bullish, each from earlier
    reports only (released strictly before)."""
    r = reports.sort_values("released").reset_index(drop=True).copy()
    gas = "hdd_dev" in r.columns and r["hdd_dev"].notna().any()
    exp, sur, scale, b_h, b_c = [], [], [], [], []
    actual = r["actual"].to_numpy(dtype=float)
    five = r["five_year"].to_numpy(dtype=float)
    raw = actual - five
    hdd = r["hdd_dev"].to_numpy(dtype=float) if gas else np.zeros(len(r))
    cdd = r["cdd_dev"].to_numpy(dtype=float) if gas else np.zeros(len(r))
    for i in range(len(r)):
        bh = bc = 0.0
        if gas and i >= MIN_HISTORY:
            x = r.loc[: i - 1, ["hdd_dev", "cdd_dev"]].to_numpy(dtype=float)
            y = raw[:i]
            ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
            if ok.sum() >= MIN_HISTORY:
                coef, *_ = np.linalg.lstsq(x[ok], y[ok], rcond=None)
                bh, bc = float(coef[0]), float(coef[1])
        adj = 0.0
        if gas:
            h, c = hdd[i], cdd[i]
            adj = bh * (h if np.isfinite(h) else 0.0) + bc * (c if np.isfinite(c) else 0.0)
        e = float(five[i]) + adj
        exp.append(e)
        sur.append(float(actual[i]) - e)
        b_h.append(bh)
        b_c.append(bc)
        prior = np.array(sur[:i], dtype=float)
        prior = prior[np.isfinite(prior)]
        scale.append(float(prior.std(ddof=1)) if len(prior) >= MIN_HISTORY else math.nan)
    r["b_hdd"], r["b_cdd"] = b_h, b_c
    r["expected"], r["surprise"], r["scale"] = exp, sur, scale
    r["z"] = r["surprise"] / r["scale"]
    r["bullish"] = r["z"] <= -THRESHOLD
    r["bearish"] = r["z"] >= THRESHOLD
    return r


def entry_session(released: pd.Timestamp) -> date:
    """The first regular session whose open is after ``released``."""
    from stoa.filings.trial17 import counts_from

    d = counts_from([released])[0]
    assert d is not None
    return d


# ---- trades ------------------------------------------------------------------------------------


@dataclass
class Trade:
    fuel: str
    symbol: str
    released: pd.Timestamp
    entry_day: date
    entry_px: float  # including slippage
    exit_day: date
    exit_px: float  # net of slippage and fee
    z: float

    @property
    def ret(self) -> float:
        return self.exit_px / self.entry_px - 1


def _px(bars: pd.DataFrame, day: date, col: str) -> float:
    return float(np.asarray(bars.loc[[pd.Timestamp(day)], col], dtype=float)[0])


def _slip(bars: pd.DataFrame, day: date) -> float:
    """One side's slippage for a trade on ``day``: 20 session dollar volume to the close
    before."""
    dv = (bars["close"] * bars["volume"]).rolling(wide.ADV_SESSIONS, min_periods=1).mean()
    prev = dv[dv.index < pd.Timestamp(day)]
    v = float(prev.iloc[-1]) if len(prev) else None
    return wide.slippage_bps(v) / 1e4


def trades_next_open(
    sig: pd.DataFrame, fuel: str, symbol: str, bars: pd.DataFrame, end: date, hold: int = HOLD
) -> list[Trade]:
    """18a: buy at the first open after each bullish release, sell at the open ``hold``
    sessions later; a bullish release while a trade is open moves its exit out."""
    days = clock.sessions_between(START, end)
    idx = {d: k for k, d in enumerate(days)}
    out: list[Trade] = []
    cur: dict[str, Any] | None = None
    bull = sig[sig["bullish"]]
    for released, z in zip(pd.to_datetime(bull["released"], utc=True), bull["z"], strict=True):
        e = entry_session(released)
        if e not in idx or e > end:
            continue
        k = idx[e]
        if cur is not None and k < cur["exit_k"]:
            cur["exit_k"] = min(k + hold, len(days) - 1)
            continue
        if cur is not None:
            out.append(_close(cur, days, bars))
        if pd.Timestamp(e) not in bars.index:
            cur = None
            continue
        s = _slip(bars, e)
        cur = {
            "fuel": fuel,
            "symbol": symbol,
            "released": released,
            "entry_day": e,
            "entry_px": _px(bars, e, "open") * (1 + s),
            "exit_k": min(k + hold, len(days) - 1),
            "z": float(z),
        }
    if cur is not None:
        out.append(_close(cur, days, bars))
    return out


def _close(cur: dict[str, Any], days: list[date], bars: pd.DataFrame) -> Trade:
    k = cur["exit_k"]
    while k > 0 and pd.Timestamp(days[k]) not in bars.index:
        k -= 1
    x = days[k]
    s = _slip(bars, x)
    px = _px(bars, x, "open") * (1 - s) * (1 - FEE)
    return Trade(
        cur["fuel"],
        cur["symbol"],
        cur["released"],
        cur["entry_day"],
        cur["entry_px"],
        x,
        px,
        cur["z"],
    )


def trades_intraday(
    sig: pd.DataFrame,
    fuel: str,
    symbol: str,
    bars: pd.DataFrame,
    minute_open: Any,
    end: date,
    side: str = "bullish",
) -> list[Trade]:
    """18b: buy at 10:35 am on the release day, sell at that day's close. ``minute_open(symbol,
    day, "HH:MM")`` gives the open of the first minute bar at or after that time (or None).
    A release after 10:35 am (a late or holiday-shifted one) trades only if 10:35 am is after
    it; otherwise that release is skipped."""
    out = []
    hits = sig[sig[side]]
    for rel, z in zip(pd.to_datetime(hits["released"], utc=True), hits["z"], strict=True):
        day = rel.tz_convert("America/New_York").date()
        if day > end or not clock.is_session(day) or pd.Timestamp(day) not in bars.index:
            continue
        t_entry = pd.Timestamp(clock.at_ny(day, INTRADAY_ENTRY))
        if t_entry <= rel:
            continue
        o = minute_open(symbol, day, INTRADAY_ENTRY)
        if o is None:
            continue
        s = _slip(bars, day)
        c = _px(bars, day, "close")
        out.append(
            Trade(fuel, symbol, rel, day, o * (1 + s), day, c * (1 - s) * (1 - FEE), float(z))
        )
    return out


# ---- the account -------------------------------------------------------------------------------


def equity_curve(
    trades: dict[str, list[Trade]],
    bars: dict[str, pd.DataFrame],
    end: date,
    intraday: bool,
    equity0: float = EQUITY,
) -> pd.Series:
    """Two sleeves of equal money (one a fuel), marked at each close. A sleeve in a trade
    holds (1 - CASH_RESERVE) of its money in the fund from the entry to the exit."""
    days = clock.sessions_between(START, end)
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    total = pd.Series(0.0, index=idx)
    per = equity0 / len(FUNDS)
    for fuel, sym in FUNDS.items():
        b = bars[sym]
        close = b["close"].reindex(idx).ffill()
        cash, units, eq = per, 0.0, []
        book = {t.entry_day: t for t in trades.get(fuel, [])}
        exits: dict[date, Trade] = {}
        for d, ts in zip(days, idx, strict=True):
            if intraday:
                t = book.get(d)
                if t is not None:
                    inv = cash * (1 - CASH_RESERVE)
                    cash += inv * (t.exit_px / t.entry_px - 1)
            else:
                t = exits.pop(d, None)
                if t is not None:
                    cash += units * t.exit_px
                    units = 0.0
                t = book.get(d)
                if t is not None:
                    inv = cash * (1 - CASH_RESERVE)
                    units = inv / t.entry_px
                    cash -= inv
                    exits[t.exit_day] = t
            eq.append(cash + units * float(close.loc[ts]) if units else cash)
        total += pd.Series(eq, index=idx)
    return total


def buy_and_hold(bars: dict[str, pd.DataFrame], end: date, equity0: float = EQUITY) -> pd.Series:
    """Half the money in each fund at the first open, held."""
    days = clock.sessions_between(START, end)
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    total = pd.Series(0.0, index=idx)
    for sym in FUNDS.values():
        b = bars[sym]
        o = float(b["open"].iloc[0]) * (1 + _slip(b, START))
        total += (equity0 / len(FUNDS)) / o * b["close"].reindex(idx).ffill()
    return total


def stats(eq: pd.Series, start: date | None = None, end: date | None = None) -> dict[str, Any]:
    if start is not None:
        eq = eq[eq.index >= pd.Timestamp(start)]
    if end is not None:
        eq = eq[eq.index <= pd.Timestamp(end)]
    r = eq.pct_change().dropna()
    dd, dd_len = metrics.max_drawdown(eq)
    return {
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "cagr": metrics.cagr(eq),
        "sharpe": metrics.sharpe(r),
        "sortino": metrics.sortino(r),
        "max_drawdown": dd,
        "max_drawdown_sessions": dd_len,
        "ann_vol": metrics.ann_vol(r),
        "exposure_days": float((r != 0).mean()),
    }


def by_year(eq: pd.Series) -> dict[str, float]:
    years = pd.DatetimeIndex(eq.index).year
    out = {}
    for y in sorted(set(years)):
        prev, cur = eq[years < y], eq[years == y]
        base = float(prev.iloc[-1]) if len(prev) else float(cur.iloc[0])
        out[str(y)] = float(cur.iloc[-1] / base - 1)
    return out


def trade_stats(ts: list[Trade]) -> dict[str, Any]:
    if not ts:
        return {"trades": 0, "hit_rate": math.nan, "avg_net": math.nan}
    r = np.array([t.ret for t in ts])
    return {
        "trades": len(ts),
        "hit_rate": float((r > 0).mean()),
        "avg_net": float(r.mean()),
        "median_net": float(np.median(r)),
        "best": float(r.max()),
        "worst": float(r.min()),
    }


def checks(
    s: dict[str, Any], bh: dict[str, Any], tune: dict[str, Any], dsr: float, ts: dict[str, Any]
) -> list[dict[str, Any]]:
    beat = sum(1 for w in tune.values() if w["sharpe"] > w["bh_sharpe"])
    n = int(ts["trades"])
    return [
        {
            "name": "drawdown_vs_buy_and_hold",
            "passed": s["max_drawdown"] < bh["max_drawdown"],
            "detail": f"{s['max_drawdown']:.2%} vs {bh['max_drawdown']:.2%}",
        },
        {
            "name": "sharpe_vs_buy_and_hold",
            "passed": s["sharpe"] > bh["sharpe"],
            "detail": f"{s['sharpe']:.2f} vs {bh['sharpe']:.2f}",
        },
        {
            "name": "sortino_vs_buy_and_hold",
            "passed": s["sortino"] > bh["sortino"],
            "detail": f"{s['sortino']:.2f} vs {bh['sortino']:.2f}",
        },
        {
            "name": "beats_cash",
            "passed": s["total_return"] > 0,
            "detail": f"total return {s['total_return']:+.2%} vs cash 0",
        },
        {
            "name": "deflated_sharpe",
            "passed": dsr >= DSR_BAR,
            "detail": f"DSR {dsr:.3f} (need {DSR_BAR})",
        },
        {
            "name": "drawdown_cap",
            "passed": s["max_drawdown"] <= DRAWDOWN_CAP,
            "detail": f"worst {s['max_drawdown']:.2%} (cap {DRAWDOWN_CAP:.0%})",
        },
        {
            "name": "tune_windows",
            "passed": beat >= 2,
            "detail": f"Sharpe above buy and hold in {beat} of 3",
        },
        {
            "name": "trade_count",
            "passed": n >= MIN_TRADES,
            "detail": f"{n} trades (need {MIN_TRADES})",
        },
        {
            "name": "hit_rate",
            "passed": n > 0 and ts["hit_rate"] > HIT_BAR,
            "detail": f"{ts['hit_rate']:.1%} of trades made money after costs "
            f"(need above {HIT_BAR:.0%})",
        },
        {
            "name": "average_trade",
            "passed": n > 0 and ts["avg_net"] > 0,
            "detail": f"average {ts['avg_net']:+.3%} a trade after costs",
        },
        {"name": "api_cost", "passed": True, "detail": "no Claude spend"},
    ]


def verdict(chk: list[dict[str, Any]], n: int) -> str:
    if all(c["passed"] for c in chk):
        return "PASS"
    return "STARVED" if n < MIN_TRADES else "FAIL"


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
