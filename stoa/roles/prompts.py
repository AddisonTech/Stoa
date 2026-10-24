"""Prompt templates with anonymization.

With anonymization on (the default), the model never sees a ticker, a calendar
date or an absolute price. Symbols become "Asset A", "Asset B" and so on in a
per-session shuffled order, dates become relative session counts, and only
scale-free features (returns, distances, volatility, RSI) are shown. That keeps
a model with a June 2026 knowledge cutoff from recognizing what it is looking at.
"""

from __future__ import annotations

import hashlib
import math
import random
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd
from pydantic import BaseModel, ConfigDict

from stoa.clock import sessions_after
from stoa.features import cell
from stoa.models import ClosedTrade, Holding

ANON_COLUMNS = (
    "ret_5",
    "ret_20",
    "ret_60",
    "ret_120",
    "dist_sma_20",
    "dist_sma_50",
    "dist_sma_200",
    "atr_pct",
    "vol_20",
    "rsi_14",
    "dist_52w_high",
    "rs_spy_60",
)
NAMED_COLUMNS = ("close", *ANON_COLUMNS, "dollar_vol_20")
MARKET_COLUMNS = ("ret_20", "ret_60", "dist_sma_200", "vol_20")

SYSTEM_PROMPT = """You are the scout on a small, long-only, daily equity trading team.
Your job is to suggest which assets look worth buying at the next open, and why.
You do not size positions, set stops or approve trades. A rule-based risk engine
and deterministic quant gates check every idea after you, and most ideas get cut.

Rules:
- Only use asset ids that appear in the feature table.
- Prefer assets in a steady uptrend with positive medium-term momentum and
  controlled volatility. Avoid assets that just spiked or look overbought.
- Do not suggest assets already held.
- conviction is an integer from 1 (weak) to 5 (strong).
- thesis is one or two plain sentences that cite the numbers you used.
- It is fine to return fewer candidates, or none, when nothing looks good.

Feature glossary (all are fractions unless noted, computed from data up to today):
ret_N: return over the last N sessions. dist_sma_N: close relative to its N day
average. atr_pct: 14 day average true range over close. vol_20: annualized 20 day
volatility. rsi_14: 14 day RSI (0 to 100). dist_52w_high: close relative to the
52 week high. rs_spy_60: 60 day return relative to the market."""


class ScoutPick(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    conviction: int
    thesis: str


class ScoutResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[ScoutPick]


@dataclass(frozen=True)
class AnonMap:
    to_alias: dict[str, str]
    to_symbol: dict[str, str]


def alias_names(n: int) -> list[str]:
    letters = string.ascii_uppercase
    out = []
    for i in range(n):
        q, r = divmod(i, 26)
        out.append(f"Asset {letters[r]}" + (str(q) if q else ""))
    return out


def build_anon_map(symbols: Sequence[str], session: date, salt: str = "stoa") -> AnonMap:
    """Shuffle deterministically per session so aliases do not stick to a ticker."""
    seed = int(hashlib.sha256(f"{salt}|{session.isoformat()}".encode()).hexdigest()[:16], 16)
    order = sorted(symbols)
    random.Random(seed).shuffle(order)
    names = alias_names(len(order))
    to_alias = dict(zip(order, names, strict=True))
    return AnonMap(to_alias=to_alias, to_symbol={v: k for k, v in to_alias.items()})


def identity_map(symbols: Sequence[str]) -> AnonMap:
    m = {s: s for s in symbols}
    return AnonMap(to_alias=m, to_symbol=dict(m))


def _fmt(value: object, decimals: int) -> str:
    if value is None:
        return "na"
    v = float(value)  # type: ignore[arg-type]
    if math.isnan(v):
        return "na"
    if abs(v) >= 1e6:
        return f"{v:.3e}"
    return f"{round(v, decimals):.{decimals}f}"


def feature_table(snap: pd.DataFrame, amap: AnonMap, anonymize: bool, decimals: int) -> str:
    cols = ANON_COLUMNS if anonymize else NAMED_COLUMNS
    header = "asset_id," + ",".join(cols)
    rows = []
    for sym in sorted(snap.index, key=lambda s: amap.to_alias[s]):
        row = snap.loc[sym]
        rows.append(amap.to_alias[sym] + "," + ",".join(_fmt(row.get(c), decimals) for c in cols))
    return "\n".join([header, *rows])


def build_user_prompt(
    *,
    session: date,
    snap: pd.DataFrame,
    benchmark: str,
    regime_bull: bool,
    holdings: Mapping[str, Holding],
    recent_trades: Sequence[ClosedTrade],
    max_candidates: int,
    anonymize: bool,
    decimals: int,
) -> tuple[str, AnonMap]:
    symbols = [s for s in snap.index if s != benchmark] if anonymize else list(snap.index)
    amap = build_anon_map(symbols, session) if anonymize else identity_map(symbols)
    table_snap = snap.loc[symbols]
    lines = []
    when = "today (day 0)" if anonymize else session.isoformat()
    lines.append(f"Decision for the close of {when}. Orders would execute at the next open.")
    if benchmark in snap.index:
        m = snap.loc[benchmark]
        market = ", ".join(f"{c}={_fmt(m.get(c), decimals)}" for c in MARKET_COLUMNS)
        label = "Market index" if anonymize else benchmark
        mood = "bullish" if regime_bull else "bearish"
        lines.append(f"{label}: {market}. Regime: {mood} (index vs its 200 day average).")
    lines.append("")
    lines.append("Feature table:")
    lines.append(feature_table(table_snap, amap, anonymize, decimals))
    lines.append("")
    held = [s for s in holdings if s in amap.to_alias]
    if held:
        lines.append("Currently held:")
        for sym in sorted(held, key=lambda s: amap.to_alias[s]):
            h = holdings[sym]
            age = sessions_after(h.meta.entry_session, session)
            close = cell(snap, sym, "close") if sym in snap.index else float(h.meta.entry_price)
            gain = close / float(h.meta.entry_price) - 1.0
            lines.append(f"- {amap.to_alias[sym]}: held {age} sessions, {gain:+.1%} since entry")
    else:
        lines.append("Currently held: nothing.")
    shown = [t for t in recent_trades if t.symbol in amap.to_alias]
    if shown:
        lines.append("")
        lines.append("Recent closed trades (most recent last):")
        for t in shown:
            ago = sessions_after(t.exit_session, session)
            lines.append(
                f"- {amap.to_alias[t.symbol]}: closed {ago} sessions ago, "
                f"{t.return_pct:+.1%}, exit reason {t.reason}"
            )
    lines.append("")
    lines.append(f"Return up to {max_candidates} candidates.")
    return "\n".join(lines), amap
