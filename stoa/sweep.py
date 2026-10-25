"""The 9:00 am overnight news sweep: read the night's news, then settle the picks.

DECIDE plans after the close but says nothing about it. At 9:00 am New York
time the sweep fetches every article published since that close on the planned
buys and the reserves (buys the rules passed but had no slot for), reads each
headline and summary with FinBERT, and:

- drops a planned buy when the night brought clearly bad news: one article at
  least ``drop_negative`` negative and a mean tone at or below ``drop_net``;
- adds a reserve when the night brought clearly good news: one article at least
  ``add_positive`` positive and a mean tone at or above ``add_net``, only into a
  free slot, at most ``max_adds``.

Everything else is left as the rules planned it. Sells are never changed. The
result is the plan EXECUTE runs at 9:45, after the usual risk checks and only
for buys approved by 9:40. If the reader cannot run, the picks go out
unchanged and the alert says so.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stoa.config import SweepConfig
from stoa.models import PlannedOrder, Side
from stoa.reader import Reader, net, text_of
from stoa.state import Plan


@dataclass(frozen=True)
class Tone:
    """The night's news on one symbol, as FinBERT read it."""

    symbol: str
    articles: int
    mean_net: float
    max_positive: float
    max_negative: float
    best: str  # the most positive headline
    worst: str  # the most negative headline

    def line(self) -> str:
        if self.articles == 0:
            return "No overnight news."
        noun = "article" if self.articles == 1 else "articles"
        return f"Overnight news: {self.articles} {noun}, tone {self.mean_net:+.2f}."


def quiet(symbol: str) -> Tone:
    return Tone(symbol, 0, 0.0, 0.0, 0.0, "", "")


def read_tones(
    news: pd.DataFrame, symbols: list[str], reader: Reader, cfg: SweepConfig
) -> dict[str, Tone]:
    """One Tone per symbol from ``news`` (columns symbol, headline, summary, known_at).

    Each symbol reads at most ``max_articles_per_symbol`` of its newest articles,
    and only articles tagged with at most ``max_symbols_per_article`` tickers.
    """
    out = {s: quiet(s) for s in symbols}
    if news.empty:
        return out
    rows = news[news["symbol"].isin(symbols)]
    if "n_symbols" in rows.columns:
        # Roundups tag many tickers; their tone is rarely about this one.
        tagged = pd.to_numeric(rows["n_symbols"], errors="coerce").fillna(1)
        rows = rows[tagged <= cfg.max_symbols_per_article]
    rows = rows.sort_values("known_at", ascending=False)
    rows = rows.groupby("symbol", sort=False).head(cfg.max_articles_per_symbol)
    if rows.empty:
        return out
    texts = [text_of(h, s) for h, s in zip(rows["headline"], rows["summary"], strict=True)]
    probs = reader.score(texts)
    tone = net(probs)
    for sym, idx in rows.groupby("symbol", sort=False).indices.items():
        p = probs[idx]
        t = tone[idx]
        heads = rows["headline"].to_numpy()[idx]
        out[str(sym)] = Tone(
            symbol=str(sym),
            articles=len(idx),
            mean_net=float(np.mean(t)),
            max_positive=float(p[:, 0].max()),
            max_negative=float(p[:, 1].max()),
            best=str(heads[int(np.argmax(p[:, 0]))]),
            worst=str(heads[int(np.argmax(p[:, 1]))]),
        )
    return out


def bad_news(t: Tone, cfg: SweepConfig) -> bool:
    return t.articles > 0 and t.max_negative >= cfg.drop_negative and t.mean_net <= cfg.drop_net


def good_news(t: Tone, cfg: SweepConfig) -> bool:
    return t.articles > 0 and t.max_positive >= cfg.add_positive and t.mean_net >= cfg.add_net


@dataclass
class Outcome:
    plan: Plan
    dropped: list[tuple[PlannedOrder, Tone]] = field(default_factory=list)
    added: list[tuple[PlannedOrder, Tone]] = field(default_factory=list)


def apply(
    plan: Plan,
    tones: Mapping[str, Tone],
    cfg: SweepConfig,
    *,
    max_positions: int,
    held: set[str],
) -> Outcome:
    """The plan after the sweep. ``held`` is what the account holds this morning."""
    sells = [o for o in plan.orders if o.side is Side.SELL]
    selling = {o.symbol for o in sells}
    out = Outcome(plan)
    buys: list[PlannedOrder] = []
    for o in plan.orders:
        if o.side is not Side.BUY:
            continue
        t = tones.get(o.symbol, quiet(o.symbol))
        if bad_news(t, cfg):
            out.dropped.append((o, t))
        else:
            buys.append(o)
    staying = len(held - selling)
    taken = {o.symbol for o in buys} | held
    for r in plan.reserves:
        if len(out.added) >= cfg.max_adds or staying + len(buys) >= max_positions:
            break
        t = tones.get(r.symbol, quiet(r.symbol))
        if r.symbol in taken or r.symbol in selling or not good_news(t, cfg):
            continue
        thesis = f"{r.thesis} Added on overnight news: {t.best}".strip()
        added = r.model_copy(update={"reason": "news_add", "thesis": thesis})
        buys.append(added)
        taken.add(r.symbol)
        out.added.append((added, t))
    out.plan = plan.model_copy(update={"orders": (*sells, *buys)})
    return out
