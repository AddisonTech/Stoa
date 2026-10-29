"""Trial 15's live second reader: Claude reads the news behind the morning's top 20.

The local open model reads every article (``eventread``); on top of it, at 9:00 am, the
existing analyst (its model, prompt, replay cache and schema, ``stoa/analyst.py``) reads the
news of the day behind the 20 stocks the brain ranks highest. Its reading is recorded beside
each pick and graded nightly against what happened. It never changes the picks: it has no
history to test it on (Claude is never used on historical news, since it knows how those
stories ended), so it earns a say only through its live record.

Spend: the one monthly cap every Claude call shares (``budget``). This reader keeps its own
ledger in the shared state (``trial15/claude_ledger.json``) and the other callers count it
(``shadow.merged_view``), so the cap covers all of them together. At the cap it skips, and the
picks go on without it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from stoa.config import Settings
from stoa.models import dec

TOP = 20
LEDGER = Path("trial15") / "claude_ledger.json"


def ledger_path(settings: Settings) -> Path:
    return settings.shared_dir / LEDGER


def spend(settings: Settings) -> dict[str, Decimal]:
    """This reader's spend by month, from the shared state."""
    p = ledger_path(settings)
    if not p.exists():
        return {}
    raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return {k: dec(v) for k, v in raw.get("spend_by_month", {}).items()}


def save_spend(settings: Settings, months: Mapping[str, Decimal]) -> None:
    p = ledger_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {"spend_by_month": {k: str(v) for k, v in sorted(months.items())}}
    p.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")


@dataclass
class Reading:
    status: str
    by_symbol: dict[str, dict[str, float]] = field(default_factory=dict)
    spent_usd: float = 0.0
    note: str = ""


def news_for(archive: pd.DataFrame, symbols: list[str], since: datetime) -> pd.DataFrame:
    """The day's articles about ``symbols`` (known since ``since``), one row per (article,
    symbol), in the analyst's columns."""
    if archive.empty:
        return pd.DataFrame(columns=["id", "symbol", "headline", "summary", "content", "known_at"])
    a = archive.explode("symbols").rename(columns={"symbols": "symbol"})
    a = a[a["symbol"].isin(symbols) & (pd.to_datetime(a["known_at"], utc=True) >= since)]
    return pd.DataFrame(
        {
            "id": a["id"].astype(str),
            "symbol": a["symbol"].astype(str),
            "headline": a["headline"].astype(str),
            "summary": "",
            "content": "",
            "known_at": a["known_at"],
        }
    )


def read(
    settings: Settings,
    api_key: str | None,
    state: Any,
    news: pd.DataFrame,
    now: Any,
    client: Any = None,
) -> Reading:
    """Claude's reading of ``news``, inside the shared monthly cap. ``state`` is the $100
    account's State (its spend and cap records); the shadow job's ledger and this reader's
    own are added, so the cap covers every caller."""
    from stoa import analyst as an
    from stoa import shadow, spendcap

    if api_key is None and client is None:
        return Reading("no_key", note="no ANTHROPIC_API_KEY")
    if news.empty:
        return Reading("no_news")
    mine = spend(settings)
    view = shadow.merged_view(state, shadow.load_ledger(settings), mine)
    month = spendcap.month_of(now())
    cap, total = spendcap.cap_for(view, settings.budget, month), spendcap.spent(view, month)
    if total >= cap:
        return Reading("paused", note=f"Claude spend {total} of the {cap} cap this month")
    other = total - mine.get(month, Decimal(0))
    ledger = an.Ledger(float(cap - other), mine, clock=now)
    cfg = settings.analyst
    items = an.items_from_news(news, cfg.text_chars, cfg.max_items_per_symbol_per_day)
    a = an.Analyst(
        cfg,
        settings.budget,
        settings.llm.prices,
        client or an.make_client(str(api_key)),
        an.ScoreCache(cfg.cache_dir),
        ledger,
    )
    rep = a.score_now(items)
    save_spend(settings, mine)
    rows = a.rows(items)
    out: dict[str, dict[str, float]] = {}
    for sym, g in rows.groupby("symbol"):
        out[str(sym)] = {
            "sentiment": float(g["sentiment"].mean()),
            "materiality": float(g["materiality"].mean()),
            "items": float(len(g)),
        }
    return Reading("read", out, rep.spent_usd, f"{rep.scored} new, {rep.failed} failed")
