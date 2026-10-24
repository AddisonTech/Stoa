"""The trial 8 universe: every listed US equity, filtered each day by price and liquidity.

Registered in docs/DECISIONS.md ("Trial 8: event-driven"):

- Candidates: every Alpaca US equity listed on NYSE, Nasdaq, NYSE American, NYSE
  Arca or Cboe BZX, active or inactive (delisted). OTC is excluded.
- Tradable on a session: close at least $5 and 20 session average dollar volume
  (close times volume, the 20 sessions ending that session) at least $5 million.
- Account fit, at each decision: fractionable at Alpaca, or a close below the
  position size (20% of current equity).
- Costs: one-way slippage in bps = 25 * sqrt($10M / 20 session dollar volume),
  between 5 and 75.
"""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

LISTED = frozenset({"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"})
# A ticker: 1 to 5 letters, optionally a share class (BRK.B). Alpaca's delisted list also
# holds CUSIP-like codes such as 0029900E0, which the bar API rejects.
TICKER = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")
WIDE_NAMESPACE = "wide"
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 5_000_000.0
ADV_SESSIONS = 20
COST_SCALE_BPS = 25.0
COST_REF_DOLLAR_VOLUME = 10_000_000.0
COST_MIN_BPS = 5.0
COST_MAX_BPS = 75.0


def candidates(assets: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Listed US equities from Alpaca's asset lists, active first (a reused ticker keeps it)."""
    out: dict[str, dict[str, Any]] = {}
    for status in ("inactive", "active"):
        for a in assets.get(status, []):
            sym = str(a.get("symbol", ""))
            listed = a.get("exchange") in LISTED and a.get("class", "us_equity") == "us_equity"
            if listed and TICKER.match(sym):
                out[sym] = {
                    "status": status,
                    "exchange": a.get("exchange"),
                    "fractionable": bool(a.get("fractionable")),
                    "name": a.get("name", ""),
                }
    return out


def fetch_robust(fetch: Any, symbols: list[str]) -> dict[str, Any]:
    """Fetch a chunk; if the provider rejects it, split it in half and retry each part.

    One bad symbol then costs only itself, not the hundred around it.
    """
    try:
        got: dict[str, Any] = fetch(symbols)
        return got
    except Exception:
        if len(symbols) == 1:
            return {}
        mid = len(symbols) // 2
        return {**fetch_robust(fetch, symbols[:mid]), **fetch_robust(fetch, symbols[mid:])}


def save_assets(path: Path, assets: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(assets), encoding="utf-8")


def load_assets(path: Path) -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = json.loads(path.read_text(encoding="utf-8"))
    return data


def dollar_volume(df: pd.DataFrame) -> pd.Series:
    """20 session average dollar volume, ending on each session (no lookahead)."""
    dv = df["close"] * df["volume"]
    return dv.rolling(ADV_SESSIONS, min_periods=ADV_SESSIONS).mean()


def liquid(df: pd.DataFrame) -> pd.Series:
    """True on sessions where the stock passes the price and liquidity floor."""
    return (df["close"] >= MIN_PRICE) & (dollar_volume(df) >= MIN_DOLLAR_VOLUME)


def fits_account(fractionable: bool, close: Decimal, equity: Decimal, max_pct: Decimal) -> bool:
    """A $100 account can hold it: fractional shares, or one share under the position size."""
    return fractionable or close < equity * max_pct


def slippage_bps(dollar_vol: float | None) -> float:
    """One-way cost, rising as liquidity falls. Unknown volume costs the most."""
    if dollar_vol is None or not math.isfinite(dollar_vol) or dollar_vol <= 0:
        return COST_MAX_BPS
    bps = COST_SCALE_BPS * math.sqrt(COST_REF_DOLLAR_VOLUME / dollar_vol)
    return min(COST_MAX_BPS, max(COST_MIN_BPS, bps))
