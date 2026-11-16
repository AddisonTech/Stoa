"""Each filing to the stock it moves, point in time, so delisted companies stay in.

Candidates for a filing by filer C, accepted on day d:

- the symbols the wide universe's SEC map gives C (``data/newsbrain/sec/ciks.parquet``: the
  SEC's ticker file, plus the symbol each insider Form 4 named for C when it was filed),
- the trading symbols C's own filings state (inline XBRL ``dei:TradingSymbol``) within 450
  days of d.

A candidate counts only if it has a daily bar in the 30 days up to d (it traded then) and no
other filer claimed it within 450 days of d (another filer's filing stated it, or a Form 4
named it for another issuer): a ticker that changed hands is never read back to the wrong
company. If C's own filings within 450 days state a symbol that is still a candidate, only
those stated symbols are kept (they are the filer's own word for that time). Of what is
left, the symbol with the most dollar volume over the 20 sessions before d wins (one share
class per company). A filing with no candidate left is unmapped and counted.

Delisted companies keep their symbol while they traded: their filings map to it and their
bars end where trading ended.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from stoa.filings import LOGS, ROOT
from stoa.filings.pull import write
from stoa.newsbrain.pace import log_progress

log = logging.getLogger(__name__)

CIKS = Path("data/newsbrain/sec/ciks.parquet")
PURCHASES = Path("data/insider/purchases.parquet")
WIDE = Path("data/bars/wide")
WINDOW_DAYS = 450
RECENT_DAYS = 30
OUT = ROOT / "filing_symbols.parquet"


class Bars:
    """Per symbol: session dates and 20 session dollar volume ending the session before."""

    def __init__(self, root: Path = WIDE) -> None:
        self.root = root
        self._cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    def get(self, sym: str) -> tuple[np.ndarray, np.ndarray] | None:
        if sym not in self._cache:
            p = self.root / f"{sym}.parquet"
            if not p.exists():
                self._cache[sym] = None
            else:
                df = pd.read_parquet(p, columns=["close", "volume"])
                idx = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
                dv = (df["close"] * df["volume"]).rolling(20, min_periods=1).mean().shift(1)
                self._cache[sym] = (idx.to_numpy(dtype="datetime64[D]"), dv.to_numpy(dtype=float))
        return self._cache[sym]

    def traded_near(self, sym: str, day: np.datetime64) -> bool:
        b = self.get(sym)
        if b is None:
            return False
        i = int(np.searchsorted(b[0], day, side="right")) - 1
        return i >= 0 and (day - b[0][i]) <= np.timedelta64(RECENT_DAYS, "D")

    def dollar_volume(self, sym: str, day: np.datetime64) -> float:
        b = self.get(sym)
        if b is None:
            return 0.0
        i = int(np.searchsorted(b[0], day, side="right")) - 1
        v = b[1][i] if i >= 0 else np.nan
        return float(v) if np.isfinite(v) else 0.0


def claimed_by_other(ev: list[tuple[np.datetime64, int]], cik: int, day: np.datetime64) -> bool:
    w = np.timedelta64(WINDOW_DAYS, "D")
    return any(c != cik and abs(d - day) <= w for d, c in ev)


def map_filings(
    filings: pd.DataFrame, book: dict[int, list[str]], purchases: pd.DataFrame, bars: Bars
) -> pd.DataFrame:
    """``filings`` has cik, accn, accepted (UTC) and symbol_tag. Adds ``symbol`` ("" when
    unmapped) and ``map_rule``."""
    f = filings.copy()
    acc = pd.to_datetime(f["accepted"], utc=True).dt.tz_convert("America/New_York")
    days = acc.dt.tz_localize(None).dt.normalize().to_numpy(dtype="datetime64[D]")
    ciks = [int(c) for c in f["cik"]]
    tags = [str(t) for t in f["symbol_tag"].fillna("")]
    ev: dict[str, list[tuple[np.datetime64, int]]] = defaultdict(list)
    own: dict[int, list[tuple[np.datetime64, str]]] = defaultdict(list)
    for c, d, tag in zip(ciks, days, tags, strict=True):
        if tag:
            ev[tag].append((d, c))
            own[c].append((d, tag))
    if len(purchases):
        pdays = pd.to_datetime(purchases["filing_date"]).to_numpy(dtype="datetime64[D]")
        for sym, d, c in zip(purchases["symbol"], pdays, purchases["issuer_cik"], strict=True):
            ev[str(sym).upper().replace("-", ".")].append((d, int(c)))
    w = np.timedelta64(WINDOW_DAYS, "D")
    symbols, rules = [], []
    for cik, day in zip(ciks, days, strict=True):
        stated = {t for d, t in own.get(cik, []) if abs(d - day) <= w}
        cands = set(book.get(cik, [])) | stated
        ok = [
            s
            for s in sorted(cands)
            if bars.traded_near(s, day) and not claimed_by_other(ev.get(s, []), cik, day)
        ]
        rule = "map"
        if stated & set(ok):
            ok = sorted(stated & set(ok))
            rule = "stated"
        if not ok:
            symbols.append("")
            rules.append("unmapped")
            continue
        best = max(ok, key=lambda s: (bars.dollar_volume(s, day), s))
        symbols.append(best)
        rules.append(rule if len(ok) == 1 else rule + ":by_volume")
    f["symbol"] = symbols
    f["map_rule"] = rules
    return f


def book_of(path: Path = CIKS) -> dict[int, list[str]]:
    df = pd.read_parquet(path)
    out: dict[int, list[str]] = {}
    for cik, sym in zip(df["cik"], df["symbol"], strict=True):
        out.setdefault(int(cik), []).append(str(sym))
    return out


def main() -> None:
    from stoa.filings.scores import SCORES_FILE

    logging.basicConfig(level=logging.INFO)
    sc = pd.read_parquet(SCORES_FILE, columns=["cik", "accn", "accepted", "symbol_tag"])
    pur = pd.read_parquet(PURCHASES) if PURCHASES.exists() else pd.DataFrame()
    out = map_filings(sc, book_of(), pur, Bars())
    write(out[["cik", "accn", "symbol", "map_rule"]], OUT)
    counts = out["map_rule"].str.split(":").str[0].value_counts().to_dict()
    log_progress(
        LOGS / "tickers.jsonl", filings=len(out), **{str(k): int(v) for k, v in counts.items()}
    )
