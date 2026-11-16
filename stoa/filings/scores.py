"""Year-on-year change scores for every filing (Cohen, Malloy and Nguyen, "Lazy Prices").

Each 10-K is compared with the same filer's 10-K a year earlier, each 10-Q with the same
quarter's 10-Q a year earlier: the prior filing is the filer's filing of the same base form
(10-K, 10-K405, 10-KT count as 10-K; 10-Q and 10-QT as 10-Q) whose period of report is 330
to 400 days before this one's, the one closest to 365 days. When no filing pairs that way
(an index entry whose period of report is missing or wrong), the filing dates are paired
by the same rule.

For the full document and each section (risk factors, MD&A, legal proceedings):

- ``cos``: cosine similarity of the term-count vectors.
- ``jac``: Jaccard similarity of the sets of distinct terms.
- ``len``: length change, words now over words a year earlier, minus 1.

A term is a run of 2 or more letters, lower case; numbers and punctuation are not terms.
No stemming, no stop list: the paper's measures on the words as written. A part that is
empty in either filing has no score (NaN).

Low similarity means the company rewrote the text (a "changer"); high similarity means it
stayed the same (a "non-changer").
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa.filings import LOGS, ROOT, TEXT
from stoa.filings.parse import SECTIONS, base_form
from stoa.filings.pull import write
from stoa.newsbrain.pace import log_progress

log = logging.getLogger(__name__)

PARTS = ("full", *SECTIONS)
LAG_MIN, LAG_MAX = 330, 400  # days between the two periods of report
SCORES = ROOT / "scores"
SCORES_FILE = ROOT / "scores.parquet"
_TERM = re.compile(r"[a-z]{2,}")


def terms(text: str) -> Counter[str]:
    return Counter(_TERM.findall(text.lower()))


def cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return math.nan
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(v * big.get(k, 0) for k, v in small.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return float(dot / (na * nb))


def jaccard(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return math.nan
    sa, sb = set(a), set(b)
    return float(len(sa & sb) / len(sa | sb))


def length_change(a: Counter[str], b: Counter[str]) -> float:
    """Words now (``a``) over words a year earlier (``b``), minus 1."""
    na, nb = sum(a.values()), sum(b.values())
    if na == 0 or nb == 0:
        return math.nan
    return float(na / nb - 1)


def period(row: Any) -> pd.Timestamp:
    rd = str(row.report_date or "")
    return pd.Timestamp(rd) if rd else pd.Timestamp(str(row.filed))


def _nearest(i: int, when: list[pd.Timestamp], forms: list[str]) -> int | None:
    best, gap = None, None
    for j in range(len(when)):
        if j == i or forms[j] != forms[i]:
            continue
        lag = (when[i] - when[j]).days
        if LAG_MIN <= lag <= LAG_MAX and (gap is None or abs(lag - 365) < gap):
            best, gap = j, abs(lag - 365)
    return best


def pair(df: pd.DataFrame) -> list[tuple[int, int | None]]:
    """(row, prior row or None) for each filing of one filer: by period of report, else
    (an index entry with no usable period) by filing date."""
    per = [period(r) for r in df.itertuples(index=False)]
    filed = [pd.Timestamp(str(f)) for f in df["filed"]]
    forms = [base_form(str(f)) for f in df["form"]]
    out: list[tuple[int, int | None]] = []
    for i in range(len(df)):
        j = _nearest(i, per, forms)
        out.append((i, j if j is not None else _nearest(i, filed, forms)))
    return out


def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    bags = {p: [terms(str(x or "")) for x in df[p]] for p in PARTS}
    rows = []
    for i, j in pair(df):
        r = df.iloc[i]
        row: dict[str, Any] = {
            "cik": int(r["cik"]),
            "accn": r["accn"],
            "form": r["form"],
            "filed": r["filed"],
            "accepted": r["accepted"],
            "report_date": r["report_date"],
            "symbol_tag": r["symbol_tag"],
            "prior_accn": df.iloc[j]["accn"] if j is not None else None,
            "prior_report_date": df.iloc[j]["report_date"] if j is not None else None,
        }
        for p in PARTS:
            a = bags[p][i]
            row[f"words_{p}"] = int(sum(a.values()))
            if j is None:
                row[f"cos_{p}"] = row[f"jac_{p}"] = row[f"len_{p}"] = math.nan
                continue
            b = bags[p][j]
            row[f"cos_{p}"] = cosine(a, b)
            row[f"jac_{p}"] = jaccard(a, b)
            row[f"len_{p}"] = length_change(a, b)
        rows.append(row)
    return pd.DataFrame(rows)


def score_file(path: Path) -> dict[str, Any]:
    t0 = time.monotonic()
    out = SCORES / path.name
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return {"cik": path.stem, "skipped": True}
    df = pd.read_parquet(path)
    res = score_frame(df) if len(df) else pd.DataFrame()
    write(res, out)
    return {"cik": path.stem, "filings": len(df), "seconds": round(time.monotonic() - t0, 1)}


def collect() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(SCORES.glob("*.parquet"))]
    parts = [p for p in parts if len(p)]
    df = pd.concat(parts, ignore_index=True)
    df["accepted"] = pd.to_datetime(df["accepted"], utc=True)
    for c in df.columns:
        if c.startswith(("cos_", "jac_", "len_")):
            df[c] = df[c].astype(np.float64)
    write(df, SCORES_FILE)
    return df


def main(workers: int = 12) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    files = sorted(TEXT.glob("*.parquet"))
    progress = LOGS / "scores.jsonl"
    n = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(score_file, files, chunksize=4):
            n += 1
            if not res.get("skipped"):
                log_progress(progress, done=n, of=len(files), **res)
    df = collect()
    log_progress(
        progress, finished=True, filings=len(df), paired=int(df["prior_accn"].notna().sum())
    )
