"""Awardee to ticker, as of the announcement date. Rules only; no language model.

Two name books, both point in time:

- **Direct**: every company in the wide universe's SEC map (``data/newsbrain/sec/ciks.parquet``)
  under each name EDGAR has on file for it (``formerNames`` with their from and to dates,
  and the current name), valid from its first day to two years after its last (contracts
  keep the old name for a while after a rename).
- **Exhibit 21**: every subsidiary listed in a 10-K's exhibit 21, under its parent, valid
  from the 10-K's EDGAR acceptance to the parent's next 10-K, at most 456 days.

A name is normalized: lower case, "&" read as "and", EDGAR's state tags ("/DE/") and
punctuation and company words (inc, corp, co, llc, lp, ltd, plc, holdings, the, ...)
dropped. An EDGAR name of three or more words ending in industries, international,
technologies, technology, worldwide or enterprises is also known without that word
("Huntington Ingalls"). The awardee is tried as written, then the part before its first
comma ("Lockheed Martin, Rotary and Mission Systems"), then any "doing business as" name,
each against the direct book and then exhibit 21; then the same names shortened word by
word from the end, against the direct book only, down to two words and never one
("General Dynamics Land Systems" is General Dynamics; one word found too many namesakes,
such as Boeing's Tapestry Solutions and Tapestry, Inc.). An exhibit 21 key needs two
words, ten letters and a first word of two letters or more.
The first try that names exactly one company wins. Two or more companies is ambiguous and
dropped; none is unmatched (private, foreign or not in the universe) and dropped.

A match also needs a tradable stock on the day: the company's symbol has a daily bar on
the entry session, and the company filed a 10-K, 10-Q, 20-F or 40-F in the 450 days before
the announcement (so a ticker another company took over later is not read back). Delisted
companies stay in while they traded. A company with several symbols trades as the one with
the most dollar volume that month.
"""

from __future__ import annotations

import bisect
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from stoa.contracts.parse import STATES
from stoa.contracts.subsidiaries import DOCS_DIR, EX21_DIR

SUBMISSIONS = Path("data/sec/submissions")
CIKS = Path("data/newsbrain/sec/ciks.parquet")
NAME_GRACE = timedelta(days=730)
EX21_DAYS = 456
ALIVE_DAYS = 450
REPORTS = ("10-K", "10-K405", "10-KT", "10-Q", "20-F", "40-F", "10-K/A", "10-Q/A")
_SUFFIX = frozenset(
    {"inc", "incorporated", "corp", "corporation", "co", "company", "companies", "ltd"}
    | {"limited", "llc", "lc", "lp", "llp", "plc", "pllc", "the", "holding", "holdings"}
    | {"sa", "nv", "ag", "se", "gmbh", "bv", "pte", "pty", "srl", "spa", "kk", "ab", "as"}
)
_JURIS = (
    {w for s in STATES for w in s.replace(".", " ").split()}
    | {"united", "kingdom", "states", "of", "america", "republic", "the", "netherlands"}
    | {"canada", "germany", "france", "japan", "china", "india", "ireland", "israel", "italy"}
    | {"spain", "australia", "singapore", "mexico", "brazil", "cayman", "islands", "bermuda"}
    | {"luxembourg", "switzerland", "sweden", "norway", "denmark", "finland", "belgium"}
    | {"austria", "poland", "korea", "taiwan", "hong", "kong", "malaysia", "thailand"}
    | {"philippines", "new", "zealand", "south", "africa", "british", "virgin", "jersey"}
    | {"guernsey", "scotland", "england", "wales", "uk", "us", "usa", "u", "s"}
    | {"percent", "owned", "wholly", "partnership", "liability", "corporation", "a", "an"}
    | {"organized", "under", "laws", "jurisdiction", "incorporation", "organization"}
)


def norm(name: str) -> str:
    """'Lockheed Martin Corp.' -> 'lockheed martin'; 'L-3 Communications, L.L.C.' ->
    'l 3 communications'."""
    t = name.lower().replace("&", " and ").replace("'s ", "s ")
    t = re.sub(r"\s*/[a-z]{1,6}/?(?=\s|$)", " ", t)  # EDGAR's state tags: "corp /de/", "co/"
    t = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", t)  # l.l.c. -> llc, u.s. -> us
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    words = [w for w in t.split() if w not in _SUFFIX]
    return " ".join(words)


_JOINERS = frozenset({"and", "of", "for", "the", "in", "at", "on", "to", "doing", "business"})


def strip_juris(key: str) -> str:
    words = key.split()
    while words and (words[-1] in _JURIS or words[-1] in _JOINERS or words[-1].isdigit()):
        words.pop()
    return " ".join(words)


def ex21_keys(text: str) -> set[str]:
    """Subsidiary names in an exhibit 21's text, normalized, jurisdiction and share removed."""
    out = set()
    for raw in text.splitlines():
        name = re.split(r",\s+an?\s+|\s+\(|\t|\s{3,}", raw.strip())[0]
        name = re.sub(r"\d+(\.\d+)?\s*%", " ", name)
        key = strip_juris(norm(name))
        words = key.split()
        if (
            2 <= len(words) <= 12
            and len(key) >= 10
            and len(words[0]) >= 2  # not "x technologies" or "m and m manufacturing"
            and "subsidiar" not in key
            and "exhibit" not in key
        ):
            out.add(key)
    return out


# A company's EDGAR name is also known without one of these at its end, when two or more
# words remain: "huntington ingalls industries" is also "huntington ingalls".
_TAIL = frozenset(
    {"industries", "international", "technologies", "technology", "worldwide", "enterprises"}
)


def aliases(key: str) -> list[str]:
    words = key.split()
    out = [key]
    if len(words) >= 3 and words[-1] in _TAIL:
        out.append(" ".join(words[:-1]))
    return out


def candidates(company: str) -> list[tuple[str, bool]]:
    """(normalized name, exact) to try, in order: as written, before the first comma, a
    'doing business as' name; then each shortened word by word."""
    raw = [company]
    if "," in company:
        # "The Boeing Co., Defense, Space & Security" is The Boeing Co.; "Rogers, Lovelock &
        # Fritz Inc." is not Rogers: the part before the comma counts when it is two words or
        # more, or ends in a company word.
        head = company.split(",", maxsplit=1)[0]
        last = re.sub(r"[^a-z]", "", head.lower().split()[-1]) if head.split() else ""
        if len(norm(head).split()) >= 2 or last in _SUFFIX:
            raw.append(head)
    for sep in (r"\bdoing business as\b", r"\bd/?b/?a\b"):
        parts = re.split(sep, company, flags=re.I)
        if len(parts) > 1:
            raw.extend(p.strip(" ,") for p in parts)
    exact: list[str] = []
    for r in raw:
        k = norm(r)
        if k and k not in exact:
            exact.append(k)
    out = [(k, True) for k in exact]
    seen = set(exact)
    for k in exact:
        words = k.split()
        # Down to two words, never one: one word found too many namesakes ("Tapestry
        # Solutions" is Boeing's, not the handbag company's; "Universal", "Cherokee", "Rafael").
        for n in range(len(words) - 1, 1, -1):
            short = " ".join(words[:n])
            if len(short) < 5 or words[n - 1] in _JOINERS:
                continue
            if short not in seen:
                seen.add(short)
                out.append((short, False))
    return out


@dataclass(frozen=True)
class Match:
    cik: int
    symbol: str
    how: str  # direct, direct_prefix, ex21, ex21_prefix
    key: str
    source: str  # the name or the exhibit 21 the match came from


@dataclass(frozen=True)
class Miss:
    reason: str  # unmatched, ambiguous, no_bar, not_reporting
    key: str
    detail: str = ""


def _ts(s: str | None) -> date | None:
    return date.fromisoformat(s[:10]) if s else None


class Book:
    """Both name books and the facts the tradability check reads."""

    def __init__(
        self,
        ciks: pd.DataFrame,
        names: dict[int, list[tuple[str, date, date]]],
        reports: dict[int, list[date]],
        ex21: dict[str, list[tuple[int, date, date, str]]],
        bars: dict[str, np.ndarray],
        dollars: dict[tuple[str, str], float] | None = None,
        sic: dict[int, int] | None = None,
    ) -> None:
        self.symbols: dict[int, list[str]] = defaultdict(list)
        for c, s in zip(ciks["cik"], ciks["symbol"], strict=True):
            self.symbols[int(c)].append(str(s))
        self.names = names
        self.by_name: dict[str, list[tuple[int, date, date]]] = defaultdict(list)
        for cik, rows in names.items():
            for n, a, b in rows:
                for alias in aliases(n):
                    if (cik, a, b) not in self.by_name[alias]:
                        self.by_name[alias].append((cik, a, b))
        self.reports = {k: sorted(v) for k, v in reports.items()}
        self.ex21 = ex21
        self.bars = bars
        self.dollars = dollars or {}
        self.sic = sic or {}

    def has_bar(self, sym: str, on: date) -> bool:
        days = self.bars.get(sym)
        if days is None or not len(days):
            return False
        d = np.datetime64(on, "D")
        i = int(np.searchsorted(days, d))
        return i < len(days) and days[i] == d

    def symbol(self, cik: int, on: date) -> str | None:
        syms = [s for s in self.symbols.get(cik, []) if self.has_bar(s, on)]
        if not syms:
            return None
        month = f"{on.year:04d}-{on.month:02d}"
        return max(syms, key=lambda s: (self.dollars.get((s, month), 0.0), s))

    def reporting(self, cik: int, day: date) -> bool:
        ds = self.reports.get(cik, [])
        i = bisect.bisect_right(ds, day)
        return i > 0 and (day - ds[i - 1]).days <= ALIVE_DAYS

    def direct(self, key: str, day: date) -> set[int]:
        return {c for c, a, b in self.by_name.get(key, []) if a <= day <= b + NAME_GRACE}

    def parents(self, key: str, day: date) -> set[int]:
        return {c for c, a, b, _ in self.ex21.get(key, []) if a <= day < b}

    def match(self, company: str, day: date, entry: date) -> Match | Miss:
        """The awardee's company on ``day`` (announcement), tradable on ``entry``."""
        last = Miss("unmatched", norm(company))
        for key, exact in candidates(company):
            # Exhibit 21 only by a whole name: a shortened name against a conglomerate's long
            # list found too many strangers ("x technologies", "johnson controls").
            for book in ("direct", "ex21") if exact else ("direct",):
                found = self.direct(key, day) if book == "direct" else self.parents(key, day)
                if not found:
                    continue
                if len(found) > 1:
                    return Miss("ambiguous", key, ",".join(str(c) for c in sorted(found)))
                cik = next(iter(found))
                how = book if exact else f"{book}_prefix"
                if not self.reporting(cik, day):
                    last = Miss("not_reporting", key, str(cik))
                    continue
                sym = self.symbol(cik, entry)
                if sym is None:
                    last = Miss("no_bar", key, str(cik))
                    continue
                src = key
                if book == "ex21":
                    src = next(d for c, a, b, d in self.ex21[key] if c == cik and a <= day < b)
                return Match(cik, sym, how, key, src)
        return last


# ---- building the book from disk ---------------------------------------------------------------


def _submissions(cik: int) -> dict[str, Any]:
    p = SUBMISSIONS / f"CIK{cik:010d}.json"
    if not p.exists():
        return {}
    body = dict(json.loads(p.read_text(encoding="utf-8")))
    recent = dict(body.get("filings", {}).get("recent", {}))
    for extra in body.get("filings", {}).get("files", []):
        q = SUBMISSIONS / str(extra.get("name", ""))
        if q.exists():
            more = json.loads(q.read_text(encoding="utf-8"))
            for k, v in more.items():
                if isinstance(v, list):
                    recent.setdefault(k, [])
                    recent[k] = list(recent[k]) + v
    body["_filings"] = recent
    return body


def name_rows(body: dict[str, Any]) -> list[tuple[str, date, date]]:
    """Each EDGAR name with the dates it was in use; the current name runs to far ahead."""
    far = date(2100, 1, 1)
    rows: list[tuple[str, date, date]] = []
    former = body.get("formerNames") or []
    last_to = date(1990, 1, 1)
    for f in former:
        a, b = _ts(f.get("from")) or date(1990, 1, 1), _ts(f.get("to")) or far
        k = norm(str(f.get("name", "")))
        if k:
            rows.append((k, a, b))
        last_to = max(last_to, b if b < far else last_to)
    cur = norm(str(body.get("name", "")))
    if cur:
        rows.append((cur, min(last_to, date.today()) if former else date(1990, 1, 1), far))
    return rows


def build(
    ciks_path: Path = CIKS,
    ex21_dir: Path = EX21_DIR,
    bars_dir: Path = Path("data/bars/wide"),
    docs_dir: Path = DOCS_DIR,
) -> Book:
    ciks = pd.read_parquet(ciks_path)
    names: dict[int, list[tuple[str, date, date]]] = {}
    reports: dict[int, list[date]] = {}
    sic: dict[int, int] = {}
    for cik in sorted({int(c) for c in ciks["cik"]}):
        body = _submissions(cik)
        if not body:
            continue
        names[cik] = name_rows(body)
        fl = body["_filings"]
        reports[cik] = [
            date.fromisoformat(d)
            for f, d in zip(fl.get("form", []), fl.get("filingDate", []), strict=False)
            if f in REPORTS and d
        ]
        try:
            sic[cik] = int(body.get("sic") or 0)
        except (TypeError, ValueError):
            sic[cik] = 0
    ex21: dict[str, list[tuple[int, date, date, str]]] = defaultdict(list)
    parts = [
        pd.read_parquet(p, columns=["cik", "accn", "accepted", "filed", "doc", "text"])
        for folder in (ex21_dir, docs_dir)
        for p in sorted(folder.glob("*.parquet"))
    ]
    parts = [p for p in parts if not p.empty]
    alldocs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(alldocs):
        alldocs["when"] = [
            (pd.Timestamp(a).date() if pd.notna(a) else date.fromisoformat(str(f)[:10]))
            for a, f in zip(alldocs["accepted"], alldocs["filed"], strict=True)
        ]
        # A filing found by both pulls, or with two exhibit files, is one list.
        alldocs = alldocs.drop_duplicates(["cik", "accn", "doc"])
    for cik_key, by_cik in alldocs.groupby("cik") if len(alldocs) else []:
        df = (
            by_cik.groupby("accn")
            .agg(when=("when", "min"), doc=("doc", "first"), text=("text", "\n".join))
            .sort_values("when")
        )
        whens = list(df["when"])
        cik = int(cast(Any, cik_key))
        for i, (when, doc, text) in enumerate(zip(df["when"], df["doc"], df["text"], strict=True)):
            nxt = whens[i + 1] if i + 1 < len(whens) else when + timedelta(days=EX21_DAYS)
            end = min(nxt, when + timedelta(days=EX21_DAYS))
            # The day after acceptance: a list accepted at 4 pm is not read for a 5 pm post
            # the same day by rule, so nothing hinges on the hour.
            start = when + timedelta(days=1)
            for key in ex21_keys(str(text)):
                ex21[key].append((cik, start, end, f"{doc} ({when})"))
    bars: dict[str, np.ndarray] = {}
    for sym in set(ciks["symbol"]):
        p = bars_dir / f"{sym}.parquet"
        if p.exists():
            idx = pd.DatetimeIndex(pd.read_parquet(p, columns=["close"]).index)
            bars[str(sym)] = np.unique(idx.tz_localize(None).normalize().values.astype("M8[D]"))
    dollars: dict[tuple[str, str], float] = {}
    plan = Path("data/newsbrain/minute_plan.parquet")
    if plan.exists():
        mp = pd.read_parquet(plan, columns=["symbol", "month", "dollars"])
        dollars = {
            (str(s), str(m)): float(v)
            for s, m, v in zip(mp["symbol"], mp["month"], mp["dollars"], strict=True)
        }
    return Book(ciks, names, reports, dict(ex21), bars, dollars, sic)
