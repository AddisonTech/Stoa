"""The ripple map: who sells to whom, typed, from the 10-K text already on disk.

Each 10-K in ``data/newsbrain/sec/tenk`` holds only the paragraphs that name customers
or suppliers with a share of sales (see ``sec_pull``). For every such filing this finds
the other companies it names, by the same normalized name book as ``panel.links``, and
types each one from the words near the name:

- ``customer``: the named company buys from the filer ("sales to", "accounted for 12%
  of net sales", "our largest customer").
- ``supplier``: the filer buys from the named company ("sole source", "purchases from",
  "contract manufacturer", "foundry", "licensed from").
- ``competitor``: "compete", "competitor", "competition" is near. Kept as a type, but it
  is not a supply link.
- ``other``: named, but with none of those words near.

The words are looked for in the sentence that holds the name first (up to 300 characters
either side), then in a window of 150 characters either side. Inside a scope the cue
closest to the name decides, so "customers such as Apple" and "suppliers such as TSMC"
in one sentence each get their own type. "Suppliers to" names a customer (the filer is
the supplier), and "Tier 1 suppliers" is not a supplier cue. A company named more than once
takes a customer or supplier cue over a competitor cue over none, the closest first. The
cue that decided is kept for audit.

False matches are held down the way ``links`` does it: a partner must be among the top
1500 names by dollar volume in the month before the 10-K became public (the first month
of the plan for older filings), its normalized name must be at least 5 characters (the
name book's rule), and the name must start with a capital letter or a digit in the text,
so "target markets" is not Target. A one-word name must also stand alone: "Southern
California" and "People's Republic" are not Southern or People. A one-word name the 10-Ks
also write in lower case (at least 5 times and 10% of the time: "target", "southern",
"shell") does not count at the start of a sentence or before a lower-case word that is not
a joining word, so "Southern division" and "Shell eggs" are dropped and "Target accounted
for 12%" is kept. Nasdaq is skipped: 10-Ks name it as their listing exchange.

An edge is valid from the 10-K's acceptance until the filer's next 10-K, and never longer
than 456 days (about 15 months), so a filer that stops filing does not keep its links
forever. Filings in the holdout are never read (``sealed.before_holdout``).

Run it with ``python -c "from stoa.newsbrain import ripplemap; ripplemap.main()"``.
"""

from __future__ import annotations

import bisect
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stoa.newsbrain import ROOT, sealed
from stoa.newsbrain.alpaca_pull import PLAN
from stoa.newsbrain.panel import TOP_NAMES, name_book, nanos
from stoa.newsbrain.sec_pull import SEC_DIR

log = logging.getLogger(__name__)

EDGES = ROOT / "ripple" / "edges.parquet"
VALID_DAYS = 456  # about 15 months
SENTENCE_REACH = 300  # characters either side of the name, inside its sentence
WINDOW = 150  # characters either side of the name, across sentences
ORIGINAL_FORMS = frozenset({"10-K", "10-K405", "10-KT"})  # amendments do not end an edge
RELATIONS = ("customer", "supplier", "competitor", "other")
FLIP = {"customer": "supplier", "supplier": "customer", "competitor": "competitor"}
COLUMNS = ["cik", "other", "relation", "accepted", "valid_to", "cue"]
COMMON_MIN = 5
COMMON_SHARE = 0.1

# "accounted for 14% of", "20 percent of our", "43% and 16%, respectively, of our
# Transmission segment": a share of something, said of the name.
_SHARE = (
    r"(?:(?:(?:accounted for|represented|comprised|made up) )?"
    r"(?:approximately |about |roughly |over |more than |nearly )?\d+(?:\.\d+)?\s*(?:%|percent)"
    r"|respectively,?) of (?:our |the company.s |its |total |consolidated |net |gross )*"
    r"(?:worldwide |[a-z]+ segment.s |[a-z]+ segment )?"
)
# "supplier to" says the filer supplies; "Tier 1 suppliers" are its customers' suppliers.
_NOT_TIER = r"(?<!tier 1 )(?<!tier 2 )(?<!tier i )(?<!tiered )"
_CUES: dict[str, re.Pattern[str]] = {
    "customer": re.compile(
        r"\b(?:largest customers?|customers?|clients?|sales to|sold to|sells? to|selling to|"
        r"shipments to|revenues? (?:derived |generated )?from|suppliers? (?:of [a-z ]{0,40})?to|"
        + _SHARE
        + r"(?:sales|revenues?))\b",
        re.I,
    ),
    "supplier": re.compile(
        r"\b(?:" + _NOT_TIER + r"suppliers?(?! (?:of [a-z ]{0,40})?to\b)|vendors?|"
        r"sole[- ]sourced?|single[- ]sourced?|"
        r"purchases? (?:of [a-z ]{0,40})?from|purchased from|buy from|bought from|"
        r"procured? from|sourced from|supplied by|manufactured by|"
        r"contract manufactur(?:er|ers|ing)|(?-i:foundry|foundries)|we rely on|relies on|"
        r"licen[cs]ed? from|licensors?|"
        + _SHARE
        + r"(?:purchases|inventory purchases|cost of (?:sales|revenues?|goods sold)))\b",
        re.I,
    ),
    # Not "competence" or "competencies".
    "competitor": re.compile(r"\bcompet(?:e|es|ed|ing|itors?|ition|itions|itive|itively)\b", re.I),
}

# norm_name's rules, applied without moving any character, so a match keeps its place.
_SUFFIX = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|companies|ltd|limited|plc|holdings?|"
    r"group|sa|nv|ag|se|llc|lp|the|class [a-c]|common stock|new)\b"
)
_PUNCT = re.compile(r"[^a-z0-9 ]")
_TOKEN = re.compile(r"\S+")
_STOP = re.compile(r"[.!?](?=[\s\"')\]]*(?:\s[A-Z0-9(\"]|$))|\n")
_WORD = re.compile(r"[A-Za-z]+")
_JOIN_BEFORE = re.compile(r"([A-Za-z][A-Za-z&]*)[ -]$")
_APOS = "'" + chr(0x2019) + chr(0xFFFD)  # straight, curly, and the replacement mark
_JOIN_AFTER = re.compile(r"(?:[" + _APOS + r"]s)?[ -]([A-Za-z][A-Za-z&]*)")
_NEXT_WORD = re.compile(r" ([A-Za-z]+)")
# Lower-case words that may follow a company's name in a sentence about it.
_LINK_WORDS = frozenset(
    {"and", "or", "as", "at", "in", "of", "for", "to", "with", "by", "on", "from", "through"}
    | {"which", "who", "is", "was", "were", "are", "has", "had", "have", "also", "each"}
    | {"accounted", "represented", "comprised", "made", "purchased", "purchases", "sales"}
    | {"revenue", "revenues", "together", "collectively", "individually", "respectively"}
    | {"alone", "combined", "accounts", "represents", "remained", "became", "continued"}
)
# Named in nearly every 10-K as the listing exchange, not as a partner.
_SKIP_NAMES = frozenset({"nasdaq"})
_CORP_WORDS = frozenset(
    {"inc", "incorporated", "corp", "corporation", "co", "company", "companies", "ltd"}
    | {"limited", "plc", "holding", "holdings", "group", "sa", "nv", "ag", "se", "llc", "lp"}
)
_ABBREV = frozenset(
    {"inc", "corp", "co", "ltd", "no", "nos", "jr", "sr", "st", "mr", "ms", "dr", "vs", "approx"}
)


@dataclass(frozen=True)
class Mention:
    other: int
    start: int  # character span of the name in the original text
    end: int


@dataclass(frozen=True)
class Typed:
    relation: str
    cue: str
    distance: int  # characters between the cue and the name; large for ``other``


class NameIndex:
    """The name book as a lookup by first word, for whole-name runs of words.

    ``common`` holds one-word names the 10-Ks mostly use as ordinary words ("people",
    "strategy", "target"); those count only away from the start of a sentence.
    """

    def __init__(self, names: dict[int, str], common: frozenset[str] = frozenset()) -> None:
        self.by_name: dict[str, int] = {}
        for c in sorted(names):
            self.by_name.setdefault(names[c], c)
        self.lengths: dict[str, set[int]] = {}
        for n in self.by_name:
            words = n.split()
            self.lengths.setdefault(words[0], set()).add(len(words))
        self.common = common

    def find(self, text: str) -> list[Mention]:
        """Every place a whole normalized name appears, capitalized in the original."""
        norm = _flat(text)
        toks = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(norm)]
        words = [t[0] for t in toks]
        out: list[Mention] = []
        for i, w in enumerate(words):
            for k in sorted(self.lengths.get(w, ()), reverse=True):
                if i + k > len(words):
                    continue
                name = " ".join(words[i : i + k])
                hit = self.by_name.get(name)
                if hit is None:
                    continue
                start, end = toks[i][1], toks[i + k - 1][2]
                if not self._proper(text, start, end, name, k):
                    continue
                out.append(Mention(hit, start, end))
                break  # the longest name at this word
        return out

    def _proper(self, text: str, start: int, end: int, name: str, k: int) -> bool:
        """Whether the words at ``start:end`` read as the company's name."""
        first = text[start]
        if not (first.isupper() or first.isdigit()):
            return False  # "target markets" is not Target
        if name in _SKIP_NAMES:
            return False
        if k > 1:
            return True
        if joined(text, start, end):
            return False  # "Southern California", "New Brunswick", "People's Republic"
        if name in self.common:
            before = text[:start].rstrip(" \t")
            if not before or before[-1] in '.!?:;\n("':
                return False  # "Strategy" or "Reliance on" opening a sentence
            after = _NEXT_WORD.match(text, end, end + 40)
            if after and after.group(1).islower() and after.group(1) not in _LINK_WORDS:
                return False  # "Southern division", "Shell eggs"
        return True


def joined(text: str, start: int, end: int) -> bool:
    """Whether a one-word name runs into a capitalized word, so it is part of another name.

    A corporate word after it ("Southern Company", "Target Corporation") and "The" before
    it do not count.
    """
    before = _JOIN_BEFORE.search(text, max(0, start - 40), start)
    if before and before.group(1)[0].isupper() and before.group(1).lower() != "the":
        return True
    after = _JOIN_AFTER.match(text, end, end + 40)
    return bool(after and after.group(1)[0].isupper() and after.group(1).lower() not in _CORP_WORDS)


def common_words(texts: list[str], names: dict[int, str]) -> frozenset[str]:
    """One-word names the texts also write in lower case, at least 5 times and 10% of the time."""
    singles = {n for n in names.values() if " " not in n}
    low: Counter[str] = Counter()
    cap: Counter[str] = Counter()
    for t in texts:
        for w in _WORD.findall(t):
            lw = w.lower()
            if lw in singles:
                (low if w.islower() else cap)[lw] += 1
    return frozenset(
        n for n in low if low[n] >= COMMON_MIN and low[n] >= COMMON_SHARE * (low[n] + cap[n])
    )


def _flat(text: str) -> str:
    """``norm_name`` with every removed character turned into a space, length kept."""
    low = text.lower()
    if len(low) != len(text):  # a few characters lower to two; keep the places right
        low = "".join(ch.lower() if len(ch.lower()) == 1 else " " for ch in text)
    low = _PUNCT.sub(" ", low)
    return _SUFFIX.sub(lambda m: " " * len(m.group()), low)


def sentence_stops(text: str) -> list[int]:
    """Positions that end a sentence: . ! ? before a capital, or a line break.

    A period after "Inc", "Corp", "Co" and the like, or after a single letter as in
    "U.S.", does not end one, so "Apple Inc. accounted for 12%" stays whole.
    """
    stops = []
    for m in _STOP.finditer(text):
        i = m.start()
        if text[i] == ".":
            j = i
            while j > 0 and text[j - 1].isalpha():
                j -= 1
            word = text[j:i].lower()
            if word in _ABBREV or len(word) == 1:
                continue
        stops.append(i)
    return stops


def _distance(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 <= b0:
        return b0 - a1
    if b1 <= a0:
        return a0 - b1
    return 0


def _closest(text: str, lo: int, hi: int, m: Mention) -> Typed | None:
    """The cue nearest the name between ``lo`` and ``hi``, if there is one."""
    best: Typed | None = None
    span = text[lo:hi]
    for rel, pat in _CUES.items():
        for c in pat.finditer(span):
            c0, c1 = lo + c.start(), lo + c.end()
            if c0 < m.end and m.start < c1:
                continue  # a cue inside the name itself, as in "Bolt Foundry"
            d = _distance(c0, c1, m.start, m.end)
            if best is None or d < best.distance:
                best = Typed(rel, c.group().lower(), d)
    return best


def type_mention(text: str, m: Mention, stops: list[int]) -> Typed:
    """The relation the words near one mention say, from the filer's point of view."""
    k = bisect.bisect_left(stops, m.start)
    s0 = stops[k - 1] + 1 if k > 0 else 0
    j = bisect.bisect_left(stops, m.end)
    s1 = stops[j] if j < len(stops) else len(text)
    lo, hi = max(s0, m.start - SENTENCE_REACH), min(s1, m.end + SENTENCE_REACH)
    found = _closest(text, lo, hi, m)
    if found is None:
        found = _closest(text, max(0, m.start - WINDOW), min(len(text), m.end + WINDOW), m)
    return found or Typed("other", "", 10**9)


def _better(a: Typed, b: Typed) -> bool:
    """Whether ``a`` decides a company's type over ``b``: a supply cue over a competitor
    cue over none; between two of one rank, the closer."""
    rank = {"customer": 0, "supplier": 0, "competitor": 1, "other": 2}
    return (rank[a.relation], a.distance) < (rank[b.relation], b.distance)


def type_filing(text: str, index: NameIndex, filer: int) -> dict[int, Typed]:
    """Each other company named in one 10-K and the type of its relation to the filer."""
    stops = sentence_stops(text)
    out: dict[int, Typed] = {}
    for m in index.find(text):
        if m.other == filer:
            continue
        t = type_mention(text, m, stops)
        if m.other not in out or _better(t, out[m.other]):
            out[m.other] = t
    return out


def top_names(ciks: pd.DataFrame, plan: pd.DataFrame) -> dict[str, set[int]]:
    """Month to the CIKs of its 1500 largest names by dollar volume."""
    sym_cik = dict(zip(ciks["symbol"], ciks["cik"], strict=True))
    out: dict[str, set[int]] = {}
    for month, g in plan.groupby("month"):
        top = g.nlargest(TOP_NAMES, "dollars")["symbol"]
        out[str(month)] = {int(sym_cik[s]) for s in top if s in sym_cik}
    return out


def allowed_for(accepted: pd.Timestamp, top: dict[str, set[int]]) -> set[int]:
    """The top names of the month before ``accepted``; the plan's first month before it."""
    if not top:
        return set()
    ny = accepted.tz_convert("America/New_York")
    year, month = (ny.year, ny.month - 1) if ny.month > 1 else (ny.year - 1, 12)
    prev = f"{year:04d}-{month:02d}"
    if prev in top:
        return top[prev]
    earlier = [m for m in sorted(top) if m <= prev]
    return top[earlier[-1]] if earlier else top[min(top)]


def valid_until(filings: pd.DataFrame) -> pd.Series:
    """For each filing, the next original 10-K's acceptance, at most 456 days on."""
    acc = pd.to_datetime(filings["accepted"], utc=True)
    t = nanos(acc)
    cap = t + pd.Timedelta(days=VALID_DAYS).value
    orig = np.sort(nanos(acc[filings["form"].isin(ORIGINAL_FORMS)]))
    if len(orig):
        i = np.searchsorted(orig, t, side="right")
        nxt = np.where(i < len(orig), orig[np.minimum(i, len(orig) - 1)], cap)
        cap = np.minimum(nxt, cap)
    return pd.Series(pd.to_datetime(cap, utc=True), index=filings.index)


def filing_edges(
    filings: pd.DataFrame, index: NameIndex, allowed: dict[str, set[int]] | None
) -> pd.DataFrame:
    """Typed edges from one filer's 10-Ks (columns cik, accn, form, accepted, text).

    ``allowed`` is month to permitted partner CIKs; ``None`` lets every name through.
    """
    f = filings.dropna(subset=["accepted"]).copy()
    if f.empty:
        return pd.DataFrame(columns=[*COLUMNS, "accn"])
    f["accepted"] = pd.to_datetime(f["accepted"], utc=True)
    f["text"] = f["text"].fillna("").astype(str)
    one = (
        f.sort_values("accepted")
        .groupby("accn", sort=False)
        .agg(
            cik=("cik", "first"),
            form=("form", "first"),
            accepted=("accepted", "first"),
            text=("text", "\n\n".join),
        )
        .reset_index()
    )
    one["valid_to"] = valid_until(one)
    rows = []
    for cik, accn, acc, valid_to, raw in zip(
        one["cik"].astype(int).tolist(),
        one["accn"].astype(str).tolist(),
        one["accepted"].tolist(),
        one["valid_to"].tolist(),
        one["text"].astype(str).tolist(),
        strict=True,
    ):
        text = raw.strip()
        if not text:
            continue
        ok = None if allowed is None else allowed_for(acc, allowed)
        for other, t in type_filing(text, index, cik).items():
            if ok is not None and other not in ok:
                continue
            rows.append((cik, other, t.relation, acc, valid_to, t.cue, accn))
    return pd.DataFrame(rows, columns=[*COLUMNS, "accn"])


def build(out: Path = EDGES) -> pd.DataFrame:
    """Every typed edge from the 10-Ks before the holdout, written to ``out``."""
    from stoa.newsbrain.pace import write_parquet

    ciks = pd.read_parquet(SEC_DIR / "ciks.parquet")
    names = name_book(ciks)
    plan = pd.read_parquet(PLAN, columns=["symbol", "month", "dollars"])
    allowed = top_names(ciks, plan)
    paths = sorted((SEC_DIR / "tenk").glob("*.parquet"))
    cols = ["cik", "accn", "form", "accepted", "text"]
    frames = []
    for path in paths:
        df = sealed.before_holdout(pd.read_parquet(path, columns=cols), "accepted")
        if not df.empty:
            frames.append(df)
    texts = [t for f in frames for t in f["text"].fillna("").astype(str) if t]
    common = common_words(texts, names)
    del texts
    index = NameIndex(names, common)
    log.info(
        "%d 10-K files, %d names in the book (%d common words), %d plan months",
        len(frames),
        len(index.by_name),
        len(common),
        len(allowed),
    )
    parts = []
    for k, df in enumerate(frames):
        parts.append(filing_edges(df, index, allowed))
        if k % 500 == 0:
            n = sum(len(p) for p in parts)
            log.info("10-K files %d of %d, %d edges so far", k, len(frames), n)
    edges = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    edges = edges.sort_values(["cik", "accepted", "other"]).reset_index(drop=True)
    edges = edges[["cik", "other", "relation", "accepted", "valid_to", "cue", "accn"]]
    write_parquet(edges, out)
    counts = edges["relation"].value_counts().to_dict()
    log.info("%d edges from %d filers: %s", len(edges), edges["cik"].nunique(), counts)
    return edges


# ---- reading the map ---------------------------------------------------------------------


def _utc(on: datetime | pd.Timestamp | str) -> pd.Timestamp:
    t = pd.Timestamp(on)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def flip(relation: str) -> str:
    """The relation seen from the other side: a customer's filer is its supplier."""
    return FLIP.get(relation, relation)


def both_ways(edges: pd.DataFrame) -> pd.DataFrame:
    """Each edge once from the filer's side and once from the named company's side.

    Columns: cik (whose view), partner, relation (from cik's view), accepted, valid_to,
    cue, said_by (the CIK whose 10-K says it).
    """
    keep = ["accepted", "valid_to", "cue"]
    own = edges[["cik", "other", "relation", *keep]].rename(columns={"other": "partner"})
    own = own.assign(said_by=edges["cik"].to_numpy())
    theirs = edges[["other", "cik", "relation", *keep]].rename(
        columns={"other": "cik", "cik": "partner"}
    )
    theirs = theirs.assign(relation=theirs["relation"].map(flip), said_by=edges["cik"].to_numpy())
    return pd.concat([own, theirs], ignore_index=True)


def partners(
    edges: pd.DataFrame, ciks: pd.DataFrame, cik: int, on: datetime | pd.Timestamp | str
) -> pd.DataFrame:
    """A company's partners on a date, from its own 10-Ks and from theirs.

    One row per (partner, the 10-K that says it, partner symbol): cik, partner,
    partner_symbol, relation (from ``cik``'s view), accepted, valid_to, cue, said_by.
    """
    t = _utc(on)
    live = edges[(edges["accepted"] <= t) & (edges["valid_to"] > t)]
    live = live[(live["cik"] == cik) | (live["other"] == cik)]
    view = both_ways(live)
    view = view[view["cik"] == cik]
    syms = ciks[["cik", "symbol"]].rename(columns={"cik": "partner", "symbol": "partner_symbol"})
    view = view.merge(syms, on="partner", how="left")
    cols = ["cik", "partner", "partner_symbol", "relation", "accepted", "valid_to", "cue"]
    return view[[*cols, "said_by"]].sort_values(["relation", "partner"]).reset_index(drop=True)


def symbol_edges(edges: pd.DataFrame, ciks: pd.DataFrame) -> pd.DataFrame:
    """The map in symbols, both ways: symbol, partner, relation (from symbol's view),
    accepted, valid_to. Every symbol of a CIK gets its edges."""
    view = both_ways(edges)
    a = ciks[["cik", "symbol"]]
    b = ciks[["cik", "symbol"]].rename(columns={"cik": "partner", "symbol": "partner_symbol"})
    out = view.merge(a, on="cik").merge(b, on="partner")
    out = out.drop(columns=["partner"]).rename(columns={"partner_symbol": "partner"})
    out = out[out["symbol"] != out["partner"]]
    cols = ["symbol", "partner", "relation", "accepted", "valid_to"]
    return out[cols].drop_duplicates().reset_index(drop=True)


def main() -> None:
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT / "logs" / "ripplemap.log", encoding="utf-8"),
        ],
    )
    build()
