"""Each filing's primary document as plain text, and three sections cut out of it.

Rules only; no language model. The HTML (or the plain text of older filings) becomes
text with paragraphs kept: scripts, styles, the hidden inline XBRL header and anything
styled ``display:none`` are dropped, tags removed, entities decoded, white space folded.

Sections, found by their headings:

- **Risk factors** (``rf``): 10-K Item 1A; 10-Q Part II Item 1A.
- **MD&A** (``mdna``): 10-K Item 7; 10-Q Part I Item 2.
- **Legal proceedings** (``legal``): 10-K Item 3; 10-Q Part II Item 1.

A section starts at a heading line that names it: its item label with the title on the
same line or the next, or the title alone as the whole line. It ends at the next heading
line: any item label, or another section's title alone on a line, or a "Part" line. The
table of contents names every section too, so of all the places a section could start,
the one that gives the longest section wins; a start with its item label always beats a
title alone (the notes to the financial statements often have their own "Legal
Proceedings" heading), which is used only when no item label names the section. A
section that is not found is left empty (a bank that puts its MD&A in exhibit 13, a 10-Q
with no legal proceedings item); the full document is always there.

The trading symbol a filing states for itself (inline XBRL ``dei:TradingSymbol``, on every
cover page since 2019 and on some before) is kept for the point-in-time ticker map.
"""

from __future__ import annotations

import html
import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from stoa.filings import LOGS, RAW, TEXT
from stoa.filings.pull import write
from stoa.newsbrain.pace import log_progress

log = logging.getLogger(__name__)

SECTIONS = ("rf", "mdna", "legal")
VERSION = "p1"  # changing a rule here is a new parse version

_DOC = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S | re.I)
_TEXTTAG = re.compile(r"<TEXT>(.*?)</TEXT>", re.S | re.I)
_SCRIPT = re.compile(r"<(script|style|head|title)\b[^>]*>.*?</\1\s*>", re.I | re.S)
_IXHEADER = re.compile(r"<ix:header\b.*?</ix:header\s*>", re.I | re.S)
_HIDDEN = re.compile(r"<(div|span|p)\b[^>]*display\s*:\s*none[^>]*>.*?</\1\s*>", re.I | re.S)
_BREAK = re.compile(r"<\s*(br|/p|/div|/tr|/li|/h\d|/table|/center|/title)\b[^>]*>", re.I)
_CELL = re.compile(r"<\s*/t[dh]\s*>", re.I)
_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t\r\f\v\xa0\u200b\u2002\u2003\u2009]+")
_LINES = re.compile(r"\n\s*\n+")
_IS_HTML = re.compile(r"<(html|body|p|div|table|font|span)\b", re.I)
_NEWLINES = re.compile(r"[\r\n]+")
_SYMBOL = re.compile(r"name=\"dei:TradingSymbol\"[^>]*>(.*?)</ix:nonNumeric", re.I | re.S)
_SYMBOL_XML = re.compile(r"<dei:TradingSymbol[^>]*>(.*?)</dei:TradingSymbol>", re.I | re.S)

_ITEM = r"item\s*(\d{1,2}\s*[a-c]?)\s*[\.\:\-\u2013\u2014\)]*\s*"
_ITEM_LINE = re.compile(r"^(?:part\s+[iv]+\s*[\.\,\-\u2013\u2014]*\s*)?" + _ITEM + r"(.*)$", re.I)
_END = r"[\s\.\:\-\u2013\u2014]*(\(continued\))?[\s\.\:]*$"

# Each section's title. After an item label it may run on; alone on a line it must be the
# whole line.
TITLES: dict[str, re.Pattern[str]] = {
    "rf": re.compile(r"^risk\s+factors\b", re.I),
    "mdna": re.compile(r"^management.{0,3}s?\s+discussion\s+and\s+analysis", re.I),
    "legal": re.compile(r"^legal\s+proceedings\b", re.I),
}
ALONE: dict[str, re.Pattern[str]] = {
    "rf": re.compile(r"^risk\s+factors" + _END, re.I),
    "mdna": re.compile(
        r"^management.{0,3}s?\s+discussion\s+and\s+analysis\s+of\s+(the\s+)?financial\s+"
        r"condition\s+and\s+results\s+of\s+operations" + _END,
        re.I,
    ),
    "legal": re.compile(r"^legal\s+proceedings" + _END, re.I),
}
ITEMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("rf", "10-K"): ("1a",),
    ("mdna", "10-K"): ("7",),
    ("legal", "10-K"): ("3",),
    ("rf", "10-Q"): ("1a",),
    ("mdna", "10-Q"): ("2",),
    ("legal", "10-Q"): ("1",),
}
# Other sections' titles alone on a line, or a "Part" line: either ends a section.
OTHER_TITLES = re.compile(
    r"^((business|unresolved\s+staff\s+comments|cybersecurity|properties|mine\s+safety\s+"
    r"disclosures?|\[?reserved\]?|exhibits(\s+and\s+financial\s+statement\s+schedules)?|"
    r"signatures?|other\s+information|controls\s+and\s+procedures|"
    r"quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market\s+risks?|"
    r"financial\s+statements(\s+and\s+supplementary\s+data)?|"
    r"unregistered\s+sales\s+of\s+equity\s+securities\s+and\s+use\s+of\s+proceeds|"
    r"defaults\s+upon\s+senior\s+securities|selected\s+(consolidated\s+)?financial\s+data|"
    r"market\s+for\s+(the\s+)?registrant.{0,120})" + _END + r"|part\s+(i|ii|iii|iv)\b.{0,80}$)",
    re.I,
)
MAX_HEADING = 160  # characters: a longer line is prose, not a heading


def html_text(raw: str) -> str:
    """Readable text from an EDGAR HTML or text document, paragraphs kept. In HTML a line
    break in the source is only a space (lines come from block tags); in a plain text
    filing it is a line."""
    m = _DOC.search(raw)
    if m:  # a full submission: the first document only
        body = m.group(1)
        inner = _TEXTTAG.search(body)
        raw = inner.group(1) if inner else body
    if _IS_HTML.search(raw[:20000]):
        raw = _NEWLINES.sub(" ", raw)
    t = _IXHEADER.sub(" ", raw)
    t = _SCRIPT.sub(" ", t)
    t = _HIDDEN.sub(" ", t)
    t = _BREAK.sub("\n", t)
    t = _CELL.sub(" ", t)
    t = html.unescape(_TAG.sub(" ", t))
    t = _SPACES.sub(" ", t)
    t = _LINES.sub("\n\n", t)
    return "\n".join(line.strip() for line in t.splitlines()).strip()


def trading_symbol(raw: str) -> str:
    """The first trading symbol the filing states for itself, or ""."""
    for rx in (_SYMBOL, _SYMBOL_XML):
        m = rx.search(raw)
        if m:
            sym = html.unescape(_TAG.sub("", m.group(1))).strip().upper()
            sym = sym.split(",")[0].split(";")[0].split()[0] if sym else ""
            if re.fullmatch(r"[A-Z]{1,5}([\.\-][A-Z]{1,2})?", sym):
                return sym.replace("-", ".")
    return ""


def _norm_item(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def heading_lines(lines: list[str]) -> list[tuple[int, str, str]]:
    """(line index, item label or "", rest of the line) for every heading-like line."""
    out = []
    for i, line in enumerate(lines):
        if not line or len(line) > MAX_HEADING:
            continue
        m = _ITEM_LINE.match(line)
        if m:
            out.append((i, _norm_item(m.group(1)), m.group(2).strip()))
        elif OTHER_TITLES.match(line) or any(rx.match(line) for rx in ALONE.values()):
            out.append((i, "", line))
    return out


def base_form(form: str) -> str:
    return "10-K" if form.startswith("10-K") else "10-Q"


def section(lines: list[str], heads: list[tuple[int, str, str]], name: str, form: str) -> str:
    """The longest span that starts at a heading naming ``name`` and runs to the next
    heading line after it."""
    items = ITEMS[(name, base_form(form))]
    # (first body line, end line, characters): item-labeled starts, then title-only ones
    best = {True: (0, 0, 0), False: (0, 0, 0)}
    for k, (i, item, rest) in enumerate(heads):
        if item:
            if item not in items:
                continue
            if rest:
                if not TITLES[name].match(rest):
                    continue
                body = i + 1
            elif i + 1 < len(lines) and TITLES[name].match(lines[i + 1]):
                body = i + 2  # "Item 1A." on its own line, the title on the next
            else:
                continue
        elif ALONE[name].match(lines[i]):
            body = i + 1
        else:
            continue
        end = next((j for j, _it, _r in heads[k + 1 :] if j >= body), len(lines))
        size = sum(len(x) + 1 for x in lines[body:end])
        labeled = bool(item)
        if size > best[labeled][2]:
            best[labeled] = (body, end, size)
    b = best[True] if best[True][2] else best[False]
    return "\n".join(lines[b[0] : b[1]]).strip() if b[2] else ""


def parse_doc(raw: str, form: str) -> dict[str, Any]:
    text = html_text(raw)
    lines = text.split("\n")
    heads = heading_lines(lines)
    out: dict[str, Any] = {"full": text, "symbol_tag": trading_symbol(raw)}
    for name in SECTIONS:
        out[name] = section(lines, heads, name, form)
    return out


def text_path(cik: int) -> Path:
    return TEXT / f"{cik}.parquet"


COLUMNS = [
    "cik",
    "accn",
    "form",
    "filed",
    "accepted",
    "report_date",
    "symbol_tag",
    "full",
    "rf",
    "mdna",
    "legal",
    "parse_version",
]


def parse_file(path: Path) -> dict[str, Any]:
    t0 = time.monotonic()
    cik = int(path.stem)
    out = text_path(cik)
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        old = pd.read_parquet(out, columns=["parse_version"])
        if set(old["parse_version"]) <= {VERSION}:
            return {"cik": cik, "skipped": True}
    raw = pd.read_parquet(path)
    raw = raw[raw["status"] == "ok"]
    rows = []
    for r in raw.itertuples(index=False):
        p = parse_doc(str(r.raw), str(r.form))
        rows.append(
            {
                "cik": cik,
                "accn": r.accn,
                "form": r.form,
                "filed": r.filed,
                "accepted": r.accepted,
                "report_date": r.report_date,
                **p,
                "parse_version": VERSION,
            }
        )
    write(pd.DataFrame(rows, columns=COLUMNS), out)
    return {"cik": cik, "filings": len(rows), "seconds": round(time.monotonic() - t0, 1)}


def main(workers: int = 12) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    files = sorted(RAW.glob("*.parquet"))
    progress = LOGS / "parse.jsonl"
    n = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(parse_file, files, chunksize=1):
            n += 1
            if not res.get("skipped"):
                log_progress(progress, done=n, of=len(files), **res)
    log_progress(progress, finished=True, files=len(files))
