"""Background pull of SEC filing text from EDGAR for the wide universe.

Three passes, each resumable per company (a file per CIK and pass):

1. ``earnings``: every 8-K with item 2.02 since 2015, its main document and its
   press release exhibits (EX-99 files).
2. ``tenk``: every 10-K since 2015, reduced to the paragraphs that name
   customers or suppliers with a share of sales (the concentration disclosures
   the ripple feature maps).
3. ``eightk``: every other 8-K since 2015, its main document.

Fair access: one process, a declared User-Agent with a contact email, at most
8 requests a second (the SEC allows 10). The filings lists come from the
submissions JSON, cached under data/sec like the rest of Stoa's SEC data.
``known_at`` for every document is the filing's EDGAR acceptance time.
"""

from __future__ import annotations

import html
import logging
import re
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from stoa.data.sec import RateLimiter, SECClient, _accepted, make_sec_client
from stoa.newsbrain import ROOT
from stoa.newsbrain.alpaca_pull import universe
from stoa.newsbrain.pace import log_progress, write_parquet

log = logging.getLogger(__name__)

SEC_DIR = ROOT / "sec"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
USER_AGENT = "Addison Smith addisontechdev@gmail.com"
SINCE = "2015-01-01"
TEXT_LIMIT = 200_000  # characters kept per document
PASSES = ("earnings", "tenk", "eightk")

_SCRIPT = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
_BREAK = re.compile(r"<\s*(br|/p|/div|/tr|/li|/h\d)[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t\r\f\v\xa0]+")
_LINES = re.compile(r"\n\s*\n+")
_EX99 = re.compile(r"ex[-_ ]?99|exhibit[-_ ]?99|ex991|dex99", re.I)
_SHARE = re.compile(r"\d+(\.\d+)?\s*(%|percent)", re.I)
_PARTY = re.compile(r"\b(customers?|clients?|suppliers?|vendors?|distributors?)\b", re.I)
_SALES = re.compile(r"\b(revenues?|net sales|sales|purchases)\b", re.I)


def html_text(raw: str) -> str:
    """Readable text from an EDGAR HTML or text document, paragraphs kept."""
    t = _SCRIPT.sub(" ", raw)
    t = _BREAK.sub("\n", t)
    t = html.unescape(_TAG.sub(" ", t))
    t = _SPACES.sub(" ", t)
    t = _LINES.sub("\n\n", t)
    return "\n".join(line.strip() for line in t.splitlines()).strip()


def decode(body: bytes) -> str:
    """EDGAR documents are UTF-8 or Windows-1252 and rarely say which."""
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("cp1252", errors="replace")


def is_ex99(name: str) -> bool:
    low = name.lower()
    return bool(_EX99.search(low)) and low.endswith((".htm", ".html", ".txt"))


def customer_paragraphs(text: str, limit: int = 40) -> str:
    """Paragraphs that tie a named customer or supplier to a share of sales."""
    keep = []
    for para in text.split("\n"):
        p = para.strip()
        if 40 <= len(p) <= 3000 and _PARTY.search(p) and _SHARE.search(p) and _SALES.search(p):
            keep.append(p)
            if len(keep) >= limit:
                break
    return "\n\n".join(keep)


def filings(recent: dict[str, Any]) -> pd.DataFrame:
    n = len(recent.get("accessionNumber", []))

    def col(name: str) -> list[Any]:
        v = list(recent.get(name, []))
        return v + [""] * (n - len(v))

    df = pd.DataFrame(
        {
            "accn": col("accessionNumber"),
            "form": col("form"),
            "filed": col("filingDate"),
            "accepted": [_accepted(str(a)) for a in col("acceptanceDateTime")],
            "items": [str(i) for i in col("items")],
            "doc": col("primaryDocument"),
        }
    )
    return df[df["filed"].astype(str) >= SINCE].reset_index(drop=True)


class Fetcher:
    """Raw documents from the EDGAR archive, paced and retried."""

    def __init__(self, client: httpx.Client, limiter: RateLimiter) -> None:
        self.client = client
        self.limiter = limiter

    def get(self, url: str) -> httpx.Response | None:
        wait = 5.0
        for _ in range(6):
            self.limiter.wait()
            try:
                resp = self.client.get(url)
            except httpx.HTTPError:
                time.sleep(wait)
                wait *= 2
                continue
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return None
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                # 403 from EDGAR usually means "slow down"; take a real pause.
                time.sleep(max(wait, 60 if resp.status_code == 403 else wait))
                wait *= 2
                continue
            return None
        return None

    def text(self, url: str) -> str:
        resp = self.get(url)
        if resp is None:
            return ""
        return html_text(decode(resp.content))[:TEXT_LIMIT]

    def exhibit_names(self, cik: int, accn: str) -> list[str]:
        folder = accn.replace("-", "")
        resp = self.get(f"{ARCHIVES}/{cik}/{folder}/index.json")
        if resp is None:
            return []
        try:
            items = resp.json().get("directory", {}).get("item", [])
        except ValueError:
            return []
        return [str(i.get("name", "")) for i in items if is_ex99(str(i.get("name", "")))]


def doc_url(cik: int, accn: str, name: str) -> str:
    return f"{ARCHIVES}/{cik}/{accn.replace('-', '')}/{name}"


def _row(cik: int, f: pd.Series, kind: str, doc: str, text: str) -> dict[str, Any]:
    return {
        "cik": cik,
        "accn": f["accn"],
        "form": f["form"],
        "items": f["items"],
        "filed": f["filed"],
        "accepted": f["accepted"],
        "kind": kind,
        "doc": doc,
        "text": text,
    }


def earnings_docs(fetch: Fetcher, cik: int, fl: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    sel = fl[fl["form"].isin(["8-K", "8-K/A"]) & fl["items"].str.contains("2.02", regex=False)]
    for _, f in sel.iterrows():
        if f["doc"]:
            rows.append(
                _row(cik, f, "main", f["doc"], fetch.text(doc_url(cik, f["accn"], f["doc"])))
            )
        for name in fetch.exhibit_names(cik, f["accn"])[:3]:
            rows.append(_row(cik, f, "ex99", name, fetch.text(doc_url(cik, f["accn"], name))))
    return rows


def tenk_docs(fetch: Fetcher, cik: int, fl: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    sel = fl[fl["form"].isin(["10-K", "10-K405", "10-KT", "10-K/A"]) & (fl["doc"] != "")]
    for _, f in sel.iterrows():
        resp = fetch.get(doc_url(cik, f["accn"], f["doc"]))
        if resp is None:
            continue
        rows.append(
            _row(
                cik, f, "customers", f["doc"], customer_paragraphs(html_text(decode(resp.content)))
            )
        )
    return rows


def eightk_docs(fetch: Fetcher, cik: int, fl: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    sel = fl[
        fl["form"].isin(["8-K", "8-K/A"])
        & ~fl["items"].str.contains("2.02", regex=False)
        & (fl["doc"] != "")
    ]
    for _, f in sel.iterrows():
        rows.append(_row(cik, f, "main", f["doc"], fetch.text(doc_url(cik, f["accn"], f["doc"]))))
    return rows


PASS_FN: dict[str, Callable[[Fetcher, int, pd.DataFrame], list[dict[str, Any]]]] = {
    "earnings": earnings_docs,
    "tenk": tenk_docs,
    "eightk": eightk_docs,
}


def company_ciks(sec: SECClient) -> dict[int, list[str]]:
    """CIK to the universe symbols it covers: the SEC map, then the insider filings."""
    from stoa.event_run import cik_map

    purchases_path = Path("data/insider/purchases.parquet")
    purchases = pd.read_parquet(purchases_path) if purchases_path.exists() else pd.DataFrame()
    ciks = cik_map(sec.tickers(), purchases)
    out: dict[int, list[str]] = {}
    for sym in universe():
        cik = ciks.get(sym)
        if cik is not None:
            out.setdefault(int(cik), []).append(sym)
    return out


def by_liquidity(ciks: dict[int, list[str]]) -> list[int]:
    """Companies whose stock traded the most dollars in any month first: they are tradable."""
    from stoa.newsbrain.alpaca_pull import PLAN

    if not PLAN.exists():
        return sorted(ciks)
    plan = pd.read_parquet(PLAN, columns=["symbol", "dollars"])
    peak = plan.groupby("symbol")["dollars"].max().to_dict()
    best = {cik: max(peak.get(s, 0.0) for s in syms) for cik, syms in ciks.items()}
    return sorted(ciks, key=lambda c: (-best[c], c))


def pull_sec(passes: tuple[str, ...] = PASSES) -> None:
    sec = make_sec_client(USER_AGENT, Path("data/sec"), 8.0)
    fetch = Fetcher(sec.client, sec.limiter)
    ciks = company_ciks(sec)
    write_parquet(
        pd.DataFrame([(c, s) for c, syms in ciks.items() for s in syms], columns=["cik", "symbol"]),
        SEC_DIR / "ciks.parquet",
    )
    progress = ROOT / "logs" / "sec.jsonl"
    order = by_liquidity(ciks)
    for name in passes:
        fn = PASS_FN[name]
        for i, cik in enumerate(order):
            path = SEC_DIR / name / f"{cik}.parquet"
            if path.exists():
                continue
            try:
                fl = filings(sec.submissions(cik))
            except Exception as exc:
                log.warning("submissions %s failed: %s", cik, type(exc).__name__)
                continue
            rows = fn(fetch, cik, fl)
            cols = ["cik", "accn", "form", "items", "filed", "accepted", "kind", "doc", "text"]
            write_parquet(pd.DataFrame(rows, columns=cols), path)
            if i % 100 == 0:
                log_progress(progress, pass_=name, done=i, of=len(ciks))
        log_progress(progress, pass_=name, finished=True)
    log_progress(progress, done=True, on=str(date.today()))
