"""Exhibit 21 (subsidiaries of the registrant) from every 10-K since 2015, for the ticker matcher.

The 10-K pass of ``stoa.newsbrain.sec_pull`` kept only the customer paragraphs, so the
subsidiary lists are pulled here, from EDGAR itself, with the same client, the same
declared User-Agent and the same pace (8 requests a second; the SEC allows 10). For each
10-K the filing's index.json names its documents; the one whose file name reads as
exhibit 21 is fetched and kept as text. Resumable per company, like the other SEC passes.

``known_at`` of a list is its 10-K's EDGAR acceptance time, so the matcher uses a list
only from the moment it was public.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from stoa.contracts import ROOT
from stoa.data.sec import make_sec_client
from stoa.newsbrain.pace import log_progress, write_parquet
from stoa.newsbrain.sec_pull import (
    ARCHIVES,
    SEC_DIR,
    USER_AGENT,
    Fetcher,
    by_liquidity,
    decode,
    doc_url,
    filings,
    html_text,
)

log = logging.getLogger(__name__)

EX21_DIR = ROOT / "ex21"
TENK_FORMS = ("10-K", "10-K405", "10-KT", "10-K/A")
COLUMNS = ["cik", "accn", "form", "filed", "accepted", "doc", "text"]
_EX21 = re.compile(r"(?:ex|exhibit)[-_ ]?21(?:[-_.]?[0-9]{1,2})?(?![0-9])", re.I)


def is_ex21(name: str) -> bool:
    """'ex21.htm', 'lmt-20231231xex21.htm', 'exhibit211.htm', 'd123dex21.htm' are exhibit 21."""
    low = name.lower()
    if not low.endswith((".htm", ".html", ".txt")):
        return False
    stem = low.rsplit(".", 1)[0]
    return bool(_EX21.search(stem))


def ex21_names(fetch: Fetcher, cik: int, accn: str) -> list[str]:
    folder = accn.replace("-", "")
    resp = fetch.get(f"{ARCHIVES}/{cik}/{folder}/index.json")
    if resp is None:
        return []
    try:
        items = resp.json().get("directory", {}).get("item", [])
    except ValueError:
        return []
    return [str(i.get("name", "")) for i in items if is_ex21(str(i.get("name", "")))]


def company_lists(fetch: Fetcher, cik: int, fl: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sel = fl[fl["form"].isin(TENK_FORMS)]
    for _, f in sel.iterrows():
        for name in ex21_names(fetch, cik, f["accn"])[:2]:
            resp = fetch.get(doc_url(cik, f["accn"], name))
            if resp is None:
                continue
            rows.append(
                {
                    "cik": cik,
                    "accn": f["accn"],
                    "form": f["form"],
                    "filed": f["filed"],
                    "accepted": f["accepted"],
                    "doc": name,
                    "text": html_text(decode(resp.content)),
                }
            )
    return rows


# ---- targeted: EDGAR full-text search for the names the matcher could not place ----------------

FTS_URL = "https://efts.sec.gov/LATEST/search-index"
FTS_DIR = ROOT / "fts"
DOCS_DIR = ROOT / "ex21_docs"


def fts_query(client: Any, limiter: Any, key: str, cache: Path = FTS_DIR) -> list[dict[str, Any]]:
    """10-K exhibit 21 and 22 documents whose text holds ``key`` as a phrase, 2015 on.
    Cached per key, so a restart asks nothing twice."""
    import hashlib
    import json

    path = cache / f"{hashlib.sha256(key.encode()).hexdigest()[:20]}.json"
    if path.exists():
        return list(json.loads(path.read_text(encoding="utf-8"))["hits"])
    params = {
        "q": f'"{key}"',
        "forms": "10-K",
        "dateRange": "custom",
        "startdt": "2015-01-01",
        "enddt": date.today().isoformat(),
    }
    body: dict[str, Any] = {}
    for wait in (5.0, 15.0, 45.0, 90.0):
        limiter.wait()
        try:
            resp = client.get(FTS_URL, params=params)
        except Exception:
            import time

            time.sleep(wait)
            continue
        if resp.status_code == 200:
            body = resp.json()
            break
        import time

        time.sleep(wait)
    hits = []
    for h in body.get("hits", {}).get("hits", []):
        s = h.get("_source", {})
        ftype = str(s.get("file_type", ""))
        if not ftype.upper().startswith(("EX-21", "EX-22")):
            continue
        accn, _, doc = str(h.get("_id", "")).partition(":")
        for c in s.get("ciks", []):
            hits.append({"cik": int(c), "accn": accn, "doc": doc, "filed": s.get("file_date")})
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "ok": bool(body), "hits": hits}), encoding="utf-8")
    return hits


def pull_targeted(keys: list[str], universe: set[int], per_second: float = 2.0) -> dict[str, int]:
    """Search each key, then fetch every exhibit found whose filer is in the price universe."""
    from stoa.contracts.scrape import keep_awake

    keep_awake()
    sec = make_sec_client(USER_AGENT, Path("data/sec"), per_second)
    fetch = Fetcher(sec.client, sec.limiter)
    found: dict[tuple[int, str, str], str] = {}
    for i, key in enumerate(keys):
        for h in fts_query(sec.client, sec.limiter, key):
            if h["cik"] in universe:
                found[(h["cik"], h["accn"], h["doc"])] = str(h["filed"])
        if i % 200 == 0:
            log_progress(ROOT / "logs" / "fts.jsonl", queried=i, of=len(keys), docs=len(found))
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    got = 0
    for (cik, accn, doc), filed in sorted(found.items()):
        path = DOCS_DIR / f"{cik}_{accn}_{doc.replace('.', '_')}.parquet"
        if path.exists():
            continue
        resp = fetch.get(doc_url(cik, accn, doc))
        text = html_text(decode(resp.content)) if resp is not None else ""
        row = {"cik": cik, "accn": accn, "form": "10-K", "filed": filed, "accepted": None,
               "doc": doc, "text": text}  # fmt: skip
        write_parquet(pd.DataFrame([row], columns=COLUMNS), path)
        got += 1
    log_progress(ROOT / "logs" / "fts.jsonl", done=True, docs=len(found), fetched=got)
    return {"keys": len(keys), "docs": len(found), "fetched": got}


def fetch_found(
    universe: set[int], part: int = 0, parts: int = 1, per_second: float = 2.0
) -> dict[str, int]:
    """Fetch every exhibit the cached searches found for a company in the price universe.
    ``parts`` processes may share it (document ``i`` goes to part ``i % parts``)."""
    import json

    from stoa.contracts.scrape import keep_awake

    keep_awake()
    found: dict[tuple[int, str, str], str] = {}
    for p in sorted(FTS_DIR.glob("*.json")):
        for h in json.loads(p.read_text(encoding="utf-8")).get("hits", []):
            if int(h["cik"]) in universe:
                found[(int(h["cik"]), str(h["accn"]), str(h["doc"]))] = str(h["filed"])
    todo = [k for i, k in enumerate(sorted(found)) if i % parts == part]
    sec = make_sec_client(USER_AGENT, Path("data/sec"), per_second)
    fetch = Fetcher(sec.client, sec.limiter)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    got = 0
    for i, (cik, accn, doc) in enumerate(todo):
        path = DOCS_DIR / f"{cik}_{accn}_{doc.replace('.', '_')}.parquet"
        if path.exists():
            continue
        resp = fetch.get(doc_url(cik, accn, doc))
        text = html_text(decode(resp.content)) if resp is not None else ""
        row = {"cik": cik, "accn": accn, "form": "10-K", "filed": found[(cik, accn, doc)],
               "accepted": None, "doc": doc, "text": text}  # fmt: skip
        write_parquet(pd.DataFrame([row], columns=COLUMNS), path)
        got += 1
        if i % 200 == 0:
            log_progress(ROOT / "logs" / f"docs-{part}.jsonl", done=i, of=len(todo))
    log_progress(ROOT / "logs" / f"docs-{part}.jsonl", done=True, fetched=got, of=len(todo))
    return {"docs": len(found), "mine": len(todo), "fetched": got}


def is_finance(cik: int) -> bool:
    """SIC 6000 to 6799 (banks, insurers, real estate, funds), from the cached submissions."""
    import json

    p = Path("data/sec/submissions") / f"CIK{cik:010d}.json"
    if not p.exists():
        return False
    try:
        sic = int(json.loads(p.read_text(encoding="utf-8")).get("sic") or 0)
    except (ValueError, TypeError):
        return False
    return 6000 <= sic < 6800


def pull_ex21(
    root: Path = EX21_DIR, part: int = 0, parts: int = 1, per_second: float = 8.0
) -> None:
    """Every company's lists. ``parts`` processes may share the work (company ``i`` goes to
    part ``i % parts``), each at ``per_second``; together they stay at 8 a second.
    Operating companies first, then finance, each by liquidity."""
    from stoa.contracts.scrape import keep_awake

    keep_awake()
    sec = make_sec_client(USER_AGENT, Path("data/sec"), per_second)
    fetch = Fetcher(sec.client, sec.limiter)
    ciks_df = pd.read_parquet(SEC_DIR / "ciks.parquet")
    ciks: dict[int, list[str]] = {}
    for c, s in zip(ciks_df["cik"], ciks_df["symbol"], strict=True):
        ciks.setdefault(int(c), []).append(str(s))
    ranked = by_liquidity(ciks)
    order = [c for c in ranked if not is_finance(c)] + [c for c in ranked if is_finance(c)]
    order = [c for i, c in enumerate(order) if i % parts == part]
    progress = ROOT / "logs" / f"ex21-{part}.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    for i, cik in enumerate(order):
        path = root / f"{cik}.parquet"
        if path.exists():
            continue
        try:
            fl = filings(sec.submissions(cik))
        except Exception as exc:
            log.warning("submissions %s failed: %s", cik, type(exc).__name__)
            continue
        write_parquet(pd.DataFrame(company_lists(fetch, cik, fl), columns=COLUMNS), path)
        if i % 100 == 0:
            log_progress(progress, done=i, of=len(order))
    log_progress(progress, done=True, on=str(date.today()))
