"""Every 10-K and 10-Q since 2014 for the wide universe's filers, from the EDGAR archive.

For each filer (a CIK in ``data/newsbrain/sec/ciks.parquet``, the wide universe's SEC map)
the submissions index is refreshed once, then every original 10-K, 10-K405, 10-KT, 10-Q
and 10-QT filed from 2014-01-01 has its primary document downloaded as EDGAR serves it and
stored, with the filing's EDGAR acceptance time (Eastern wall clock in the index, stored
in UTC), its period of report and its filing date, in ``data/filings/raw/<cik>.parquet``
(zstd). Amendments (``/A``) are not pulled: most restate one part, and the trial compares
each periodic report with the same report a year earlier.

Fair access: one process, a declared User-Agent read from ``STOA_SEC_USER_AGENT`` in the
git-ignored ``.env``, at most 7 requests started a second across every worker (EDGAR
allows 10; a large document takes seconds to arrive, so up to 24 are in flight),
gzip on the wire, and a real pause after any 403 or 429; 20 of those stop the pull so a
person can look. Every non-200 answer is counted in the progress log. Filers whose stock
ever met the liquidity floor go first, most traded first, so the trial's universe is
complete early.
A filer whose file exists and holds every filing in its index is skipped, so the pull
can stop and start.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from stoa import envfile, wide
from stoa.data.sec import DATA_URL, RateLimiter, _accepted, make_sec_client
from stoa.filings import LOGS, PARQUET_LEVEL, RAW
from stoa.newsbrain.pace import log_progress

log = logging.getLogger(__name__)

ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
SINCE = "2014-01-01"
FORMS = ("10-K", "10-K405", "10-KT", "10-Q", "10-QT")
PER_SECOND = 7.0
WORKERS = 6  # filers in flight
DOWNLOADS = 24  # documents in flight across them
CIKS = Path("data/newsbrain/sec/ciks.parquet")
WIDE_BARS = Path("data/bars/wide")
SUBMISSIONS = Path("data/sec")
RAW_ROW_GROUP = 4
RAW_COLUMNS = ["cik", "accn", "form", "filed", "accepted", "report_date", "doc", "status", "raw"]


def user_agent() -> str:
    envfile.need("STOA_SEC_USER_AGENT")
    ua = os.environ["STOA_SEC_USER_AGENT"]
    if "@" not in ua:
        raise SystemExit("STOA_SEC_USER_AGENT must carry a contact address, as EDGAR asks")
    return ua


def periodic(recent: dict[str, Any], since: str = SINCE) -> pd.DataFrame:
    """The filer's original 10-K and 10-Q filings since ``since``, one row each."""
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
            "report_date": col("reportDate"),
            "doc": col("primaryDocument"),
        }
    )
    df = df[df["form"].isin(FORMS) & (df["filed"].astype(str) >= since)]
    return df.drop_duplicates("accn").sort_values("filed").reset_index(drop=True)


def decode(body: bytes) -> str:
    """EDGAR documents are UTF-8 or Windows-1252 and rarely say which."""
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("cp1252", errors="replace")


def doc_url(cik: int, accn: str, doc: str) -> str:
    folder = accn.replace("-", "")
    if not doc:  # no primary document named: the whole submission as one text file
        return f"{ARCHIVES}/{cik}/{folder}/{accn}.txt"
    return f"{ARCHIVES}/{cik}/{folder}/{doc}"


class Fetcher:
    """Raw documents, paced by the shared limiter, retried, with a real pause on refusal."""

    def __init__(self, client: httpx.Client, limiter: RateLimiter) -> None:
        self.client = client
        self.limiter = limiter
        self.refusals = 0  # 403 and 429: EDGAR asking us to slow down
        self.codes: Counter[str] = Counter()  # every answer that was not a 200, and errors
        self._lock = threading.Lock()

    def _note(self, code: str) -> None:
        with self._lock:
            self.codes[code] += 1
            if code in ("403", "429"):
                self.refusals += 1

    def get(self, url: str) -> tuple[str, bytes]:
        wait = 5.0
        for _ in range(7):
            self.limiter.wait()
            try:
                resp = self.client.get(url)
            except httpx.HTTPError as exc:
                self._note(type(exc).__name__)
                time.sleep(wait)
                wait = min(wait * 2, 300)
                continue
            if resp.status_code == 200:
                return "ok", resp.content
            self._note(str(resp.status_code))
            if resp.status_code == 404:
                return "missing", b""
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                # EDGAR's 403 means "slow down": stand still for a minute or more.
                time.sleep(max(wait, 60.0) if resp.status_code in (403, 429) else wait)
                wait = min(wait * 2, 600)
                continue
            return f"http{resp.status_code}", b""
        return "gave_up", b""


def ever_liquid(start: str = "2015-12-01") -> dict[str, float]:
    """Symbol to its peak 20 session dollar volume, for symbols that met the floor (close at
    least $5 and 20 session dollar volume at least $5M) on any session since ``start``."""
    out: dict[str, float] = {}
    for p in WIDE_BARS.glob("*.parquet"):
        df = pd.read_parquet(p, columns=["close", "volume"])
        df = df[pd.DatetimeIndex(df.index).tz_localize(None) >= pd.Timestamp(start)]
        if len(df) < wide.ADV_SESSIONS:
            continue
        ok = wide.liquid(df)
        if bool(ok.any()):
            out[p.stem] = float(wide.dollar_volume(df)[ok].max())
    return out


def order(ciks: dict[int, list[str]], liquid: dict[str, float]) -> list[int]:
    """Filers with an ever-liquid symbol first (most traded first), then the rest."""

    def key(c: int) -> tuple[int, float, int]:
        best = max((liquid.get(s, 0.0) for s in ciks[c]), default=0.0)
        return (0 if best > 0 else 1, -best, c)

    return sorted(ciks, key=key)


def filers() -> dict[int, list[str]]:
    df = pd.read_parquet(CIKS)
    out: dict[int, list[str]] = {}
    for cik, sym in zip(df["cik"], df["symbol"], strict=True):
        out.setdefault(int(cik), []).append(str(sym))
    return out


def raw_path(cik: int) -> Path:
    return RAW / f"{cik}.parquet"


def have(cik: int) -> set[str]:
    p = raw_path(cik)
    if not p.exists():
        return set()
    df = pd.read_parquet(p, columns=["accn", "status"])
    return set(df.loc[df["status"] == "ok", "accn"])


def write(df: pd.DataFrame, path: Path, row_group_size: int | None = None) -> None:
    from stoa.files import replace_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(
        tmp,
        index=False,
        compression="zstd",
        compression_level=PARQUET_LEVEL,
        row_group_size=row_group_size,
    )
    replace_file(tmp, path)


class Puller:
    def __init__(self, refresh: bool = True) -> None:
        self.sec = make_sec_client(user_agent(), SUBMISSIONS, PER_SECOND)
        self.fetch = Fetcher(self.sec.client, self.sec.limiter)
        self.refresh = refresh
        self.progress = LOGS / "pull.jsonl"
        # EDGAR takes seconds to serve a large document, so many downloads wait at once;
        # the shared limiter still lets at most PER_SECOND requests start each second.
        self.downloads = ThreadPoolExecutor(max_workers=DOWNLOADS)

    def index(self, cik: int) -> pd.DataFrame:
        """The filer's index: the recent block refreshed once, older pages from the cache."""
        name = f"submissions/CIK{cik:010d}.json"
        body = self.sec.get_json(f"{DATA_URL}/submissions/CIK{cik:010d}.json", name, self.refresh)
        if not isinstance(body, dict):
            return periodic({})
        fl = body.get("filings", {})
        recent = dict(fl.get("recent", {}))
        for extra in fl.get("files", []):
            fname = str(extra.get("name", ""))
            if not fname:
                continue
            more = self.sec.get_json(f"{DATA_URL}/submissions/{fname}", f"submissions/{fname}")
            if isinstance(more, dict):
                for k, v in more.items():
                    if isinstance(v, list):
                        recent[k] = list(recent.get(k, [])) + v
        return periodic(recent)

    def one(self, cik: int) -> dict[str, Any]:
        t0 = time.monotonic()
        try:
            idx = self.index(cik)
        except Exception as exc:  # one filer's index failing must not stop the pull
            return {"cik": cik, "error": f"index {type(exc).__name__}"}
        done = have(cik)
        todo = idx[~idx["accn"].isin(done)]
        if todo.empty:
            return {"cik": cik, "filings": len(idx), "fetched": 0, "skipped": True}
        old = pd.read_parquet(raw_path(cik)) if raw_path(cik).exists() else None
        rows = []
        items = list(todo.itertuples(index=False))
        got = self.downloads.map(lambda f: self.fetch.get(doc_url(cik, f.accn, f.doc)), items)
        for f, (status, body) in zip(items, got, strict=True):
            rows.append(
                {
                    "cik": cik,
                    "accn": f.accn,
                    "form": f.form,
                    "filed": f.filed,
                    "accepted": f.accepted,
                    "report_date": f.report_date,
                    "doc": f.doc,
                    "status": status,
                    "raw": decode(body) if status == "ok" else "",
                }
            )
        new = pd.DataFrame(rows, columns=RAW_COLUMNS)
        if old is not None:
            new = pd.concat([old[old["accn"].isin(done)], new], ignore_index=True)
        new["accepted"] = pd.to_datetime(new["accepted"], utc=True)
        # small row groups: the parse can then hold one filing at a time
        write(new.sort_values("filed").reset_index(drop=True), raw_path(cik), RAW_ROW_GROUP)
        return {
            "cik": cik,
            "filings": len(idx),
            "fetched": len(rows),
            "ok": int((new["status"] == "ok").sum()),
            "seconds": round(time.monotonic() - t0, 1),
        }

    def run(self, ciks: list[int], workers: int = WORKERS) -> None:
        log_progress(self.progress, started=len(ciks), at_utc=datetime.now(UTC).isoformat())
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(self.one, ciks):
                done += 1
                log_progress(
                    self.progress, done=done, of=len(ciks), codes=dict(self.fetch.codes), **res
                )
                if self.fetch.refusals >= 20:
                    log_progress(
                        self.progress,
                        stopped="20 slow-down answers from EDGAR",
                        codes=dict(self.fetch.codes),
                    )
                    raise SystemExit("EDGAR said slow down 20 times: stopping to look")
        log_progress(self.progress, finished=True, codes=dict(self.fetch.codes))


def main(limit: int | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    import json

    book = filers()
    cached = LOGS.parent / "liquid_symbols.json"
    if cached.exists():
        liquid = json.loads(cached.read_text(encoding="utf-8"))
    else:
        liquid = ever_liquid()
        cached.write_text(json.dumps(liquid), encoding="utf-8")
    queue = order(book, liquid)
    n_liquid = sum(1 for c in book if any(s in liquid for s in book[c]))
    log_progress(LOGS / "pull.jsonl", filers=len(book), ever_liquid_filers=n_liquid)
    Puller().run(queue[:limit] if limit else queue)
