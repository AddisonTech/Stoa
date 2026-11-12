"""Pull every daily contract announcement page from war.gov, 2016 to today.

The listing at /News/Contracts/?Page=N shows 10 announcements a page, newest first;
each links to one article per business day ("Contracts for Nov. 10, 2026"). The old
defense.gov addresses redirect to war.gov, so the archive is read from war.gov only,
never from a third-party mirror.

Manners: one request a second at most (every redirect hop counts), a plain crawler
user agent that names itself, robots.txt allows both the listing and the articles.
A refusal (403 or 429) backs off and retries; four refusals in a row mean the site is
blocking the scraper, and the pull stops with status "blocked" for a person to read.

Resumable: article pages are cached by article id under data/contracts/raw/articles
and never fetched twice; the listing walk saves the next page in state.json, and a
restart steps back two pages, because new posts push older ones down the listing.

Run it with ``python -m stoa.contracts pull``.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from stoa.contracts import ROOT

log = logging.getLogger(__name__)

BASE = "https://www.war.gov"
LISTING = BASE + "/News/Contracts/?Page={page}"
USER_AGENT = "Mozilla/5.0 (compatible; StoaResearch/1.0)"
START = date(2016, 1, 1)
INTERVAL = 1.0  # seconds between request starts
REFUSALS_TO_STOP = 4
BACKOFF = (30.0, 60.0, 120.0, 240.0)
MAX_HOPS = 5

# The listing carries each link in an article-url="..." attribute; older pages use href.
_ARTICLE = re.compile(
    r"""["']((?:https?://[^"'\s]+)?/News/Contracts/Contract/Article/(\d+)/([^"'/?#\s]+)/?)["']""",
    re.I,
)
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "june": 6, "jul": 7,
    "july": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}  # fmt: skip
_SLUG_DATE = re.compile(r"contracts?-for-([a-z]+)-(\d{1,2})-(\d{4})", re.I)
_TITLE_DATE = re.compile(r"Contracts? [Ff]or\s+([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})")


class Blocked(RuntimeError):
    """The site refused the scraper repeatedly: stop and report, do not work around it."""


@dataclass(frozen=True)
class Link:
    id: int
    slug: str
    url: str


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (a no-op elsewhere)."""
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def _month_day_year(month: str, day: str, year: str) -> date | None:
    m = _MONTHS.get(month.lower().rstrip("."))
    if m is None:
        return None
    try:
        return date(int(year), m, int(day))
    except ValueError:
        return None


def slug_date(slug: str) -> date | None:
    """'contracts-for-oct-28-2026' is 2026-10-28."""
    got = _SLUG_DATE.search(slug)
    return _month_day_year(*got.groups()) if got else None


def title_date(html: str) -> date | None:
    """The date in the article's own title, 'Contracts For Oct. 28, 2026'."""
    got = _TITLE_DATE.search(html)
    return _month_day_year(*got.groups()) if got else None


def listing_links(html: str) -> list[Link]:
    """Contract announcement links on a listing page, in page order, once each."""
    seen: dict[int, Link] = {}
    for href, aid, slug in _ARTICLE.findall(html):
        i = int(aid)
        if i not in seen:
            seen[i] = Link(i, slug.lower(), urljoin(BASE, href))
    return list(seen.values())


class Polite:
    """GET with at most one request a second, redirects followed by hand so each hop is paced."""

    def __init__(
        self,
        client: httpx.Client,
        interval: float = INTERVAL,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        backoff: tuple[float, ...] = BACKOFF,
    ) -> None:
        self.client = client
        self.interval = interval
        self.sleep = sleep
        self.monotonic = monotonic
        self.backoff = backoff
        self._last = -1e18
        self.requests = 0

    def _pace(self) -> None:
        wait = self._last + self.interval - self.monotonic()
        if wait > 0:
            self.sleep(wait)
        self._last = self.monotonic()

    def _once(self, url: str) -> httpx.Response:
        for _ in range(MAX_HOPS):
            self._pace()
            self.requests += 1
            resp = self.client.get(url, follow_redirects=False)
            if resp.status_code in (301, 302, 303, 307, 308) and "location" in resp.headers:
                url = urljoin(url, resp.headers["location"])
                continue
            return resp
        raise httpx.TooManyRedirects(f"more than {MAX_HOPS} redirects from {url}")

    def get(self, url: str) -> httpx.Response | None:
        """The page, None on 404; raises Blocked after repeated refusals."""
        refusals = 0
        errors = 0
        while True:
            try:
                resp = self._once(url)
            except httpx.HTTPError as exc:
                errors += 1
                if errors > len(self.backoff):
                    raise
                log.warning("%s on %s, retrying", type(exc).__name__, url)
                self.sleep(self.backoff[errors - 1])
                continue
            code = resp.status_code
            if code == 200:
                return resp
            if code == 404:
                return None
            if code in (403, 429):
                refusals += 1
                if refusals >= REFUSALS_TO_STOP:
                    raise Blocked(f"{code} {refusals} times in a row on {url}")
                self.sleep(self.backoff[min(refusals, len(self.backoff)) - 1])
                continue
            errors += 1
            if errors > len(self.backoff):
                resp.raise_for_status()
                return None
            self.sleep(self.backoff[errors - 1])


def make_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0)


# ---- the archive on disk ---------------------------------------------------------------


class Archive:
    """Raw pages, the link index and the walk's state under one folder."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.articles = root / "raw" / "articles"
        self.listing = root / "raw" / "listing"
        self.index_path = root / "index.jsonl"
        self.state_path = root / "state.json"
        self.log_path = root / "logs" / "pull.jsonl"

    def setup(self) -> None:
        for p in (self.articles, self.listing, self.log_path.parent):
            p.mkdir(parents=True, exist_ok=True)

    def state(self) -> dict[str, Any]:
        if self.state_path.exists():
            return dict(json.loads(self.state_path.read_text(encoding="utf-8")))
        return {"next_page": 1, "listing_done": False}

    def save_state(self, st: dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)

    def index(self) -> dict[int, Link]:
        out: dict[int, Link] = {}
        if self.index_path.exists():
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    out[int(d["id"])] = Link(int(d["id"]), d["slug"], d["url"])
        return out

    def add_links(self, links: list[Link], known: dict[int, Link]) -> int:
        new = [lk for lk in links if lk.id not in known]
        if new:
            with self.index_path.open("a", encoding="utf-8") as fh:
                for lk in new:
                    fh.write(json.dumps(asdict(lk)) + "\n")
                    known[lk.id] = lk
        return len(new)

    def article_path(self, aid: int) -> Path:
        return self.articles / f"{aid}.html"

    def has(self, aid: int) -> bool:
        return self.article_path(aid).exists()

    def write_article(self, aid: int, body: str) -> None:
        p = self.article_path(aid)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(p)

    def log(self, **kw: Any) -> None:
        kw = {"at": datetime.now(UTC).isoformat(timespec="seconds"), **kw}
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(kw, default=str) + "\n")


def link_date(lk: Link) -> date | None:
    return slug_date(lk.slug)


def walk_listing(
    web: Polite, arc: Archive, start: date = START, max_pages: int | None = None
) -> dict[str, Any]:
    """Collect announcement links page by page until the walk passes ``start``."""
    st = arc.state()
    known = arc.index()
    if st.get("listing_done"):
        return st
    page = max(1, int(st.get("next_page", 1)) - 2)
    walked = 0
    while max_pages is None or walked < max_pages:
        resp = web.get(LISTING.format(page=page))
        walked += 1
        links = listing_links(resp.text) if resp is not None else []
        if resp is not None:
            (arc.listing / f"page-{page:04d}.html").write_text(resp.text, encoding="utf-8")
        new = arc.add_links(links, known)
        dates = [d for d in (link_date(lk) for lk in links) if d is not None]
        arc.log(step="listing", page=page, links=len(links), new=new,
                oldest=min(dates) if dates else None)  # fmt: skip
        if not links or (dates and max(dates) < start):
            st = {"next_page": page + 1, "listing_done": True}
            arc.save_state(st)
            break
        page += 1
        st = {"next_page": page, "listing_done": False}
        arc.save_state(st)
    return st


def refresh_listing(web: Polite, arc: Archive, pages: int = 2) -> int:
    """The newest announcements: the first pages again, for the nightly run and a final top-up."""
    known = arc.index()
    added = 0
    for page in range(1, pages + 1):
        resp = web.get(LISTING.format(page=page))
        if resp is None:
            break
        added += arc.add_links(listing_links(resp.text), known)
    return added


def wanted(arc: Archive, start: date = START, end: date | None = None) -> list[Link]:
    """Indexed announcements in [start, end] not yet on disk, oldest first."""
    out = []
    for lk in arc.index().values():
        d = link_date(lk)
        if d is not None and d >= start and (end is None or d <= end) and not arc.has(lk.id):
            out.append(lk)
    return sorted(out, key=lambda lk: (link_date(lk) or START, lk.id))


def fetch_articles(web: Polite, arc: Archive, links: list[Link]) -> int:
    got = 0
    for i, lk in enumerate(links):
        resp = web.get(lk.url)
        if resp is None:
            arc.log(step="article", id=lk.id, slug=lk.slug, status=404)
            continue
        arc.write_article(lk.id, resp.text)
        got += 1
        if i % 50 == 0:
            arc.log(step="article", done=i, of=len(links), slug=lk.slug)
    return got


def pull(
    root: Path = ROOT,
    start: date = START,
    client: httpx.Client | None = None,
    web: Polite | None = None,
    max_pages: int | None = None,
) -> dict[str, Any]:
    """The whole archive, resumable. Returns a status dict; ``status`` is ok or blocked."""
    keep_awake()
    arc = Archive(root)
    arc.setup()
    web = web or Polite(client or make_client())
    try:
        st = walk_listing(web, arc, start, max_pages)
        if st.get("listing_done"):
            refresh_listing(web, arc)
        todo = wanted(arc, start)
        arc.log(step="articles", todo=len(todo))
        got = fetch_articles(web, arc, todo)
    except Blocked as exc:
        arc.log(step="stop", status="blocked", reason=str(exc))
        return {"status": "blocked", "reason": str(exc), "requests": web.requests}
    out = {
        "status": "ok",
        "listing_done": bool(st.get("listing_done")),
        "fetched": got,
        "indexed": len(arc.index()),
        "requests": web.requests,
    }
    arc.log(step="done", **out)
    return out
