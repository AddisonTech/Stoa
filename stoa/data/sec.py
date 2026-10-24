"""SEC EDGAR: point-in-time fundamentals and historical earnings release times.

Sources (free, no key):
- ``https://www.sec.gov/files/company_tickers.json``: ticker to CIK.
- ``https://data.sec.gov/submissions/CIK##########.json``: every filing with its
  form, filing date, acceptance time and 8-K item numbers. Older filings are
  in the extra files listed under ``filings.files``.
- ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json``: every XBRL
  fact a company has filed, each tagged with the filing it came from
  (``accn``) and the date it was filed.

SEC fair access rules: declare a User-Agent with a contact email, and stay
under 10 requests per second. Stoa throttles to ``sources.sec_rate_per_second``
and caches every response on disk, so a second run does not hit the SEC.

Point in time:
- A fact's ``known_at`` is the acceptance time of the filing that reported it,
  looked up by accession number in the submissions file. If the accession is
  not found, the ``filed`` date is used at the end of that New York day.
- An 8-K with item 2.02 ("Results of Operations and Financial Condition") is
  the earnings release. Its acceptance time is the moment the market learned
  the numbers. That gives a point-in-time history of earnings dates.

Restatements: companyfacts keeps every version of a value, each with its own
filing. Using the earliest filing for each period end means a later
restatement never leaks backwards.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from stoa import pit
from stoa.data.base import DataError
from stoa.files import replace_file
from stoa.net import request_with_retry

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DATA_URL = "https://data.sec.gov"
EARNINGS_ITEM = "2.02"
PERIODIC_FORMS = frozenset({"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"})

# XBRL tags, in order of preference, for each line item. Companies use
# different tags for the same concept; the first one present wins.
TAGS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ),
    "cost_of_revenue": (
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ),
    "gross_profit": ("GrossProfit",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "long_term_debt": ("LongTermDebtNoncurrent", "LongTermDebt"),
}
FLOW_ITEMS = ("revenue", "cost_of_revenue", "gross_profit", "net_income", "operating_cash_flow")
STOCK_ITEMS = ("assets", "liabilities", "equity", "long_term_debt")
QUALITY_COLUMNS = (
    "q_gross_profitability",
    "q_roe",
    "q_accruals",
    "q_leverage",
    "q_earnings_stability",
)


class RateLimiter:
    def __init__(
        self,
        per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = 1.0 / per_second
        self.clock = clock
        self.sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self.clock()
            if now < self._next:
                self.sleep(self._next - now)
                now = self._next
            self._next = now + self.interval


class SECClient:
    """Cached, throttled access to the EDGAR JSON APIs."""

    def __init__(
        self,
        client: httpx.Client,
        cache_dir: Path | None,
        limiter: RateLimiter,
        *,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.cache_dir = cache_dir
        self.limiter = limiter
        self.max_retries = max_retries
        self.sleep = sleep

    def _cached(self, name: str) -> Any | None:
        if self.cache_dir is None:
            return None
        p = self.cache_dir / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def _store(self, name: str, body: Any) -> None:
        if self.cache_dir is None:
            return
        p = self.cache_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(body), encoding="utf-8")
        replace_file(tmp, p)

    def get_json(self, url: str, cache_name: str | None, refresh: bool = False) -> Any | None:
        """GET a JSON document. Returns None on 404 (a company with no XBRL facts)."""
        if cache_name and not refresh:
            hit = self._cached(cache_name)
            if hit is not None:
                return hit
        self.limiter.wait()
        resp = request_with_retry(
            self.client, "GET", url, max_retries=self.max_retries, sleep=self.sleep
        )
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise DataError(f"SEC request {httpx.URL(url).path} failed: HTTP {resp.status_code}")
        body = resp.json()
        if cache_name:
            self._store(cache_name, body)
        return body

    def tickers(self, refresh: bool = False) -> dict[str, int]:
        body = self.get_json(TICKERS_URL, "company_tickers.json", refresh)
        if not isinstance(body, dict):
            raise DataError("SEC company_tickers.json is not a mapping")
        out: dict[str, int] = {}
        for row in body.values():
            out[str(row["ticker"]).upper().replace("-", ".")] = int(row["cik_str"])
        return out

    def submissions(self, cik: int, refresh: bool = False) -> dict[str, Any]:
        name = f"submissions/CIK{cik:010d}.json"
        body = self.get_json(f"{DATA_URL}/submissions/CIK{cik:010d}.json", name, refresh)
        if not isinstance(body, dict):
            return {}
        filings = body.get("filings", {})
        recent = dict(filings.get("recent", {}))
        for extra in filings.get("files", []):
            fname = str(extra.get("name", ""))
            if not fname:
                continue
            more = self.get_json(f"{DATA_URL}/submissions/{fname}", f"submissions/{fname}", refresh)
            if isinstance(more, dict):
                for k, v in more.items():
                    if isinstance(v, list):
                        recent.setdefault(k, [])
                        recent[k] = list(recent[k]) + v
        return recent

    def companyfacts(self, cik: int, refresh: bool = False) -> dict[str, Any] | None:
        name = f"companyfacts/CIK{cik:010d}.json"
        body = self.get_json(f"{DATA_URL}/api/xbrl/companyfacts/CIK{cik:010d}.json", name, refresh)
        return body if isinstance(body, dict) else None


def make_sec_client(user_agent: str, cache_dir: Path | None, per_second: float) -> SECClient:
    if "@" not in user_agent:
        raise ValueError("SEC requires a User-Agent with a contact email")
    client = httpx.Client(
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=30.0,
        follow_redirects=True,
    )
    return SECClient(client, cache_dir, RateLimiter(per_second))


# ---- parsing ------------------------------------------------------------------------


def _accepted(value: str) -> datetime | None:
    """EDGAR acceptance times are Eastern wall clock with a misleading Z suffix."""
    if not value:
        return None
    try:
        naive = datetime.fromisoformat(value.replace("Z", "").replace(".000", ""))
    except ValueError:
        return None
    return pit.eastern(naive.replace(tzinfo=None))


def filings_table(recent: dict[str, Any]) -> pd.DataFrame:
    """One row per filing: accession, form, filing date, acceptance time (UTC), items."""
    n = len(recent.get("accessionNumber", []))
    if n == 0:
        return pd.DataFrame(columns=["accn", "form", "filed", "accepted", "items"])

    def col(name: str) -> list[Any]:
        v = list(recent.get(name, []))
        return v + [""] * (n - len(v))

    df = pd.DataFrame(
        {
            "accn": col("accessionNumber"),
            "form": col("form"),
            "filed": col("filingDate"),
            "accepted": [_accepted(str(a)) for a in col("acceptanceDateTime")],
            "items": col("items"),
        }
    )
    return df


def earnings_releases(symbol: str, filings: pd.DataFrame) -> pd.DataFrame:
    """Earnings release times from 8-K item 2.02. ``known_at`` is the acceptance time."""
    if filings.empty:
        return pd.DataFrame(columns=["symbol", "event_time", "known_at", "source"])
    is_8k = filings["form"].astype(str).isin(["8-K", "8-K/A"])
    has_item = (
        filings["items"]
        .astype(str)
        .str.split(",")
        .apply(lambda xs: EARNINGS_ITEM in [x.strip() for x in xs])
    )
    rows = filings.loc[is_8k & has_item & filings["accepted"].notna()]
    out = pd.DataFrame(
        {
            "symbol": symbol,
            "event_time": rows["accepted"].to_list(),
            "known_at": rows["accepted"].to_list(),
            "source": "sec_8k_2.02",
        }
    )
    return out.sort_values("event_time").drop_duplicates("event_time").reset_index(drop=True)


@dataclass(frozen=True)
class Fact:
    item: str
    end: date
    start: date | None
    value: float
    accn: str
    filed: date
    form: str


def _facts(cf: dict[str, Any], item: str) -> list[Fact]:
    gaap = cf.get("facts", {}).get("us-gaap", {})
    for tag in TAGS[item]:
        node = gaap.get(tag)
        if not node:
            continue
        units = node.get("units", {})
        rows = units.get("USD") or []
        out = []
        for r in rows:
            try:
                out.append(
                    Fact(
                        item=item,
                        end=date.fromisoformat(str(r["end"])),
                        start=date.fromisoformat(str(r["start"])) if r.get("start") else None,
                        value=float(r["val"]),
                        accn=str(r.get("accn", "")),
                        filed=date.fromisoformat(str(r["filed"])),
                        form=str(r.get("form", "")),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue
        if out:
            return out
    return []


def _duration_days(f: Fact) -> int | None:
    return (f.end - f.start).days if f.start else None


def ttm_and_stocks(cf: dict[str, Any]) -> pd.DataFrame:
    """Per period end: trailing twelve month flows and period-end stocks, first-reported.

    Flows use annual values (about 365 days) directly, or sum the last four
    quarterly values (about 91 days) when all four are known. Only the first
    filing of each value is used, so later restatements cannot leak back.
    """
    by_item: dict[str, dict[date, Fact]] = {}
    for item in (*FLOW_ITEMS, *STOCK_ITEMS):
        first: dict[tuple[date, int | None], Fact] = {}
        for f in _facts(cf, item):
            if f.form not in PERIODIC_FORMS:
                continue
            key = (f.end, _duration_days(f))
            if key not in first or f.filed < first[key].filed:
                first[key] = f
        chosen: dict[date, Fact] = {}
        if item in STOCK_ITEMS:
            for (end, _), f in first.items():
                if end not in chosen or f.filed < chosen[end].filed:
                    chosen[end] = f
        else:
            annual = {k[0]: f for k, f in first.items() if k[1] and 350 <= k[1] <= 380}
            quarters = sorted(
                (f for k, f in first.items() if k[1] and 80 <= k[1] <= 100), key=lambda f: f.end
            )
            for end, f in annual.items():
                chosen[end] = f
            for i in range(3, len(quarters)):
                last4 = quarters[i - 3 : i + 1]
                span = (last4[-1].end - last4[0].end).days
                if 250 <= span <= 300 and last4[-1].end not in chosen:
                    chosen[last4[-1].end] = Fact(
                        item=item,
                        end=last4[-1].end,
                        start=None,
                        value=sum(q.value for q in last4),
                        accn=last4[-1].accn,
                        filed=max(q.filed for q in last4),
                        form=last4[-1].form,
                    )
        by_item[item] = chosen
    ends = sorted({e for d in by_item.values() for e in d})
    rows = []
    for end in ends:
        row: dict[str, Any] = {"period_end": end}
        accns, filed = [], []
        for item, d in by_item.items():
            got = d.get(end)
            row[item] = got.value if got else np.nan
            if got:
                accns.append(got.accn)
                filed.append(got.filed)
        if not filed:
            continue
        row["accn"] = accns[0]
        row["filed"] = max(filed)
        rows.append(row)
    return pd.DataFrame(rows)


def quality_features(symbol: str, cf: dict[str, Any], filings: pd.DataFrame) -> pd.DataFrame:
    """Quality factor inputs per filing, each stamped with when it became public.

    - q_gross_profitability: gross profit over total assets (Novy-Marx).
    - q_roe: net income over equity.
    - q_accruals: (net income - operating cash flow) over assets. Lower is better.
    - q_leverage: liabilities over assets. Lower is better.
    - q_earnings_stability: minus the standard deviation of the last 8 TTM ROE values.
    """
    t = ttm_and_stocks(cf)
    if t.empty:
        return pd.DataFrame(columns=["symbol", "known_at", "period_end", *QUALITY_COLUMNS])
    t = t.sort_values("period_end").reset_index(drop=True)
    gp = t["gross_profit"].where(t["gross_profit"].notna(), t["revenue"] - t["cost_of_revenue"])
    assets = t["assets"].where(t["assets"] > 0)
    equity = t["equity"].where(t["equity"] > 0)
    out = pd.DataFrame(
        {
            "symbol": symbol,
            "period_end": t["period_end"],
            "q_gross_profitability": gp / assets,
            "q_roe": t["net_income"] / equity,
            "q_accruals": (t["net_income"] - t["operating_cash_flow"]) / assets,
            "q_leverage": t["liabilities"] / assets,
        }
    )
    out["q_earnings_stability"] = -out["q_roe"].rolling(8, min_periods=4).std(ddof=1)
    accepted = (
        dict(zip(filings["accn"], filings["accepted"], strict=False)) if not filings.empty else {}
    )
    known = []
    for accn, filed in zip(t["accn"], t["filed"], strict=True):
        stamp = accepted.get(accn)
        known.append(stamp if isinstance(stamp, datetime) else pit.end_of_day(filed))
    out["known_at"] = known
    return out


def fetch_fundamentals(
    sec: SECClient,
    symbols: Iterable[str],
    *,
    refresh: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Quality features and earnings releases for every symbol the SEC knows.

    Returns (quality, earnings, problems by symbol).
    """
    ciks = sec.tickers(refresh)
    quality, earnings = [], []
    problems: dict[str, str] = {}
    for sym in symbols:
        cik = ciks.get(sym.upper()) or ciks.get(sym.upper().replace(".", "-"))
        if cik is None:
            problems[sym] = "no CIK (ETF, delisted before the mapping, or renamed)"
            continue
        try:
            filings = filings_table(sec.submissions(cik, refresh))
            earnings.append(earnings_releases(sym, filings))
            cf = sec.companyfacts(cik, refresh)
            if cf is None:
                problems[sym] = "no XBRL company facts"
                continue
            quality.append(quality_features(sym, cf, filings))
        except (DataError, httpx.HTTPError, ValueError) as exc:
            problems[sym] = f"{type(exc).__name__}: {str(exc)[:120]}"
        if progress:
            progress(sym)
    q = pd.concat(quality, ignore_index=True) if quality else pd.DataFrame()
    e = pd.concat(earnings, ignore_index=True) if earnings else pd.DataFrame()
    return q, e, problems


def now_utc() -> datetime:
    return datetime.now(UTC)
