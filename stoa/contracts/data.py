"""Trial 16's tables, built from disk: awards, matches, prices, features and labels.

Every award is known at 5 pm New York time on its announcement date (the time war.gov
posts the day's contracts; the pages carry no time of their own). Nothing is used before
then: the trade enters at the open of the first session after the post, the "prior close"
is the close of the last session at or before the announcement date, and every SEC fact
(shares, revenue, subsidiary lists) must have been filed before the announcement date.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import statistics
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd

from stoa import clock, wide
from stoa.contracts import ROOT
from stoa.contracts.parse import frame, parse_page
from stoa.contracts.scrape import Archive
from stoa.contracts.tickers import Book, Match, Miss
from stoa.newsbrain.panel import FEE

log = logging.getLogger(__name__)

AWARDS = ROOT / "awards.parquet"
MATCHED = ROOT / "matched.parquet"
PANEL = ROOT / "panel.parquet"
BARS = Path("data/bars/wide")
FACTS = Path("data/sec/companyfacts")
POST_TIME = "17:00"
HOLDS = (0, 1, 5)  # sessions after entry: 0 is the entry session's close (the primary exit)
RECENT_DAYS = 30
REVENUE_CONCEPTS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
)
# SIC divisions, as trial 15's event memory groups them.
SIC_DIVISIONS = (
    (100, 999, "agriculture"),
    (1000, 1499, "mining"),
    (1500, 1799, "construction"),
    (2000, 3999, "manufacturing"),
    (4000, 4999, "transport_utilities"),
    (5000, 5199, "wholesale"),
    (5200, 5999, "retail"),
    (6000, 6799, "finance"),
    (7000, 8999, "services"),
    (9100, 9999, "public_admin"),
)


def known_at(day: date) -> pd.Timestamp:
    """When the announcement for ``day`` is public: 5 pm New York time that day."""
    return pd.Timestamp(clock.at_ny(day, POST_TIME))


def entry_session(day: date) -> date:
    """The first session that opens after the 5 pm post: never the announcement day itself."""
    nxt = clock.next_session(day)
    assert pd.Timestamp(clock.session(nxt).open) > known_at(day), (day, nxt)
    return nxt


def prior_close_session(day: date) -> date:
    """The last session that closed before the post (the announcement day if it traded)."""
    return day if clock.is_session(day) else clock.previous_session(day)


def sic_division(sic: int) -> str:
    for lo, hi, name in SIC_DIVISIONS:
        if lo <= sic <= hi:
            return name
    return "unknown"


# ---- parsing --------------------------------------------------------------------------------


def parse_all(root: Path = ROOT, out: Path | None = AWARDS) -> pd.DataFrame:
    arc = Archive(root)
    rows = []
    for p in sorted(arc.articles.glob("*.html")):
        rows.extend(parse_page(p.read_text(encoding="utf-8"), int(p.stem)))
    df = frame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    return df


# ---- matching -------------------------------------------------------------------------------


def as_date(x: Any) -> date:
    out: date = x.date() if isinstance(x, (pd.Timestamp, datetime)) else x
    return out


def match_all(awards: pd.DataFrame, book: Book) -> pd.DataFrame:
    """Each award row with its company and symbol, or why it has none."""
    out: dict[str, list[Any]] = {k: [] for k in ("cik", "symbol", "how", "key", "source", "miss")}
    entries = []
    cache: dict[tuple[str, date], Match | Miss] = {}
    awards = awards.assign(date=[as_date(d) for d in awards["date"]])
    for company, day in zip(awards["company"], awards["date"], strict=True):
        entry = entry_session(day)
        entries.append(entry)
        k = (str(company), day)
        if k not in cache:
            cache[k] = book.match(str(company), day, entry)
        m = cache[k]
        if isinstance(m, Match):
            vals = [m.cik, m.symbol, m.how, m.key, m.source, ""]
        else:
            vals = [None, None, None, m.key, m.detail, m.reason]
        for col, v in zip(out, vals, strict=True):
            out[col].append(v)
    df = awards.copy()
    df["entry"] = entries
    for col, v in out.items():
        df[col] = v
    df["cik"] = df["cik"].astype("Int64")
    return df


# ---- SEC facts ------------------------------------------------------------------------------


def _facts(cik: int) -> dict[str, Any] | None:
    p = FACTS / f"CIK{cik:010d}.json"
    if not p.exists():
        return None
    return dict(json.loads(p.read_text(encoding="utf-8")))


def _rows(facts: dict[str, Any], taxonomy: str, concept: str, unit: str) -> list[dict[str, Any]]:
    node = facts.get("facts", {}).get(taxonomy, {}).get(concept, {})
    return list(node.get("units", {}).get(unit, []))


def shares_on(facts: dict[str, Any], day: date) -> float | None:
    """Shares outstanding from the latest filing before ``day``, all classes of that filing."""
    rows = [
        r
        for r in _rows(facts, "dei", "EntityCommonStockSharesOutstanding", "shares")
        if r.get("filed") and r["filed"] < day.isoformat() and r.get("val")
    ]
    if not rows:
        rows = [
            r
            for r in _rows(facts, "us-gaap", "CommonStockSharesOutstanding", "shares")
            if r.get("filed") and r["filed"] < day.isoformat() and r.get("val")
        ]
    if not rows:
        return None
    last = max((r["filed"], str(r.get("accn", ""))) for r in rows)
    same = [r for r in rows if (r["filed"], str(r.get("accn", ""))) == last]
    # A filing reports each share class once; the same number twice is one fact.
    distinct = {(str(r.get("end")), float(r["val"])) for r in same}
    total = sum(v for _, v in distinct)
    return total if total > 0 else None


def shares_filed(facts: dict[str, Any], day: date) -> tuple[float | None, str | None]:
    """``shares_on`` and the filing date it came from."""
    got = shares_on(facts, day)
    if got is None:
        return None, None
    cut = day.isoformat()
    filed = [
        r["filed"]
        for concept in ("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding")
        for tax in ("dei", "us-gaap")
        for r in _rows(facts, tax, concept, "shares")
        if r.get("filed") and r["filed"] < cut and r.get("val")
    ]
    return got, max(filed) if filed else None


SPLIT_MIN = 1.4  # a restated share count at least this much bigger (or 1/this smaller)
SPLIT_MAX = 100.0  # beyond this it is an XBRL scale error, not a split


def split_events(facts: dict[str, Any]) -> list[tuple[str, float]]:
    """Stock splits, from the SEC's own restatements: after a split, later filings restate
    earlier periods' weighted average share counts by its ratio. Each event is (the filing
    date the restated counts first appear, the ratio, new shares per old share)."""
    rows = _rows(facts, "us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic", "shares")
    if not rows:
        rows = _rows(facts, "us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares")
    periods: dict[tuple[str, str], dict[str, float]] = {}
    for r in rows:
        if r.get("start") and r.get("end") and r.get("filed") and r.get("val"):
            periods.setdefault((r["start"], r["end"]), {})[r["filed"]] = float(r["val"])
    found: dict[str, list[float]] = {}
    for by_filed in periods.values():
        seq = sorted(by_filed.items())
        for (_, a), (filed_b, b) in itertools.pairwise(seq):
            if a <= 0:
                continue
            ratio = b / a
            if SPLIT_MIN <= ratio <= SPLIT_MAX or 1 / SPLIT_MAX <= ratio <= 1 / SPLIT_MIN:
                found.setdefault(filed_b, []).append(ratio)
    raw = sorted((filed, float(statistics.median(rs))) for filed, rs in found.items())
    # One split is restated period by period over the next year of filings: the same ratio
    # within 15 months is one split, dated at the first filing that shows it.
    events: list[tuple[str, float]] = []
    for filed, ratio in raw:
        if events:
            last_filed, last_ratio = events[-1]
            close = abs(math.log(ratio / last_ratio)) < 0.05
            days = (date.fromisoformat(filed) - date.fromisoformat(last_filed)).days
            if close and days <= 456:
                continue
        events.append((filed, ratio))
    return events


def split_factor(facts: dict[str, Any], filed: str | None) -> float:
    """New shares per share counted in a filing of date ``filed``: every split whose restated
    counts first appear in a later filing. The daily bars are adjusted for all of them."""
    if filed is None:
        return 1.0
    out = 1.0
    for when, ratio in split_events(facts):
        if when > filed:
            out *= ratio
    return out


DIVIDEND_CONCEPTS = ("CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid")


def dividends(facts: dict[str, Any]) -> list[tuple[date, float]]:
    """Cash dividends per share, dated at the end of the period they first appear in, as first
    reported (the share basis of the day). Each fiscal year is rebuilt from its year-to-date
    figures (first quarter, half, nine months, year: all start on the fiscal year's first
    day), because companies tag the same dividend in more than one three-month period. A
    company with no annual figures falls back to its three-month figures."""
    for concept in DIVIDEND_CONCEPTS:
        first: dict[tuple[str, str], tuple[str, float]] = {}
        for r in _rows(facts, "us-gaap", concept, "USD/shares"):
            if r.get("start") and r.get("end") and r.get("filed") and r.get("val") is not None:
                k = (str(r["start"]), str(r["end"]))
                if k not in first or r["filed"] < first[k][0]:
                    first[k] = (str(r["filed"]), float(r["val"]))
        if not first:
            continue
        years = {
            s
            for (s, e) in first
            if 350 <= (date.fromisoformat(e) - date.fromisoformat(s)).days <= 380
        }
        # The year under way has no annual figure yet: its start is an anniversary of one.
        known = [date.fromisoformat(y) for y in years]
        for s, _ in first:
            sd = date.fromisoformat(s)
            if any(abs((sd - y).days - 365 * k) <= 7 for y in known for k in (1, 2)):
                years.add(s)
        out: list[tuple[date, float]] = []
        if years:
            for s in sorted(years):
                ytd = sorted((e, v) for (s2, e), (_, v) in first.items() if s2 == s)
                before = 0.0
                for e, v in ytd:
                    if v - before > 1e-9:
                        out.append((date.fromisoformat(e), v - before))
                    before = max(before, v)
        else:
            for (s, e), (_, v) in first.items():
                if 80 <= (date.fromisoformat(e) - date.fromisoformat(s)).days <= 100 and v > 0:
                    out.append((date.fromisoformat(e), v))
        return sorted(out)
    return []


def dividend_adjustment(facts: dict[str, Any], bars: pd.DataFrame, day: date) -> float:
    """How much the bars' dividend adjustment scaled the price on ``day``: the product over
    later dividends of (1 - dividend / price before it), each priced at its period end in the
    raw terms of that day. Raw price = adjusted price / this (and times the split factor)."""
    if bars.empty:
        return 1.0
    last = bars.index[-1].date()
    closes = bars["close"]
    after = 1.0
    out_at_day = 1.0
    for end, dps in sorted(dividends(facts), reverse=True):
        if end > last:
            continue  # the bars stop before this dividend; it is not in their adjustment
        ts = pd.Timestamp(end)
        got = closes.asof(ts)
        px = float(cast(Any, got)) if got is not None else math.nan
        if not math.isfinite(px) or px <= 0:
            continue
        split = split_factor(facts, end.isoformat())
        raw = px * split / after
        if dps >= raw:
            continue  # a scale error in the filing, not a dividend
        after *= 1 - dps / raw
        if end > day:
            out_at_day = after
    return out_at_day


def _duration(r: dict[str, Any]) -> int:
    try:
        return (date.fromisoformat(r["end"]) - date.fromisoformat(r["start"])).days
    except (KeyError, ValueError, TypeError):
        return -1


def revenue_ttm(facts: dict[str, Any], day: date) -> float | None:
    """Trailing 12 month revenue from what was filed before ``day``: the latest fiscal year,
    rolled forward by the latest year-to-date figure and its year-earlier twin when both are on
    file."""
    cut = day.isoformat()
    best: float | None = None
    best_end = ""
    for concept in REVENUE_CONCEPTS:
        rows = [
            r
            for r in _rows(facts, "us-gaap", concept, "USD")
            if r.get("filed") and r["filed"] < cut and r.get("start") and r.get("end")
        ]
        annual = [r for r in rows if 350 <= _duration(r) <= 380]
        if not annual:
            continue
        a = max(annual, key=lambda r: (r["end"], r["filed"]))
        ttm, end = float(a["val"]), str(a["end"])
        ytd = [r for r in rows if 80 <= _duration(r) <= 290 and r["end"] > a["end"]]
        if ytd:
            y = max(ytd, key=lambda r: (r["end"], r["filed"]))
            y_end = date.fromisoformat(y["end"])
            twin = [
                r
                for r in rows
                if abs((date.fromisoformat(r["end"]) - (y_end - timedelta(days=365))).days) <= 10
                and abs(_duration(r) - _duration(y)) <= 10
            ]
            if twin:
                t = max(twin, key=lambda r: r["filed"])
                ttm = ttm + float(y["val"]) - float(t["val"])
                end = str(y["end"])
        if end > best_end:
            best, best_end = ttm, end
    return best if best is not None and best > 0 else None


def pull_facts(ciks: Iterable[int], per_second: float = 1.0) -> int:
    """XBRL company facts for matched companies not yet on disk, from EDGAR, at a pace that
    keeps the total under the SEC's 10 a second while the exhibit 21 pull runs."""
    from stoa.data.sec import make_sec_client
    from stoa.newsbrain.sec_pull import USER_AGENT

    sec = make_sec_client(USER_AGENT, Path("data/sec"), per_second)
    got = 0
    for cik in sorted({int(c) for c in ciks}):
        if (FACTS / f"CIK{cik:010d}.json").exists():
            continue
        try:
            if sec.companyfacts(cik) is not None:
                got += 1
        except Exception as exc:
            log.warning("companyfacts %s failed: %s", cik, type(exc).__name__)
    return got


# ---- prices ---------------------------------------------------------------------------------


def load_bars(symbol: str) -> pd.DataFrame | None:
    p = BARS / f"{symbol}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["open", "close", "volume"])
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated()].sort_index()
    df["adv"] = (
        (df["close"] * df["volume"])
        .rolling(wide.ADV_SESSIONS, min_periods=wide.ADV_SESSIONS)
        .mean()
        .shift(1)
    )
    return df


def legs(df: pd.DataFrame, entry: date) -> dict[str, float]:
    """Entry open, the closes HOLDS sessions on, and the entry session's dollar volume (through
    the close before)."""
    nan = math.nan
    out: dict[str, float] = {"open": nan, "adv": nan, "prev_close": nan}
    out.update({f"close_{h}": nan for h in HOLDS})
    ts = pd.Timestamp(entry)
    if ts not in df.index:
        return out
    i = int(df.index.searchsorted(ts))  # the index is sorted and unique, and ts is in it
    out["open"] = float(df["open"].iloc[i])
    out["adv"] = float(df["adv"].iloc[i])
    out["prev_close"] = float(df["close"].iloc[i - 1]) if i > 0 else nan
    for h in HOLDS:
        if i + h < len(df):
            out[f"close_{h}"] = float(df["close"].iloc[i + h])
    return out


def build_panel(matched: pd.DataFrame, book: Book) -> pd.DataFrame:
    """One row per (company, announcement date) for the matched awards: summed new value,
    summed modification value, features and labels."""
    matched = matched.assign(
        date=[as_date(d) for d in matched["date"]], entry=[as_date(d) for d in matched["entry"]]
    )
    m = matched[matched["symbol"].notna()].copy()
    m["value"] = pd.to_numeric(m["value"], errors="coerce")
    spy = load_bars("SPY")
    assert spy is not None, "no SPY bars"
    rows = []
    every = matched[matched["cik"].notna()][["cik", "date", "group"]]
    seen_days: dict[int, list[date]] = {}
    for cik, d in zip(every["cik"], every["date"], strict=True):
        seen_days.setdefault(int(cik), []).append(d)
    seen_days = {k: sorted(set(v)) for k, v in seen_days.items()}
    facts_cache: dict[int, dict[str, Any] | None] = {}
    bars_cache: dict[str, pd.DataFrame | None] = {}
    for (cik_key, day_key, group), g in m.groupby(["cik", "date", "group"], sort=True):
        cik = int(cast(Any, cik_key))
        day = cast(date, day_key)
        sym = str(g["symbol"].iloc[0])
        entry = g["entry"].iloc[0]
        if sym not in bars_cache:
            bars_cache[sym] = load_bars(sym)
        bars = bars_cache[sym]
        if bars is None:
            continue
        if cik not in facts_cache:
            facts_cache[cik] = _facts(cik)
        facts = facts_cache[cik]
        lg = legs(bars, entry)
        sp = legs(spy, entry)
        prior = prior_close_session(day)
        pc = bars["close"].get(pd.Timestamp(prior), math.nan)
        shares, filed = shares_filed(facts, day) if facts else (None, None)
        # The bars are split adjusted to today; the share count is as filed. Bring the count
        # to today's basis so price times shares is the market value on the day.
        split = split_factor(facts, filed) if facts else 1.0
        # And undo the dividend adjustment: the bars' prices before a dividend are scaled down.
        divadj = dividend_adjustment(facts, bars, day) if facts else 1.0
        rev = revenue_ttm(facts, day) if facts else None
        value = float(g["value"].sum(min_count=1)) if g["value"].notna().any() else math.nan
        mcap = shares * split * float(pc) / divadj if shares and math.isfinite(pc) else math.nan
        earlier = [
            x for x in seen_days.get(cik, []) if day - timedelta(days=RECENT_DAYS) <= x < day
        ]
        row = {
            "cik": cik,
            "symbol": sym,
            "date": day,
            "entry": entry,
            "group": group,
            "types": ",".join(sorted(set(g["type"]))),
            "awards": len(g),
            "value": value,
            "small_business": bool(g["small_business"].any()),
            "how": ",".join(sorted(set(g["how"].astype(str)))),
            "sector": sic_division(book.sic.get(cik, 0)),
            "sic": book.sic.get(cik, 0),
            "shares": shares if shares else math.nan,
            "split_factor": split,
            "dividend_adjustment": divadj,
            "prior_close": float(pc) if math.isfinite(pc) else math.nan,
            "mcap": mcap,
            "revenue_ttm": rev if rev else math.nan,
            "value_to_mcap": value / mcap
            if mcap and math.isfinite(mcap) and mcap > 0
            else math.nan,
            "value_to_revenue": value / rev if rev else math.nan,
            "recent_award": bool(earlier),
            **{f"stk_{k}": v for k, v in lg.items()},
            **{f"spy_{k}": v for k, v in sp.items()},
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return add_labels(df)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Net excess return against SPY after trial 15's costs, for each hold."""
    df = df.copy()
    df["slip"] = [wide.slippage_bps(v if math.isfinite(v) else None) / 1e4 for v in df["stk_adv"]]
    for h in HOLDS:
        r = df[f"stk_close_{h}"] / df["stk_open"] - 1
        b = df[f"spy_close_{h}"] / df["spy_open"] - 1
        df[f"ret_{h}"] = r
        df[f"spy_{h}"] = b
        df[f"net_{h}"] = r - b - 2 * df["slip"] - FEE
    df["gap"] = df["stk_open"] / df["prior_close"] - 1
    return df


def cap_bucket(mcap: float) -> str:
    if not math.isfinite(mcap):
        return "unknown"
    return "small" if mcap < 2e9 else ("mid" if mcap < 1e10 else "large")


def ratio_bucket(x: float) -> str:
    if not math.isfinite(x):
        return "unknown"
    for hi, name in ((0.001, "<0.1%"), (0.005, "0.1-0.5%"), (0.02, "0.5-2%"), (0.05, "2-5%")):
        if x < hi:
            return name
    return ">=5%"


def years(df: pd.DataFrame, col: str = "date") -> pd.Series:
    return pd.Series([d.year for d in df[col]], index=df.index)


def counts_by(df: pd.DataFrame, cols: Iterable[str]) -> dict[str, Any]:
    return {str(k): int(v) for k, v in df.groupby(list(cols)).size().items()}


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")
