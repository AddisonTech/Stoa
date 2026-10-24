"""Trial 8 data and wiring: build the point-in-time event data once, then score it once.

``stoa events build`` (``build``) reads only data up to the end of the scoreboard
windows (2026-06-30): the wide bar cache, the SEC insider purchases, and SEC
earnings release times and SIC codes for every symbol that was ever liquid.
It writes ``data/events/``. ``stoa score --strategy event`` (``source``) turns
that into the strategy, bars, universe and cost model the scoreboard runs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from stoa import clock, event_signals, wide
from stoa.config import Settings
from stoa.data.base import Bars, DataError
from stoa.data.cache import BarCache, load_cached_bars
from stoa.data.sec import SECClient, earnings_releases, filings_table
from stoa.earnings import EarningsCalendar, empty_calendar, volume_proxy, with_known_lead
from stoa.event_strategy import EventStrategy, liquidity_slippage, sector_of_sic
from stoa.insider import known_session
from stoa.scoreboard import ModelSource
from stoa.universe import AssetInfo, Universe

log = logging.getLogger(__name__)

EVENTS_DIR = Path("data/events")
WARMUP_DAYS = 420  # calendar days of bars before a window: the 260 session warmup and more


def events_dir() -> Path:
    return EVENTS_DIR


class LazyBars(Mapping[str, pd.DataFrame]):
    """Bars read from the cache one symbol at a time, so thousands never sit in memory."""

    def __init__(self, cache: BarCache, symbols: list[str], start: date, end: date) -> None:
        self.cache, self.symbols, self.start, self.end = cache, symbols, start, end

    def __getitem__(self, symbol: str) -> pd.DataFrame:
        try:
            return self.cache.read([symbol], self.start, self.end)[symbol]
        except DataError as exc:
            raise KeyError(symbol) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self.symbols)

    def __len__(self) -> int:
        return len(self.symbols)


def ever_liquid(cache: BarCache, symbols: list[str], start: date, end: date) -> set[str]:
    """Symbols that pass the price and liquidity floor on at least one session."""
    out = set()
    for sym in symbols:
        try:
            df = cache.read([sym], start, end)[sym]
        except (DataError, ValueError):
            continue
        if len(df) and bool(wide.liquid(df).any()):
            out.add(sym)
    return out


def cik_map(current: Mapping[str, int], purchases: pd.DataFrame) -> dict[str, int]:
    """Symbol to CIK: the SEC's current map, else the issuer CIK the insider filings named."""
    out: dict[str, int] = {}
    if len(purchases):
        p = purchases.sort_values("filing_date")
        for sym, cik in zip(p["symbol"], p["issuer_cik"], strict=True):
            out[event_signals.normalize_symbol(str(sym))] = int(cik)
    out.update(current)  # the current map wins for a ticker in use today
    return out


def _sic(sec: SECClient, cik: int) -> str | None:
    body = sec.get_json(
        f"https://data.sec.gov/submissions/CIK{cik:010d}.json", f"submissions/CIK{cik:010d}.json"
    )
    return str(body.get("sic") or "") or None if isinstance(body, dict) else None


@dataclass
class BuildReport:
    candidates: int
    ever_liquid: int
    with_cik: int
    with_releases: int
    releases: int
    drift_signals: int
    cluster_signals: int
    cluster_signals_with_bars: int


def build(
    settings: Settings,
    sec: SECClient,
    *,
    end: date,
    progress: Callable[[str], None] = lambda _m: None,
) -> BuildReport:
    """Everything trial 8 needs, point in time, through ``end``. Writes data/events/."""
    s = settings
    out = events_dir()
    out.mkdir(parents=True, exist_ok=True)
    cache = BarCache(s.data.cache_dir, wide.WIDE_NAMESPACE)
    cands = wide.candidates(wide.load_assets(cache.dir / "_assets.json"))
    start = s.data.history_start
    liquid = ever_liquid(cache, sorted(cands), start, end)
    progress(f"{len(liquid)} of {len(cands)} candidates were liquid on at least one session")

    purchases = pd.read_parquet(Path("data/insider/purchases.parquet"))
    purchases = purchases.loc[[known_session(d) <= end for d in purchases["filing_date"]]]
    ciks = cik_map(sec.tickers(), purchases)
    rel_parts, sics, with_cik = [], {}, 0
    for i, sym in enumerate(sorted(liquid)):
        cik = ciks.get(sym)
        if cik is None:
            continue
        with_cik += 1
        try:
            filings = filings_table(sec.submissions(cik))
            sics[sym] = _sic(sec, cik)
        except Exception as exc:
            log.warning("SEC submissions for %s failed: %s", sym, type(exc).__name__)
            continue
        rel = earnings_releases(sym, filings)
        rel = rel.loc[pd.to_datetime(rel["event_time"], utc=True).dt.date <= end]
        if len(rel):
            rel_parts.append(rel)
        if i % 500 == 0:
            progress(f"SEC filings: {i}/{len(liquid)} symbols")
    releases = pd.concat(rel_parts, ignore_index=True) if rel_parts else empty_calendar()
    releases.to_parquet(out / "earnings.parquet")
    (out / "sic.json").write_text(json.dumps(sics, indent=1), encoding="utf-8")

    bars = LazyBars(cache, sorted(liquid | {s.benchmark}), start, end)
    drift = event_signals.earnings_drift(releases, bars, s.benchmark)
    sessions = clock.sessions_between(date(2015, 6, 1), end)
    clusters = event_signals.insider_clusters(purchases, sessions)
    clusters_kept = [c for c in clusters if c.symbol in liquid and c.session <= end]
    signals = [x for x in [*drift, *clusters_kept] if x.session <= end]
    event_signals.to_frame(signals).to_parquet(out / "signals.parquet")
    report = BuildReport(
        candidates=len(cands),
        ever_liquid=len(liquid),
        with_cik=with_cik,
        with_releases=releases["symbol"].nunique() if len(releases) else 0,
        releases=len(releases),
        drift_signals=len(drift),
        cluster_signals=len(clusters),
        cluster_signals_with_bars=len(clusters_kept),
    )
    (out / "build.json").write_text(json.dumps(report.__dict__, indent=1), encoding="utf-8")
    return report


# ---- the scoreboard source ------------------------------------------------------------


def load_signals(end: date) -> list[event_signals.Signal]:
    df = pd.read_parquet(events_dir() / "signals.parquet")
    out = []
    for r in df.to_dict("records"):
        day = pd.Timestamp(str(r["session"])).date()
        if day <= end:
            out.append(
                event_signals.Signal(
                    str(r["symbol"]), day, str(r["kind"]), float(str(r["score"])), str(r["detail"])
                )
            )
    return out


def source(settings: Settings, base_bars: Bars, end: date) -> ModelSource:
    s = settings
    cache = BarCache(s.data.cache_dir, wide.WIDE_NAMESPACE)
    cands = wide.candidates(wide.load_assets(cache.dir / "_assets.json"))
    signals = [x for x in load_signals(end) if x.symbol in cands]
    by_day = event_signals.by_session(signals)
    symbols = sorted({x.symbol for x in signals})
    sics: dict[str, Any] = json.loads((events_dir() / "sic.json").read_text(encoding="utf-8"))
    releases = pd.read_parquet(events_dir() / "earnings.parquet")
    hist = (
        with_known_lead(releases, s.earnings.confirm_lead_sessions) if len(releases) else releases
    )
    have_history = set(hist["symbol"]) if len(hist) else set()

    def window_bars(start: date, finish: date) -> Bars:
        first = clock.previous_session(start)
        names = sorted({x.symbol for x in signals if first <= x.session <= finish})
        lo = start - timedelta(days=WARMUP_DAYS)
        out: Bars = {
            k: v.loc[pd.Timestamp(lo) : pd.Timestamp(finish)] for k, v in base_bars.items()
        }
        for sym in names:
            try:
                out[sym] = cache.read([sym], lo, finish)[sym]
            except (DataError, ValueError):
                continue
        return out

    calendar_bars = LazyBars(cache, symbols, s.data.history_start, end)
    no_history = {x for x in symbols if x not in have_history}
    cached = cache_symbols(cache)
    proxy = volume_proxy(
        {k: calendar_bars[k] for k in no_history if k in cached},
        no_history,
        s.earnings.heuristic_volume_multiple,
    )
    table = pd.concat([hist, proxy], ignore_index=True) if len(proxy) else hist
    strategy = EventStrategy(
        settings=s,
        signals=by_day,
        earnings=EarningsCalendar(table if len(table) else empty_calendar()),
        sectors={k: sector_of_sic(sics.get(k)) for k in symbols},
    )
    assets = {
        k: AssetInfo(k, tradable=True, fractionable=bool(cands[k]["fractionable"])) for k in symbols
    }
    for k in base_bars:
        assets.setdefault(k, AssetInfo(k, tradable=True, fractionable=True))
    universe = Universe(
        watchlist=tuple(sorted(set(symbols) | {s.benchmark})),
        benchmark=s.benchmark,
        assets=assets,
        whole_shares=frozenset(k for k in symbols if not cands[k]["fractionable"]),
    )
    slip_bars: dict[str, pd.DataFrame] = dict(base_bars)
    slip = _lazy_slippage(cache, slip_bars, s.data.history_start, end)
    return ModelSource(
        strategy=strategy,
        bars=base_bars,
        universe=universe,
        name="event",
        slippage=slip,
        window_bars=window_bars,
    )


def cache_symbols(cache: BarCache) -> set[str]:
    return {p.stem for p in cache.dir.glob("*.parquet")}


def _lazy_slippage(
    cache: BarCache, known: dict[str, pd.DataFrame], start: date, end: date
) -> Callable[[str, date | None], float]:
    """The registered cost model, loading a symbol's dollar volume the first time it trades."""
    fns: dict[str, Callable[[str, date | None], float]] = {}

    def slip(symbol: str, session: date | None) -> float:
        if symbol not in fns:
            df = known.get(symbol)
            if df is None:
                try:
                    df = cache.read([symbol], start, end)[symbol]
                except (DataError, ValueError):
                    df = None
            fns[symbol] = liquidity_slippage({symbol: df} if df is not None else {})
        return fns[symbol](symbol, session)

    return slip


def base_bars(settings: Settings, end: date) -> Bars:
    """SPY, the watchlist and MTUM from the main cache, for every other strategy."""
    s = settings
    names = list(s.watchlist)
    bars = load_cached_bars(s.data.cache_dir, names, s.data.history_start, end)
    for r in s.sources.reference_symbols:
        try:
            bars.update(load_cached_bars(s.data.cache_dir, [r], s.data.history_start, end))
        except DataError:
            continue
    return bars
