"""One place that assembles the model pipeline's inputs, so every caller does it the same way.

Callers: `stoa model train|report|promote`, the monthly retrain, the
scoreboard, the holdout, and DECIDE. Each one needs bars for the training
universe, the membership history, fundamentals, earnings dates and (when
enabled) analyst scores, read from the local stores that the fetch commands
fill.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stoa import clock
from stoa.config import Settings
from stoa.data.base import Bars, DataError
from stoa.data.cache import DEFAULT_NAMESPACE, BarCache
from stoa.data.synthetic import synthetic_universe
from stoa.dataset import FeatureSet
from stoa.earnings import (
    EXEMPT_DEFAULT,
    EarningsCalendar,
    empty_calendar,
    synthetic_calendar,
    volume_proxy,
    with_known_lead,
)
from stoa.membership import Membership, default_membership
from stoa.training import TrainingInputs

log = logging.getLogger(__name__)

UNIVERSE_NAMESPACE = "universe"
COVERAGE_FILE = "coverage.json"


def feature_set(settings: Settings, text: bool | None = None) -> FeatureSet:
    m = settings.model
    return FeatureSet(quality=m.use_quality, text=m.use_text if text is None else text)


def _read_parquet(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def universe_cache(settings: Settings) -> BarCache:
    return BarCache(settings.data.cache_dir, UNIVERSE_NAMESPACE)


def load_universe_bars(settings: Settings, symbols: list[str], start: date, end: date) -> Bars:
    """Whatever the universe cache holds for ``symbols``; missing symbols are left out."""
    out: Bars = {}
    for cache in (BarCache(settings.data.cache_dir, DEFAULT_NAMESPACE), universe_cache(settings)):
        for sym in symbols:
            if sym in out or not cache.path(sym).exists():
                continue
            try:
                out.update(cache.read([sym], start, end))
            except DataError:
                continue
    return out


def coverage(settings: Settings) -> dict[str, Any]:
    p = settings.data.cache_dir / UNIVERSE_NAMESPACE / COVERAGE_FILE
    if not p.exists():
        return {}
    data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return data


def write_coverage(settings: Settings, report: dict[str, Any]) -> Path:
    p = settings.data.cache_dir / UNIVERSE_NAMESPACE / COVERAGE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return p


def coverage_report(
    membership: Membership, have: set[str], start: date, end: date
) -> dict[str, Any]:
    """How much of the point-in-time universe has bars, and how much of the gap is delistings."""
    ever = membership.ever_members(start, end)
    left = membership.left_between(start, end)
    missing = sorted(ever - have)
    missing_left = sorted(set(missing) & left)
    by_year = {}
    for y in range(start.year, end.year + 1):
        d = date(y, 6, 30) if date(y, 6, 30) <= end else end
        if d < start:
            continue
        members = membership.members(d)
        by_year[str(y)] = {
            "members": len(members),
            "with_bars": len(members & have),
            "coverage": round(len(members & have) / max(1, len(members)), 4),
        }
    return {
        "window": [str(start), str(end)],
        "ever_members": len(ever),
        "with_bars": len(ever & have),
        "left_index": len(left),
        "left_index_with_bars": len(left & have),
        "missing": missing,
        "missing_that_left": missing_left,
        "by_year": by_year,
    }


@dataclass
class Inputs:
    training: TrainingInputs
    calendar: EarningsCalendar
    tradable: set[str]
    synthetic: bool


def load_inputs(settings: Settings, end: date, *, synthetic: bool = False) -> Inputs:
    """Bars, membership, fundamentals, earnings and text for training and backtests."""
    s = settings
    refs = tuple(r for r in s.sources.reference_symbols if r not in s.watchlist)
    if synthetic:
        bars, membership = synthetic_universe(
            list(s.watchlist),
            date(2016, 1, 4),
            end,
            seed=s.metrics.seed,
            references=refs,
            signal=s.demo.signal,
        )
        if s.model.universe == "watchlist":
            membership = Membership([])
        cal_table = with_known_lead(
            synthetic_calendar(bars, EXEMPT_DEFAULT | {s.benchmark, *refs}),
            s.earnings.confirm_lead_sessions,
        )
        fundamentals = None
        text = None
    else:
        membership = default_membership()
        if s.model.universe == "sp500":
            symbols = sorted(
                membership.ever_members(s.data.history_start, end) | set(s.watchlist) | set(refs)
            )
        else:
            symbols = sorted(set(s.watchlist) | set(refs))
        bars = load_universe_bars(s, symbols, s.data.history_start, end)
        missing_core = [x for x in s.watchlist if x not in bars]
        if missing_core:
            raise DataError(
                f"no cached bars for watchlist symbols {missing_core}; run "
                "`stoa data fetch` and `stoa universe build`"
            )
        fundamentals = _read_parquet(s.fundamentals_path)
        hist = _read_parquet(s.earnings_path)
        hist = hist if hist is not None else empty_calendar()
        hist = with_known_lead(hist, s.earnings.confirm_lead_sessions) if not hist.empty else hist
        no_history = (set(bars) - set(hist["symbol"]) if not hist.empty else set(bars)) - (
            EXEMPT_DEFAULT | {s.benchmark, *refs}
        )
        proxy = volume_proxy(bars, no_history, s.earnings.heuristic_volume_multiple)
        cal_table = pd.concat([hist, proxy], ignore_index=True) if not proxy.empty else hist
        text = _read_parquet(s.analyst_scores_path) if s.model.use_text else None
    if s.model.universe == "watchlist":
        membership_used: Membership | None = None
    else:
        membership_used = membership
    training = TrainingInputs(
        bars=bars,
        membership=membership_used,
        fundamentals=fundamentals,
        text_scores=text,
        extra_symbols=tuple(s.watchlist),
        exclude=refs,
    )
    tradable = set(s.watchlist) if s.strategy.trade_universe == "watchlist" else set(bars)
    return Inputs(
        training=training,
        calendar=EarningsCalendar(cal_table, s.job_windows.execute_after_open_minutes),
        tradable=tradable - {s.benchmark, *refs},
        synthetic=synthetic,
    )


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def last_session(now: datetime | None = None) -> date:
    return clock.last_closed_session(now or clock.now_utc())


def walkforward_model_source(settings: Settings, inputs: Inputs, end: date) -> Any:
    """The model strategy for backtests, driven only by out-of-sample walk-forward predictions.

    Every probability a backtest sees on a session came from a model trained
    on data that ended before that session's test slice, with the purge and
    embargo applied. Sessions before the first test slice have no predictions,
    so the model makes no entries there.
    """
    from stoa.model_strategy import ModelStrategy
    from stoa.scoreboard import ModelSource
    from stoa.training import predictor_from_walkforward, run_walkforward
    from stoa.universe import Universe

    validation = settings.backtest.windows.get("validation")
    wf = run_walkforward(
        settings,
        inputs.training,
        feature_set(settings),
        end=end,
        select_before=validation.start if validation is not None else None,
    )
    strategy = ModelStrategy(
        settings,
        predictor_from_walkforward(wf),
        inputs.calendar,
        tradable=inputs.tradable,
    )
    tradable = tuple(sorted(inputs.tradable | {settings.benchmark}))
    universe = Universe.assume_all_fractionable(tradable, settings.benchmark)
    # The backtest only needs what can be traded plus the baselines' symbols.
    keep = set(tradable) | set(settings.watchlist) | set(settings.sources.reference_symbols)
    bars = {k: v for k, v in inputs.training.bars.items() if k in keep}
    return ModelSource(strategy=strategy, bars=bars, universe=universe), wf
