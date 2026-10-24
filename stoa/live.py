"""Builds the model strategy for a live DECIDE, from the promoted model and fresh data.

The promoted model was trained by `stoa model train` (or the monthly
retrain). DECIDE does not retrain. It fetches fresh bars for the watchlist
(the tradable set) and computes today's cross-sectional features on the same
universe the model was trained on, as far as bars allow, then asks the model
for today's probabilities.

Live cross-sections: standardizing across only the 29 watchlist names would
put features on a different scale than training (which used about 500 index
members). So DECIDE also fetches the current index members' bars when
``model.universe`` is ``sp500``. If that fetch is incomplete, the run falls
back to the watchlist cross-section and says so in the summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from stoa.config import Settings
from stoa.data.base import BarProvider, Bars, DataError
from stoa.dataset import FeatureSet, build_dataset
from stoa.earnings import (
    EXEMPT_DEFAULT,
    EarningsCalendar,
    empty_calendar,
    volume_proxy,
    with_known_lead,
)
from stoa.membership import default_membership
from stoa.model import ModelStore
from stoa.model_strategy import ModelPredictor, ModelStrategy

log = logging.getLogger(__name__)

MIN_UNIVERSE_FRACTION = 0.8


@dataclass
class LiveModel:
    strategy: ModelStrategy
    version: str
    notes: list[str]


def _read(path: object) -> pd.DataFrame | None:
    from pathlib import Path

    p = Path(str(path))
    return pd.read_parquet(p) if p.exists() else None


def live_calendar(
    settings: Settings,
    bars: Bars,
    upcoming: pd.DataFrame | None,
) -> EarningsCalendar:
    s = settings
    hist = _read(s.earnings_path)
    hist = with_known_lead(hist, s.earnings.confirm_lead_sessions) if hist is not None else None
    parts = [p for p in (hist, upcoming) if p is not None and not p.empty]
    table = pd.concat(parts, ignore_index=True) if parts else empty_calendar()
    known = set(table["symbol"]) if not table.empty else set()
    no_history = set(bars) - known - EXEMPT_DEFAULT - {s.benchmark}
    proxy = volume_proxy(bars, no_history, s.earnings.heuristic_volume_multiple)
    if not proxy.empty:
        table = pd.concat([table, proxy], ignore_index=True)
    return EarningsCalendar(table, s.job_windows.execute_after_open_minutes)


def build_live_model(
    settings: Settings,
    provider: BarProvider,
    bars: Bars,
    session: date,
    upcoming: pd.DataFrame | None = None,
) -> LiveModel | None:
    """The model strategy for today, or None when no model has been promoted yet."""
    s = settings
    store = ModelStore(s.artifact_dir)
    current = store.load_current()
    if current is None:
        return None
    model, meta = current
    notes: list[str] = []
    cross: Bars = dict(bars)
    membership = None
    if s.model.universe == "sp500":
        members = sorted(default_membership().members(session) - set(bars))
        start = session - timedelta(days=s.data.live_lookback_days)
        try:
            extra = provider.fetch(members, start, session) if members else {}
        except DataError as exc:
            extra = {}
            notes.append(f"index bars unavailable ({str(exc)[:80]})")
        fresh = {k: v for k, v in extra.items() if len(v) and v.index[-1] == pd.Timestamp(session)}
        share = (len(fresh) + len(bars)) / max(1, len(members) + len(bars))
        if share >= MIN_UNIVERSE_FRACTION:
            cross.update(fresh)
            membership = default_membership()
        else:
            notes.append(
                f"only {share:.0%} of index members had today's bar; ranked on the watchlist only"
            )
    fs_model = FeatureSet.from_columns(model.features, normalize=s.model.normalize)
    ds = build_dataset(
        cross,
        s.benchmark,
        model.horizon,
        membership=membership,
        extra_symbols=s.watchlist,
        fundamentals=_read(s.fundamentals_path) if fs_model.quality else None,
        text_scores=_read(s.analyst_scores_path) if fs_model.text else None,
        features=fs_model,
        min_price=s.model.min_price,
        min_dollar_volume=s.model.min_dollar_volume_20,
        text_lookback_days=s.model.text_lookback_days,
        start=session - timedelta(days=10),
    )
    predictor = ModelPredictor(model=model, dataset=ds, name=f"model:{meta.version}")
    cal = live_calendar(s, cross, upcoming)
    strategy = ModelStrategy(s, predictor, cal, tradable=set(s.watchlist) - {s.benchmark})
    return LiveModel(strategy=strategy, version=meta.version, notes=notes)
