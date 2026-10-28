"""The news brain for one live session: trial 13's picks for the big paper account.

The rows are built by the same code that built trial 9's panel (``panel.build``), over
the 400 sessions up to the live session, so every feature is computed exactly as the
frozen model was trained on: the news baselines, the first-break test, the filings of
the last 5 sessions and the 15 month supplier links. The wall's ``live`` mode lets it
read up to the session; nothing is labelled or graded.

The data must be current first: news and its FinBERT scores up to 9:00 am, SEC filings
and the daily bars up to the close before. ``freshness`` says what is missing, and no
picks are made from stale data.
"""

from __future__ import annotations

import logging
import pickle
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from stoa import clock, event_signals
from stoa.newsbrain import ROOT, sealed
from stoa.newsbrain import spycore as sc
from stoa.newsbrain import trial as t9
from stoa.newsbrain.alpaca_pull import NEWS_DIR, WIDE_BARS
from stoa.newsbrain.panel import build
from stoa.newsbrain.score import SCORE_DIR

log = logging.getLogger(__name__)

SESSIONS_BACK = 400  # more than the 315 session supplier links need
FROZEN = ROOT / "trial9"


class StaleData(RuntimeError):
    """The news brain's data is not current enough for this session's picks."""


def freshness(session: date, news_dir: Path = NEWS_DIR, bars_dir: Path = WIDE_BARS) -> list[str]:
    """What is not current for picks on ``session``: empty when everything is."""
    problems: list[str] = []
    month = f"{session.year:04d}-{session.month:02d}"
    news = news_dir / f"{month}.parquet"
    if not news.exists():
        problems.append(f"no news file for {month}")
    else:
        scores = SCORE_DIR / "news" / news.name
        if not scores.exists() or scores.stat().st_mtime < news.stat().st_mtime:
            problems.append(f"the news for {month} is not scored yet")
        last = pd.read_parquet(news, columns=["created_at"])["created_at"].max()
        since = pd.Timestamp(clock.at_ny(clock.previous_session(session), "16:00"))
        if pd.isna(last) or pd.Timestamp(last) < since:
            problems.append(f"the newest article is {last}, before yesterday's close")
    spy = bars_dir / "SPY.parquet"
    before = clock.previous_session(session)
    if not spy.exists():
        problems.append("no daily bars for SPY")
    else:
        idx = pd.DatetimeIndex(pd.read_parquet(spy, columns=["close"]).index)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        if pd.Timestamp(before) not in idx.normalize():
            problems.append(f"the daily bars stop before {before}")
    return problems


def rows_for(session: date, sessions_back: int = SESSIONS_BACK) -> pd.DataFrame:
    """The news brain's rows for ``session``, built as trial 9's panel was."""
    start = clock.sessions_between(date(2015, 1, 2), session)[-sessions_back]
    if session > sealed.HOLDOUT_END:
        with sealed.live(session):
            rows = build(start, session)
    else:
        rows = build(start, session)  # a past session, for checking against the panel
    return rows[rows["session"] == session].reset_index(drop=True)


def model(frozen_dir: Path = FROZEN) -> Any:
    return pickle.loads((frozen_dir / f"model_h{sc.GRADED}.pkl").read_bytes())


def signals(session: date, frozen_dir: Path = FROZEN) -> list[event_signals.Signal]:
    """Up to 10 picks at p >= 0.55, best first, filed under the close before ``session``."""
    problems = freshness(session)
    if problems:
        raise StaleData("; ".join(problems))
    rows = rows_for(session)
    if rows.empty:
        return []
    t9.register_kind(sc.GRADED)
    return sc.signals(rows, model(frozen_dir).predict(rows), sc.GRADED)
