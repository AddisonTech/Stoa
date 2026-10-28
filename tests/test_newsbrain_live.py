"""The news brain for one live session (the big paper account's morning picks)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from stoa import clock
from stoa.newsbrain import live, sealed

LIVE = date(2026, 11, 3)


def test_live_reads_reach_the_session_and_never_past_it() -> None:
    with sealed.live(LIVE):
        sealed.check(date(2026, 8, 3), "a baseline read inside the spent holdout")
        sealed.check(LIVE, "the live session")
        with pytest.raises(sealed.HoldoutBreach):
            sealed.check(date(2026, 11, 4), "tomorrow")
        assert sealed.month_allowed("2026-11") and not sealed.month_allowed("2026-12")
        df = pd.DataFrame({"t": pd.to_datetime(["2026-11-03 20:00", "2026-11-04 06:00"], utc=True)})
        assert len(sealed.before_holdout(df, "t")) == 1
    with pytest.raises(sealed.HoldoutBreach):
        sealed.check(date(2026, 8, 3), "after the live read")  # the wall is back


def test_live_mode_is_only_for_sessions_after_the_holdout() -> None:
    with pytest.raises(sealed.HoldoutBreach), sealed.live(sealed.HOLDOUT_END):
        pass
    with pytest.raises(sealed.HoldoutBreach), sealed.live(date(2026, 8, 3)):
        pass
    assert sealed.live_session() is None


def test_live_mode_never_runs_inside_a_trial_opening(tmp_path: Path) -> None:
    with (
        sealed.opened(sealed.HOLDOUT_KEY, tmp_path / "spent.json"),
        pytest.raises(sealed.HoldoutBreach),
        sealed.live(LIVE),
    ):
        pass


def test_freshness_names_what_is_missing(tmp_path: Path) -> None:
    news, bars = tmp_path / "news", tmp_path / "bars"
    news.mkdir()
    bars.mkdir()
    problems = live.freshness(LIVE, news, bars)
    assert any("no news file for 2026-11" in p for p in problems)
    assert any("SPY" in p for p in problems)
    idx = pd.DatetimeIndex([pd.Timestamp(clock.previous_session(LIVE))])
    pd.DataFrame({"close": [500.0]}, index=idx).to_parquet(bars / "SPY.parquet")
    pd.DataFrame({"created_at": pd.to_datetime(["2026-11-03 12:30"], utc=True)}).to_parquet(
        news / "2026-11.parquet"
    )
    problems = live.freshness(LIVE, news, bars)
    assert problems == ["the news for 2026-11 is not scored yet"]
