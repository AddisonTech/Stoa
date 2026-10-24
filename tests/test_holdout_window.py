"""The holdout is a fixed window: nothing but `stoa holdout score` may read it."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stoa import holdout, scoreboard
from stoa.config import BacktestConfig, Window
from stoa.data.base import Bars
from stoa.dataset import FeatureSet
from stoa.pipeline import load_inputs
from stoa.training import TrainingInputs, dataset_for
from tests.conftest import SMALL_WATCHLIST, make_settings

START, END = date(2026, 7, 1), date(2026, 10, 23)


def scored_lock(repo: Path) -> None:
    holdout.write_lock(
        repo,
        holdout.HoldoutLock(
            commit="c",
            code_hash="h",
            config_hash="h",
            scoreboard_hash="h",
            strategy_hash="h",
            holdout_start=START,
            holdout_end=END,
            sealed_at=datetime(2026, 11, 1, tzinfo=UTC),
            scored_at=datetime(2026, 12, 1, tzinfo=UTC),
        ),
    )


def test_the_shipped_holdout_is_fixed() -> None:
    s = make_settings()
    assert (s.backtest.holdout_start, s.backtest.holdout_end) == (START, END)
    assert holdout.holdout_dates(s) == (START, END)


def test_cutoff_before_and_after_the_end_and_after_scoring(tmp_path: Path) -> None:
    s = make_settings()
    # Before the holdout ends, training stops the session before it starts.
    assert holdout.training_cutoff(tmp_path, s, date(2026, 9, 1)) == date(2026, 6, 30)
    assert holdout.training_cutoff(tmp_path, s, END) == date(2026, 6, 30)
    # After it ends, later sessions are usable; the holdout rows stay blocked.
    assert holdout.training_cutoff(tmp_path, s, date(2026, 11, 20)) == date(2026, 11, 20)
    assert holdout.blocked_window(tmp_path, s) == (START, END)
    # Once scored, the holdout is spent and nothing is blocked.
    scored_lock(tmp_path)
    assert holdout.training_cutoff(tmp_path, s, date(2026, 9, 1)) == date(2026, 9, 1)
    assert holdout.blocked_window(tmp_path, s) is None


def test_windows_may_not_overlap_the_holdout() -> None:
    def cfg(start: date, end: date) -> BacktestConfig:
        return BacktestConfig(
            holdout_start=START, holdout_end=END, windows={"w": Window(start=start, end=end)}
        )

    assert cfg(date(2025, 1, 1), date(2026, 6, 30))  # before: fine
    assert cfg(date(2026, 10, 26), date(2027, 3, 31))  # after the end: fine
    for a, b in ((date(2026, 6, 1), date(2026, 7, 1)), (date(2026, 10, 23), date(2027, 1, 1))):
        with pytest.raises(ValueError, match="overlaps the sealed holdout"):
            cfg(a, b)
    with pytest.raises(ValueError, match="holdout_end must be after"):
        BacktestConfig(holdout_start=START, holdout_end=START)


def test_holdout_scoring_stops_at_the_end() -> None:
    s = make_settings()
    assert scoreboard.holdout_window(s, date(2027, 2, 1)) == (START, END)


def test_load_inputs_blocks_the_holdout_by_default() -> None:
    s = make_settings(watchlist=",".join(SMALL_WATCHLIST))
    assert load_inputs(s, date(2021, 12, 31), synthetic=True).training.blocked == (START, END)
    free = load_inputs(s, date(2021, 12, 31), synthetic=True, blocked=None)
    assert free.training.blocked is None


def wreck(bars: Bars, lo: date, hi: date) -> Bars:
    """Every price inside [lo, hi] multiplied by a different random factor."""
    rng = np.random.default_rng(3)
    out = {}
    for sym, df in bars.items():
        d = df.copy()
        inside = (d.index >= pd.Timestamp(lo)) & (d.index <= pd.Timestamp(hi))
        f = rng.uniform(0.3, 3.0, inside.sum())
        for c in ("open", "high", "low", "close"):
            d.loc[inside, c] = d.loc[inside, c].to_numpy() * f
        out[sym] = d
    return out


def test_nothing_in_training_reads_the_holdout_window(bars: Bars) -> None:
    lo, hi = date(2020, 3, 2), date(2020, 5, 29)  # a stand-in holdout inside the synthetic years
    s = make_settings(watchlist=",".join(SMALL_WATCHLIST), model__min_dollar_volume_20="0")
    fs = FeatureSet(quality=False)

    def dataset(b: Bars) -> pd.DataFrame:
        inputs = TrainingInputs(bars=b, extra_symbols=tuple(SMALL_WATCHLIST), blocked=(lo, hi))
        return dataset_for(s, inputs, fs, horizon=10).panel

    clean, dirty = dataset(bars), dataset(wreck(bars, lo, hi))
    dates = clean.index.get_level_values("date")
    assert not ((dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))).any()
    before = dates < pd.Timestamp(lo)
    assert (pd.to_datetime(clean.loc[before, "label_end"]) < pd.Timestamp(lo)).all()
    assert clean.index.equals(dirty.index)
    # Rows before the holdout are untouched in every column: features look back, labels end before.
    pd.testing.assert_frame_equal(clean.loc[before], dirty.loc[before])
    # Rows after it: every label uses only later prices. Features may look back over the window,
    # as every live decision does, but no label, test row or score comes from it.
    after = ~before
    labels = ["fwd_ret", "fwd_excess", "y", "label_end"]
    pd.testing.assert_frame_equal(clean.loc[after, labels], dirty.loc[after, labels])
    # And the window really was in the data: without the block its rows are there.
    unblocked = dataset_for(s, TrainingInputs(bars=bars, extra_symbols=tuple(SMALL_WATCHLIST)), fs)
    u = unblocked.panel.index.get_level_values("date")
    assert ((u >= pd.Timestamp(lo)) & (u <= pd.Timestamp(hi))).any()
