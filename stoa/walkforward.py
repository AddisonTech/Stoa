"""Purged, embargoed walk-forward splits. Train on the past, test on the next slice.

For each fold:
- The test slice is ``test_sessions`` consecutive sessions.
- Training rows are decision sessions strictly before the test slice whose
  label is complete before the test slice starts (purging), and which end at
  least ``embargo`` sessions before it (the embargo gap). With an embargo of
  at least the horizon, no training label can overlap a test label.
- ``expanding`` uses all history before that point; ``rolling`` keeps only the
  last ``rolling_sessions``.

Every fold carries its own indices so a test can prove there is no overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Fold:
    number: int
    train_start: date
    train_end: date  # last decision session in training
    test_start: date
    test_end: date
    train_sessions: tuple[date, ...]
    test_sessions: tuple[date, ...]


def make_folds(
    sessions: list[date],
    *,
    horizon: int,
    embargo: int,
    test_sessions: int,
    min_train_sessions: int,
    window: str = "expanding",
    rolling_sessions: int = 756,
    end: date | None = None,
) -> list[Fold]:
    """Folds over ``sessions`` (sorted, unique). The last fold may be shorter."""
    if embargo < horizon:
        raise ValueError("embargo must be at least the label horizon")
    days = [d for d in sessions if end is None or d <= end]
    folds: list[Fold] = []
    # Train decisions at index i have labels ending at i + horizon. The gap
    # between the last training decision and the first test decision is the
    # embargo, so the last training label ends before the test slice starts.
    first_test = min_train_sessions + embargo
    k = 0
    while first_test < len(days):
        test = days[first_test : first_test + test_sessions]
        train_hi = first_test - embargo  # exclusive
        train_lo = 0 if window == "expanding" else max(0, train_hi - rolling_sessions)
        train = days[train_lo:train_hi]
        if len(train) >= min(min_train_sessions, rolling_sessions):
            folds.append(
                Fold(
                    number=k,
                    train_start=train[0],
                    train_end=train[-1],
                    test_start=test[0],
                    test_end=test[-1],
                    train_sessions=tuple(train),
                    test_sessions=tuple(test),
                )
            )
            k += 1
        first_test += test_sessions
    return folds


def label_end_index(sessions: list[date], i: int, horizon: int) -> int:
    return i + horizon
