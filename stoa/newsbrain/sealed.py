"""The holdout wall for the news brain trials.

Nothing in the news brain may read a session from the holdout start on: not news,
not filings, not daily bars, not minute bars, not 9:45 am prices. Every loader
calls ``check`` or filters with ``before_holdout`` first, and ``HoldoutBreach`` is
raised before any data from the window is read. The dates match
``backtest.holdout_start`` and ``holdout_end`` in config/default.yaml (a test
holds them together).

The owner opened the holdout once, for trial 12 as pre-registered (docs/DECISIONS.md,
"Trial 12 on the holdout"). ``opened`` lifts the wall for that one run only, up to the
holdout's last day, and refuses for any other key and for any second run. After that the
holdout is spent and the wall stays shut.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

HOLDOUT_START = date(2026, 7, 1)
HOLDOUT_END = date(2026, 10, 23)
# Midnight New York time on the first holdout day, for timestamps.
CUTOFF = pd.Timestamp(HOLDOUT_START.isoformat(), tz="America/New_York").tz_convert("UTC")
# Midnight New York time after the holdout's last day: the most an open wall lets through.
END_CUTOFF = pd.Timestamp("2026-10-24", tz="America/New_York").tz_convert("UTC")
HOLDOUT_KEY = "trial12:holdout:e75fa8775469de92"  # the one run allowed to open it
SPENT = Path("data/newsbrain/holdout_spent.json")
_STATE: dict[str, str | None] = {"open_for": None}  # set only inside ``opened``


class HoldoutBreach(RuntimeError):
    """Something asked for data from inside the sealed holdout."""


def is_open() -> bool:
    return _STATE["open_for"] is not None


def check(day: date, what: str) -> None:
    if day >= HOLDOUT_START and not (is_open() and day <= HOLDOUT_END):
        raise HoldoutBreach(
            f"{what} for {day} is inside the holdout ({HOLDOUT_START} to {HOLDOUT_END}) "
            "or after it, which the trials never read"
        )


def month_allowed(stem: str) -> bool:
    """Whether a monthly file named YYYY-MM may be read: wholly before the holdout, or
    up to its last month while the wall is open."""
    if is_open():
        return stem <= f"{HOLDOUT_END.year:04d}-{HOLDOUT_END.month:02d}"
    return stem < f"{HOLDOUT_START.year:04d}-{HOLDOUT_START.month:02d}"


def before_holdout(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Rows whose timestamp is before the holdout's first day, New York time (before the
    day after its last day while the wall is open)."""
    ts = pd.to_datetime(df[column], utc=True)
    return df[ts < (END_CUTOFF if is_open() else CUTOFF)]


@contextmanager
def opened(key: str, spent: Path = SPENT, *, owner_rerun: bool = False) -> Iterator[None]:
    """Lift the wall for the one pre-registered holdout run, then put it back.

    A marker is written the moment the wall opens. A finished run makes the holdout
    spent for good; a run that stopped part way can open it again only with the owner's
    word (``owner_rerun``).
    """
    if key != HOLDOUT_KEY:
        raise HoldoutBreach("only trial 12's pre-registered holdout run may open the holdout")
    if spent.exists():
        status = json.loads(spent.read_text(encoding="utf-8")).get("status")
        if status == "done" or not owner_rerun:
            raise HoldoutBreach(
                f"the holdout was opened before ({spent}, {status}); no run may read it again"
            )
    _write(spent, {"status": "running"})
    _STATE["open_for"] = key
    try:
        yield
    finally:
        _STATE["open_for"] = None


def mark_spent(summary: dict[str, object], spent: Path = SPENT) -> None:
    """Record that the one holdout run finished; ``opened`` refuses for good from then on."""
    _write(spent, {"status": "done", **summary})


def _write(spent: Path, body: dict[str, object]) -> None:
    spent.parent.mkdir(parents=True, exist_ok=True)
    full = {"key": HOLDOUT_KEY, "at": datetime.now(UTC).isoformat(timespec="seconds"), **body}
    spent.write_text(json.dumps(full, indent=1, default=str), encoding="utf-8")
