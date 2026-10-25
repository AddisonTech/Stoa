"""Request pacing shared by the background pulls.

Each pull runs at its own rate (requests per minute), read from
``data/newsbrain/limits.json`` once a minute so the split between pulls can be
changed while they run. All Alpaca pulls share one key, whose limit is 200 a
minute, and the live jobs use the same key: every Alpaca pull pauses during the
live windows on weekdays (New York time), so the 9:00 am sweep, the 9:45 am
EXECUTE and the 5:15 pm DECIDE always have the key to themselves.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from datetime import time as tod
from pathlib import Path

from stoa.clock import NY

log = logging.getLogger(__name__)

# (start, end) New York time on weekdays when Alpaca pulls stand aside.
LIVE_WINDOWS = ((tod(8, 45), tod(10, 20)), (tod(17, 0), tod(17, 50)), (tod(19, 35), tod(20, 30)))


def in_live_window(now: datetime) -> bool:
    ny = now.astimezone(NY)
    if ny.weekday() >= 5:
        return False
    t = ny.time()
    return any(a <= t < b for a, b in LIVE_WINDOWS)


class Pace:
    """At most ``rate`` calls a minute for this pull, plus the live-window pause."""

    def __init__(
        self,
        name: str,
        default_rate: float,
        limits_path: Path,
        *,
        respect_live: bool = True,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.default_rate = default_rate
        self.limits_path = limits_path
        self.respect_live = respect_live
        self.now = now
        self.sleep = sleep
        self.monotonic = monotonic
        self.rate = default_rate
        self._read_at = -1e18
        self._next = 0.0

    def _refresh(self) -> None:
        t = self.monotonic()
        if t - self._read_at < 60:
            return
        self._read_at = t
        try:
            body = json.loads(self.limits_path.read_text(encoding="utf-8"))
            self.rate = float(body.get(self.name, self.default_rate))
        except (OSError, ValueError):
            self.rate = self.default_rate

    def wait(self) -> None:
        while self.respect_live and in_live_window(self.now()):
            log.info("%s: live window, pausing", self.name)
            self.sleep(60)
        self._refresh()
        while self.rate <= 0:
            self.sleep(60)
            self._read_at = -1e18
            self._refresh()
        gap = 60.0 / self.rate
        t = self.monotonic()
        if t < self._next:
            self.sleep(self._next - t)
            t = self._next
        self._next = t + gap

    def backoff(self, seconds: float) -> None:
        """After a 429 or a server error: stand still, then carry on at the same rate."""
        self.sleep(seconds)
        self._next = self.monotonic()


def write_parquet(df: object, path: Path) -> None:
    """Write atomically: a crash never leaves a half file that looks done."""
    import pandas as pd

    from stoa.files import replace_file
    from stoa.newsbrain import PARQUET_LEVEL

    assert isinstance(df, pd.DataFrame)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd", compression_level=PARQUET_LEVEL)
    replace_file(tmp, path)


def log_progress(path: Path, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"at": datetime.now(UTC).isoformat(timespec="seconds"), **fields})
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
