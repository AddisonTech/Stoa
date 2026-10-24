"""The rename step of an atomic write, shared by every temp-file-then-replace writer.

On Windows an antivirus scanner or the search indexer can hold a freshly written
file open for a moment, and os.replace then fails with PermissionError (WinError 5)
even though nothing is wrong. That case is retried a few times with a short backoff.
Every other error, and the last failed attempt, is raised as before. On Linux the
rename does not fail this way, so behavior there is unchanged.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

REPLACE_ATTEMPTS = 5
REPLACE_BACKOFF = 0.05


def replace_file(
    src: Path,
    dst: Path,
    *,
    attempts: int = REPLACE_ATTEMPTS,
    backoff: float = REPLACE_BACKOFF,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """os.replace(src, dst), retrying only on PermissionError."""
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            wait = backoff * (2**attempt)
            log.debug("replace %s -> %s locked, retrying in %.2fs", src, dst, wait)
            sleep(wait)
