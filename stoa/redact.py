"""Keeps secret values out of logs and alert text.

Every secret Stoa reads from the environment is registered here. A logging
filter replaces any registered value in log records, and :func:`redact` scrubs
text before it goes into an alert (for example an exception message).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable

MASK = "<redacted>"
MIN_SECRET_LEN = 6  # shorter values are too likely to match ordinary text

_lock = threading.Lock()
_secrets: set[str] = set()


def register(values: Iterable[str | None]) -> None:
    """Remember secret values so they are scrubbed from logs and alerts."""
    with _lock:
        for v in values:
            if v and len(v) >= MIN_SECRET_LEN:
                _secrets.add(v)


def clear() -> None:
    """Forget every registered secret (tests only)."""
    with _lock:
        _secrets.clear()


def redact(text: str) -> str:
    """Replace every registered secret value in ``text``. Longest first."""
    with _lock:
        values = sorted(_secrets, key=len, reverse=True)
    for v in values:
        if v in text:
            text = text.replace(v, MASK)
    return text


class RedactingFilter(logging.Filter):
    """Rewrites a log record's final message with secrets masked."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            return True
        cleaned = redact(message)
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        if cleaned != message or record.args:
            record.msg = cleaned
            record.args = None
        return True


def install(logger: logging.Logger | None = None) -> None:
    """Attach the filter to every handler of ``logger`` (the root logger by default)."""
    target = logger or logging.getLogger()
    for handler in target.handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())
