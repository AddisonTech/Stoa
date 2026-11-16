"""Read keys from the git-ignored ``.env`` file into the process environment.

Research pulls on the Dell (trials 17 and 18) take their keys from ``.env`` at the
repository root: ``KEY=value`` lines, ``#`` comments and blank lines ignored, quotes
around a value dropped. A variable already set in the environment wins. Values are
never printed, logged or written anywhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(".env")


def parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load(path: Path = ENV_FILE) -> list[str]:
    """Set every variable from ``path`` that is not set yet. Returns the names set."""
    if not path.exists():
        return []
    added = []
    for key, value in parse(path.read_text(encoding="utf-8")).items():
        if key not in os.environ:
            os.environ[key] = value
            added.append(key)
    return added


def need(*names: str, path: Path = ENV_FILE) -> None:
    """Load ``path`` and stop with the missing names (never the values) if any are unset."""
    load(path)
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(f"missing in {path} or the environment: {', '.join(missing)}")
