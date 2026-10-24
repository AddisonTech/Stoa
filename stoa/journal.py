"""Append-only JSONL decision log. One record per step. Never rewritten."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _default(obj: object) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    return str(obj)


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def hash_obj(obj: object) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


class Journal:
    """Writes to <root>/<YYYY-MM>.jsonl, opened in append mode only."""

    def __init__(self, root: Path | None) -> None:
        self.root = root
        self.records: list[dict[str, Any]] = []

    def write(self, ts: datetime, job: str, session: date | None, step: str, **data: Any) -> None:
        record = {
            "ts": ts.isoformat(),
            "job": job,
            "session": session.isoformat() if session else None,
            "step": step,
            "data": data,
        }
        self.records.append(record)
        if self.root is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{ts:%Y-%m}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(record) + "\n")


def read_journal(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for path in sorted(root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            out.extend(json.loads(line) for line in fh if line.strip())
    return out
