from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from stoa.journal import Journal, hash_obj, read_journal


def test_journal_appends_and_never_rewrites(tmp_path: Path) -> None:
    j = Journal(tmp_path)
    ts = datetime(2026, 3, 2, 22, 0, tzinfo=UTC)
    j.write(ts, "decide", date(2026, 3, 2), "scout", candidates=["A"])
    path = tmp_path / "2026-03.jsonl"
    first = path.read_text()
    Journal(tmp_path).write(ts, "decide", date(2026, 3, 2), "risk", orders=[])
    assert path.read_text().startswith(first)
    recs = read_journal(tmp_path)
    assert [r["step"] for r in recs] == ["scout", "risk"]
    assert recs[0]["data"] == {"candidates": ["A"]}


def test_memory_journal_keeps_records() -> None:
    j = Journal(None)
    j.write(datetime(2026, 3, 2, tzinfo=UTC), "x", None, "s", a=1)
    assert j.records[0]["session"] is None


def test_hash_is_order_independent() -> None:
    assert hash_obj({"a": 1, "b": 2}) == hash_obj({"b": 2, "a": 1})
    assert hash_obj({"a": 1}) != hash_obj({"a": 2})
