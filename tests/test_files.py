from __future__ import annotations

import os
from pathlib import Path

import pytest

from stoa import files


def test_replace_retries_once_on_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src, dst = tmp_path / "j2.tmp", tmp_path / "j2.json"
    src.write_text("new", encoding="utf-8")
    dst.write_text("old", encoding="utf-8")
    real_replace = os.replace
    calls: list[tuple[str, str]] = []

    def flaky(a: str, b: str) -> None:
        calls.append((str(a), str(b)))
        if len(calls) == 1:
            raise PermissionError(13, "Access is denied")
        real_replace(a, b)

    monkeypatch.setattr(files.os, "replace", flaky)
    slept: list[float] = []
    files.replace_file(src, dst, sleep=slept.append)
    assert len(calls) == 2 and slept == [files.REPLACE_BACKOFF]
    assert dst.read_text(encoding="utf-8") == "new" and not src.exists()


def test_replace_gives_up_after_the_last_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def locked(a: str, b: str) -> None:
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(files.os, "replace", locked)
    slept: list[float] = []
    with pytest.raises(PermissionError):
        files.replace_file(tmp_path / "a.tmp", tmp_path / "a.json", attempts=3, sleep=slept.append)
    assert len(slept) == 2


def test_replace_does_not_retry_other_errors(tmp_path: Path) -> None:
    slept: list[float] = []
    with pytest.raises(FileNotFoundError):
        files.replace_file(tmp_path / "missing.tmp", tmp_path / "a.json", sleep=slept.append)
    assert slept == []
