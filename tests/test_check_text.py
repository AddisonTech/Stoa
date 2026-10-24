from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.conftest import ROOT

SCRIPT = ROOT / "scripts" / "check_text.py"
DASH = "—"


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def make_repo(tmp_path: Path, text: str, message: str) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "AddisonTech")
    git(tmp_path, "config", "user.email", "stalor54@yahoo.com")
    git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "a.md").write_text(text, encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", message)
    return tmp_path


def check(repo: Path, *args: str) -> int:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=repo, capture_output=True, check=False
    )
    return proc.returncode


def test_clean_repo_passes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, "plain text, with commas.\n", "Add notes")
    assert check(repo, "files") == 0
    assert check(repo, "commits", "HEAD") == 0


def test_em_dash_in_file_fails(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, f"one {DASH} two\n", "Add notes")
    assert check(repo, "files") == 1


def test_trailers_in_commit_fail(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, "ok\n", "Add notes\n\nSigned-off-by: someone <x@y>")
    assert check(repo, "commits", "HEAD") == 1


def test_this_repository_is_clean() -> None:
    assert check(ROOT, "files") == 0
