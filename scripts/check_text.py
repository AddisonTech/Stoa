"""Fail if any tracked file contains an em dash, or a commit message has trailers
that name a co-author or sign-off. Run from the repository root.

    python scripts/check_text.py files
    python scripts/check_text.py commits <git range>
"""

from __future__ import annotations

import subprocess
import sys

EM_DASH = "—"
BANNED_TRAILERS = ("co-authored-by", "signed-off-by")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], check=True, capture_output=True).stdout
    return [p for p in out.decode().split("\0") if p]


def check_files() -> int:
    bad = []
    for path in tracked_files():
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if EM_DASH in line:
                bad.append(f"{path}:{lineno}")
    for item in bad:
        print(f"em dash (U+2014) found at {item}")
    return 1 if bad else 0


def check_commits(rev_range: str) -> int:
    out = subprocess.run(
        ["git", "log", "--format=%H%x00%B%x01", rev_range], check=True, capture_output=True
    ).stdout.decode()
    bad = 0
    for entry in out.split("\x01"):
        if not entry.strip():
            continue
        sha, _, body = entry.strip().partition("\x00")
        lowered = body.lower()
        for trailer in BANNED_TRAILERS:
            if trailer in lowered:
                print(f"commit {sha[:10]} contains a {trailer} line")
                bad = 1
        if EM_DASH in body:
            print(f"commit {sha[:10]} message contains an em dash")
            bad = 1
    return bad


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "files":
        return check_files()
    if len(argv) >= 3 and argv[1] == "commits":
        return check_commits(argv[2])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
