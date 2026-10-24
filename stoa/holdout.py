"""Seal and one-time scoring of the holdout window.

`stoa holdout seal` writes HOLDOUT.lock with the commit, a hash of the config
and the pass bar, and the date. `stoa holdout score` refuses unless the git
tree is clean and the commit, code and config still match the seal, and it
refuses a second time once a score is recorded. One shot, enforced in code.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from stoa.config import Secrets, Settings, file_sha256

LOCK_NAME = "HOLDOUT.lock"


class HoldoutError(RuntimeError):
    """The holdout cannot be sealed or scored in the current state."""


class HoldoutLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit: str
    code_hash: str
    config_hash: str
    scoreboard_hash: str
    strategy_hash: str
    holdout_start: date
    sealed_at: datetime
    scored_at: datetime | None = None
    score: dict[str, Any] | None = None


def _git(repo: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True, timeout=30
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise HoldoutError(f"git {' '.join(args)} failed: {exc}") from exc
    return out.stdout.strip()


def head_commit(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def dirty_paths(repo: Path) -> list[str]:
    """Tracked or untracked changes, ignoring the lock file itself."""
    lines = _git(repo, "status", "--porcelain", "--untracked-files=all").splitlines()
    return [ln[3:] for ln in lines if ln[3:].strip() != LOCK_NAME]


def code_hash(repo: Path) -> str:
    """Hash of every tracked file under stoa/ at HEAD, from git's own object ids."""
    listing = _git(repo, "ls-tree", "-r", "HEAD", "--", "stoa")
    return hashlib.sha256(listing.encode()).hexdigest()


def lock_path(repo: Path) -> Path:
    return repo / LOCK_NAME


def read_lock(repo: Path) -> HoldoutLock | None:
    p = lock_path(repo)
    if not p.exists():
        return None
    return HoldoutLock.model_validate_json(p.read_text(encoding="utf-8"))


def write_lock(repo: Path, lock: HoldoutLock) -> None:
    lock_path(repo).write_text(lock.model_dump_json(indent=2) + "\n", encoding="utf-8")


def seal(
    repo: Path,
    config_path: Path,
    scoreboard_path: Path,
    now: datetime,
    settings: Settings | None = None,
) -> HoldoutLock:
    if read_lock(repo) is not None:
        raise HoldoutError(f"{LOCK_NAME} already exists; the holdout is sealed")
    dirty = dirty_paths(repo)
    if dirty:
        raise HoldoutError(f"commit your changes before sealing: {', '.join(dirty[:5])}")
    from stoa.config import load_settings

    s = settings or load_settings(repo / config_path)
    lock = HoldoutLock(
        commit=head_commit(repo),
        code_hash=code_hash(repo),
        config_hash=file_sha256(repo / config_path),
        scoreboard_hash=file_sha256(repo / scoreboard_path),
        strategy_hash=s.strategy_hash(),
        holdout_start=s.backtest.holdout_start,
        sealed_at=now,
    )
    write_lock(repo, lock)
    return lock


def verify(repo: Path, config_path: Path, scoreboard_path: Path, settings: Settings) -> HoldoutLock:
    """Raise HoldoutError unless the holdout may be scored right now."""
    lock = read_lock(repo)
    if lock is None:
        raise HoldoutError("the holdout is not sealed; run `stoa holdout seal` first")
    if lock.scored_at is not None:
        raise HoldoutError(
            f"the holdout was already scored on {lock.scored_at:%Y-%m-%d}; it is one shot"
        )
    dirty = dirty_paths(repo)
    if dirty:
        raise HoldoutError(f"the working tree is dirty: {', '.join(dirty[:5])}")
    problems = []
    if code_hash(repo) != lock.code_hash:
        problems.append("code under stoa/ changed")
    if file_sha256(repo / config_path) != lock.config_hash:
        problems.append("config changed")
    if file_sha256(repo / scoreboard_path) != lock.scoreboard_hash:
        problems.append("scoreboard changed")
    if settings.strategy_hash() != lock.strategy_hash:
        problems.append("effective strategy settings changed (environment overrides?)")
    if problems:
        raise HoldoutError("the seal does not match: " + "; ".join(problems))
    return lock


def record(repo: Path, lock: HoldoutLock, score: dict[str, Any], now: datetime) -> HoldoutLock:
    updated = lock.model_copy(update={"scored_at": now, "score": score})
    write_lock(repo, updated)
    return updated


def score(
    repo: Path,
    config_path: Path,
    scoreboard_path: Path,
    settings: Settings,
    secrets: Secrets,
    now: datetime,
    *,
    llm: bool = False,
    synthetic: bool = False,
    end: date | None = None,
) -> dict[str, Any]:
    """Verify the seal, run the three strategies over the holdout, record the result once."""
    from stoa import clock, scoreboard

    lock = verify(repo, config_path, scoreboard_path, settings)
    finish = end or clock.last_closed_session(now)
    start, finish = scoreboard.holdout_window(settings, finish)
    board = scoreboard.load_scoreboard(repo / scoreboard_path)
    bars = scoreboard.load_bars(settings, finish, synthetic=synthetic)
    runs = {
        s: scoreboard.run_one(
            settings, secrets, bars, s, "holdout", start, finish, llm=llm and s == "agents"
        )
        for s in scoreboard.STRATEGIES
    }
    a, spy, sma = (runs[s].metrics for s in scoreboard.STRATEGIES)
    c = board.criteria
    checks = {
        "beats_spy": a.total_return > spy.total_return,
        "beats_sma": a.total_return > sma.total_return,
        "drawdown_vs_spy": a.max_drawdown <= spy.max_drawdown,
        "drawdown_cap": a.max_drawdown <= c.max_drawdown_cap,
        "profit_factor": a.profit_factor >= c.min_profit_factor,
        "api_cost": runs["agents"].max_monthly_api_cost < c.max_monthly_api_cost_usd,
    }
    result: dict[str, Any] = {
        "window": {"start": start.isoformat(), "end": finish.isoformat()},
        "data_source": "synthetic" if synthetic else "cache",
        "llm": llm,
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {s: runs[s].metrics.to_dict() for s in scoreboard.STRATEGIES},
    }
    record(repo, lock, result, now)
    return result
