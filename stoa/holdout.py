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
    holdout_end: date | None = None
    sealed_at: datetime
    model_version: str = ""  # the promoted model at seal time; scoring uses exactly this one
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
    from stoa.exam import read_walkforward

    s = settings or load_settings(repo / config_path)
    if s.exam.require_walkforward_pass:
        wf = read_walkforward(s.shared_dir)
        if wf is None:
            raise HoldoutError(
                "walk-forward results do not exist yet; run `stoa model train` first. "
                "The exam order is walk-forward, then the holdout, then paper."
            )
        if not wf.passed:
            raise HoldoutError(
                "walk-forward did not pass (" + "; ".join(wf.reasons) + "). "
                "Sealing the holdout now would spend it on a model that already failed."
            )
    lock = HoldoutLock(
        commit=head_commit(repo),
        code_hash=code_hash(repo),
        config_hash=file_sha256(repo / config_path),
        scoreboard_hash=file_sha256(repo / scoreboard_path),
        strategy_hash=s.strategy_hash(),
        holdout_start=s.backtest.holdout_start,
        holdout_end=s.backtest.holdout_end,
        sealed_at=now,
        model_version=current_model_version(s),
    )
    write_lock(repo, lock)
    return lock


def current_model_version(settings: Settings) -> str:
    from stoa.model import ModelStore

    return ModelStore(settings.artifact_dir).current_version() or ""


def is_scored(repo: Path) -> bool:
    lock = read_lock(repo)
    return lock is not None and lock.scored_at is not None


def training_cutoff(repo: Path, settings: Settings, last_session: date) -> date:
    """The last session training may use.

    Until the holdout has been scored, nothing inside it may train or evaluate
    a model. While the holdout has not ended, that means stopping the session
    before its start. After ``backtest.holdout_end``, later sessions are usable
    again, and the holdout rows themselves are removed by ``blocked_window``.
    Once scored, the holdout is spent and every session may be used.
    """
    from stoa import clock

    if is_scored(repo):
        return last_session
    end = settings.backtest.holdout_end
    if end is not None and last_session > end:
        return last_session
    before = clock.previous_session(settings.backtest.holdout_start)
    return min(last_session, before)


def blocked_window(repo: Path, settings: Settings) -> tuple[date, date] | None:
    """The holdout dates training and evaluation must not touch, or None once it is scored."""
    if is_scored(repo):
        return None
    return holdout_dates(settings)


def holdout_dates(settings: Settings) -> tuple[date, date]:
    b = settings.backtest
    return b.holdout_start, b.holdout_end or date.max


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
    if lock.model_version and current_model_version(settings) != lock.model_version:
        problems.append(
            f"the promoted model changed since the seal ({lock.model_version} at seal time)"
        )
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
    model: Any | None = None,
) -> dict[str, Any]:
    """Verify the seal, run every strategy over the holdout, record the result once.

    The pass bar is the same risk bar as the scoreboard: lower max drawdown,
    higher Sharpe and Sortino than SPY, the drawdown cap and the API cost cap.
    MTUM is reported, and the verdict says to buy it if Stoa does not beat it.
    """
    from stoa import clock, scoreboard
    from stoa.baselines import BuyAndHold

    lock = verify(repo, config_path, scoreboard_path, settings)
    finish = end or clock.last_closed_session(now)
    start, finish = scoreboard.holdout_window(settings, finish)
    board = scoreboard.load_scoreboard(repo / scoreboard_path)
    bars = (
        model.bars
        if model is not None
        else scoreboard.load_bars(settings, finish, synthetic=synthetic)
    )
    runs: dict[str, scoreboard.Run] = {}
    for s in scoreboard.STRATEGIES:
        runs[s] = scoreboard.run_one(
            settings, secrets, bars, s, "holdout", start, finish, llm=llm and s == "agents"
        )
    if model is not None:
        runs["model"] = scoreboard.run_one(
            settings,
            secrets,
            bars,
            model.strategy,
            "holdout",
            start,
            finish,
            universe=model.universe,
        )
    if "MTUM" in bars:
        from stoa.universe import Universe

        runs["mtum"] = scoreboard.run_one(
            settings,
            secrets,
            bars,
            BuyAndHold("MTUM", name="mtum"),
            "holdout",
            start,
            finish,
            universe=Universe.assume_all_fractionable(
                (*settings.watchlist, "MTUM"), settings.benchmark
            ),
        )
    graded = "model" if model is not None else "agents"
    g, spy = runs[graded].metrics, runs["spy"].metrics
    c = board.criteria
    checks = {
        "drawdown_vs_spy": g.max_drawdown < spy.max_drawdown,
        "sharpe_vs_spy": g.sharpe > spy.sharpe,
        "sortino_vs_spy": g.sortino > spy.sortino,
        "drawdown_cap": g.max_drawdown <= c.max_drawdown_cap,
        "api_cost": runs[graded].max_monthly_api_cost < c.max_monthly_api_cost_usd,
    }
    verdict = []
    if "mtum" in runs:
        mt = runs["mtum"].metrics
        beats = scoreboard.beats_on_risk(g, mt)
        checks_mtum = {"beats_mtum": beats}
        if c.beat_mtum_for_pass:
            checks.update(checks_mtum)
        verdict.append(
            "Stoa beats MTUM on the risk-adjusted bar."
            if beats
            else "Stoa does not beat MTUM on the risk-adjusted bar. Recommendation: buy MTUM."
        )
    result: dict[str, Any] = {
        "window": {"start": start.isoformat(), "end": finish.isoformat()},
        "data_source": "synthetic" if synthetic else "cache",
        "llm": llm,
        "graded": graded,
        "passed": all(checks.values()),
        "checks": checks,
        "verdict": verdict,
        "metrics": {s: r.metrics.to_dict() for s, r in runs.items()},
    }
    record(repo, lock, result, now)
    return result
