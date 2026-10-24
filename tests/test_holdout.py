from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from stoa import holdout
from stoa.config import Secrets, load_settings
from stoa.exam import WalkForwardRecord, write_walkforward
from tests.conftest import ROOT, SMALL_WATCHLIST

CFG = Path("config/default.yaml")
BOARD = Path("config/scoreboard.yaml")
NOW = datetime(2026, 10, 20, 22, 0, tzinfo=UTC)
ENV: dict[str, str] = {"STOA__WATCHLIST": ",".join(SMALL_WATCHLIST)}


def passed_walkforward(state_dir: Path, passed: bool = True) -> None:
    write_walkforward(
        state_dir,
        WalkForwardRecord(
            created="2026-10-01T00:00:00+00:00",
            passed=passed,
            reasons=[] if passed else ["deflated Sharpe 0.400 under 0.95"],
            checks={},
            report_path="",
            model_version="v1",
        ),
    )


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    shutil.copytree(ROOT / "stoa", r / "stoa", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "config", r / "config")
    git(r, "init", "-q")
    git(r, "config", "user.name", "AddisonTech")
    git(r, "config", "user.email", "stalor54@yahoo.com")
    git(r, "config", "commit.gpgsign", "false")
    (r / ".gitignore").write_text("/state/\n")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "init")
    ENV["STOA_STATE_DIR"] = str(tmp_path / "state")
    passed_walkforward(tmp_path / "state")
    return r


def settings(repo: Path):  # type: ignore[no-untyped-def]
    return load_settings(repo / CFG, env=ENV)


def score(repo: Path):  # type: ignore[no-untyped-def]
    return holdout.score(
        repo,
        CFG,
        BOARD,
        settings(repo),
        Secrets.from_env({}),
        NOW,
        synthetic=True,
        end=date(2026, 10, 16),
    )


def test_seal_records_commit_and_hashes(repo: Path) -> None:
    lock = holdout.seal(repo, CFG, BOARD, NOW, settings(repo))
    assert lock.commit == holdout.head_commit(repo)
    assert lock.holdout_start == date(2026, 7, 1)
    assert lock.scored_at is None
    assert holdout.read_lock(repo) == lock
    with pytest.raises(holdout.HoldoutError, match="already"):
        holdout.seal(repo, CFG, BOARD, NOW, settings(repo))


def test_seal_refuses_dirty_tree(repo: Path) -> None:
    (repo / "stoa" / "extra.py").write_text("x = 1\n")
    with pytest.raises(holdout.HoldoutError, match="commit"):
        holdout.seal(repo, CFG, BOARD, NOW, settings(repo))


def test_score_is_one_shot(repo: Path) -> None:
    holdout.seal(repo, CFG, BOARD, NOW, settings(repo))
    result = score(repo)
    assert set(result["metrics"]) == {"agents", "spy", "sma", "mtum"}
    assert result["window"]["start"] == "2026-07-01"
    lock = holdout.read_lock(repo)
    assert lock is not None and lock.scored_at == NOW and lock.score == result
    with pytest.raises(holdout.HoldoutError, match="already scored"):
        score(repo)


def test_score_refuses_without_seal(repo: Path) -> None:
    with pytest.raises(holdout.HoldoutError, match="not sealed"):
        score(repo)


def test_score_refuses_dirty_tree(repo: Path) -> None:
    holdout.seal(repo, CFG, BOARD, NOW, settings(repo))
    (repo / "notes.txt").write_text("tweak\n")
    with pytest.raises(holdout.HoldoutError, match="dirty"):
        score(repo)


def test_score_refuses_changed_code(repo: Path) -> None:
    holdout.seal(repo, CFG, BOARD, NOW, settings(repo))
    risk = repo / "stoa" / "risk.py"
    risk.write_text(risk.read_text() + "\n# tuned after seeing the holdout\n")
    git(repo, "commit", "-qam", "tweak")
    with pytest.raises(holdout.HoldoutError, match="code"):
        score(repo)


def test_score_refuses_changed_config(repo: Path) -> None:
    holdout.seal(repo, CFG, BOARD, NOW, settings(repo))
    cfg = repo / CFG
    cfg.write_text(cfg.read_text().replace("max_positions: 5", "max_positions: 4"))
    git(repo, "commit", "-qam", "tweak")
    with pytest.raises(holdout.HoldoutError, match="config"):
        score(repo)


def test_score_refuses_changed_scoreboard(repo: Path) -> None:
    holdout.seal(repo, CFG, BOARD, NOW, settings(repo))
    b = repo / BOARD
    b.write_text(b.read_text().replace("min_trades: 30", "min_trades: 10"))
    git(repo, "commit", "-qam", "tweak")
    with pytest.raises(holdout.HoldoutError, match="scoreboard"):
        score(repo)


def test_score_refuses_env_override_of_strategy(repo: Path) -> None:
    holdout.seal(repo, CFG, BOARD, NOW, settings(repo))
    tweaked = load_settings(repo / CFG, env={**ENV, "STOA__RISK__MAX_POSITIONS": "3"})
    with pytest.raises(holdout.HoldoutError, match="strategy"):
        holdout.score(
            repo,
            CFG,
            BOARD,
            tweaked,
            Secrets.from_env({}),
            NOW,
            synthetic=True,
            end=date(2026, 10, 16),
        )


def test_seal_requires_walkforward_results(repo: Path, tmp_path: Path) -> None:
    (tmp_path / "state" / "exam" / "walkforward.json").unlink()
    with pytest.raises(holdout.HoldoutError, match="model train"):
        holdout.seal(repo, CFG, BOARD, NOW, settings(repo))


def test_seal_refuses_failed_walkforward(repo: Path, tmp_path: Path) -> None:
    passed_walkforward(tmp_path / "state", passed=False)
    with pytest.raises(holdout.HoldoutError, match="did not pass"):
        holdout.seal(repo, CFG, BOARD, NOW, settings(repo))


def test_holdout_graded_on_risk_with_mtum_verdict(repo: Path) -> None:
    holdout.seal(repo, CFG, BOARD, NOW, settings(repo))
    result = score(repo)
    assert set(result["checks"]) == {
        "drawdown_vs_spy",
        "sharpe_vs_spy",
        "sortino_vs_spy",
        "drawdown_cap",
        "api_cost",
    }
    assert "mtum" in result["metrics"] and result["verdict"]
