"""The v0.2 commands, end to end on synthetic data, offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from stoa.cli import main
from stoa.state import StateStore
from stoa.tracking import activate
from tests.conftest import ROOT, SMALL_WATCHLIST

CONFIG = str(ROOT / "config" / "default.yaml")


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    return {
        "STOA_STATE_DIR": str(tmp_path / "state"),
        "STOA_NTFY_TOPIC": "",
        "STOA_NTFY_APPROVAL_TOPIC": "",
        "ALPACA_PAPER_KEY_ID": "",
        "ALPACA_PAPER_SECRET": "",
        "ANTHROPIC_API_KEY": "",
        "STOA_MODE": "paper",
        "STOA__WATCHLIST": ",".join(SMALL_WATCHLIST),
        "STOA__DEMO__SIGNAL": "1.0",
        "STOA__MODEL__MIN_TRAIN_SESSIONS": "300",
        "STOA__MODEL__TEST_SESSIONS": "126",
        "STOA__MODEL__CANDIDATES": "[logistic]",
    }


def run(env: dict[str, str], *args: str) -> tuple[int, str]:
    result = CliRunner().invoke(main, ["--config", CONFIG, *args], env=env, catch_exceptions=False)
    return result.exit_code, result.output


def test_help_lists_every_v2_command(env: dict[str, str]) -> None:
    code, out = run(env, "--help")
    assert code == 0
    for cmd in (
        "universe",
        "fundamentals",
        "news",
        "analyst",
        "model",
        "experiments",
        "exam",
        "fallback",
        "approve",
    ):
        assert cmd in out


def test_model_train_report_promote_experiments_and_exam(
    env: dict[str, str], tmp_path: Path
) -> None:
    results = str(tmp_path / "results")
    code, out = run(env, "model", "train", "--synthetic", "--results", results, "--repo", str(ROOT))
    assert code == 0, out
    assert "Out of sample: log loss" in out and "Deflated Sharpe" in out
    reports = list(Path(results).glob("*-model/calibration.md"))
    assert reports and "Walk-forward verdict" in reports[0].read_text()

    code, out = run(env, "model", "promote")
    assert code == 0 and "logistic" in out

    code, out = run(env, "experiments", "list")
    assert code == 0 and "distinct configurations tried" in out and "walkforward" in out

    code, out = run(env, "exam", "status", "--repo", str(ROOT))
    assert code == 0 and "Walk-forward" in out and "Live allowed        no" in out

    code, out = run(
        env, "model", "report", "--synthetic", "--results", results, "--repo", str(ROOT)
    )
    assert code == 0 and "10 sessions" in out and "5 sessions" in out


def test_model_promote_by_hand_requires_a_known_version(env: dict[str, str]) -> None:
    result = CliRunner().invoke(
        main, ["--config", CONFIG, "model", "promote", "nope", "--yes"], env=env
    )
    assert result.exit_code != 0 and "no model nope" in result.output


def test_fallback_status_and_reset(env: dict[str, str], tmp_path: Path) -> None:
    code, out = run(env, "fallback", "status")
    assert code == 0 and out.startswith("off")
    store = StateStore(tmp_path / "state")
    state = store.load()
    from datetime import date

    activate(state.track, "rolling Sharpe below the band", date(2026, 9, 1))
    store.save(state)
    code, out = run(env, "fallback", "status")
    assert "ACTIVE since 2026-09-01" in out
    code, out = run(env, "fallback", "reset", "--yes")
    assert code == 0 and "reset" in out
    assert not store.load().track.fallback_active


def test_approve_test_needs_the_topic(env: dict[str, str]) -> None:
    result = CliRunner().invoke(main, ["--config", CONFIG, "approve", "test"], env=env)
    assert result.exit_code != 0 and "STOA_NTFY_APPROVAL_TOPIC" in result.output


def test_analyst_backfill_estimate_without_key(env: dict[str, str], tmp_path: Path) -> None:
    import pandas as pd

    from stoa.config import load_settings

    s = load_settings(
        Path(CONFIG), env={**env, "STOA__SOURCES__NEWS_CACHE_DIR": str(tmp_path / "news")}
    )
    s.news_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": ["a1", "a2"],
            "symbol": ["AAPL", "MSFT"],
            "known_at": pd.to_datetime(["2026-08-03T14:00:00Z", "2026-08-03T15:00:00Z"]),
            "created_at": pd.to_datetime(["2026-08-03T14:00:00Z", "2026-08-03T15:00:00Z"]),
            "headline": ["Apple raises outlook", "Microsoft beats"],
            "summary": ["s", "s"],
            "content": ["c" * 200, "c" * 200],
            "source": ["benzinga", "benzinga"],
            "kind": ["news", "news"],
        }
    ).to_parquet(s.news_path)
    e = {
        **env,
        "STOA__SOURCES__NEWS_CACHE_DIR": str(tmp_path / "news"),
        "STOA__ANALYST__CACHE_DIR": str(tmp_path / "ac"),
    }
    code, out = run(e, "analyst", "backfill", "--estimate")
    assert code == 0 and "2 to score" in out and "Estimated cost $" in out
    result = CliRunner().invoke(main, ["--config", CONFIG, "analyst", "backfill"], env=e)
    assert result.exit_code != 0 and "ANTHROPIC_API_KEY" in result.output


def test_status_shows_v2_fields(env: dict[str, str]) -> None:
    code, out = run(env, "status")
    assert code == 0
    for field in (
        "fallback        off",
        "approval topic  missing",
        "healthcheck     missing",
        "live model      none",
    ):
        assert field in out


def test_score_synthetic_with_model_grades_the_model(env: dict[str, str], tmp_path: Path) -> None:
    code, out = run(env, "score", "--synthetic", "--results", str(tmp_path / "r"))
    assert code == 0, out
    assert "Graded strategy: `model`" in out and "| validation | model |" in out
    assert "MTUM" in out and "Deflated Sharpe" in out
    data = json.loads(next((tmp_path / "r").glob("*/report.json")).read_text())
    assert data["graded"] == "model" and data["deflated"]["trials"] >= 1
