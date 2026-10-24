from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from stoa.cli import main
from tests.test_cli import CONFIG, env, run

__all__ = ["env"]


def test_backtest_synthetic(env: dict[str, str], tmp_path: Path) -> None:
    code, out = run(
        env,
        "backtest",
        "--strategy",
        "spy",
        "--window",
        "tune_a",
        "--synthetic",
        "--out",
        str(tmp_path / "r"),
    )
    assert code == 0 and "spy tune_a" in out and "total_return" in out
    assert (tmp_path / "r" / "spy-tune_a.json").exists()


def test_backtest_rejects_holdout_window(env: dict[str, str]) -> None:
    result = CliRunner().invoke(
        main, ["--config", CONFIG, "backtest", "--window", "holdout"], env=env
    )
    assert result.exit_code != 0 and "holdout score" in result.output
