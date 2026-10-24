from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

from stoa import __version__
from stoa.cli import main
from stoa.state import KillSwitch, StateStore
from tests.conftest import ROOT

CONFIG = str(ROOT / "config" / "default.yaml")
Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    return {
        "STOA_STATE_DIR": str(tmp_path / "state"),
        "STOA_NTFY_TOPIC": "",
        "ALPACA_PAPER_KEY_ID": "",
        "ALPACA_PAPER_SECRET": "",
        "ANTHROPIC_API_KEY": "",
        "STOA_MODE": "paper",
    }


def run(env: dict[str, str], *args: str) -> tuple[int, str]:
    result = CliRunner().invoke(main, ["--config", CONFIG, *args], env=env, catch_exceptions=False)
    return result.exit_code, result.output


def test_version_and_help(env: dict[str, str]) -> None:
    code, out = run(env, "--version")
    assert code == 0 and __version__ in out
    code, out = run(env, "--help")
    for cmd in (
        "data",
        "backtest",
        "score",
        "holdout",
        "decide",
        "execute",
        "status",
        "alert",
        "kill-switch",
        "money",
    ):
        assert cmd in out


def test_status_shows_locks(env: dict[str, str]) -> None:
    code, out = run(env, "status")
    assert code == 0
    assert "mode            paper" in out
    assert "live lock       locked" in out
    assert "pending plan    none" in out


def _alpaca_status(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch, handler: Handler
) -> tuple[int, str]:
    def mock_client(base_url: str, key_id: str, secret: str, timeout: float = 30.0) -> httpx.Client:
        return httpx.Client(base_url=base_url, transport=httpx.MockTransport(handler))

    monkeypatch.setattr("stoa.jobs.make_trading_client", mock_client)
    return run({**env, "ALPACA_PAPER_KEY_ID": "PKTEST", "ALPACA_PAPER_SECRET": "SKTEST"}, "status")


def test_status_shows_account_when_keys_are_set(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        return httpx.Response(
            200, json={"equity": "100", "cash": "100", "buying_power": "200", "status": "ACTIVE"}
        )

    code, out = _alpaca_status(env, monkeypatch, h)
    assert code == 0 and seen == ["/v2/account"]
    assert "equity          $100.00" in out
    assert "cash            $100.00" in out
    assert "buying power    $200.00" in out


def test_status_warns_and_continues_when_account_fails(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    code, out = _alpaca_status(env, monkeypatch, lambda _req: httpx.Response(403, json={}))
    lines = [line for line in out.splitlines() if line.startswith("account ")]
    assert code == 0 and len(lines) == 1 and "unavailable" in lines[0]
    assert "equity " not in out
    assert "pending plan    none" in out and "live model" in out


def test_status_skips_account_without_keys(env: dict[str, str]) -> None:
    code, out = run(env, "status")
    assert code == 0 and "equity " not in out and "account " not in out


def test_alert_test_console_fallback(env: dict[str, str]) -> None:
    code, out = run(env, "alert", "test")
    assert code == 0 and "STOA_NTFY_TOPIC is not set" in out and "Stoa test alert" in out


def test_kill_switch_status_and_reset(env: dict[str, str]) -> None:
    store = StateStore(Path(env["STOA_STATE_DIR"]))
    state = store.load()
    state.kill_switch = KillSwitch(active=True, reason="drawdown limit")
    state.high_water_mark = Decimal(200)
    store.save(state)
    code, out = run(env, "kill-switch", "status")
    assert code == 0 and "ACTIVE" in out
    code, out = run(env, "kill-switch", "reset", "--equity", "140", "--yes")
    assert code == 0 and "200 -> 140" in out
    after = store.load()
    assert not after.kill_switch.active and after.high_water_mark == Decimal(140)
    code, out = run(env, "kill-switch", "reset", "--equity", "140", "--yes")
    assert "not active" in out


def test_money_status_and_confirm(env: dict[str, str]) -> None:
    code, out = run(env, "money", "status")
    assert code == 0 and "withdraw at           $200.00" in out
    code, out = run(env, "money", "confirm-withdrawal", "--equity", "105", "--yes")
    assert code == 0 and "Withdrawal confirmed" in out
    state = StateStore(Path(env["STOA_STATE_DIR"])).load()
    assert state.principal_withdrawn and state.high_water_mark == Decimal(105)
    code, out = run(env, "money", "confirm-withdrawal", "--equity", "105", "--yes")
    assert "already confirmed" in out


def test_decide_refuses_without_keys_when_forced(env: dict[str, str]) -> None:
    code, out = run(env, "decide", "--force")
    assert code == 2
    assert "ALPACA_PAPER_KEY_ID" in out
