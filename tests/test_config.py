from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from stoa.config import Secrets, Settings
from tests.conftest import ROOT, make_settings


def test_default_config_loads(settings: Settings) -> None:
    assert settings.mode == "paper"
    assert settings.benchmark == "SPY"
    assert "SPY" in settings.watchlist and "QQQ" in settings.watchlist
    assert len(settings.watchlist) == 29
    assert settings.risk.max_position_pct == Decimal("0.20")
    assert settings.risk.max_positions == 5
    assert settings.risk.kill_switch_drawdown_pct == Decimal("0.25")
    assert settings.llm.model == "claude-haiku-5-5"
    assert settings.backtest.starting_equity == Decimal("100")


def test_env_override_nested_and_shortcuts() -> None:
    s = make_settings(risk__max_positions="4", mode="signal")
    assert s.risk.max_positions == 4
    assert s.mode == "signal"


def test_state_dir_shortcut() -> None:
    from stoa.config import load_settings

    s = load_settings(ROOT / "config" / "default.yaml", env={"STOA_STATE_DIR": "/tmp/x"})
    assert s.state.dir == Path("/tmp/x")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("risk__max_gross_exposure_pct", "1.5"),
        ("risk__max_position_pct", "0"),
        ("risk__min_order_usd", "0.5"),
        ("risk__kill_switch_drawdown_pct", "1.2"),
        ("risk__max_orders_per_day", "2"),
        ("mode", "yolo"),
        ("benchmark", "TSLA"),
        ("money__milestone_multiple", "1"),
    ],
)
def test_invalid_values_rejected(key: str, value: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{key: value})


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(risk__max_leverage="2")


def test_window_may_not_touch_holdout() -> None:
    with pytest.raises(ValidationError):
        make_settings(backtest__windows__tune_a__end="2026-07-15")


def test_secrets_from_env_hide_values() -> None:
    s = Secrets.from_env(
        {
            "ALPACA_PAPER_KEY_ID": "PKXYZ123",
            "ALPACA_PAPER_SECRET": "s3cr3tvalue",
            "STOA_NTFY_TOPIC": " ",
        }
    )
    assert s.has_paper_keys()
    assert not s.has_live_keys()
    assert s.ntfy_topic is None
    assert "s3cr3tvalue" not in repr(s) and "PKXYZ123" not in repr(s)
    assert "s3cr3tvalue" not in str(s.model_dump())


def test_live_confirm_phrase_exact() -> None:
    assert Secrets.from_env({"STOA_LIVE_CONFIRM": "I_UNDERSTAND"}).live_confirmed()
    assert not Secrets.from_env({"STOA_LIVE_CONFIRM": "yes"}).live_confirmed()
    assert not Secrets.from_env({}).live_confirmed()


def test_strategy_hash_tracks_risk_but_not_mode() -> None:
    base = make_settings()
    assert base.strategy_hash() == make_settings(mode="signal").strategy_hash()
    assert base.strategy_hash() != make_settings(risk__max_positions="4").strategy_hash()
