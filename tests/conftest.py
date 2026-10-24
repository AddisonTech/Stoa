from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from stoa.config import Settings, load_settings
from stoa.data.base import Bars
from stoa.data.synthetic import synthetic_bars

ROOT = Path(__file__).resolve().parents[1]
SMALL_WATCHLIST = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "XOM", "KO"]


def make_settings(**overrides: str) -> Settings:
    env = {f"STOA__{k.upper()}": v for k, v in overrides.items()}
    return load_settings(ROOT / "config" / "default.yaml", env=env)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return make_settings()


@pytest.fixture(scope="session")
def small_settings() -> Settings:
    return make_settings(watchlist=",".join(SMALL_WATCHLIST))


@pytest.fixture(scope="session")
def bars() -> Bars:
    return synthetic_bars(SMALL_WATCHLIST, date(2016, 1, 4), date(2021, 12, 31), seed=11)


@pytest.fixture(scope="session")
def full_bars(settings: Settings) -> Bars:
    return synthetic_bars(list(settings.watchlist), date(2016, 1, 4), date(2020, 12, 31), seed=5)
