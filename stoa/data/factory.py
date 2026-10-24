"""Builds the configured bar provider."""

from __future__ import annotations

import httpx

from stoa.config import Secrets, Settings
from stoa.data.alpaca import AlpacaDataProvider
from stoa.data.base import BarProvider, DataError
from stoa.data.csv import CSVProvider
from stoa.data.synthetic import SyntheticProvider
from stoa.data.yahoo import YahooProvider


def alpaca_data_client(settings: Settings, secrets: Secrets) -> httpx.Client:
    """Market data client. Uses paper keys when present, else live keys (read-only use)."""
    if secrets.has_paper_keys():
        assert secrets.alpaca_paper_key_id and secrets.alpaca_paper_secret
        key, secret = secrets.alpaca_paper_key_id, secrets.alpaca_paper_secret
    elif secrets.has_live_keys():
        assert secrets.alpaca_live_key_id and secrets.alpaca_live_secret
        key, secret = secrets.alpaca_live_key_id, secrets.alpaca_live_secret
    else:
        raise DataError("Alpaca data needs ALPACA_PAPER_KEY_ID/ALPACA_PAPER_SECRET")
    return httpx.Client(
        base_url=settings.alpaca.data_url,
        headers={
            "APCA-API-KEY-ID": key.get_secret_value(),
            "APCA-API-SECRET-KEY": secret.get_secret_value(),
        },
        timeout=settings.data.timeout_seconds,
    )


def make_provider(settings: Settings, secrets: Secrets) -> BarProvider:
    kind = settings.data.provider
    if kind == "alpaca":
        return AlpacaDataProvider(
            alpaca_data_client(settings, secrets),
            feed=settings.data.feed,
            fallback_feed=settings.data.fallback_feed or None,
            max_retries=settings.data.max_retries,
        )
    if kind == "yahoo":
        return YahooProvider()
    if kind == "csv":
        return CSVProvider(settings.data.csv_dir)
    return SyntheticProvider(benchmark=settings.benchmark)
