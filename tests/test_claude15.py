"""Trial 15's live Claude reader: the news behind the top 20, inside the one monthly cap
shared with every other Claude call, recorded beside the picks and never changing them."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd

from stoa import shadow
from stoa.newsbrain import claude15
from stoa.state import State
from tests.conftest import make_settings
from tests.test_analyst import FakeClient

NOW = datetime(2026, 10, 30, 13, 5, tzinfo=UTC)


def archive() -> pd.DataFrame:
    t = pd.Timestamp("2026-10-30 12:00", tz="UTC")
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "symbols": [["AAPL"], ["MSFT", "AAPL"], ["KO"]],
            "headline": ["Apple beats", "Microsoft and Apple deal", "Coke news"],
            "known_at": [t, t, pd.Timestamp("2026-10-28 12:00", tz="UTC")],
        }
    )


def test_the_news_is_the_days_behind_the_top_symbols() -> None:
    since = datetime(2026, 10, 29, 13, 0, tzinfo=UTC)
    got = claude15.news_for(archive(), ["AAPL", "KO"], since)
    assert sorted(got["symbol"]) == ["AAPL", "AAPL"]  # KO's article is too old


def test_it_reads_inside_the_cap_and_counts_in_the_shared_ledger(tmp_path: Path) -> None:
    s = make_settings(
        state__shared_dir=str(tmp_path / "shared"), analyst__cache_dir=str(tmp_path / "cache")
    )
    news = claude15.news_for(archive(), ["AAPL"], datetime(2026, 10, 29, tzinfo=UTC))
    got = claude15.read(s, None, State(), news, lambda: NOW, FakeClient())
    assert got.status == "read" and got.by_symbol["AAPL"]["items"] == 2
    spent = claude15.spend(s)["2026-10"]
    assert spent > 0
    # The shadow job and the dashboard see it in the one cap.
    view = shadow.merged_view(State(), shadow.load_ledger(s), claude15.spend(s))
    assert view.llm_spend_by_month["2026-10"] == spent


def test_at_the_cap_it_skips_and_spends_nothing(tmp_path: Path) -> None:
    s = make_settings(
        state__shared_dir=str(tmp_path / "shared"), analyst__cache_dir=str(tmp_path / "cache")
    )
    full = State(llm_spend_by_month={"2026-10": Decimal("25.0")})
    news = claude15.news_for(archive(), ["AAPL"], datetime(2026, 10, 29, tzinfo=UTC))
    client = FakeClient()
    got = claude15.read(s, None, full, news, lambda: NOW, client)
    assert got.status == "paused" and client.messages.calls == []
    assert claude15.read(s, None, State(), news, lambda: NOW).status == "no_key"
