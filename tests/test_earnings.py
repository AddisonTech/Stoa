"""Earnings rules, SEC parsing, news parsing and the forward sources, all offline."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

from stoa import clock
from stoa.config import EarningsConfig
from stoa.data import news as news_mod
from stoa.data import sec
from stoa.data.external import FinnhubClient
from stoa.earnings import (
    EarningsCalendar,
    entry_block,
    exit_signals,
    last_safe_session,
    volume_proxy,
    with_known_lead,
)

CFG = EarningsConfig()


def at(day: str, hh: int, mm: int) -> datetime:
    from datetime import time

    return datetime.combine(date.fromisoformat(day), time(hh, mm), tzinfo=clock.NY).astimezone(UTC)


def cal(*rows: tuple[str, datetime, datetime]) -> EarningsCalendar:
    df = pd.DataFrame(
        [{"symbol": s, "event_time": e, "known_at": k, "source": "test"} for s, e, k in rows]
    )
    return EarningsCalendar(df)


def test_last_safe_session_before_open_after_close_and_weekend() -> None:
    # Before the open on Thursday: the last safe open is Wednesday.
    assert last_safe_session(at("2024-07-25", 7, 0), 15) == date(2024, 7, 24)
    # After the close on Thursday: Thursday's open is still safe.
    assert last_safe_session(at("2024-07-25", 16, 5), 15) == date(2024, 7, 25)
    # During the day after Stoa trades (10:30): Thursday's open was safe.
    assert last_safe_session(at("2024-07-25", 10, 30), 15) == date(2024, 7, 25)
    # 9:40 is before Stoa's 9:45 execution: Thursday is not safe.
    assert last_safe_session(at("2024-07-25", 9, 40), 15) == date(2024, 7, 24)
    # Saturday: the last safe session is Friday.
    assert last_safe_session(at("2024-07-27", 8, 0), 15) == date(2024, 7, 26)


def test_exit_before_confirmed_earnings() -> None:
    known = at("2024-07-01", 12, 0)
    c = cal(("AAPL", at("2024-07-25", 16, 5), known), ("MSFT", at("2024-07-26", 7, 0), known))
    # AAPL reports Thu after the close: Thu's open is the last safe exit.
    # MSFT reports Fri before the open: Thu's open is also its last safe exit.
    # Both must be sold at Thu's open, which is decided at Wed's close.
    sigs = exit_signals(c, ["AAPL", "MSFT"], date(2024, 7, 24), CFG)
    assert [s.symbol for s in sigs] == ["AAPL", "MSFT"]
    assert all(s.reason == "earnings_exit" and "earnings" in s.detail for s in sigs)
    # A day earlier, neither exit is due yet.
    assert exit_signals(c, ["AAPL", "MSFT"], date(2024, 7, 23), CFG) == []
    off = EarningsConfig(exit_before=False)
    assert exit_signals(c, ["AAPL"], date(2024, 7, 24), off) == []


def test_exit_not_triggered_when_report_is_further_out_or_unknown() -> None:
    c = cal(("AAPL", at("2024-07-25", 16, 5), at("2024-07-01", 12, 0)))
    assert exit_signals(c, ["AAPL"], date(2024, 7, 22), CFG) == []
    # Date not known yet at decision time: no exit (the no-entry rule handles new buys).
    late = cal(("AAPL", at("2024-07-25", 16, 5), at("2024-07-24", 21, 0)))
    assert exit_signals(late, ["AAPL"], date(2024, 7, 24), CFG) == []
    assert exit_signals(c, ["SPY"], date(2024, 7, 24), CFG) == []


def test_no_entry_with_earnings_inside_holding_window() -> None:
    known = at("2024-06-01", 12, 0)
    c = cal(
        ("AAPL", at("2024-04-30", 16, 5), at("2024-04-01", 12, 0)),
        ("AAPL", at("2024-07-25", 16, 5), known),
    )
    # Decide 2024-07-12 (Fri): entry Mon 15th; last safe session Thu 25th is 8
    # sessions after entry: inside the 10 session window.
    b = entry_block(c, "AAPL", date(2024, 7, 12), CFG)
    assert b is not None and b[0] == "earnings" and "holding window" in b[1]
    # Decide 2024-07-02: the release is about 16 sessions out: allowed.
    assert entry_block(c, "AAPL", date(2024, 7, 2), CFG) is None


def test_unknown_date_rule_is_conservative() -> None:
    # Last report 2024-04-30, nothing upcoming known.
    c = cal(("AAPL", at("2024-04-30", 16, 5), at("2024-04-01", 12, 0)))
    assert entry_block(c, "AAPL", date(2024, 6, 3), CFG) is None  # 34 days: fine
    b = entry_block(c, "AAPL", date(2024, 7, 22), CFG)  # 83 days: stale
    assert b is not None and "no confirmed date" in b[1]
    relaxed = EarningsConfig(block_when_unknown=False)
    assert entry_block(c, "AAPL", date(2024, 7, 22), relaxed) is None
    # No history at all.
    assert entry_block(c, "ZZZ", date(2024, 7, 22), CFG) is not None
    # ETFs never report.
    assert entry_block(c, "SPY", date(2024, 7, 22), CFG) is None


def test_with_known_lead_sets_known_at_before_the_release() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "A",
                "event_time": at("2024-07-25", 16, 5),
                "known_at": at("2024-07-25", 16, 5),
                "source": "sec_8k_2.02",
            },
            {
                "symbol": "A",
                "event_time": at("2024-10-24", 16, 5),
                "known_at": at("2024-10-01", 9, 0),
                "source": "finnhub",
            },
        ]
    )
    out = with_known_lead(df, 10)
    k0 = out["known_at"].iloc[0].to_pydatetime()
    assert k0 < at("2024-07-25", 0, 0)
    assert clock.sessions_after(clock.to_ny(k0).date(), date(2024, 7, 25)) == 11
    assert out["known_at"].iloc[1] == pd.Timestamp(at("2024-10-01", 9, 0))


def test_volume_proxy_finds_spikes() -> None:
    days = clock.sessions_between(date(2024, 1, 2), date(2024, 6, 28))
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    vol = pd.Series(1e6, index=idx)
    vol.iloc[[30, 32, 95]] = 5e6
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": vol})
    out = volume_proxy({"X": df}, {"X"}, multiple=2.5)
    assert len(out) == 2 and (out["source"] == "proxy").all()


# ---- SEC parsing ---------------------------------------------------------------------------


SUBMISSIONS = {
    "filings": {
        "recent": {
            "accessionNumber": ["0001-24-000010", "0001-24-000009", "0001-24-000005"],
            "form": ["8-K", "10-Q", "8-K"],
            "filingDate": ["2024-08-01", "2024-08-02", "2024-05-02"],
            "acceptanceDateTime": [
                "2024-08-01T16:30:12.000Z",
                "2024-08-02T18:01:00.000Z",
                "2024-05-02T16:31:00.000Z",
            ],
            "items": ["2.02,9.01", "", "5.02"],
        },
        "files": [],
    }
}


def fact(
    end: str, val: float, accn: str, filed: str, form: str = "10-Q", start: str | None = None
) -> dict[str, object]:
    out: dict[str, object] = {"end": end, "val": val, "accn": accn, "filed": filed, "form": form}
    if start:
        out["start"] = start
    return out


COMPANYFACTS = {
    "facts": {
        "us-gaap": {
            "GrossProfit": {
                "units": {
                    "USD": [
                        fact(
                            "2024-06-29", 40.0, "0001-24-000009", "2024-08-02", start="2023-07-02"
                        ),
                        fact(
                            "2024-06-29",
                            99.0,
                            "0001-25-000001",
                            "2025-02-01",
                            "10-K/A",
                            start="2023-07-02",
                        ),
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        fact(
                            "2024-06-29", 20.0, "0001-24-000009", "2024-08-02", start="2023-07-02"
                        ),
                    ]
                }
            },
            "NetCashProvidedByUsedInOperatingActivities": {
                "units": {
                    "USD": [
                        fact(
                            "2024-06-29", 25.0, "0001-24-000009", "2024-08-02", start="2023-07-02"
                        ),
                    ]
                }
            },
            "Assets": {
                "units": {"USD": [fact("2024-06-29", 200.0, "0001-24-000009", "2024-08-02")]}
            },
            "Liabilities": {
                "units": {"USD": [fact("2024-06-29", 120.0, "0001-24-000009", "2024-08-02")]}
            },
            "StockholdersEquity": {
                "units": {"USD": [fact("2024-06-29", 80.0, "0001-24-000009", "2024-08-02")]}
            },
        }
    }
}


def test_earnings_releases_from_8k_item_202_with_acceptance_time() -> None:
    filings = sec.filings_table(SUBMISSIONS["filings"]["recent"])
    rel = sec.earnings_releases("AAPL", filings)
    assert len(rel) == 1
    # Acceptance 16:30 Eastern on 2024-08-01 is 20:30 UTC (EDT).
    assert rel["event_time"].iloc[0] == datetime(2024, 8, 1, 20, 30, 12, tzinfo=UTC)
    assert rel["known_at"].iloc[0] == rel["event_time"].iloc[0]


def test_quality_features_known_at_is_filing_acceptance_and_ignores_restatements() -> None:
    filings = sec.filings_table(SUBMISSIONS["filings"]["recent"])
    q = sec.quality_features("AAPL", COMPANYFACTS, filings)
    row = q.iloc[0]
    assert row["q_gross_profitability"] == pytest.approx(40 / 200)  # not the 99 restatement
    assert row["q_roe"] == pytest.approx(20 / 80)
    assert row["q_accruals"] == pytest.approx((20 - 25) / 200)
    assert row["q_leverage"] == pytest.approx(120 / 200)
    assert row["known_at"] == datetime(2024, 8, 2, 22, 1, tzinfo=UTC)


def test_sec_client_caches_throttles_and_sends_user_agent(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path.endswith("company_tickers.json"):
            return httpx.Response(200, json={"0": {"cik_str": 320193, "ticker": "AAPL"}})
        if "submissions" in req.url.path:
            return httpx.Response(200, json=SUBMISSIONS)
        if "companyfacts" in req.url.path:
            return httpx.Response(200, json=COMPANYFACTS)
        return httpx.Response(404)

    waits: list[float] = []
    t = [0.0]

    def fake_sleep(sec_: float) -> None:
        waits.append(sec_)
        t[0] += sec_

    limiter = sec.RateLimiter(10, clock=lambda: t[0], sleep=fake_sleep)
    client = httpx.Client(
        transport=httpx.MockTransport(h), headers={"User-Agent": "Stoa test stalor54@yahoo.com"}
    )
    sc = sec.SECClient(client, tmp_path, limiter, sleep=lambda _s: None)
    q, e, problems = sec.fetch_fundamentals(sc, ["AAPL", "SPY"])
    assert len(q) == 1 and len(e) == 1 and "SPY" in problems
    assert all("stalor54@yahoo.com" in r.headers["User-Agent"] for r in seen)
    assert waits and all(w == pytest.approx(0.1) for w in waits)
    n = len(seen)
    sec.fetch_fundamentals(sc, ["AAPL"])
    assert len(seen) == n  # second run served from the disk cache
    with pytest.raises(ValueError):
        sec.make_sec_client("no email here", None, 8)


# ---- news ------------------------------------------------------------------------------------


def test_news_parse_known_at_is_latest_edit_and_html_is_stripped() -> None:
    rows = [
        {
            "id": 1,
            "headline": "Acme <b>beats</b> &amp; raises",
            "summary": "s",
            "content": "<p>Body</p>",
            "symbols": ["ACME", "OTHER"],
            "source": "benzinga",
            "created_at": "2024-07-25T20:05:00Z",
            "updated_at": "2024-07-26T13:00:00Z",
        }
    ]
    out = news_mod.parse_articles(rows, {"ACME"})
    assert len(out) == 1
    r = out[0]
    assert r["headline"] == "Acme beats & raises" and r["content"] == "Body"
    assert r["known_at"] == datetime(2024, 7, 26, 13, 0, tzinfo=UTC)


def test_news_client_paginates(tmp_path: Path) -> None:
    pages = {
        None: {
            "news": [
                {
                    "id": 1,
                    "headline": "a",
                    "symbols": ["X"],
                    "created_at": "2024-01-02T15:00:00Z",
                    "updated_at": "2024-01-02T15:00:00Z",
                }
            ],
            "next_page_token": "p2",
        },
        "p2": {
            "news": [
                {
                    "id": 2,
                    "headline": "b",
                    "symbols": ["X"],
                    "created_at": "2024-01-03T15:00:00Z",
                    "updated_at": "2024-01-03T15:00:00Z",
                }
            ],
            "next_page_token": None,
        },
    }

    def h(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1beta1/news"
        return httpx.Response(200, json=pages[req.url.params.get("page_token")])

    c = httpx.Client(base_url="https://data.alpaca.markets", transport=httpx.MockTransport(h))
    df = news_mod.AlpacaNewsClient(c, sleep=lambda _s: None).fetch(
        ["X"], date(2024, 1, 1), date(2024, 1, 5)
    )
    assert list(df["headline"]) == ["a", "b"]
    store = news_mod.NewsStore(tmp_path / "news.parquet")
    store.merge(df)
    assert len(store.merge(df)) == 2  # merging twice never duplicates


# ---- forward sources ---------------------------------------------------------------------------


def test_finnhub_quote_and_calendar_keep_token_out_of_the_url() -> None:
    seen: list[httpx.Request] = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path.endswith("/quote"):
            return httpx.Response(200, json={"c": 101.5, "t": 1721923200})
        return httpx.Response(
            200,
            json={"earningsCalendar": [{"date": "2024-07-25", "hour": "amc", "symbol": "AAPL"}]},
        )

    now = datetime(2024, 7, 25, 16, 10, tzinfo=UTC)
    c = httpx.Client(base_url="https://finnhub.io/api/v1", transport=httpx.MockTransport(h))
    fh = FinnhubClient("tok-secret-123", c, now=lambda: now)
    from decimal import Decimal

    assert fh.price("AAPL") == Decimal("101.5")
    up = fh.upcoming(["AAPL"], date(2024, 7, 20), date(2024, 8, 10))
    assert up["event_time"].iloc[0] == pd.Timestamp(at("2024-07-25", 16, 5))
    assert all("tok-secret-123" not in str(r.url) for r in seen)
    assert all(r.headers["X-Finnhub-Token"] == "tok-secret-123" for r in seen)
    assert "tok-secret-123" not in repr(fh)


def test_finnhub_stale_or_bad_quote_is_unavailable() -> None:
    now = datetime(2024, 7, 25, 16, 10, tzinfo=UTC)

    def stale(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"c": 101.5, "t": 1721800000})

    c = httpx.Client(base_url="https://x", transport=httpx.MockTransport(stale))
    assert FinnhubClient("t", c, now=lambda: now).price("AAPL") is None
    c2 = httpx.Client(
        base_url="https://x", transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    assert FinnhubClient("t", c2, now=lambda: now).price("AAPL") is None
    c3 = httpx.Client(
        base_url="https://x",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=json.dumps({"c": 0}))),
    )
    assert FinnhubClient("t", c3, now=lambda: now).price("AAPL") is None


def test_unknown_date_exit_after_stale_days() -> None:
    c = cal(("AAPL", at("2024-04-30", 16, 5), at("2024-04-01", 12, 0)))
    assert exit_signals(c, ["AAPL"], date(2024, 7, 15), CFG) == []  # 76 days
    sigs = exit_signals(c, ["AAPL"], date(2024, 7, 22), CFG)  # 83 days
    assert [s.reason for s in sigs] == ["earnings_unknown"]
    assert (
        exit_signals(c, ["AAPL"], date(2024, 7, 22), EarningsConfig(block_when_unknown=False)) == []
    )
