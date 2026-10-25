from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stoa.newsbrain import alpaca_pull as ap
from stoa.newsbrain import sec_pull as sp
from stoa.newsbrain.pace import Pace, in_live_window
from stoa.newsbrain.score import first_section


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def test_pace_spaces_calls_and_reads_the_limits_file(tmp_path: Path) -> None:
    limits = tmp_path / "limits.json"
    limits.write_text('{"news": 120}', encoding="utf-8")
    c = Clock()
    p = Pace("news", 10, limits, respect_live=False, sleep=c.sleep, monotonic=c.monotonic)
    for _ in range(3):
        p.wait()
    assert p.rate == 120 and c.slept == [0.5, 0.5]


def test_live_windows_are_weekday_new_york_times() -> None:
    monday_910 = datetime(2026, 10, 26, 13, 10, tzinfo=UTC)  # 9:10 am EDT
    monday_noon = datetime(2026, 10, 26, 16, 0, tzinfo=UTC)
    saturday_910 = datetime(2026, 10, 24, 13, 10, tzinfo=UTC)
    assert in_live_window(monday_910)
    assert not in_live_window(monday_noon) and not in_live_window(saturday_910)


def test_pace_waits_out_a_live_window(tmp_path: Path) -> None:
    c = Clock()
    times = iter(
        [datetime(2026, 10, 26, 13, 10, tzinfo=UTC)] * 2
        + [datetime(2026, 10, 26, 16, 0, tzinfo=UTC)]
    )
    p = Pace(
        "x",
        60,
        tmp_path / "none.json",
        now=lambda: next(times),
        sleep=c.sleep,
        monotonic=c.monotonic,
    )
    p.wait()
    assert c.slept[:2] == [60, 60]


def test_news_rows_keep_universe_articles_with_all_their_symbols() -> None:
    articles = [
        {
            "id": 1,
            "created_at": "2016-03-01T23:08:01Z",
            "updated_at": None,
            "symbols": ["keQu"],
            "headline": "<b>Kewaunee</b> up",
            "summary": "Strong &amp; steady",
        },
        {
            "id": 2,
            "created_at": "2016-03-01T23:09:01Z",
            "symbols": ["ZZZZ"],
            "headline": "x",
            "summary": "",
        },
    ]
    rows = ap.news_rows(articles, {"KEQU"})
    assert len(rows) == 1
    assert rows[0]["symbols"] == ["KEQU"] and rows[0]["headline"] == "Kewaunee up"
    assert (
        rows[0]["summary"] == "Strong & steady" and rows[0]["updated_at"] == "2016-03-01T23:08:01Z"
    )


def test_minute_groups_pack_liquid_names_tightly() -> None:
    rows = pd.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(400)],
            "days": 21,
            "dollars": np.arange(400, 0, -1) * 1e6,
        }
    )
    groups = ap.groups_for(rows)
    assert groups[0][0] == "S0" and len(groups[0]) <= 7  # 900 bars a day x 21 days
    assert sum(len(g) for g in groups) == 400
    assert all(len(g) <= ap.GROUP_MAX for g in groups)


def test_minute_frame_and_months() -> None:
    pages = [
        {
            "AAPL": [
                {
                    "t": "2025-03-03T14:30:00Z",
                    "o": 1,
                    "h": 2,
                    "l": 0.5,
                    "c": 1.5,
                    "v": 10,
                    "n": 2,
                    "vw": 1.2,
                }
            ]
        }
    ]
    df = ap.frame_of(pages)
    assert list(df["symbol"]) == ["AAPL"] and df["t"].iloc[0] == pd.Timestamp(
        "2025-03-03 14:30", tz="UTC"
    )
    assert ap.months(date(2025, 11, 15), date(2026, 2, 1)) == [
        date(2025, 11, 1),
        date(2025, 12, 1),
        date(2026, 1, 1),
        date(2026, 2, 1),
    ]
    assert ap.month_end(date(2024, 2, 1)) == date(2024, 2, 29)


def test_edgar_text_exhibits_and_customer_paragraphs() -> None:
    raw = (
        "<html><style>p{}</style><p>Acme reports record quarter&#8217;s revenue</p>"
        "<div>Walmart accounted for 18% of our net sales in fiscal 2024, and our ten "
        "largest customers together accounted for 45% of revenue.</div>"
        "<p>We have many suppliers.</p></html>"
    )
    text = sp.html_text(raw)
    assert "p{}" not in text and "quarter" + chr(0x2019) + "s" in text
    paras = sp.customer_paragraphs(text)
    assert paras.startswith("Walmart accounted for 18%") and "many suppliers" not in paras
    assert sp.is_ex99("a8-kex991q32026.htm") and sp.is_ex99("dex991.htm")
    assert not sp.is_ex99("aapl-20260730.htm") and not sp.is_ex99("ex99.jpg")
    assert sp.decode("caf\xe9".encode("cp1252")) == "caf\xe9"


def test_filings_keep_2015_on_with_primary_documents() -> None:
    recent = {
        "accessionNumber": ["0001-26-000001", "0001-14-000001"],
        "form": ["8-K", "8-K"],
        "filingDate": ["2026-07-30", "2014-12-31"],
        "acceptanceDateTime": ["2026-07-30T16:30:21.000Z", "2014-12-31T09:00:00.000Z"],
        "items": ["2.02,9.01", "8.01"],
        "primaryDocument": ["x.htm", "y.htm"],
    }
    fl = sp.filings(recent)
    assert list(fl["accn"]) == ["0001-26-000001"] and fl["doc"].iloc[0] == "x.htm"
    assert fl["accepted"].iloc[0] == datetime(2026, 7, 30, 20, 30, 21, tzinfo=UTC)


def test_first_section_skips_cover_lines() -> None:
    text = (
        "EX-99.1\n2\nExhibit 99.1\n"
        + "Apple reports third quarter results with record revenue. " * 3
    )
    out = first_section(text)
    assert out.startswith("Apple reports") and "EX-99.1" not in out
