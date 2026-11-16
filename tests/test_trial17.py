"""Trial 17: the look-ahead check (known answers), the text rules, the scores and the ticker
map."""

from __future__ import annotations

from collections import Counter
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stoa import envfile
from stoa.filings import known, parse, scores, tickers
from stoa.filings import trial17 as t17

# ---- the look-ahead check -------------------------------------------------------------------


def test_known_answer_is_honest() -> None:
    res = known.known_answer()
    failed = {k: v for k, v in res.items() if isinstance(v, dict) and not v["passed"]}
    assert res["verdict"] == "honest", failed


def test_known_answer_catches_a_date_only_filing_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """The usual bug: reading a filing from its acceptance date's open, not after its time.
    B (accepted 9:31 am on the rebalance day) then counts that morning, and the check must
    say so."""

    def by_date(accepted: object) -> list[date]:
        return [pd.Timestamp(t).tz_convert("America/New_York").date() for t in accepted]  # type: ignore[attr-defined]

    monkeypatch.setattr(t17, "counts_from", by_date)
    res = known.known_answer()
    assert res["verdict"] == "bug"
    assert not res["no_early_filing"]["passed"]


def test_counts_from_after_the_close_and_on_weekends() -> None:
    ny = known._ny
    got = t17.counts_from(
        [
            ny("2024-03-15 16:05"),  # Friday after the close: Monday
            ny("2024-03-16 11:00"),  # Saturday: Monday
            ny("2024-03-18 09:30"),  # at the open exactly: the next session
            ny("2024-03-18 09:29"),  # before the open: that session
            ny("2024-03-28 18:00"),  # Thursday before Good Friday: the next Monday
        ]
    )
    assert got == [
        date(2024, 3, 18),
        date(2024, 3, 18),
        date(2024, 3, 19),
        date(2024, 3, 18),
        date(2024, 4, 1),
    ]


def test_rebalance_days_are_month_starts() -> None:
    days = t17.rebalance_days(date(2024, 1, 1), date(2024, 4, 30))
    assert days == [date(2024, 1, 2), date(2024, 2, 1), date(2024, 3, 1), date(2024, 4, 1)]


def test_quintiles_split_evenly_lowest_first() -> None:
    filings, prices = known.fixture()
    df = t17.eligible(filings, prices, date(2020, 4, 1))
    assert list(df["quintile"]) == [1, 1, 2, 3, 3, 4, 5]
    assert list(df["cos_full"]) == sorted(df["cos_full"])


def test_spec_hash_is_stable_and_names_the_window() -> None:
    a = t17.spec_a()
    assert t17.spec_hash(a) == t17.spec_hash(t17.spec_a())
    assert "2016-01-04" in a["window"] and "2026-11-13" in a["window"]
    assert "Sharpe strictly higher than the equal weight" in a["pass"]


# ---- text --------------------------------------------------------------------------------


TENK = """<html><body>
<ix:header><ix:hidden>secret hidden facts</ix:hidden></ix:header>
<div style="display:none">hidden div</div>
<p>TABLE OF CONTENTS</p>
<table><tr><td>Item 1A.</td><td>Risk Factors</td><td>12</td></tr>
<tr><td>Item 3.</td><td>Legal Proceedings</td><td>30</td></tr>
<tr><td>Item 7.</td><td>Management's Discussion and Analysis</td><td>40</td></tr></table>
<p>Item 1. Business</p><p>We make widgets.</p>
<p>Item 1A. Risk Factors</p>
<p>Business interruptions could hurt us. This line starts with business and is a risk.</p>
<p>Our suppliers may fail.</p>
<p>Item 1B. Unresolved Staff Comments</p><p>None.</p>
<p>Item 3. Legal Proceedings</p><p>See Note 12.</p>
<p>Item 4. Mine Safety Disclosures</p><p>Not applicable.</p>
<p>Item 7.</p>
<p>Management&#8217;s Discussion and Analysis of Financial Condition and Results of
Operations</p>
<p>Sales rose. Costs fell.</p>
<p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p>
<p>Rates.</p>
<p>Note 12. Contingencies</p><p>Legal Proceedings</p>
<p>A long note about many lawsuits that goes on.</p>
<ix:nonNumeric name="dei:TradingSymbol" contextRef="c">wdgt</ix:nonNumeric>
</body></html>"""


def test_sections_found_by_heading_not_by_table_of_contents() -> None:
    p = parse.parse_doc(TENK, "10-K")
    assert "secret" not in p["full"] and "hidden div" not in p["full"]
    assert p["rf"].startswith("Business interruptions") and p["rf"].endswith("may fail.")
    assert p["legal"] == "See Note 12."  # the item, not the longer note heading
    assert p["mdna"] == "Sales rose. Costs fell."
    assert p["symbol_tag"] == "WDGT"


def test_tenq_sections() -> None:
    doc = """<p>PART I</p><p>Item 1. Financial Statements</p><p>Numbers.</p>
<p>Item 2. Management's Discussion and Analysis of Financial Condition</p>
<p>Quarter went well.</p>
<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p><p>Rates.</p>
<p>PART II. OTHER INFORMATION</p><p>Item 1. Legal Proceedings</p>
<p>A new suit was filed.</p>
<p>Item 1A. Risk Factors</p><p>No material changes.</p>
<p>Item 2. Unregistered Sales</p><p>None.</p>"""
    p = parse.parse_doc(doc, "10-Q")
    assert p["mdna"] == "Quarter went well."
    assert p["legal"] == "A new suit was filed."
    assert p["rf"] == "No material changes."


# ---- scores ---------------------------------------------------------------------------------


def test_similarity_measures_known_values() -> None:
    a = scores.terms("the cat sat. The cat 42 ran")
    b = scores.terms("the dog sat")
    assert a == Counter({"the": 2, "cat": 2, "sat": 1, "ran": 1})
    dot = 2 * 1 + 1 * 1  # the, sat
    assert scores.cosine(a, b) == pytest.approx(dot / (np.sqrt(4 + 4 + 1 + 1) * np.sqrt(3)))
    assert scores.jaccard(a, b) == pytest.approx(2 / 5)
    assert scores.length_change(a, b) == pytest.approx(6 / 3 - 1)
    assert np.isnan(scores.cosine(a, Counter()))
    assert scores.cosine(a, a) == pytest.approx(1.0)


def test_pairs_same_form_a_year_apart() -> None:
    df = pd.DataFrame(
        {
            "form": ["10-K", "10-Q", "10-Q", "10-K", "10-Q", "10-Q"],
            "report_date": [
                "2020-12-31",
                "2021-03-31",
                "2021-06-30",
                "2021-12-31",
                "2022-03-31",
                "",
            ],
            "filed": [
                "2021-02-20",
                "2021-05-01",
                "2021-08-01",
                "2022-02-20",
                "2022-05-01",
                "2022-08-02",
            ],
        }
    )
    assert scores.pair(df) == [(0, None), (1, None), (2, None), (3, 0), (4, 1), (5, 2)]


# ---- tickers ---------------------------------------------------------------------------------


class FakeBars(tickers.Bars):
    def __init__(self, spans: dict[str, tuple[str, str, float]]) -> None:
        super().__init__()
        for s, (a, b, dv) in spans.items():
            days = pd.bdate_range(a, b).to_numpy(dtype="datetime64[D]")
            self._cache[s] = (days, np.full(len(days), dv))


def test_ticker_map_point_in_time() -> None:
    bars = FakeBars(
        {
            "OLD": ("2016-01-01", "2018-06-30", 1e7),
            "NEW": ("2016-01-01", "2026-01-01", 5e7),
            "CLSA": ("2016-01-01", "2026-01-01", 9e7),
            "CLSB": ("2016-01-01", "2026-01-01", 1e6),
            "GONE": ("2016-01-01", "2019-03-29", 2e7),
        }
    )
    book = {1: ["NEW"], 2: ["NEW"], 3: ["CLSA", "CLSB"], 4: ["GONE"]}
    f = pd.DataFrame(
        {
            "cik": [1, 1, 2, 3, 4],
            "accn": ["a", "b", "c", "d", "e"],
            "accepted": pd.to_datetime(
                ["2017-03-01", "2024-03-01", "2017-02-01", "2020-03-01", "2019-02-01"], utc=True
            ),
            "symbol_tag": ["", "NEW", "", "", ""],
        }
    )
    pur = pd.DataFrame({"symbol": ["NEW"], "issuer_cik": [2], "filing_date": ["2017-01-15"]})
    out = tickers.map_filings(f, book, pur, bars)
    got = dict(zip(out["accn"], out["symbol"], strict=True))
    # filer 1's 2017 filing: NEW was named for filer 2 then, so it is not read back
    assert got["a"] == ""
    assert got["b"] == "NEW"
    assert got["c"] == "NEW"
    assert got["d"] == "CLSA"  # the most traded share class
    assert got["e"] == "GONE"  # delisted later, mapped while it traded


def test_envfile_parse() -> None:
    got = envfile.parse("# c\nA=1\nexport B = 'two'\n\nC=\"x=y\"\nbad line\n")
    assert got == {"A": "1", "B": "two", "C": "x=y"}
