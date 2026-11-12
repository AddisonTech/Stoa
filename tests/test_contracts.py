"""Trial 16, the contract announcements: the scraper's manners, the award parser, the type
classifier, the point-in-time ticker matcher and the no-lookahead rule."""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd
import pytest

from stoa import clock
from stoa.contracts import data, parse, scrape, tickers, trial16

# ---- the scraper -------------------------------------------------------------------------------

LISTING = """
<div item-index="1" article-id="4645969" article-title="Contracts for Nov. 10, 2026"
 article-url="https://www.war.gov/News/Contracts/Contract/Article/4645969/contracts-for-nov-10-2026/"
 article-url-or-link="https://www.war.gov/News/Contracts/Contract/Article/4645969/contracts-for-nov-10-2026/">
<a href="/News/Contracts/Contract/Article/1043211/contracts-for-sept-6-2016/">old</a>
<a href="https://www.war.gov/News/Releases/Release/Article/4646284/readout/">not a contract</a>
"""


def test_listing_links_read_article_url_attributes_and_hrefs_once_each() -> None:
    links = scrape.listing_links(LISTING)
    assert [(lk.id, lk.slug) for lk in links] == [
        (4645969, "contracts-for-nov-10-2026"),
        (1043211, "contracts-for-sept-6-2016"),
    ]
    assert links[1].url.startswith("https://www.war.gov/News/Contracts/")
    assert scrape.slug_date("contracts-for-sept-6-2016") == date(2016, 9, 6)
    assert scrape.slug_date("contracts-for-oct-28-2026") == date(2026, 10, 28)
    assert scrape.title_date("<title>Contracts For Dec. 2, 2019</title>") == date(2019, 12, 2)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s

    def monotonic(self) -> float:
        return self.t


def polite(handler: object) -> tuple[scrape.Polite, Clock]:
    c = Clock()
    client = httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return scrape.Polite(client, sleep=c.sleep, monotonic=c.monotonic, backoff=(1.0,) * 4), c


def test_requests_are_paced_one_a_second_and_redirect_hops_count() -> None:
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        if "defense.gov" in str(req.url):
            return httpx.Response(301, headers={"location": "https://www.war.gov/x"})
        return httpx.Response(200, text="ok")

    web, c = polite(handler)
    for _ in range(3):
        assert web.get("https://www.war.gov/a") is not None
    web.get("https://www.defense.gov/old")
    assert web.requests == 5 and len(calls) == 5
    assert c.slept == [1.0, 1.0, 1.0, 1.0]  # one second between every request start


def test_repeated_refusals_stop_the_pull_as_blocked() -> None:
    web, _ = polite(lambda req: httpx.Response(403))
    with pytest.raises(scrape.Blocked):
        web.get("https://www.war.gov/News/Contracts/")
    assert web.requests == scrape.REFUSALS_TO_STOP


# ---- the parser ---------------------------------------------------------------------------------

PAGE = """<html><head><title>Contracts For Nov. 10, 2026</title></head><body>
<div class="body">
<p style="text-align: center;"><strong>ARMY</strong></p>
<p>Shift5 Inc.,* Arlington, Virginia, was awarded a $99,000,000 firm-fixed-price,
indefinite-delivery/indefinite-quantity contract for commercial offerings. Army Contracting
Command, Aberdeen Proving Ground, Maryland, is the contracting activity (W9128Z-26-D-A020).</p>
<p>Oshkosh Defense LLC, Oshkosh, Wisconsin, was awarded a $31,972,227 modification (P00018) to
contract W56HZV-22-D-0004 for systems technical support services.</p>
<p>American Ordnance LLC,* Middletown, Iowa (W52P1J-17-D-0002); and Day &amp; Zimmermann Inc.,
Parsons, Kansas (W52P1J-17-D-0003), will share in a $150,000,000 firm-fixed-price contract
for ammunition.</p>
<p style="text-align: center;"><strong>NAVY</strong></p>
<p>Core Services Group Inc.,* Virginia Beach, Virginia, is awarded $23,873,454 for a
firm-fixed-price, indefinite-delivery/indefinite-quantity contract. The contract includes a
six-month option which if exercised will bring the total to $26,385,606.</p>
<p>The Boeing Co., St. Louis, Missouri, is awarded a $12.5 million firm-fixed-price modification
to previously awarded contract N00019-22-C-0015 to exercise an option for spares.</p>
<p>Booz Allen Hamilton Inc., Washington, D.C., was awarded a $40,000,000 cost-plus-fixed-fee
task order (N0001926F2222) for engineering services.</p>
<p>CORRECTION: The $33,141,075 modification announced on Oct. 13, 2026, to L3Harris
Technologies Inc. was not released.</p>
<p>*Small business</p>
</div></article></body></html>"""


def test_the_parser_reads_every_award_row_and_skips_notes() -> None:
    rows = parse.parse_page(PAGE, 7)
    got = [(a.company, a.city, a.state, a.value, a.type, a.branch) for a in rows]
    assert got == [
        ("Shift5 Inc.", "Arlington", "Virginia", 99_000_000.0, "idiq", "Army"),
        ("Oshkosh Defense LLC", "Oshkosh", "Wisconsin", 31_972_227.0, "modification", "Army"),
        ("American Ordnance LLC", "Middletown", "Iowa", 150_000_000.0, "multiple_award", "Army"),
        ("Day & Zimmermann Inc.", "Parsons", "Kansas", 150_000_000.0, "multiple_award", "Army"),
        ("Core Services Group Inc.", "Virginia Beach", "Virginia", 23_873_454.0, "idiq", "Navy"),
        ("The Boeing Co.", "St. Louis", "Missouri", 12_500_000.0, "option", "Navy"),
        ("Booz Allen Hamilton Inc.", "Washington", "D.C.", 40_000_000.0, "order", "Navy"),
    ]
    shift5, oshkosh, ammo1, ammo2, _core, boeing, _booz = rows
    assert shift5.small_business and shift5.contract == "W9128Z26DA020" and shift5.group == "new"
    assert oshkosh.mod_number == "P00018" and oshkosh.contract == "W56HZV22D0004"
    assert oshkosh.group == "mod" and boeing.group == "mod"
    assert ammo1.n_awardees == 2 and ammo1.contract == "W52P1J17D0002"
    assert ammo2.contract == "W52P1J17D0003" and ammo1.group == "multi"
    assert all(a.date == date(2026, 11, 10) and a.article == 7 for a in rows)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("X was awarded a $10,000,000 firm-fixed-price contract for parts.", "new"),
        ("X was awarded a $10,000,000 indefinite-delivery/indefinite-quantity contract.", "idiq"),
        (
            "X was awarded a $10,000,000 ceiling IDIQ contract with task orders contemplated.",
            "idiq",
        ),
        ("X was awarded a $10,000,000 delivery order (N0001926F2222) for spares.", "order"),
        ("X was awarded a $10,000,000 modification (P00012) to contract W1.", "modification"),
        (
            "X is awarded $10,000,000 bringing the cumulative face value to $90,000,000.",
            "modification",
        ),
        ("X was awarded $10,000,000 to exercise options for logistics support.", "option"),
        ("X was awarded a $10,000,000 modification exercising an option on contract W1.", "option"),
        ("X is awarded a $10,000,000 firm-fixed-price option for year three of support.", "option"),
        ("X was awarded a $10,000,000 multiple-award contract for repairs.", "multiple_award"),
        (
            "X was awarded a $10,000,000 contract with an option which if exercised adds a year.",
            "new",
        ),
        ("X was awarded a $10,000,000 contract for C-130 aircraft modification services.", "new"),
    ],
)
def test_the_type_classifier_rules(text: str, kind: str) -> None:
    verb = parse._VERB.search(text)
    assert verb is not None
    assert parse.classify(text, 1, parse.award_phrase(text[verb.start() :])) == kind


def test_multiple_awardees_are_one_type_whatever_the_wording() -> None:
    assert parse.classify("A and B were awarded $5,000,000 contracts.", 2) == "multiple_award"


def test_hand_check_fixes_of_2026_11_12() -> None:
    """Three rules the 30 row hand check of 2026-11-12 found wrong."""
    new_multi = (
        "A, B and C will compete for each order of the $200,000,000 firm-fixed-price contract "
        "for security engineering. This total cumulative face value is $200,000,000."
    )
    assert parse.classify(new_multi, 3) == "multiple_award"
    mod = "X was awarded $9,000,000 bringing the total cumulative face value of the contract to $90"
    assert parse.classify(mod, 1) == "modification"
    option = (
        "X has been awarded a maximum $25,400,632 modification (P00121) exercising the second "
        "one-year option period of a one-year base contract (SPE1C1-14-D-1048)."
    )
    assert parse.classify(option, 1) == "option"
    lead = (
        "Laguna Components,* Laguna Beach, California (N68936-16-D-0030); Impact Components,*; "
        "San Diego, California (N68936-16-D-0032); and Space Vector Corp.,* Chatsworth, "
        "California (N68936-16-D-0033)"
    )
    names = parse.split_awardees(lead)
    assert len(names) == 3 and names[1].startswith("Impact Components")
    assert parse.split_place(names[1])[:3] == ("Impact Components", "San Diego", "California")
    # The second sample of 30.
    task = (
        "Booz Allen Hamilton Inc., McLean, Virginia, is awarded $17,559,070 for a firm-fixed-"
        "price, one-year task order (M95494-18-F-0009) under a previously awarded GSA OASIS "
        "contract for professional services."
    )
    verb = parse._VERB.search(task)
    assert verb is not None
    assert parse.classify(task, 1, parse.award_phrase(task[verb.start() :])) == "order"
    letter = (
        "Lockheed Martin Corp., Fort Worth, Texas, is awarded a not-to-exceed $152,329,105 "
        "fixed-price-incentive-firm-target undefinitized indefinite-delivery/indefinite-"
        "quantity contract. This contract provides for logistics systems."
    )
    verb = parse._VERB.search(letter)
    assert verb is not None
    assert parse.classify(letter, 1, parse.award_phrase(letter[verb.start() :])) == "idiq"
    boa = "X was awarded a $7,612,416 firm-fixed-price basic ordering agreement order for blades."
    verb = parse._VERB.search(boa)
    assert verb is not None
    assert parse.classify(boa, 1, parse.award_phrase(boa[verb.start() :])) == "order"
    fuel = "$35,123,700 Hunt Refining Co., Tuscaloosa, Alabama (SPE600-17-D-0479), $33,058,662"
    assert parse.split_place(fuel)[:3] == ("Hunt Refining Co.", "Tuscaloosa", "Alabama")
    # The third sample of 30.
    order = "is being awarded $9,975,624 for firm-fixed-price order N0001917F1568 against a BOA"
    assert parse.classify("Insitu Inc. " + order, 1, parse.award_phrase(order)) == "order"
    multi = "were awarded a one-year base plus four one-year options contract for analytics"
    assert parse.classify("A; B; and C " + multi, 3, parse.award_phrase(multi)) == "multiple_award"
    solicited = (
        "in response to solicitation number N66604-18-R-0881. The Naval Undersea Warfare "
        "Center is the contracting activity (N66604-22-C-0102)."
    )
    assert parse.piid(solicited) == "N6660422C0102"
    under_mac = (
        "is being awarded $12,867,312 for firm-fixed-price task order 0002 under a previously "
        "awarded multiple award construction contract (N62478-16-D-4015) for drydock repair."
    )
    assert parse.classify("GSI Pacific Inc. " + under_mac, 1, parse.award_phrase(under_mac)) == (
        "order"
    )


# ---- the ticker matcher --------------------------------------------------------------------------

D = date


def book(bars: dict[str, tuple[date, date]] | None = None) -> tickers.Book:
    ciks = pd.DataFrame({"cik": [1, 2, 3, 4, 5], "symbol": ["LMT", "OA", "NOC", "CUB", "ACME"]})
    names = {
        1: [("lockheed martin", D(1995, 1, 1), D(2100, 1, 1))],
        2: [("orbital atk", D(2015, 2, 10), D(2100, 1, 1))],
        3: [("northrop grumman", D(2001, 1, 1), D(2100, 1, 1))],
        4: [("lionheart", D(2024, 6, 1), D(2100, 1, 1))],  # took the CUB ticker in 2024
        5: [("acme systems", D(2010, 1, 1), D(2100, 1, 1))],
    }
    reports = {
        1: [D(2015, 2, 1) + timedelta(days=90 * i) for i in range(48)],
        2: [D(2015, 3, 1) + timedelta(days=90 * i) for i in range(14)],  # stops in 2018
        3: [D(2015, 2, 1) + timedelta(days=90 * i) for i in range(48)],
        4: [D(2024, 8, 1) + timedelta(days=90 * i) for i in range(8)],
        5: [D(2015, 2, 1) + timedelta(days=90 * i) for i in range(48)],
    }
    ex21 = {
        "sikorsky aircraft": [(1, D(2016, 2, 2), D(2017, 2, 2), "lmt-ex21.htm (2016-02-01)")],
        "orbital atk": [(3, D(2019, 2, 1), D(2020, 2, 1), "noc-ex21.htm (2019-01-31)")],
        "acme systems": [(1, D(2016, 2, 2), D(2017, 2, 2), "lmt-ex21.htm (2016-02-01)")],
    }
    spans = bars or {
        "LMT": (D(2016, 1, 4), D(2026, 6, 30)),
        "OA": (D(2016, 1, 4), D(2018, 6, 5)),  # delisted when Northrop bought it
        "NOC": (D(2016, 1, 4), D(2026, 6, 30)),
        "CUB": (D(2016, 1, 4), D(2026, 6, 30)),  # Cubic Corp's bars, then the new company's
        "ACME": (D(2016, 1, 4), D(2026, 6, 30)),
    }
    arrays = {
        s: np.array(pd.bdate_range(a, b).values.astype("M8[D]")) for s, (a, b) in spans.items()
    }
    return tickers.Book(ciks, names, reports, ex21, arrays)


def entry(day: date) -> date:
    return data.entry_session(day)


def test_a_subsidiary_maps_to_its_parent_only_while_the_parent_lists_it() -> None:
    b = book()
    m = b.match("Sikorsky Aircraft Corp., Stratford", D(2016, 6, 1), entry(D(2016, 6, 1)))
    assert isinstance(m, tickers.Match)
    assert (m.symbol, m.how, m.key) == ("LMT", "ex21", "sikorsky aircraft")
    # Before the 10-K that listed it was public, the same name is not matched.
    early = b.match("Sikorsky Aircraft Corp.", D(2016, 1, 20), entry(D(2016, 1, 20)))
    assert isinstance(early, tickers.Miss) and early.reason == "unmatched"


def test_a_division_name_matches_its_parent_by_the_name_before_the_comma() -> None:
    m = book().match(
        "Lockheed Martin, Rotary and Mission Systems", D(2020, 3, 2), entry(D(2020, 3, 2))
    )
    assert isinstance(m, tickers.Match) and m.symbol == "LMT" and m.how == "direct"
    m2 = book().match("Lockheed Martin Aeronautics Co.", D(2020, 3, 2), entry(D(2020, 3, 2)))
    assert isinstance(m2, tickers.Match) and m2.how == "direct_prefix"


def test_a_delisted_company_stays_in_while_it_traded_then_maps_to_its_buyer() -> None:
    b = book()
    before = b.match("Orbital ATK Inc.", D(2017, 5, 1), entry(D(2017, 5, 1)))
    assert isinstance(before, tickers.Match) and before.symbol == "OA"
    gone = b.match("Orbital ATK Inc.", D(2018, 9, 4), entry(D(2018, 9, 4)))
    assert isinstance(gone, tickers.Miss) and gone.reason == "no_bar"
    later = b.match("Orbital ATK Inc.", D(2019, 6, 3), entry(D(2019, 6, 3)))
    assert isinstance(later, tickers.Match) and later.symbol == "NOC" and later.how == "ex21"


def test_a_ticker_taken_over_later_is_not_read_back() -> None:
    # Cubic Corp traded as CUB until 2021; a company named Lionheart took CUB in 2024. An
    # award to "Lionheart" before Lionheart reported anything is not matched to CUB's old bars.
    m = book().match("Lionheart Inc.", D(2019, 4, 1), entry(D(2019, 4, 1)))
    assert isinstance(m, tickers.Miss)


def test_two_companies_for_one_name_is_ambiguous_and_dropped() -> None:
    m = book().match("Acme Systems LLC", D(2016, 6, 1), entry(D(2016, 6, 1)))
    assert isinstance(m, tickers.Match) and m.symbol == "ACME"  # the direct book comes first
    b = book()
    b.by_name["sikorsky aircraft"].append((5, D(2010, 1, 1), D(2100, 1, 1)))
    b.ex21["sikorsky aircraft"].append((5, D(2016, 2, 2), D(2017, 2, 2), "x"))
    got = b.match("Sikorsky Aircraft", D(2016, 6, 1), entry(D(2016, 6, 1)))
    assert isinstance(got, tickers.Match) and got.symbol == "ACME"
    b.by_name["sikorsky aircraft"].append((1, D(2010, 1, 1), D(2100, 1, 1)))
    amb = b.match("Sikorsky Aircraft", D(2016, 6, 1), entry(D(2016, 6, 1)))
    assert isinstance(amb, tickers.Miss) and amb.reason == "ambiguous"


def test_an_exact_subsidiary_listing_beats_a_one_word_prefix() -> None:
    # Boeing's exhibit 21 lists Tapestry Solutions; Tapestry, Inc. (the handbag company, Coach
    # until 2017) must not get its contracts by the shared first word.
    ciks = pd.DataFrame({"cik": [10, 11], "symbol": ["BA", "TPR"]})
    names = {
        10: [("boeing", D(1990, 1, 1), D(2100, 1, 1))],
        11: [("tapestry", D(2017, 10, 31), D(2100, 1, 1))],
    }
    reports = {c: [D(2018, 1, 1) + timedelta(days=90 * i) for i in range(20)] for c in (10, 11)}
    ex21 = {"tapestry solutions": [(10, D(2018, 2, 10), D(2019, 2, 10), "ba-ex21.htm")]}
    days = np.array(pd.bdate_range(D(2016, 1, 4), D(2022, 12, 30)).values.astype("M8[D]"))
    b = tickers.Book(ciks, names, reports, ex21, {"BA": days, "TPR": days})
    m = b.match("Tapestry Solutions Inc., a Boeing Co.", D(2018, 10, 1), entry(D(2018, 10, 1)))
    assert isinstance(m, tickers.Match) and (m.symbol, m.how) == ("BA", "ex21")
    # Without the listing, a one-word prefix would have guessed the handbag company; one word
    # is never enough, so the name stays unmatched.
    b2 = tickers.Book(ciks, names, reports, {}, {"BA": days, "TPR": days})
    guess = b2.match("Tapestry Solutions Inc.", D(2018, 10, 1), entry(D(2018, 10, 1)))
    assert isinstance(guess, tickers.Miss) and guess.reason == "unmatched"


def test_one_word_before_a_comma_is_not_a_company_unless_it_says_so() -> None:
    # The 50 match hand check of 2026-11-12 found "Rogers, Lovelock & Fritz Inc." (architects)
    # matched to Rogers Corp. by the part before the comma.
    assert "rogers" not in [k for k, _ in tickers.candidates("Rogers, Lovelock & Fritz Inc.")]
    keys = [k for k, _ in tickers.candidates("The Boeing Co., Defense, Space & Security")]
    assert "boeing" in keys
    assert "lockheed martin" in [k for k, _ in tickers.candidates("Lockheed Martin, RMS")]


def test_a_name_is_never_shortened_to_one_word() -> None:
    keys = [k for k, _ in tickers.candidates("General Atomics Aeronautical Systems Inc.")]
    assert "general" not in keys and "general atomics" in keys
    assert [k for k, _ in tickers.candidates("Raytheon Missile Systems")] == [
        "raytheon missile systems",
        "raytheon missile",
    ]
    assert tickers.norm("L-3 Communications, L.L.C.") == "l 3 communications"
    assert tickers.ex21_keys("Electric Boat Corporation Delaware\nCACI, INC.-FEDERAL Delaware") == {
        "electric boat",
        "caci federal",
    }


# ---- no lookahead ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (D(2024, 3, 8), D(2024, 3, 11)),  # Friday 5 pm post, bought Monday at the open
        (D(2024, 3, 11), D(2024, 3, 12)),  # never the announcement day itself
        (D(2024, 3, 29), D(2024, 4, 1)),  # Good Friday, market shut, page still posts
        (D(2024, 11, 11), D(2024, 11, 12)),  # Veterans Day: market open, bought next day
    ],
)
def test_no_award_is_used_before_its_5pm_post(day: date, expected: date) -> None:
    got = data.entry_session(day)
    assert got == expected
    assert pd.Timestamp(clock.session(got).open) > data.known_at(day)
    assert data.known_at(day) == pd.Timestamp(clock.at_ny(day, "17:00"))
    assert data.prior_close_session(day) <= day


def test_sec_facts_filed_on_or_after_the_announcement_day_are_not_read() -> None:
    facts = {
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"end": "2023-12-31", "val": 100, "filed": "2024-02-01", "accn": "a"},
                            {"end": "2024-03-31", "val": 300, "filed": "2024-05-01", "accn": "b"},
                        ]
                    }
                }
            },
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "val": 1000,
                                "filed": "2024-02-01",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-03-31",
                                "val": 400,
                                "filed": "2024-05-01",
                            },
                            {
                                "start": "2023-01-01",
                                "end": "2023-03-31",
                                "val": 200,
                                "filed": "2024-05-01",
                            },
                        ]
                    }
                }
            },
        }
    }
    assert data.shares_on(facts, D(2024, 5, 1)) == 100  # the 2024-05-01 filing is not yet read
    assert data.shares_on(facts, D(2024, 5, 2)) == 300
    assert data.revenue_ttm(facts, D(2024, 5, 1)) == 1000
    assert data.revenue_ttm(facts, D(2024, 5, 2)) == 1000 + 400 - 200  # rolled forward


def test_market_value_undoes_the_bars_split_and_dividend_adjustment() -> None:
    """The daily bars are adjusted for every later split and dividend; SEC share counts are as
    filed. A company that splits 5 for 1 later must not look 5 times smaller before it (that
    would favour stocks that went on to rise), so the split comes from the SEC's own
    restatement and the dividend from its reported dividends per share."""
    weighted = [
        # FY2021 as first reported, then restated by the 10-K filed after a 5 for 1 split.
        {"start": "2021-01-01", "end": "2021-12-31", "val": 54e6, "filed": "2022-02-10"},
        {"start": "2021-01-01", "end": "2021-12-31", "val": 270e6, "filed": "2024-11-19"},
        {"start": "2022-01-01", "end": "2022-12-31", "val": 53e6, "filed": "2023-02-10"},
        {"start": "2022-01-01", "end": "2022-12-31", "val": 265e6, "filed": "2024-11-19"},
        {"start": "2023-01-01", "end": "2023-12-31", "val": 52e6, "filed": "2024-02-10"},
        {"start": "2023-01-01", "end": "2023-12-31", "val": 260e6, "filed": "2025-11-19"},
    ]
    facts = {
        "facts": {
            "us-gaap": {
                "WeightedAverageNumberOfSharesOutstandingBasic": {"units": {"shares": weighted}},
                "CommonStockDividendsPerShareDeclared": {
                    "units": {
                        "USD/shares": [
                            {
                                "start": "2025-01-01",
                                "end": "2025-03-31",
                                "val": 2.0,
                                "filed": "2025-05-01",
                            },
                        ]
                    }
                },
            }
        }
    }
    # One split, though two later filings each restate a year for the first time.
    assert data.split_events(facts) == [("2024-11-19", 5.0)]
    assert data.split_factor(facts, "2022-02-10") == 5.0
    assert data.split_factor(facts, "2024-11-19") == 1.0
    days = pd.bdate_range("2021-01-04", "2025-06-30")
    bars = pd.DataFrame({"close": 20.0}, index=days)  # adjusted: $100 before the split / 5
    # A $2 dividend when the raw price was $20 scales every earlier price by 0.9.
    assert data.dividend_adjustment(facts, bars, D(2025, 3, 1)) == pytest.approx(0.9)
    assert data.dividend_adjustment(facts, bars, D(2025, 4, 1)) == 1.0


def test_an_exhibit_21_is_read_from_the_day_after_its_10k_was_accepted() -> None:
    b = book()
    on_day = b.match("Sikorsky Aircraft Corp.", D(2016, 2, 1), entry(D(2016, 2, 1)))
    assert isinstance(on_day, tickers.Miss)
    next_day = b.match("Sikorsky Aircraft Corp.", D(2016, 2, 2), entry(D(2016, 2, 2)))
    assert isinstance(next_day, tickers.Match)


def test_the_holdout_is_never_graded() -> None:
    from stoa.newsbrain import sealed

    panel = pd.DataFrame({"entry": [D(2026, 7, 1)]})
    with pytest.raises(sealed.HoldoutBreach):
        trial16.window(panel, D(2026, 1, 1), D(2026, 7, 31))


# ---- the grader --------------------------------------------------------------------------------


def test_the_grader_counts_hits_edge_and_a_zero_on_days_with_no_trade() -> None:
    t = pd.DataFrame(
        {
            "entry": [D(2024, 3, 11), D(2024, 3, 11), D(2024, 3, 13)],
            "net_0": [0.02, -0.01, 0.03],
            "ret_0": [0.03, 0.0, 0.04],
            "spy_0": [0.01, 0.01, 0.01],
        }
    )
    g = trial16.grade(t, D(2024, 3, 11), D(2024, 3, 13))
    assert g.trades == 3 and g.hit_rate == pytest.approx(2 / 3)
    assert g.avg_net == pytest.approx(0.04 / 3)
    s = trial16.daily_series(t, D(2024, 3, 11), D(2024, 3, 13))
    assert list(s) == pytest.approx([0.005, 0.0, 0.03])
    assert trial16.verdict(trial16.checks(g, 0.99), g.trades) == "STARVED"


def test_the_registered_spec_is_stable_and_says_what_is_excluded() -> None:
    s = trial16.spec()
    assert "multiple-award" in s["excluded"] and "never mixed" in s["excluded"]
    assert trial16.spec_hash() == trial16.spec_hash()
    assert s["pass"].count("above") >= 2 and "54%" in s["pass"]
