"""The ripple map types each named company from the words near it, and reads both ways."""

from __future__ import annotations

import pandas as pd

from stoa.newsbrain import ripplemap as rm

FILER = 1
ACME, BOLT, CRANE, DELTA = 10, 20, 30, 40
NAMES = {
    FILER: "filer works",
    ACME: "acme widgets",
    BOLT: "bolt foundry",
    CRANE: "crane systems",
    DELTA: "delta parts",
}
INDEX = rm.NameIndex(NAMES)


def typed(text: str) -> dict[int, tuple[str, str]]:
    return {c: (t.relation, t.cue) for c, t in rm.type_filing(text, INDEX, FILER).items()}


def ts(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def test_a_customer_cue() -> None:
    text = "Sales to Acme Widgets, Inc. accounted for 14% of our net sales in 2023."
    assert typed(text)[ACME] == ("customer", "sales to")


def test_a_share_of_sales_is_a_customer_cue() -> None:
    text = "In 2023 Acme Widgets Inc. accounted for 14% of net sales."
    assert typed(text)[ACME] == ("customer", "accounted for 14% of net sales")
    text = "Acme Widgets Corp. each accounted for between 15 and 20 percent of our revenue."
    assert typed(text)[ACME] == ("customer", "20 percent of our revenue")
    text = "Acme Widgets and Delta had 43% and 16%, respectively, of our Pipe segment revenue."
    assert typed(text)[ACME] == ("customer", "respectively, of our pipe segment revenue")
    text = "Delta Parts Ltd. accounted for 11% of our purchases in 2023."
    assert typed(text)[DELTA] == ("supplier", "accounted for 11% of our purchases")


def test_a_supplier_cue() -> None:
    text = "Our chips are manufactured by Delta Parts Ltd. under a long term agreement."
    assert typed(text)[DELTA] == ("supplier", "manufactured by")


def test_a_competitor() -> None:
    text = "We compete with Crane Systems Corporation in every market we serve."
    assert typed(text)[CRANE] == ("competitor", "compete")
    text = "Crane Systems Corp. helped us build our core competencies."
    assert typed(text)[CRANE] == ("other", "")


def test_named_with_no_cue_is_other() -> None:
    text = "Our headquarters lease is with Crane Systems Corp. and runs to 2030."
    assert typed(text)[CRANE] == ("other", "")


def test_the_closer_cue_wins() -> None:
    text = "Our customers include Delta Parts Ltd.; our suppliers include Acme Widgets."
    out = typed(text)
    assert out[DELTA] == ("customer", "customers")
    assert out[ACME] == ("supplier", "suppliers")


def test_a_cue_inside_the_name_does_not_count() -> None:
    text = "We buy from many makers; Bolt Foundry Inc. is one of them."
    assert typed(text)[BOLT] == ("supplier", "buy from")


def test_a_company_name_period_does_not_end_the_sentence() -> None:
    text = "Delta Parts Inc. and others. Acme Widgets Inc. accounted for 31% of our revenues."
    assert typed(text)[ACME][0] == "customer"
    assert rm.sentence_stops("Apple Inc. Supplies phones. Next one.") == [26, 36]


def test_lowercase_words_and_the_filer_are_not_matches() -> None:
    text = "Filer Works sells acme widgets to customers; Acme Widgets is one of them."
    out = rm.type_filing(text, INDEX, FILER)
    assert set(out) == {ACME}
    mentions = INDEX.find(text)
    assert [m.start for m in mentions if m.other == ACME] == [text.index("Acme Widgets")]


def filings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cik": [FILER] * 4,
            "accn": ["a1", "a2", "a3", "a4"],
            "form": ["10-K", "10-K/A", "10-K", "10-K"],
            "accepted": [ts("2020-03-10"), ts("2020-06-10"), ts("2021-03-10"), ts("2024-03-10")],
            "text": [
                "Sales to Acme Widgets accounted for 20% of revenues.",
                "Our suppliers include Delta Parts.",
                "",
                "Sales to Acme Widgets accounted for 18% of revenues.",
            ],
        }
    )


def test_valid_to_is_the_next_original_10k_or_15_months() -> None:
    e = rm.filing_edges(filings(), INDEX, None).set_index("accn")
    assert e.loc["a1", "valid_to"] == ts("2021-03-10")  # the amendment does not end it
    assert e.loc["a2", "valid_to"] == ts("2021-03-10")
    assert e.loc["a4", "valid_to"] == ts("2024-03-10") + pd.Timedelta(days=rm.VALID_DAYS)
    assert e.loc["a2", "relation"] == "supplier"
    assert "a3" not in e.index  # an empty filing names no one, but still ends the old edges


def test_a_long_gap_is_capped_at_15_months() -> None:
    f = filings()
    f = f[f["accn"].isin(["a1", "a4"])]
    e = rm.filing_edges(f, INDEX, None).set_index("accn")
    assert e.loc["a1", "valid_to"] == ts("2020-03-10") + pd.Timedelta(days=rm.VALID_DAYS)


def test_the_top_names_guard_drops_partners_outside_it() -> None:
    allowed = {"2020-02": {DELTA}, "2020-05": {ACME}, "2024-02": {DELTA}}
    e = rm.filing_edges(filings(), INDEX, allowed)
    assert list(e["accn"]) == []  # a1 named Acme but only Delta was allowed the month before
    allowed = {"2020-02": {ACME}, "2020-05": {DELTA}, "2024-02": {ACME}}
    assert sorted(rm.filing_edges(filings(), INDEX, allowed)["accn"]) == ["a1", "a2", "a4"]


def edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cik": [FILER, FILER, CRANE],
            "other": [ACME, DELTA, FILER],
            "relation": ["customer", "supplier", "competitor"],
            "accepted": [ts("2020-03-01"), ts("2020-03-01"), ts("2020-05-01")],
            "valid_to": [ts("2021-03-01"), ts("2021-03-01"), ts("2020-06-01")],
            "cue": ["sales to", "suppliers", "compete"],
            "accn": ["a1", "a1", "c1"],
        }
    )


CIKS = pd.DataFrame(
    {
        "cik": [FILER, ACME, ACME, DELTA, CRANE],
        "symbol": ["FLR", "ACM", "ACM.B", "DLT", "CRN"],
    }
)


def test_partners_reverse_the_view() -> None:
    p = rm.partners(edges(), CIKS, ACME, "2020-07-01")
    assert list(p["partner_symbol"]) == ["FLR"]
    assert p["relation"].iloc[0] == "supplier"  # FLR says ACM is its customer
    assert p["said_by"].iloc[0] == FILER
    mine = rm.partners(edges(), CIKS, FILER, ts("2020-05-15"))
    got = dict(zip(mine["partner_symbol"], mine["relation"], strict=True))
    assert got == {"ACM": "customer", "ACM.B": "customer", "DLT": "supplier", "CRN": "competitor"}


def test_partners_respect_the_valid_window() -> None:
    assert rm.partners(edges(), CIKS, ACME, "2020-02-28").empty
    assert rm.partners(edges(), CIKS, ACME, "2021-03-01").empty  # valid_to is exclusive
    late = rm.partners(edges(), CIKS, FILER, "2020-07-01")
    assert "CRN" not in set(late["partner_symbol"])


def test_symbol_edges_expand_every_symbol_both_ways() -> None:
    s = rm.symbol_edges(edges(), CIKS)
    assert list(s.columns) == ["symbol", "partner", "relation", "accepted", "valid_to"]
    pairs = {(a, b): r for a, b, r in zip(s["symbol"], s["partner"], s["relation"], strict=True)}
    assert pairs[("FLR", "ACM")] == "customer"
    assert pairs[("FLR", "ACM.B")] == "customer"
    assert pairs[("ACM", "FLR")] == "supplier"
    assert pairs[("ACM.B", "FLR")] == "supplier"
    assert pairs[("DLT", "FLR")] == "customer"
    assert pairs[("FLR", "CRN")] == pairs[("CRN", "FLR")] == "competitor"
    assert len(s) == 8


def test_the_filer_as_a_supplier_names_a_customer() -> None:
    text = "We are one of a limited number of suppliers to OEMs such as Acme Widgets Corp."
    assert typed(text)[ACME] == ("customer", "suppliers to")
    text = "Our parts reach carmakers through their Tier 1 suppliers, such as Delta Parts Ltd."
    assert typed(text)[DELTA] == ("other", "")
    text = "Our wafer partners are Acme Widgets and Global Foundries."
    assert typed(text)[ACME] == ("other", "")  # a capitalized "Foundries" is a name


def test_a_one_word_name_inside_a_longer_name_is_not_a_match() -> None:
    index = rm.NameIndex({ACME: "southern", DELTA: "people"}, common=frozenset({"people"}))
    assert index.find("Southern California Edison is our customer.") == []
    assert index.find("We sell in the People's Republic of China.") == []
    assert len(index.find("Our largest customer is The Southern Company.")) == 1
    assert index.find("People are our largest asset.") == []
    assert len(index.find("Our customers include People, Inc. and others.")) == 1
    common = rm.NameIndex({ACME: "southern"}, common=frozenset({"southern"}))
    assert common.find("Revenue in the Southern division fell.") == []
    assert len(common.find("Our customers are Southern and others.")) == 1
    assert len(common.find("Sales to Southern accounted for 12% of revenue.")) == 1
    assert rm.NameIndex({ACME: "nasdaq"}).find("Our shares trade on the Nasdaq.") == []


def test_common_words_are_the_names_written_in_lower_case_most() -> None:
    names = {ACME: "target", DELTA: "walmart"}
    texts = ["target markets, target customers", "Target, Walmart and Walmart", "our target"]
    assert rm.common_words(texts * 2, names) == frozenset({"target"})  # 6 lower, 2 capital
    assert rm.common_words(texts, names) == frozenset()  # too few to say
