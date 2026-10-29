"""Trial 15's new brain inputs: the day's events, the event memory (earlier sessions only)
and training with recent rows weighted more."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stoa.newsbrain import brain15 as b
from stoa.newsbrain import eventread


def test_the_memory_reads_only_rows_two_sessions_back_or_more() -> None:
    hist = pd.DataFrame(
        {
            "k": [1, 1, 1, 2],
            "sidx": [10, 11, 12, 10],
            "x": [0.01, 0.03, 0.05, 9.0],
            "hit": [1.0, 1.0, 0.0, 1.0],
        }
    )
    query = pd.DataFrame({"k": [1, 1, 1, 1, 2], "sidx": [11, 12, 13, 20, 13]})
    got = b.asof(hist, query, "k")
    # sidx 11 sees nothing (10 is only one back); 12 sees 10; 13 sees 10 and 11; 20 sees all.
    assert list(got["n"]) == [0, 1, 2, 3, 1]
    assert got["s"].tolist() == pytest.approx([0.0, 0.01, 0.04, 0.09, 9.0])
    assert list(got["h"]) == [0, 1, 2, 2, 1]


def test_a_thin_cell_leans_on_its_parent() -> None:
    n = np.array([0.0, 50.0, 5000.0])
    own = np.array([np.nan, 0.02, 0.02])
    parent = np.array([0.01, 0.0, 0.0])
    got = b.shrunk(n, own, parent)
    assert got.tolist() == pytest.approx([0.01, 0.01, 0.02 * 5000 / 5050])


def test_the_days_main_event_and_direction() -> None:
    labels = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "event": pd.Categorical(
                ["other", "earnings_beat", "other"], categories=eventread.CODES
            ),
            "p_event": [0.9, 0.8, 0.7],
            "direction": [0.0, 0.9, -0.5],
            "size": [1.0, 2.0, 1.0],
        }
    )
    t = pd.Timestamp("2024-03-01 13:00", tz="UTC")
    times = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "known_at": [t, t, t],
            "sidx": [5, 5, 5],
        }
    )
    ev = b.day_events(labels, times).set_index("symbol")
    beat = eventread.CODES.index("earnings_beat")
    assert ev.loc["AAPL", "ev_type"] == beat and ev.loc["AAPL", "ev_sign"] == 1
    assert ev.loc["AAPL", "ev_count"] == 1 and ev.loc["AAPL", "ev_size"] == 2.0
    assert ev.loc["AAPL", "ev_dir"] == pytest.approx(0.9 * 1.6 / (0.9 + 1.6))
    assert ev.loc["MSFT", "ev_type"] == eventread.CODES.index("other")
    assert ev.loc["MSFT", "ev_sign"] == -1


def test_recent_rows_weigh_more_halving_every_two_years() -> None:
    w = b.recency(np.array([1000, 1000 - 504, 1000 - 1008]), 1000)
    assert w.tolist() == pytest.approx([1.0, 0.5, 0.25])


def test_the_sector_is_the_sic_division() -> None:
    assert b.division("3571") == 4  # manufacturing
    assert b.division("6021") == 8  # finance
    assert b.division(None) == 0 and b.division("99") == 0


def test_the_top_5_and_their_grade() -> None:
    rows = pd.DataFrame(
        {
            "sidx": [1] * 6 + [2] * 2,
            "symbol": [*"ABCDEF", "A", "B"],
            "excess_1": [0.02, -0.01, 0.01, 0.0, 0.03, 0.05, -0.02, 0.01],
            "y_1": [1, 0, 1, 0, 1, 1, 0, 1],
        }
    )
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.4, 0.3])
    picks = b.top5(rows, score)
    assert list(picks[picks["sidx"] == 1]["symbol"]) == ["A", "B", "C", "D", "E"]
    g = b.grade(picks)
    assert g["picks"] == 7 and g["days"] == 2
    assert g["hit_rate"] == pytest.approx(4 / 7)
    assert g["day_hit_rate"] == pytest.approx(0.5)  # day 1 up, day 2 down


def test_ripples_reach_a_company_from_its_customers_news() -> None:
    cut = np.arange(10, dtype=np.int64) * 10
    t = pd.Timestamp(0, unit="ns", tz="UTC")
    edges = pd.DataFrame(
        {
            "symbol": ["SUP", "SUP"],
            "partner": ["BIG", "OLD"],
            "relation": ["customer", "customer"],
            "accepted": [t, t],
            "valid_to": [t + pd.Timedelta(55, unit="ns"), t + pd.Timedelta(15, unit="ns")],
        }
    )
    beat = eventread.CODES.index("earnings_beat")
    events = pd.DataFrame(
        {
            "symbol": ["BIG", "BIG", "OLD"],
            "sidx": [2, 4, 4],
            "ev_dir": [0.8, -0.6, 0.9],
            "ev_type": [beat, beat, beat],
            "ev_sign": [1, -1, 1],
            "ev_size": [2.0, 2.0, 2.0],
        }
    )
    rows = pd.DataFrame(
        {
            "symbol": ["SUP", "SUP", "SUP"],
            "sidx": [3, 5, 8],
            "excess_1": [0.01, -0.02, 0.0],
            "y_1": [1.0, 0.0, 0.0],
        }
    )
    got = b.with_ripple(rows, events, edges, cut).set_index("sidx")
    # Session 3 sees BIG's news of session 2; session 5 sees BIG's of 4 (OLD's link ended).
    assert got.loc[3, "rip_cust"] == pytest.approx(0.8)
    assert got.loc[5, "rip_cust"] == pytest.approx(-0.6)
    assert np.isnan(got.loc[8, "rip_cust"]) and np.isnan(got.loc[3, "rip_supp"])
    assert got.loc[3, "rip_mem"] == pytest.approx(0.0)  # no history yet: the parent's 0


def test_the_premarket_flag_needs_news_before_the_open_on_a_small_thin_company() -> None:
    rows = pd.DataFrame(
        {
            "symbol": ["S", "M", "L", "T"],
            "sidx": [7, 7, 7, 7],
            "log_adv": [8.0, 15.0, 20.0, 9.0],
            "coverage": [np.log1p(1 / 60), 0.0, 0.0, np.log1p(10 / 60)],
        }
    )
    pre = pd.Series([pd.Timestamp("2024-03-01 12:00", tz="UTC")] * 4)
    flag = b.premarket_flag(rows, pre)
    assert list(flag) == [1.0, 0.0, 0.0, 0.0]  # M and L are not small, T is not thin
    assert list(b.premarket_flag(rows, pd.Series([pd.NaT] * 4))) == [0.0] * 4
