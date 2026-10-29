from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stoa import clock
from stoa.newsbrain import catchup as cu
from stoa.newsbrain import panel as pn

DAYS = clock.sessions_between(date(2024, 1, 2), date(2024, 12, 31))
CUT = np.array([pd.Timestamp(clock.at_ny(d, "09:00")).value for d in DAYS], dtype=np.int64)
D = date(2024, 3, 12)  # a Tuesday: previous 2024-03-11, next 2024-03-13
PREV, NEXT = date(2024, 3, 11), date(2024, 3, 13)


def ny(d: date, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(clock.at_ny(d, hhmm))


def minutes(rows: list[tuple[date, str, float, float]]) -> cu.Minutes:
    """(day, hh:mm, open, close) per minute bar."""
    rows = sorted(rows, key=lambda r: ny(r[0], r[1]))
    return cu.Minutes(
        np.array([ny(d, m).value for d, m, _, _ in rows], dtype=np.int64),
        np.array([r[2] for r in rows], dtype=float),
        np.array([r[3] for r in rows], dtype=float),
    )


# ---- prices -----------------------------------------------------------------------------------


def test_price_is_the_open_at_the_minute_or_the_first_bar_within_30_minutes() -> None:
    m = minutes([(D, "09:30", 10.0, 10.1), (D, "09:52", 11.0, 11.2), (D, "10:40", 12.0, 12.0)])
    assert cu.price_at(m, ny(D, "09:30").value) == 10.0
    assert cu.price_at(m, ny(D, "09:45").value) == 11.0  # 7 minutes late: still taken
    assert math.isnan(cu.price_at(m, ny(D, "10:05").value))  # next bar is 35 minutes on
    edge = minutes([(D, "10:15", 9.0, 9.0)])
    assert cu.price_at(edge, ny(D, "09:45").value) == 9.0  # 9:45 to 10:15, as the scoreboard
    assert math.isnan(cu.price_at(minutes([(D, "10:16", 9.0, 9.0)]), ny(D, "09:45").value))
    assert math.isnan(cu.price_at(cu.NO_BARS, ny(D, "09:45").value))


def test_last_close_is_the_last_regular_bar_of_the_session() -> None:
    m = minutes([(PREV, "15:58", 9.0, 9.5), (PREV, "15:59", 9.6, 10.0), (D, "09:30", 11.0, 11.0)])
    assert cu.last_close(m, PREV) == 10.0
    assert math.isnan(cu.last_close(minutes([(D, "09:30", 11.0, 11.0)]), PREV))
    # An early close (2024-11-29, 1:00 pm): a later bar is after hours and not the close.
    ec = date(2024, 11, 29)
    m2 = minutes([(ec, "12:59", 5.0, 5.5), (ec, "13:00", 6.0, 6.0)])
    assert cu.last_close(m2, ec) == 5.5


def day_bars(extra: list[tuple[date, str, float, float]] | None = None) -> cu.Minutes:
    rows = [
        (PREV, "15:59", 9.9, 10.0),  # prev_close 10
        (D, "09:30", 10.2, 10.3),  # p_open 10.2
        (D, "09:31", 10.5, 10.7),
        (D, "09:33", 10.75, 10.85),  # close 10.85 is within 20% of the move (>= 10.8)
        (D, "09:45", 10.6, 10.6),  # p_945 10.6
        (D, "15:59", 10.9, 11.0),  # p_close 11
        (NEXT, "09:47", 11.13, 11.2),  # p_next_945 11.13 (first bar after 9:45)
    ]
    return minutes(rows + (extra or []))


def test_gap_rest_and_minutes_to_80() -> None:
    got = cu.measure(day_bars(), D, PREV, NEXT)
    assert got["prev_close"] == 10.0
    assert got["p_open"] == 10.2
    assert got["p_945"] == 10.6
    assert got["p_close"] == 11.0
    assert got["p_next_945"] == 11.13
    assert got["gap_945"] == pytest.approx(10.6 / 10.0 - 1)
    assert got["rest"] == pytest.approx(11.13 / 10.6 - 1)
    assert got["minutes_to_80"] == 4.0  # the 9:33 bar's close, known at 9:34


def test_minutes_to_80_counts_an_open_at_its_own_minute() -> None:
    m = day_bars([(D, "09:32", 10.8, 10.4)])  # the 9:32 open is already within the band
    assert cu.measure(m, D, PREV, NEXT)["minutes_to_80"] == 2.0
    m0 = day_bars([(D, "09:30", 10.95, 10.3)])  # the opening print: zero minutes
    assert cu.minutes_to_80(m0, D, 10.0, 11.0) == 0.0


def test_minutes_to_80_falls_and_the_under_1pct_rule() -> None:
    down = minutes([(D, "09:30", 9.9, 9.8), (D, "10:00", 9.3, 9.1), (D, "15:59", 9.0, 9.0)])
    # Move 10 -> 9; within 20% means at most 9.2: the 10:00 close, known at 10:01.
    assert cu.minutes_to_80(down, D, 10.0, 9.0) == 31.0
    assert math.isnan(cu.minutes_to_80(down, D, 10.0, 10.09))  # a 0.9% move
    assert math.isnan(cu.minutes_to_80(down, D, 10.0, 9.91))  # a 0.9% fall
    assert cu.minutes_to_80(down, D, 10.0, 9.88) == 0.0  # 1.2%: the 9.9 open is within 0.024
    assert math.isnan(cu.minutes_to_80(down, D, math.nan, 9.0))
    # A quiet session: gap and rest are still measured, minutes_to_80 is not.
    quiet = minutes(
        [(PREV, "15:59", 10.0, 10.0), (D, "09:45", 10.6, 10.6), (D, "15:59", 10.05, 10.05)]
    )
    got = cu.measure(quiet, D, PREV, NEXT)
    assert got["p_close"] == 10.05 and math.isnan(got["minutes_to_80"])
    assert got["gap_945"] == pytest.approx(0.06)


def test_missing_prices_leave_nan_not_errors() -> None:
    m = minutes([(D, "09:30", 10.0, 10.0), (D, "15:59", 10.0, 10.5)])
    got = cu.measure(m, D, PREV, NEXT)
    assert math.isnan(got["prev_close"]) and math.isnan(got["gap_945"])
    assert math.isnan(got["p_945"]) and math.isnan(got["rest"])
    assert math.isnan(got["p_next_945"])


def test_the_holdout_is_never_read() -> None:
    last, first_sealed = date(2026, 6, 30), date(2026, 7, 1)
    m = minutes([(last, "09:45", 10.0, 10.0), (first_sealed, "09:45", 99.0, 99.0)])
    got = cu.measure(m, last, clock.previous_session(last), first_sealed)
    assert got["p_945"] == 10.0
    assert math.isnan(got["p_next_945"]) and math.isnan(got["rest"])
    inside = cu.measure(m, first_sealed, last, clock.next_session(first_sealed))
    assert all(math.isnan(v) for v in inside.values())


# ---- which events ------------------------------------------------------------------------------


def news_rows(rows: list[tuple[str, pd.Timestamp]]) -> pd.DataFrame:
    """(symbol, known time) per article, placed on sessions as panel.load_news places them."""
    df = pd.DataFrame({"symbol": [s for s, _ in rows], "known_at": [t for _, t in rows]})
    df["known_at"] = pd.to_datetime(df["known_at"], utc=True)
    df["sidx"] = pn.to_session(df["known_at"], CUT)
    return df


def test_premarket_window_is_after_the_previous_close_and_by_9am() -> None:
    k = DAYS.index(D)
    mon = DAYS.index(date(2024, 3, 18))
    after_ec = DAYS.index(date(2024, 12, 2))
    news = news_rows(
        [
            ("AAA", ny(PREV, "15:30")),  # in the session's news, but before the close
            ("BBB", ny(PREV, "16:30")),  # after the close: pre-market
            ("BBB", ny(D, "08:59")),  # pre-market, later
            ("CCC", ny(D, "09:10")),  # after 9:00: the next session's, and intraday there
            ("DDD", ny(date(2024, 3, 16), "11:00")),  # Saturday: Monday's pre-market
            ("EEE", ny(date(2024, 11, 29), "14:00")),  # after a 1:00 pm early close
        ]
    )
    pre = cu.premarket_news(news, DAYS)
    got = {(r.symbol, r.sidx): (r.first_known, r.n_pre) for r in pre.itertuples()}
    assert set(got) == {("BBB", k), ("DDD", mon), ("EEE", after_ec)}
    assert got[("BBB", k)] == (ny(PREV, "16:30"), 2)
    assert ("CCC", k + 1) not in got


def test_prior_articles_matches_the_panel_coverage() -> None:
    times_ = [ny(DAYS[i], "08:00") for i in (3, 10, 10, 40, 69, 70, 71, 75, 75, 75, 140, 141, 200)]
    news = news_rows([("AAA", t) for t in times_] + [("BBB", ny(DAYS[70], "08:00"))])
    news["created_at"] = news["known_at"]
    news["headline"] = [f"story {i} about something new {i}" for i in range(len(news))]
    news["net"] = np.float32(0.1)
    day = pn.daily_news(news.sort_values(["symbol", "created_at"]).reset_index(drop=True), 250)
    from_cov = np.rint(np.expm1(day["coverage"].to_numpy()) * pn.BASELINE_SESSIONS)
    recount = cu.prior_articles(news, day[["symbol", "sidx"]])
    assert recount.tolist() == from_cov.tolist()
    # AAA on session 70: sessions 10 to 69 hold 2 + 1 + 1 = 4 articles; 3 is too old.
    at = day[(day["symbol"] == "AAA") & (day["sidx"] == 70)].index[0]
    assert recount[at] == 4
    keys = pd.DataFrame({"symbol": ["ZZZ", "BBB"], "sidx": [100, 70]})
    assert cu.prior_articles(news, keys).tolist() == [0, 0]


def test_small_and_thin_filters() -> None:
    thin, busy = math.log1p(2 / 60), math.log1p(3 / 60)
    panel = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E", "F", "G"],
            "sidx": [0, 0, 0, 0, 0, 0, 1],
            "log_adv": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0],
            "coverage": [thin, busy, thin, thin, 0.0, thin, 0.0],
        }
    )
    pre = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "G"],
            "sidx": [0, 0, 0, 0, 1],
            "first_known": pd.to_datetime(["2024-03-12 12:00"] * 5, utc=True),
            "n_pre": [1, 1, 1, 1, 1],
        }
    )
    got = cu.select_events(panel, pre)
    # Session 0's bottom third is log_adv <= 2.67 (A, B); B has 3 prior articles; C to F are
    # not small; G is alone in its session, so it is its own bottom third.
    assert got["symbol"].tolist() == ["A", "G"]
    assert got["prior_60"].tolist() == [2.0, 0.0]


# ---- a month from files --------------------------------------------------------------------------


def write_month(root: Path, month: str, rows: list[tuple[str, date, str, float]]) -> None:
    """(symbol, day, hh:mm, price) per bar, as float32 like the compacted files."""
    df = pd.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "t": [ny(r[1], r[2]).tz_convert("UTC") for r in rows],
            "o": np.array([r[3] for r in rows], dtype="float32"),
            "c": np.array([r[3] for r in rows], dtype="float32"),
        }
    ).sort_values(["symbol", "t"])
    df["t"] = df["t"].dt.as_unit("us")
    (root / month).mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / month / "part-000.parquet", index=False)


def test_measure_month_reads_edge_sessions_from_the_next_and_previous_months(
    tmp_path: Path,
) -> None:
    feb29, mar1 = date(2024, 2, 29), date(2024, 3, 1)
    mar27, mar28, apr1 = date(2024, 3, 27), date(2024, 3, 28), date(2024, 4, 1)  # Good Friday
    write_month(tmp_path, "2024-02", [("BBB", feb29, "15:59", 20.0)])
    write_month(
        tmp_path,
        "2024-03",
        [
            ("BBB", mar1, "09:29", 99.0),  # pre-market: never a regular price
            ("BBB", mar1, "09:45", 22.1),
            ("AAA", mar27, "15:59", 5.0),
            ("AAA", mar28, "09:45", 5.5),
            ("CCC", mar28, "09:45", 1.0),
        ],
    )
    write_month(tmp_path, "2024-04", [("AAA", apr1, "10:00", 6.05), ("BBB", apr1, "09:45", 1.0)])
    ev = pd.DataFrame(
        {
            "symbol": ["BBB", "AAA"],
            "session": [mar1, mar28],
            "first_known": [ny(mar1, "07:00"), ny(mar28, "06:30")],
        }
    )
    got = cu.measure_month(ev, tmp_path).set_index("symbol")
    assert got.loc["BBB", "prev_close"] == 20.0
    assert got.loc["BBB", "p_945"] == pytest.approx(22.1)  # float32 back to 4 decimals
    assert got.loc["BBB", "p_open"] == pytest.approx(22.1)  # 9:45 is within 30 minutes of 9:30
    assert got.loc["BBB", "gap_945"] == pytest.approx(22.1 / 20 - 1)
    assert got.loc["AAA", "p_next_945"] == pytest.approx(6.05)
    assert got.loc["AAA", "rest"] == pytest.approx(6.05 / 5.5 - 1)
    assert got.loc["AAA", "hours_before_open"] == pytest.approx(3.0)
    assert got.loc["BBB", "hours_before_open"] == pytest.approx(2.5)


def test_summary_splits_by_the_sign_of_the_gap() -> None:
    df = pd.DataFrame(
        {
            "session": [D, D, date(2023, 5, 1)],
            "gap_945": [0.10, 0.30, -0.2],
            "rest": [0.02, -0.02, 0.05],
            "minutes_to_80": [5.0, 7.0, math.nan],
            "prev_close": [10.0, 10.0, 10.0],
            "prev_close_daily": [10.0, 10.1, 10.0],
            "p_open": [1.0] * 3,
            "p_945": [1.0] * 3,
            "p_close": [1.0] * 3,
            "p_next_945": [1.0] * 3,
            "prior_60": [0.0, 1.0, 2.0],
            "prior_recount": [0.0, 1.0, 2.0],
        }
    )
    s = cu.summary(df)
    assert s["events_per_year"] == {2023: 1, 2024: 2}
    assert s["by_sign"]["up"]["events"] == 2
    assert s["by_sign"]["up"]["share_done"] == pytest.approx(0.4 / 0.4)
    assert s["by_sign"]["down"]["share_done"] == pytest.approx(-0.2 / -0.15)
    assert s["minutes_to_80_median"] == 6.0
