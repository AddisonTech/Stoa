"""Point-in-time visibility, survivorship-free membership and the event calendars."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from stoa import clock, events, pit
from stoa.membership import Interval, Membership, default_membership, sector_of


def test_decision_cutoff_is_the_official_close() -> None:
    assert pit.decision_cutoff(date(2024, 6, 3)) == datetime(2024, 6, 3, 20, 0, tzinfo=UTC)
    # Early close the day after Thanksgiving: 1 pm ET.
    assert pit.decision_cutoff(date(2024, 11, 29)) == datetime(2024, 11, 29, 18, 0, tzinfo=UTC)


def test_record_known_after_cutoff_is_invisible() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "value": [1.0, 2.0, 3.0],
            "known_at": [
                datetime(2024, 6, 3, 12, 0, tzinfo=UTC),  # morning: visible
                datetime(2024, 6, 3, 20, 0, tzinfo=UTC),  # exactly at the close: visible
                datetime(2024, 6, 3, 20, 0, 1, tzinfo=UTC),  # one second late: invisible
            ],
        }
    )
    cut = pit.decision_cutoff(date(2024, 6, 3))
    assert list(pit.visible(frame, cut)["value"]) == [1.0, 2.0]
    assert pit.latest_visible(frame, cut).loc["AAPL", "value"] == 2.0
    nxt = pit.decision_cutoff(date(2024, 6, 4))
    assert pit.latest_visible(frame, nxt).loc["AAPL", "value"] == 3.0


def test_date_only_records_wait_for_the_next_close() -> None:
    stamp = pit.end_of_day(date(2024, 6, 3))
    frame = pd.DataFrame({"symbol": ["X"], "value": [1.0], "known_at": [stamp]})
    assert pit.visible(frame, pit.decision_cutoff(date(2024, 6, 3))).empty
    assert len(pit.visible(frame, pit.decision_cutoff(date(2024, 6, 4)))) == 1


def test_naive_cutoff_and_missing_column_fail_loudly() -> None:
    frame = pd.DataFrame({"symbol": ["X"], "known_at": [datetime(2024, 1, 2, tzinfo=UTC)]})
    with pytest.raises(ValueError):
        pit.visible(frame, datetime(2024, 1, 3))
    with pytest.raises(KeyError):
        pit.visible(pd.DataFrame({"symbol": ["X"]}), pit.decision_cutoff(date(2024, 1, 3)))


def test_asof_panel_matches_latest_visible_at_every_session() -> None:
    rows = []
    for i, d in enumerate(clock.sessions_between(date(2024, 1, 2), date(2024, 3, 28))[::7]):
        rows.append({"symbol": "A", "value": float(i), "known_at": pit.end_of_day(d)})
        rows.append({"symbol": "B", "value": 100.0 + i, "known_at": pit.decision_cutoff(d)})
    frame = pd.DataFrame(rows)
    sessions = clock.sessions_between(date(2024, 1, 2), date(2024, 4, 5))
    panel = pit.asof_panel(frame, sessions, ["value"])
    for s in sessions:
        expect = pit.latest_visible(frame, pit.decision_cutoff(s))
        for sym in ("A", "B"):
            got = panel.loc[(pd.Timestamp(s), sym), "value"]
            if sym in expect.index:
                assert got == expect.loc[sym, "value"]
            else:
                assert pd.isna(got)


def test_survivorship_delisted_name_in_universe_before_and_not_after() -> None:
    m = Membership(
        [
            Interval("LIVE", date(2010, 1, 4), None),
            Interval("GONE", date(2012, 3, 1), date(2020, 5, 4)),
            Interval("BACK", date(2011, 1, 3), date(2015, 1, 2)),
            Interval("BACK", date(2018, 6, 1), None),
        ]
    )
    assert m.members(date(2020, 5, 1)) == {"LIVE", "GONE", "BACK"}
    assert m.members(date(2020, 5, 4)) == {"LIVE", "BACK"}
    assert m.members(date(2016, 1, 4)) == {"LIVE", "GONE"}
    assert m.left_between(date(2020, 1, 1), date(2020, 12, 31)) == {"GONE"}
    assert m.left_between(date(2014, 1, 1), date(2019, 1, 1)) == set()  # BACK came back
    assert m.ever_members(date(2019, 1, 1), date(2021, 1, 1)) == {"LIVE", "GONE", "BACK"}


def test_vendored_membership_has_real_delistings() -> None:
    m = default_membership()
    # Silicon Valley Bank failed in March 2023 and left the index on 2023-03-15.
    assert m.is_member("SIVB", date(2023, 3, 14))
    assert not m.is_member("SIVB", date(2023, 3, 15))
    assert m.is_member("TWTR", date(2021, 6, 1)) and not m.is_member("TWTR", date(2023, 1, 3))
    for d in (date(2016, 1, 4), date(2020, 3, 2), date(2024, 7, 1), date(2026, 8, 18)):
        assert 495 <= len(m.members(d)) <= 510
    assert sector_of("NVDA") == "Information Technology"
    assert sector_of("SIVB") == "Unknown"
    assert sector_of("SPY") == "ETF"


def test_event_calendars_cover_2016_to_2027() -> None:
    lo, hi = events.coverage()
    assert lo.year == 2016 and hi.year == 2027
    assert len(events.fomc_days()) >= 8 * 12
    assert len(events.cpi_days()) >= 11 * 12
    assert date(2026, 10, 28) in events.fomc_days()
    assert date(2026, 10, 14) in events.cpi_days()
    assert date(2025, 11, 13) not in events.cpi_days()  # cancelled in the 2025 lapse


def test_go_light_on_event_day_and_the_day_before() -> None:
    # FOMC statement on Wednesday 2026-10-28: decide Tue 27th (execute on the 28th)
    # and decide Wed 28th (the day itself) are both light. Thursday is not.
    assert events.light_reason(date(2026, 10, 27)) == "FOMC 2026-10-28"
    assert events.light_reason(date(2026, 10, 28)) == "FOMC 2026-10-28"
    assert events.light_reason(date(2026, 10, 29)) is None
    assert events.light_reason(date(2026, 10, 13)) == "CPI 2026-10-14"
    # The Sunday 2020-03-15 emergency cut lands on Monday the 16th.
    assert "FOMC 2020-03-15" in (events.light_reason(date(2020, 3, 13)) or "")
