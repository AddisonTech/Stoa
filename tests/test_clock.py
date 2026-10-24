from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from stoa import clock

AFTER = 30
END = "23:59"
OPEN_AFTER = 15
WINDOW = 120


def utc(y: int, m: int, d: int, hh: int, mm: int) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


def decide(ts: datetime) -> date | None:
    return clock.decide_session(ts, AFTER, END)


def execute(ts: datetime) -> date | None:
    return clock.execute_session(ts, OPEN_AFTER, WINDOW)


@pytest.mark.parametrize(
    "day",
    [
        date(2026, 11, 26),
        date(2026, 12, 25),
        date(2026, 4, 3),
        date(2026, 1, 1),
        date(2026, 10, 24),
    ],
)
def test_holidays_and_weekends_are_not_sessions(day: date) -> None:
    assert not clock.is_session(day)


def test_early_close_day_after_thanksgiving() -> None:
    s = clock.session(date(2026, 11, 27))
    assert s.early_close
    assert clock.to_ny(s.close).hour == 13
    assert not clock.session(date(2026, 11, 25)).early_close


def test_decide_window_edt_both_slots() -> None:
    # 2026-06-15 is a Monday in EDT: close 20:00 UTC, window opens 20:30 UTC.
    assert decide(utc(2026, 6, 15, 21, 20)) == date(2026, 6, 15)
    assert decide(utc(2026, 6, 15, 22, 20)) == date(2026, 6, 15)
    assert decide(utc(2026, 6, 15, 20, 29)) is None
    assert decide(utc(2026, 6, 15, 20, 30)) == date(2026, 6, 15)


def test_decide_window_est_both_slots() -> None:
    # 2026-12-14 is a Monday in EST: close 21:00 UTC, window opens 21:30 UTC.
    assert decide(utc(2026, 12, 14, 21, 20)) is None
    assert decide(utc(2026, 12, 14, 22, 20)) == date(2026, 12, 14)


def test_decide_window_ends_at_midnight_ny() -> None:
    # 23:59 ET in EDT is 03:59 UTC the next day, still the same NY session date.
    assert decide(utc(2026, 6, 16, 3, 59)) == date(2026, 6, 15)
    assert decide(utc(2026, 6, 16, 4, 1)) is None


def test_decide_on_early_close_opens_earlier() -> None:
    # 2026-11-27 closes 13:00 ET = 18:00 UTC; window from 18:30 UTC.
    assert decide(utc(2026, 11, 27, 18, 31)) == date(2026, 11, 27)
    assert decide(utc(2026, 11, 27, 21, 20)) == date(2026, 11, 27)


def test_decide_skips_holidays() -> None:
    assert decide(utc(2026, 11, 26, 22, 20)) is None
    assert decide(utc(2026, 12, 25, 22, 20)) is None
    assert decide(utc(2026, 4, 3, 22, 20)) is None


def test_execute_window_edt_both_slots() -> None:
    # EDT open 13:30 UTC; window 13:45 to 15:45 UTC.
    assert execute(utc(2026, 6, 16, 13, 50)) == date(2026, 6, 16)
    assert execute(utc(2026, 6, 16, 14, 50)) == date(2026, 6, 16)
    assert execute(utc(2026, 6, 16, 13, 44)) is None
    assert execute(utc(2026, 6, 16, 15, 46)) is None


def test_execute_window_est_both_slots() -> None:
    # EST open 14:30 UTC; window 14:45 to 16:45 UTC. The 13:50 slot is before the open.
    assert execute(utc(2026, 12, 15, 13, 50)) is None
    assert execute(utc(2026, 12, 15, 14, 50)) == date(2026, 12, 15)


def test_dst_transition_march() -> None:
    # DST starts Sunday 2026-03-08. Friday 03-06 is EST, Monday 03-09 is EDT.
    assert execute(utc(2026, 3, 6, 13, 50)) is None
    assert execute(utc(2026, 3, 6, 14, 50)) == date(2026, 3, 6)
    assert execute(utc(2026, 3, 9, 13, 50)) == date(2026, 3, 9)
    assert decide(utc(2026, 3, 6, 21, 20)) is None
    assert decide(utc(2026, 3, 9, 21, 20)) == date(2026, 3, 9)


def test_dst_transition_november() -> None:
    # DST ends Sunday 2026-11-01. Friday 10-30 is EDT, Monday 11-02 is EST.
    assert execute(utc(2026, 10, 30, 13, 50)) == date(2026, 10, 30)
    assert execute(utc(2026, 11, 2, 13, 50)) is None
    assert execute(utc(2026, 11, 2, 14, 50)) == date(2026, 11, 2)
    assert decide(utc(2026, 10, 30, 21, 20)) == date(2026, 10, 30)
    assert decide(utc(2026, 11, 2, 21, 20)) is None
    assert decide(utc(2026, 11, 2, 22, 20)) == date(2026, 11, 2)


def test_execute_skips_holidays() -> None:
    assert execute(utc(2026, 11, 26, 14, 50)) is None
    assert execute(utc(2026, 12, 25, 14, 50)) is None
    assert execute(utc(2026, 4, 3, 13, 50)) is None


def test_next_and_previous_session_skip_holidays() -> None:
    assert clock.next_session(date(2026, 4, 2)) == date(2026, 4, 6)  # Good Friday
    assert clock.previous_session(date(2026, 11, 27)) == date(2026, 11, 25)  # Thanksgiving
    assert clock.next_session(date(2026, 12, 24)) == date(2026, 12, 28)


def test_last_closed_session() -> None:
    assert clock.last_closed_session(utc(2026, 6, 15, 19, 0)) == date(2026, 6, 12)
    assert clock.last_closed_session(utc(2026, 6, 15, 20, 0)) == date(2026, 6, 15)
    assert clock.last_closed_session(utc(2026, 11, 26, 15, 0)) == date(2026, 11, 25)


def test_sessions_after_counts_sessions_not_days() -> None:
    assert clock.sessions_after(date(2026, 4, 2), date(2026, 4, 6)) == 1
    assert clock.sessions_after(date(2026, 4, 6), date(2026, 4, 2)) == 0
    assert clock.sessions_after(date(2026, 6, 1), date(2026, 6, 1)) == 0


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError):
        clock.to_ny(datetime(2026, 1, 2, 10, 0))
