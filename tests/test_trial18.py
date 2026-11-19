"""Trial 18: the surprise signal uses only earlier reports, and trades start after release."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stoa.commod import trial18 as t18


def reports(n: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    released = pd.date_range("2016-01-07 15:30", periods=n, freq="7D", tz="UTC")
    hdd = rng.normal(0, 20, n)
    cdd = rng.normal(0, 5, n)
    five = rng.normal(0, 50, n)
    actual = five - 1.5 * hdd - 0.5 * cdd + rng.normal(0, 10, n)
    return pd.DataFrame(
        {"released": released, "actual": actual, "five_year": five, "hdd_dev": hdd, "cdd_dev": cdd}
    )


def test_a_later_report_never_changes_an_earlier_signal() -> None:
    base = t18.surprises(reports())
    later = reports()
    later.loc[120:, "actual"] += 1e6  # a wild future
    moved = t18.surprises(later)
    cols = ["expected", "surprise", "scale", "z", "b_hdd", "b_cdd"]
    pd.testing.assert_frame_equal(base.loc[:119, cols], moved.loc[:119, cols])


def test_weather_coefficients_are_learned_and_scale_waits_for_history() -> None:
    s = t18.surprises(reports())
    assert s["scale"].iloc[: t18.MIN_HISTORY].isna().all()
    assert not s["bullish"].iloc[: t18.MIN_HISTORY].any()
    assert abs(s["b_hdd"].iloc[-1] + 1.5) < 0.2 and abs(s["b_cdd"].iloc[-1] + 0.5) < 0.5


def test_entry_is_the_first_open_after_release() -> None:
    thu = pd.Timestamp("2024-03-14 10:30", tz="America/New_York").tz_convert("UTC")
    assert t18.entry_session(thu) == date(2024, 3, 15)
    early = pd.Timestamp("2024-03-14 09:00", tz="America/New_York").tz_convert("UTC")
    assert t18.entry_session(early) == date(2024, 3, 14)


def test_reports_before_the_window_feed_history_but_never_trade() -> None:
    """EIA history reaches back decades, before the trading calendar starts: those reports
    feed the averages and the scaling only, and no trade function looks them up."""
    from stoa import clock

    rel = pd.to_datetime(["1988-11-16 15:30", "2010-01-07 15:30", "2016-03-10 15:30"], utc=True)
    sig = pd.DataFrame({"released": rel, "z": [-3.0, -3.0, -3.0], "bullish": True})
    days = clock.sessions_between(date(2016, 1, 4), date(2016, 4, 29))
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    bars = pd.DataFrame({"open": 10.0, "close": 10.0, "volume": 1e8}, index=idx)
    a = t18.trades_next_open(sig, "gas", "UNG", bars, date(2016, 4, 29))
    b = t18.trades_intraday(sig, "gas", "UNG", bars, lambda *_: 10.0, date(2016, 4, 29))
    assert [t.entry_day for t in a] == [date(2016, 3, 11)]
    assert [t.entry_day for t in b] == [date(2016, 3, 10)]


def test_known_answer_is_honest() -> None:
    from stoa.commod.known18 import known_answer

    res = known_answer()
    failed = {k: v for k, v in res.items() if isinstance(v, dict) and not v["passed"]}
    assert res["verdict"] == "honest", failed


def test_known_answer_catches_entering_at_the_release_day_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The usual bug: buying at the open of the release day, before a 10:30 am release."""
    from stoa.commod.known18 import known_answer

    monkeypatch.setattr(
        t18, "entry_session", lambda ts: pd.Timestamp(ts).tz_convert("America/New_York").date()
    )
    res = known_answer()
    assert res["verdict"] == "bug" and not res["no_early_use"]["passed"]


def test_conservative_calendar_never_earlier_than_the_normal_time() -> None:
    from stoa.commod import eia

    hol = eia.holidays()
    for p in pd.date_range("2016-01-01", "2026-11-06", freq="7D"):
        for fuel in ("gas", "oil"):
            ts, shifted = eia.release(fuel, p.date(), hol)
            ny = ts.tz_convert("America/New_York")
            assert ny.weekday() == eia.RELEASE_WEEKDAY[fuel] or shifted
            assert (ny.hour, ny.minute) == ((12, 0) if shifted else (10, 30))
            assert ny.date() > p.date()


def test_trial_log_lists_every_trial_on_the_dashboard() -> None:
    from stoa.dashboard import render_trial_log
    from stoa.trial_log import rows

    html = render_trial_log()
    numbers = [t.number for t in rows()]
    assert numbers[0] == "1 to 5" and "16" in numbers
    assert all(t.verdict for t in rows())
    assert "Defense contract awards" in html and "<table" in html
