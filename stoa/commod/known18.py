"""Trial 18's known-answer harness: made-up weekly reports whose right answers are known,
pushed through the trial's own code (``eia.release``, ``eia.five_year``, ``eia.degree_days``
rules, ``trial18.surprises``, ``trades_next_open``, ``trades_intraday``).

1. Release calendar. A normal gas week releases Thursday 10:30 am; Thanksgiving week 2024
   (Thursday 2024-11-28) releases no earlier than noon Friday; an oil week with Monday a
   holiday (Labor Day 2024) releases no earlier than noon Thursday.
2. The 5-year average. Weekly stocks whose changes are known: the average for a week is the
   mean of the same week's changes in the 5 prior years exactly, and a wild change planted
   after the week (or in the same week of a later year) does not move it.
3. No number before its release. A planted report with a huge draw is released Thursday
   10:30 am. 18a must enter at Friday's open (never Thursday's), 18b at 10:35 am Thursday
   (never at the open), and a copy of the report marked released at 11:00 am must not trade
   at 10:35 am. The signal of every earlier report is the same with or without it.
4. Degree days count only when NOAA posted them before the release.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from stoa import clock
from stoa.commod import eia
from stoa.commod import trial18 as t18


def _ny(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")


def known_answer() -> dict[str, Any]:
    out: dict[str, Any] = {}
    hol = eia.holidays()

    # 1. the calendar
    normal, h1 = eia.release("gas", date(2024, 3, 8), hol)  # Friday week end
    thanks, h2 = eia.release("gas", date(2024, 11, 22), hol)
    labor, h3 = eia.release("oil", date(2024, 8, 30), hol)
    out["calendar"] = {
        "passed": normal == _ny("2024-03-14 10:30")
        and not h1
        and thanks == _ny("2024-11-29 12:00")
        and h2
        and labor == _ny("2024-09-05 12:00")
        and h3,
        "expected": "2024-03-14 10:30, 2024-11-29 12:00 (holiday), 2024-09-05 12:00 (holiday)",
        "actual": [str(normal), str(thanks), str(labor)],
    }

    # 2. the 5-year average
    periods = pd.date_range("2010-01-01", "2020-12-25", freq="7D")
    rng = np.random.default_rng(7)
    ch = pd.Series(rng.normal(0, 30, len(periods)), index=periods)
    fy = eia.five_year(ch)
    p = periods[-1]
    want = float(np.mean([ch.iloc[len(periods) - 1 - 52 * k] for k in range(1, 6)]))
    ch2 = ch.copy()
    ch2.iloc[-1] = 1e9  # the week itself
    fy2 = eia.five_year(ch2)
    out["five_year"] = {
        "passed": bool(
            abs(fy.loc[p] - want) < 1e-9
            and fy2.iloc[:-1].equals(fy.iloc[:-1])
            and abs(fy2.loc[p] - want) < 1e-9
        ),
        "expected": f"{want:.6f}, unmoved by the week's own change",
        "actual": [float(fy.loc[p]), float(fy2.loc[p])],
    }

    # 3. no number before its release
    n = 120
    cal = [
        eia.release("gas", d.date(), hol) for d in pd.date_range("2022-01-07", periods=n, freq="7D")
    ]
    rep = pd.DataFrame(
        {
            "released": [c[0] for c in cal],
            "actual": rng.normal(0, 10, n),
            "five_year": 0.0,
            "hdd_dev": 0.0,
            "cdd_dev": 0.0,
            "holiday_week": [c[1] for c in cal],
        }
    )
    k = next(i for i in range(n - 10, n) if not cal[i][1])  # a Thursday 10:30 am release
    rep.loc[k, "actual"] = -500.0  # a huge draw: bullish
    sig = t18.surprises(rep)
    day = pd.Timestamp(str(sig["released"].iloc[k])).tz_convert("America/New_York").date()
    sessions = clock.sessions_between(date(2022, 1, 3), date(2024, 6, 28))
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in sessions])
    bars = pd.DataFrame({"open": 10.0, "close": 10.0, "volume": 1e8}, index=idx)
    nxt = t18.trades_next_open(sig.iloc[: k + 1], "gas", "UNG", bars, date(2024, 6, 28))
    late = [t for t in nxt if t.released == sig.loc[k, "released"]]

    def opens(_s: str, _d: date, hhmm: str) -> float:
        return 10.0 if hhmm == t18.INTRADAY_ENTRY else float("nan")

    intra = t18.trades_intraday(sig.iloc[: k + 1], "gas", "UNG", bars, opens, date(2024, 6, 28))
    moved = sig.iloc[: k + 1].copy()
    moved.loc[k, "released"] = pd.Timestamp(f"{day} 11:00", tz="America/New_York").tz_convert("UTC")
    intra_late = t18.trades_intraday(moved, "gas", "UNG", bars, opens, date(2024, 6, 28))
    without = t18.surprises(rep.drop(index=k).reset_index(drop=True))
    same_before = np.allclose(
        sig.loc[: k - 1, "z"].to_numpy(dtype=float),
        without.loc[: k - 1, "z"].to_numpy(dtype=float),
        equal_nan=True,
    )
    out["no_early_use"] = {
        "passed": bool(
            sig.loc[k, "bullish"]
            and len(late) == 1
            and late[0].entry_day == clock.next_session(day)
            and any(t.entry_day == day for t in intra)
            and not any(t.entry_day == day for t in intra_late)
            and same_before
        ),
        "expected": f"18a enters {clock.next_session(day)} (after the {day} 10:30 release); 18b "
        f"enters 10:35 on {day}; a release at 11:00 is not traded at 10:35; earlier signals "
        "unchanged",
        "actual": {
            "18a": [str(t.entry_day) for t in late],
            "18b": [str(t.entry_day) for t in intra if t.entry_day == day],
            "18b_late": [str(t.entry_day) for t in intra_late if t.entry_day == day],
            "earlier_signals_unchanged": bool(same_before),
        },
    }

    # 4. degree days only if posted before the release (posted Monday 7 am; a release before
    #    that Monday must not see them)
    periods4 = pd.Series([pd.Timestamp("2024-01-05")])
    noaa = pd.DataFrame(
        {
            "week_end": [pd.Timestamp("2024-01-06")],
            "hdd": [200],
            "hdd_dev": [25],
            "hdd_posted": ["08-Jan-2024 06:59"],
            "cdd": [0],
            "cdd_dev": [0],
            "cdd_posted": ["08-Jan-2024 06:59"],
        }
    )
    ok = eia.degree_days(periods4, pd.Series([_ny("2024-01-11 10:30")]), noaa)
    early = eia.degree_days(periods4, pd.Series([_ny("2024-01-07 10:30")]), noaa)
    h_ok = float(ok["hdd_dev"].to_numpy(dtype=float)[0])
    h_early = float(early["hdd_dev"].to_numpy(dtype=float)[0])
    out["degree_days_known_at"] = {
        "passed": bool(np.isfinite(h_ok) and np.isnan(h_early)),
        "expected": "the 2024-01-06 week's departure used by the 2024-01-11 release and not by a "
        "release before it was posted",
        "actual": [h_ok, h_early],
    }
    out["verdict"] = "honest" if all(v["passed"] for v in out.values()) else "bug"
    return out
