"""The weekly EIA numbers for trial 18, from the EIA API v2 (free; key in ``.env`` as
``EIA_API_KEY``), with a release time for each week and the 5-year average rebuilt point in
time.

Series (API v2 ``seriesid`` route, weekly):

- gas: ``NG.NW2_EPG0_SWO_R48_BCF.W``, working gas in underground storage, Lower 48, Bcf.
- oil: ``PET.WCESTUS1.W``, U.S. ending stocks of crude oil excluding the SPR, thousand
  barrels.

The week's change is this week's stocks less last week's.

**Expected (the 5-year average change).** EIA's own yardstick, rebuilt from its weekly
history: the average of the same week's change in each of the 5 prior years, where "the same
week" is the week whose period ends nearest 52 x k weeks before (k = 1 to 5, within 3 days).
Only those earlier years are used, so nothing after the report can reach it. The values are
the API's (today's, revised), not the figure printed that week; weekly revisions are rare
and small. The owner chose this on 2026-11-16 over third-party archives of the printed page.

**Release time.** EIA's API carries none, and EIA lists holiday shifts only for the current
year, so the time is a conservative calendar (owner's choice, 2026-11-16): oil on the
Wednesday and gas on the Thursday after the week ends, 10:30 am New York time; in a release
week with a U.S. federal holiday on or before the normal day, the release is taken as the
latest EIA's holiday shifts reach, noon on the next business day after the normal day, and
the week is marked ``holiday_week`` (18b does not trade it). The time can be late, never
early, so no number is used before EIA published it.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from stoa import envfile
from stoa.commod import LOGS, ROOT
from stoa.newsbrain.pace import log_progress, write_parquet

log = logging.getLogger(__name__)

API = "https://api.eia.gov/v2/seriesid/"
SERIES = {"gas": "NG.NW2_EPG0_SWO_R48_BCF.W", "oil": "PET.WCESTUS1.W"}
RELEASE_WEEKDAY = {"oil": 2, "gas": 3}  # Wednesday, Thursday
RELEASE_TIME = "10:30"
HOLIDAY_TIME = "12:00"
FIVE_YEARS = 5
MATCH_DAYS = 3
TABLE = ROOT / "eia" / "reports.parquet"
RAW = ROOT / "eia" / "raw"
HISTORY_START = "2008-01-01"  # five years before the first week trial 18 can use, and more
NOAA_TABLE = ROOT / "noaa" / "degree_days.parquet"


def fetch(series: str, key: str) -> pd.DataFrame:
    """Every weekly value of one series: period (week end) and value."""
    rows: list[dict[str, Any]] = []
    offset = 0
    with httpx.Client(timeout=60) as c:
        while True:
            r = c.get(
                API + series,
                params={
                    "api_key": key,
                    "start": HISTORY_START,
                    "offset": offset,
                    "length": 5000,
                    "sort[0][column]": "period",
                    "sort[0][direction]": "asc",
                },
            )
            r.raise_for_status()
            resp = r.json().get("response", {})
            data = resp.get("data", [])
            rows.extend(data)
            offset += len(data)
            if not data or offset >= int(resp.get("total", 0)):
                break
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "period": pd.to_datetime(df["period"]),
            "value": pd.to_numeric(df["value"], errors="coerce"),
        }
    )
    return out.dropna().drop_duplicates("period").sort_values("period").reset_index(drop=True)


def holidays(start: str = "2007-01-01", end: str = "2027-12-31") -> set[date]:
    return {d.date() for d in USFederalHolidayCalendar().holidays(start, end)}


def release(fuel: str, period: date, hol: set[date]) -> tuple[pd.Timestamp, bool]:
    """(release time in UTC, holiday week) by the conservative calendar."""
    monday = period + timedelta(days=(7 - period.weekday()) % 7 or 7)  # the Monday after
    normal = monday + timedelta(days=RELEASE_WEEKDAY[fuel])
    shifted = any(monday + timedelta(days=k) in hol for k in range(RELEASE_WEEKDAY[fuel] + 1))
    if not shifted:
        day, hhmm = normal, RELEASE_TIME
    else:
        day = normal + timedelta(days=1)
        while day.weekday() >= 5 or day in hol:
            day += timedelta(days=1)
        hhmm = HOLIDAY_TIME
    ts = pd.Timestamp(f"{day.isoformat()} {hhmm}", tz="America/New_York").tz_convert("UTC")
    return ts, shifted


def five_year(changes: pd.Series) -> pd.Series:
    """For each week (index: period end), the average change of the same week in each of the
    5 prior years; NaN unless all 5 are found."""
    idx = pd.DatetimeIndex(changes.index)
    vals = changes.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i, p in enumerate(idx):
        got = []
        for k in range(1, FIVE_YEARS + 1):
            target = p - pd.Timedelta(weeks=52 * k)
            j = int(np.argmin(np.abs((idx - target).days)))
            if abs((idx[j] - target).days) <= MATCH_DAYS and idx[j] < p and np.isfinite(vals[j]):
                got.append(vals[j])
        if len(got) == FIVE_YEARS:
            out[i] = float(np.mean(got))
    return pd.Series(out, index=changes.index)


def degree_days(
    periods: pd.Series, released: pd.Series, table: pd.DataFrame | None = None
) -> pd.DataFrame:
    """For each gas week (period ends Friday), NOAA's week ending the next day (Saturday),
    kept only if NOAA posted it before the release."""
    dd = (pd.read_parquet(NOAA_TABLE) if table is None else table).copy()
    dd["week_end"] = pd.to_datetime(dd["week_end"])
    out = []
    for p, rel in zip(pd.to_datetime(periods), released, strict=True):
        row = dd[dd["week_end"] == p + pd.Timedelta(days=1)]
        h = c = np.nan
        if len(row):
            r = row.iloc[0]
            for kind in ("hdd", "cdd"):
                posted = r[f"{kind}_posted"]
                if posted is None or pd.isna(posted):
                    continue
                at = pd.Timestamp(pd.to_datetime(str(posted), format="%d-%b-%Y %H:%M"))
                if at.tz_localize("America/New_York").tz_convert("UTC") < rel:
                    if kind == "hdd":
                        h = float(r["hdd_dev"])
                    else:
                        c = float(r["cdd_dev"])
        out.append({"hdd_dev": h, "cdd_dev": c})
    return pd.DataFrame(out)


def build(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hol = holidays()
    parts = []
    for fuel, df in raw.items():
        s = df.set_index("period")["value"]
        ch = s.diff()
        fy = five_year(ch)
        rel = [release(fuel, p.date(), hol) for p in s.index]
        t = pd.DataFrame(
            {
                "fuel": fuel,
                "period": s.index,
                "stocks": s.to_numpy(),
                "actual": ch.to_numpy(),
                "five_year": fy.to_numpy(),
                "released": [r[0] for r in rel],
                "holiday_week": [r[1] for r in rel],
            }
        )
        if fuel == "gas":
            t = pd.concat(
                [
                    t.reset_index(drop=True),
                    degree_days(t["period"], t["released"]).reset_index(drop=True),
                ],
                axis=1,
            )
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    return out.dropna(subset=["actual"]).reset_index(drop=True)


def pull() -> pd.DataFrame:
    envfile.need("EIA_API_KEY")
    key = os.environ["EIA_API_KEY"]
    raw = {}
    for fuel, series in SERIES.items():
        df = fetch(series, key)
        write_parquet(df, RAW / f"{fuel}.parquet")
        raw[fuel] = df
        log_progress(
            LOGS / "eia.jsonl",
            fuel=fuel,
            weeks=len(df),
            first=str(df["period"].min().date()),
            last=str(df["period"].max().date()),
        )
    table = build(raw)
    write_parquet(table, TABLE)
    return table


def load(path: Path = TABLE) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["released"] = pd.to_datetime(df["released"], utc=True)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    t = pull()
    print(t.groupby("fuel").size().to_dict())
