"""Weekly population-weighted heating and cooling degree days for the United States, from
NOAA's Climate Prediction Center (free, no key).

Source: the CPC's weekly state summaries,
``ftp.cpc.ncep.noaa.gov/htdocs/products/analysis_monitoring/cdus/degree_days/archives/``
under ``Heating degree Days/weekly states/<year>/`` and ``Cooling Degree Days/weekly
cooling degree days state/<year>/``, one text file a week (two weeks a kind, 2022-01-01
and 2023-07-01, are posted without the national line and stay missing). Each names the
last day of its week (a Saturday) and gives, for the population-weighted national average
("UNITED STATES"), the week's degree days and their departure from normal and from last
year.

Every file is cached raw under ``data/commod/noaa/raw/``; the table
``data/commod/noaa/degree_days.parquet`` has one row per week: ``week_end``, ``hdd``,
``hdd_dev`` (departure from normal), ``cdd``, ``cdd_dev``, and the archive's own time stamp
for each file (``hdd_posted``, ``cdd_posted``, as the directory listing shows it).
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote, unquote

import httpx
import pandas as pd

from stoa.commod import LOGS, ROOT
from stoa.newsbrain.pace import log_progress, write_parquet

log = logging.getLogger(__name__)

BASE = (
    "https://ftp.cpc.ncep.noaa.gov/htdocs/products/analysis_monitoring/cdus/degree_days/archives/"
)
KINDS = {
    "hdd": "Heating degree Days/weekly states",
    "cdd": "Cooling Degree Days/weekly cooling degree days state",
}
RAW = ROOT / "noaa" / "raw"
TABLE = ROOT / "noaa" / "degree_days.parquet"
USER_AGENT = "StoaResearch/1.0"
_ROW = re.compile(
    r'href="([^"?/][^"]*\.txt)">[^<]*</a></td><td[^>]*>\s*(\d{2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2})'
)
_LAST = re.compile(r"LAST DATE OF DATA COLLECTION PERIOD IS\s+([A-Z]{3})\s+(\d{1,2}),\s+(\d{4})")
_US = re.compile(r"^\s*UNITED STATES\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)", re.M)


def listing(client: httpx.Client, kind: str, year: int) -> list[tuple[str, str]]:
    url = f"{BASE}{quote(KINDS[kind])}/{year}/"
    r = client.get(url)
    if r.status_code != 200:
        return []
    return [(unquote(name), stamp) for name, stamp in _ROW.findall(r.text)]


def parse(text: str) -> dict[str, object] | None:
    """The week's last day and the national population-weighted line (first such line:
    the population weighting, before the gas-customer weighting)."""
    m = _LAST.search(text)
    u = _US.search(text)
    if not m or not u:
        return None
    week_end = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y").date()
    return {"week_end": week_end, "value": int(u.group(1)), "dev": int(u.group(2))}


def pull(start_year: int = 2014, end_year: int | None = None) -> pd.DataFrame:
    end_year = end_year or date.today().year
    client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True)
    rows: dict[tuple[str, date], dict[str, object]] = {}
    progress = LOGS / "noaa.jsonl"
    for kind, folder in KINDS.items():
        for year in range(start_year, end_year + 1):
            files = listing(client, kind, year)
            time.sleep(1.0)
            for name, stamp in files:
                path = RAW / kind / str(year) / name.replace(",", "").replace(" ", "_")
                if not path.exists():
                    url = f"{BASE}{quote(folder)}/{year}/{quote(name)}"
                    r = client.get(url)
                    time.sleep(1.0)
                    if r.status_code != 200:
                        log_progress(progress, kind=kind, file=name, status=r.status_code)
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(r.text, encoding="utf-8")
                got = parse(path.read_text(encoding="utf-8"))
                if got is None:
                    log_progress(progress, kind=kind, file=name, status="unparsed")
                    continue
                rows[(kind, got["week_end"])] = {**got, "posted": stamp}  # type: ignore[index]
            log_progress(progress, kind=kind, year=year, files=len(files))
    table = table_of(rows)
    write_parquet(table, TABLE)
    log_progress(progress, finished=True, weeks=len(table))
    return table


def table_of(rows: dict[tuple[str, date], dict[str, object]]) -> pd.DataFrame:
    weeks = sorted({w for _k, w in rows})
    out = []
    for w in weeks:
        h, c = rows.get(("hdd", w)), rows.get(("cdd", w))
        out.append(
            {
                "week_end": pd.Timestamp(w),
                "hdd": h["value"] if h else None,
                "hdd_dev": h["dev"] if h else None,
                "hdd_posted": h["posted"] if h else None,
                "cdd": c["value"] if c else None,
                "cdd_dev": c["dev"] if c else None,
                "cdd_posted": c["posted"] if c else None,
            }
        )
    return pd.DataFrame(out)


def load(path: Path = TABLE) -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(pull().tail())
