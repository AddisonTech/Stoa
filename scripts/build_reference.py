"""Rebuild the vendored reference files from their upstream sources.

    python scripts/build_reference.py <path to a clone of github.com/fja05680/sp500>

Writes:
  stoa/reference/sp500_membership.csv  symbol,start,end  (end is the first date the
                                       symbol was no longer a member; blank if still in)
  stoa/reference/sectors.csv           symbol,sector  (GICS sector of current members,
                                       plus the ETFs Stoa trades)

The membership history is the "S&P 500 Historical Components & Changes (Updated).csv"
file from fja05680/sp500 (MIT license). Each row there is a full member list on a
change date; a symbol's interval runs from the first row that lists it to the first
later row that does not.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "stoa" / "reference"
HISTORY = "S&P 500 Historical Components & Changes (Updated).csv"
SINCE = "2015-01-01"  # intervals that ended before this date are not needed
ETFS = {"SPY": "ETF", "MTUM": "ETF", "QQQ": "Information Technology"}


def intervals(rows: list[tuple[str, set[str]]]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    open_: dict[str, str] = {}
    for day, members in rows:
        for sym in sorted(members - set(open_)):
            open_[sym] = day
        for sym in sorted(set(open_) - members):
            out.append((sym, open_.pop(sym), day))
    out.extend((sym, start, "") for sym, start in open_.items())
    return sorted(i for i in out if i[2] == "" or i[2] >= SINCE)


def main(src: Path) -> int:
    with (src / HISTORY).open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        rows = [(r[0], {t.strip() for t in r[1].split(",") if t.strip()}) for r in reader]
    rows.sort()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "sp500_membership.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["symbol", "start", "end"])
        w.writerows(intervals(rows))
    sectors = dict(ETFS)
    with (src / "sp500.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sectors.setdefault(r["Symbol"].strip(), r["GICS Sector"].strip())
    with (OUT / "sectors.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["symbol", "sector"])
        w.writerows(sorted(sectors.items()))
    print(f"membership through {rows[-1][0]}, {len(sectors)} sectors written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
