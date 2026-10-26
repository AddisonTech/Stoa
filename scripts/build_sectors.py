"""Build stoa/reference/sectors_wide.csv: a sector for every wide-universe symbol.

    python scripts/build_sectors.py

The sector comes from the company's SIC code in its SEC submissions file, mapped
to the GICS sector names in stoa/reference/sectors.csv (membership.SIC_SECTORS).
S&P 500 members keep their GICS sector from sectors.csv, which wins at lookup.
Inputs, all git-ignored: the wide bar files (data/bars/wide/*.parquet, one per
symbol), the symbol to CIK map from the news brain SEC pull
(data/newsbrain/sec/ciks.parquet) and the SEC submissions cache (data/sec/). A
submissions file missing from the cache is fetched at 2 requests a second, so this
can run beside the SEC pull and stay under the SEC's 10 a second.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stoa.data.sec import make_sec_client  # noqa: E402
from stoa.membership import UNKNOWN_SECTOR, sic_sector  # noqa: E402
from stoa.newsbrain.sec_pull import USER_AGENT  # noqa: E402

OUT = ROOT / "stoa" / "reference" / "sectors_wide.csv"


def main() -> int:
    universe = sorted(p.stem for p in (ROOT / "data" / "bars" / "wide").glob("*.parquet"))
    ciks = pd.read_parquet(ROOT / "data" / "newsbrain" / "sec" / "ciks.parquet")
    cik_of = {str(s): int(c) for s, c in zip(ciks["symbol"], ciks["cik"], strict=True)}
    sec = make_sec_client(USER_AGENT, ROOT / "data" / "sec", 2.0)
    rows = []
    for sym in universe:
        cik = cik_of.get(sym)
        if cik is None:
            continue
        body = sec.get_json(
            f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
            f"submissions/CIK{cik:010d}.json",
        )
        sic = str(body.get("sic") or "") if isinstance(body, dict) else ""
        sector = sic_sector(sic)
        if sector != UNKNOWN_SECTOR:
            rows.append((sym, sector, sic))
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["symbol", "sector", "sic"])
        w.writerows(rows)
    counts = Counter(r[1] for r in rows)
    print(f"{len(rows)} of {len(universe)} symbols have a sector ({len(cik_of)} with a CIK)")
    for name, n in counts.most_common():
        print(f"  {name}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
