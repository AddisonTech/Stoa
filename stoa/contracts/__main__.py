"""python -m stoa.contracts <step>: the trial 16 pipeline, one step at a time."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stoa.contracts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pull", help="pull the announcement archive from war.gov (resumable)")
    ex = sub.add_parser("ex21", help="pull Exhibit 21 subsidiary lists from EDGAR (resumable)")
    ex.add_argument("--part", type=int, default=0)
    ex.add_argument("--parts", type=int, default=1)
    tg = sub.add_parser("targeted", help="EDGAR full-text search for unmatched awardee names")
    tg.add_argument("--reverse", action="store_true", help="work the list from its end")
    fd = sub.add_parser("docs", help="fetch the exhibits the cached searches found")
    fd.add_argument("--part", type=int, default=0)
    fd.add_argument("--parts", type=int, default=1)
    sub.add_parser("parse", help="read every cached page into data/contracts/awards.parquet")
    sub.add_parser("match", help="map awardees to tickers into data/contracts/matched.parquet")
    sub.add_parser("facts", help="pull company facts for matched companies (1 a second)")
    sub.add_parser("panel", help="features and labels into data/contracts/panel.parquet")
    sub.add_parser("preregister", help="write trial 16's rules to the registry, once")
    sub.add_parser("trial", help="harness, tuning on 2016 to 2022, then the one test")
    args = ap.parse_args(argv)
    if args.cmd in ("preregister", "trial"):
        from pathlib import Path

        import pandas as pd

        from stoa.config import load_settings
        from stoa.contracts import data, trial16
        from stoa.experiments import Registry

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        registry = Registry(load_settings(Path("config/default.yaml")).registry_path)
        if args.cmd == "preregister":
            wrote = trial16.preregister(registry)
            print(json.dumps({"written": wrote, "spec_hash": trial16.spec_hash()}))
            return 0
        out = trial16.RESULTS / data.stamp()
        try:
            rep = trial16.run(pd.read_parquet(data.PANEL), registry, out)
        except trial16.Stop as exc:
            print(json.dumps({"stopped": str(exc), "out": str(out)}))
            return 3
        print(json.dumps({"verdict": rep["test"]["verdict"], "out": str(out)}))
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd == "docs":
        import pandas as pd

        from stoa.contracts import subsidiaries, tickers

        universe = {int(c) for c in pd.read_parquet(tickers.CIKS)["cik"]}
        print(json.dumps(subsidiaries.fetch_found(universe, args.part, args.parts)))
        return 0
    if args.cmd == "targeted":
        import pandas as pd

        from stoa.contracts import data, subsidiaries, tickers

        m = pd.read_parquet(data.MATCHED)
        # Names not placed, and names placed only by a shortened name: an exact subsidiary
        # list ("Tapestry Solutions, Inc." in Boeing's) must get the chance to win over a
        # prefix guess (Tapestry, Inc., the handbag company).
        unsure = (m["miss"] == "unmatched") | (m["how"] == "direct_prefix")
        miss = m[unsure & m["group"].isin(["new", "mod"])]
        found: dict[str, None] = {}
        for company in sorted(set(miss["company"].astype(str))):
            for key, exact in tickers.candidates(company):
                if exact and len(key) >= 6:
                    found.setdefault(key)
        keys = list(found)[:: -1 if args.reverse else 1]
        universe = {int(c) for c in pd.read_parquet(tickers.CIKS)["cik"]}
        print(json.dumps(subsidiaries.pull_targeted(keys, universe)))
        return 0
    if args.cmd in ("parse", "match", "facts", "panel"):
        import pandas as pd

        from stoa.contracts import data, tickers

        if args.cmd == "parse":
            df = data.parse_all()
            print(json.dumps({"rows": len(df), "pages": int(df["article"].nunique())}))
            return 0
        if args.cmd == "facts":
            m = pd.read_parquet(data.MATCHED)
            print(json.dumps({"pulled": data.pull_facts(m["cik"].dropna().astype(int))}))
            return 0
        book = tickers.build()
        if args.cmd == "match":
            m = data.match_all(pd.read_parquet(data.AWARDS), book)
            m.to_parquet(data.MATCHED, index=False)
            print(json.dumps({"rows": len(m), "matched": int(m["symbol"].notna().sum())}))
            return 0
        p = data.build_panel(pd.read_parquet(data.MATCHED), book)
        p.to_parquet(data.PANEL, index=False)
        print(json.dumps({"rows": len(p)}))
        return 0
    if args.cmd == "ex21":
        from stoa.contracts.subsidiaries import pull_ex21

        pull_ex21(part=args.part, parts=args.parts, per_second=8.0 / args.parts)
        return 0
    if args.cmd == "pull":
        from stoa.contracts.scrape import pull

        status = pull()
        print(json.dumps(status, default=str))
        return 0 if status["status"] == "ok" else 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
