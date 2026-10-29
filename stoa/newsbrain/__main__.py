"""Run one background job: ``python -m stoa.newsbrain <job>``.

Jobs: news, minute, sec, score, panel, trial9, trial10, preregister11, trial11,
news_timing, harness, preregister12, trial12, preregister12h,
holdout12, preregister13, trial13, preregister14, trial14, coverage, feedseed <session>,
events, preregister15.
Each logs to data/newsbrain/logs/ and can be stopped and restarted at any time.

``feedseed`` is the big paper feed's one seed build on the Dell: the rolling state as of a
session's 9:00 am cutoff, from the full history, into ``big_paper.feed_dir``. ``feedpush``
uploads that folder once to ``big_paper.feed_repo`` (HF_TOKEN in the environment only;
``scripts/push-feed-seed.ps1`` asks for it without saving it). ``feedreset <commit>`` puts
the dataset's live state back to an earlier commit (an id or a message such as
"seed 2026-10-26"), run from the bigpaper workflow (job ``feed-reset``). ``feedpull``
downloads the whole dataset, the dry runs' pulls and checks included, into
``data/newsbrain/feedcheck/hf`` for the Dell's rebuild to compare
(``scripts/pull-feed.ps1``, a read token, never saved).

``events`` runs the local event reader over every news month not yet labelled, newest
first, on the GPU (stoa/newsbrain/eventread.py, trial 15).

``retrain <last session>`` refits trial 12's model on every session labelled by the open
of that session (stoa/newsbrain/retrain.py). ``backfill <version> <session>...`` scores
those sessions' Dell rows with that model for the shadow scoreboard, marked backfill.
``modelpush <version>`` uploads the model and its backfill to ``big_paper.feed_repo``
(``scripts/push-model.ps1``, a write token, never saved).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from typing import Any

from stoa.newsbrain import ROOT

START_NEWS = date(2015, 1, 1)
START_MINUTE = date(2016, 1, 1)
END = date(2026, 10, 23)  # the last full session before the pulls began


def run_trial9(trial: int = 9) -> None:
    """Trial 9 (or 10) end to end. Needs the panel and the Alpaca keys (9:45 am prices)."""
    import json
    from pathlib import Path

    import pandas as pd

    from stoa.config import load_settings
    from stoa.experiments import Registry
    from stoa.newsbrain.alpaca_pull import LIMITS, client_from_env
    from stoa.newsbrain.pace import Pace
    from stoa.newsbrain.panel import PANEL, PANEL_END, PANEL_START, calendar
    from stoa.newsbrain.trial import EntryPrices, run_trial

    settings = load_settings(Path("config/default.yaml"))
    rows = pd.read_parquet(PANEL)
    days, _ = calendar(PANEL_START, PANEL_END)
    entry = EntryPrices(client_from_env(), Pace("entry", 60, LIMITS), ROOT / "entry")
    fills: Any = entry
    frozen_dir = None
    if trial == 10:
        from stoa.newsbrain.timing import CoolOffPrices, MinuteStore

        fills = CoolOffPrices(MinuteStore(), entry.at_945)
        frozen_dir = ROOT / "trial9"
    report = run_trial(
        settings,
        rows,
        days,
        fills,
        Registry(settings.registry_path),
        ROOT / f"trial{trial}",
        trial=trial,
        frozen_dir=frozen_dir,
    )
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]}, indent=1))


def run_trial11() -> None:
    """Trial 11 end to end: its rows (with pre-market fills), then the trial."""
    import json
    from pathlib import Path

    import pandas as pd

    from stoa.config import load_settings
    from stoa.experiments import Registry
    from stoa.newsbrain.alpaca_pull import LIMITS, client_from_env
    from stoa.newsbrain.pace import Pace, write_parquet
    from stoa.newsbrain.panel import PANEL, PANEL_END, PANEL_START, calendar
    from stoa.newsbrain.premarket import OUT, build_rows, preregister, run_trial11
    from stoa.newsbrain.trial import EntryPrices

    settings = load_settings(Path("config/default.yaml"))
    registry = Registry(settings.registry_path)
    preregister(registry)  # already there; never rewritten
    days, _ = calendar(PANEL_START, PANEL_END)
    path = OUT / "rows.parquet"
    if path.exists():
        rows = pd.read_parquet(path)
    else:
        rows = build_rows(pd.read_parquet(PANEL), days)
        write_parquet(rows, path)
    entry = EntryPrices(client_from_env(), Pace("entry", 60, LIMITS), ROOT / "entry")
    report = run_trial11(settings, rows, days, entry.at_945, registry)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]}, indent=1))


def preregister11() -> None:
    from pathlib import Path

    from stoa.config import load_settings
    from stoa.experiments import Registry
    from stoa.newsbrain.premarket import preregister, spec_hash

    settings = load_settings(Path("config/default.yaml"))
    preregister(Registry(settings.registry_path))
    print(f"trial 11 pre-registered in {settings.registry_path} (spec {spec_hash()})")


def main(argv: list[str]) -> int:
    job = argv[0] if argv else ""
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(ROOT / "logs" / f"{job or 'main'}.log", encoding="utf-8")],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if job == "news":
        from stoa.newsbrain.alpaca_pull import pull_news

        pull_news(START_NEWS, END)
    elif job == "minute":
        from stoa.newsbrain.alpaca_pull import pull_minutes

        pull_minutes(START_MINUTE, END)
    elif job == "sec":
        from stoa.newsbrain.sec_pull import pull_sec

        pull_sec()
    elif job == "score":
        from stoa.newsbrain.score import score_forever

        score_forever()
    elif job == "panel":
        from stoa.newsbrain.panel import PANEL, PANEL_END, PANEL_START, build

        build(PANEL_START, PANEL_END, out=PANEL)
    elif job == "trial9":
        run_trial9()
    elif job == "trial10":
        run_trial9(10)
    elif job == "preregister11":
        preregister11()
    elif job == "trial11":
        run_trial11()
    elif job == "news_timing":
        from stoa.newsbrain.news_timing import run as news_timing

        print(json.dumps(news_timing(), indent=1, default=str))
    elif job == "harness":
        from pathlib import Path

        import pandas as pd

        from stoa.config import load_settings
        from stoa.experiments import Registry
        from stoa.newsbrain.harness import run as harness
        from stoa.newsbrain.panel import PANEL

        settings = load_settings(Path("config/default.yaml"))
        legs = tuple(argv[1].split(",")) if len(argv) > 1 else ("spy", "cheater", "random")
        got = harness(settings, pd.read_parquet(PANEL), Registry(settings.registry_path), legs=legs)
        print(json.dumps(got["verdict"], indent=1))
    elif job in ("preregister12", "trial12"):
        from pathlib import Path

        import pandas as pd

        from stoa.config import load_settings
        from stoa.experiments import Registry
        from stoa.newsbrain import spycore
        from stoa.newsbrain.panel import PANEL

        settings = load_settings(Path("config/default.yaml"))
        registry = Registry(settings.registry_path)
        spycore.preregister(registry)  # written once, never rewritten
        if job == "preregister12":
            print(
                f"trial 12 pre-registered in {settings.registry_path} (spec {spycore.spec_hash()})"
            )
        else:
            from stoa.newsbrain.alpaca_pull import LIMITS, client_from_env
            from stoa.newsbrain.pace import Pace
            from stoa.newsbrain.trial import EntryPrices

            entry = EntryPrices(client_from_env(), Pace("entry", 60, LIMITS), ROOT / "entry")
            got = spycore.run_trial12(settings, pd.read_parquet(PANEL), entry.at_945, registry)
            print(json.dumps({"verdict": got["verdict"], "checks": got["checks"]}, indent=1))
    elif job in ("preregister12h", "holdout12"):
        from pathlib import Path

        from stoa.config import load_settings
        from stoa.experiments import Registry
        from stoa.newsbrain import holdout12

        settings = load_settings(Path("config/default.yaml"))
        registry = Registry(settings.registry_path)
        holdout12.preregister(registry)  # written once, never rewritten
        if job == "preregister12h":
            print(f"trial 12 holdout pre-registered (spec {holdout12.spec_hash()})")
        else:
            from stoa.newsbrain.alpaca_pull import LIMITS, client_from_env
            from stoa.newsbrain.pace import Pace
            from stoa.newsbrain.trial import EntryPrices

            entry = EntryPrices(client_from_env(), Pace("entry", 60, LIMITS), ROOT / "entry")
            got = holdout12.run(settings, entry.at_945, registry)
            print(json.dumps({"verdict": got["verdict"], "checks": got["checks"]}, indent=1))
    elif job in ("preregister13", "trial13"):
        from pathlib import Path

        import pandas as pd

        from stoa.config import load_settings
        from stoa.experiments import Registry
        from stoa.newsbrain import trial13
        from stoa.newsbrain.panel import PANEL

        settings = load_settings(Path("config/default.yaml"))
        registry = Registry(settings.registry_path)
        if job == "preregister13":
            trial13.preregister(registry)  # written once, never rewritten
            print(
                f"trial 13 pre-registered in {settings.registry_path} (spec {trial13.spec_hash()})"
            )
        else:
            # 9:45 am prices as the harness reads them: trial 12's cached price, else the
            # open of the first minute bar from 9:45 to 10:15 on disk (the same rule).
            from stoa.newsbrain.harness import Prices
            from stoa.newsbrain.timing import MinuteStore

            prices = Prices(ROOT / "entry", MinuteStore())
            got = trial13.run_trial13(settings, pd.read_parquet(PANEL), prices.at_945, registry)
            path = trial13.OUT / "report.json"
            body = json.loads(path.read_text(encoding="utf-8"))
            body["price_sources"] = prices.sources
            path.write_text(json.dumps(body, indent=1, default=str), encoding="utf-8")
            print(json.dumps({"verdict": got["verdict"], "checks": got["checks"]}, indent=1))
    elif job in ("preregister14", "trial14"):
        from pathlib import Path

        import pandas as pd

        from stoa.config import load_settings
        from stoa.experiments import Registry
        from stoa.newsbrain import trial14
        from stoa.newsbrain.panel import PANEL

        settings = load_settings(Path("config/default.yaml"))
        registry = Registry(settings.registry_path)
        if job == "preregister14":
            trial14.preregister(registry)  # written once, never rewritten
            print(
                f"trial 14 pre-registered in {settings.registry_path} (spec {trial14.spec_hash()})"
            )
        else:
            # 9:45 am prices as trial 13 read them (the harness's rule).
            from stoa.newsbrain.harness import Prices
            from stoa.newsbrain.timing import MinuteStore

            prices = Prices(ROOT / "entry", MinuteStore())
            got = trial14.run_trial14(settings, pd.read_parquet(PANEL), prices.at_945, registry)
            path = trial14.OUT / "report.json"
            body = json.loads(path.read_text(encoding="utf-8"))
            body["price_sources"] = prices.sources
            path.write_text(json.dumps(body, indent=1, default=str), encoding="utf-8")
            print(json.dumps({"verdict": got["verdict"], "checks": got["checks"]}, indent=1))
    elif job == "coverage":
        from stoa.newsbrain.coverage import report

        print(report())
    elif job == "feedseed":
        from datetime import datetime
        from pathlib import Path

        from stoa.config import load_settings
        from stoa.newsbrain import feed

        settings = load_settings(Path("config/default.yaml"))
        session = date.fromisoformat(argv[1])
        first = json.loads((ROOT / "logs" / "sec.jsonl").read_text(encoding="utf-8").split("\n")[0])
        sec_pulled = datetime.fromisoformat(first["at"])  # the last SEC pull's first line
        st = feed.seed_from_dell(feed.Feed(settings.big_paper.feed_dir), session, sec_pulled)
        print(json.dumps({"seeded": st.session.isoformat(), "dir": str(st.root)}))
    elif job == "feedpush":
        import os
        from pathlib import Path

        from stoa.config import load_settings
        from stoa.newsbrain import feed

        settings = load_settings(Path("config/default.yaml"))
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            print("HF_TOKEN is not set in this process; nothing sent")
            return 2
        fd = feed.Feed(settings.big_paper.feed_dir)
        seeded = fd.record().get("seeded")
        feed.push_state(fd, settings.big_paper.feed_repo, token, f"seed {seeded}")
        print(json.dumps({"pushed": str(fd.root), "to": settings.big_paper.feed_repo}))
    elif job == "retrain":
        from stoa.newsbrain import retrain

        meta = retrain.run(date.fromisoformat(argv[1]))
        print(json.dumps({k: meta[k] for k in ("version", "train_window", "rows", "sha256")}))
    elif job == "backfill":
        from stoa.newsbrain import retrain

        days = [date.fromisoformat(a) for a in argv[2:]]
        for meta in retrain.backfill(argv[1], days):
            print(json.dumps({k: meta[k] for k in ("session", "rows", "above", "picks")}))
    elif job == "modelpush":
        import os
        from pathlib import Path

        from stoa.config import load_settings
        from stoa.newsbrain import retrain

        settings = load_settings(Path("config/default.yaml"))
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            print("HF_TOKEN is not set in this process; nothing sent")
            return 2
        retrain.push(argv[1], settings.big_paper.feed_repo, token)
        print(json.dumps({"pushed": argv[1], "to": settings.big_paper.feed_repo}))
    elif job == "feedreset":
        import os
        import tempfile
        from pathlib import Path

        from stoa.config import load_settings
        from stoa.newsbrain import feed

        settings = load_settings(Path("config/default.yaml"))
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            print("HF_TOKEN is not set in this process; nothing changed")
            return 2
        with tempfile.TemporaryDirectory() as tmp:
            rev = feed.reset_state(
                feed.Feed(Path(tmp)), settings.big_paper.feed_repo, token, argv[1]
            )
        print(json.dumps({"reset": settings.big_paper.feed_repo, "to": argv[1], "commit": rev}))
    elif job == "preregister15":
        from pathlib import Path

        from stoa.config import load_settings
        from stoa.experiments import Registry
        from stoa.newsbrain import trial15

        settings = load_settings(Path("config/default.yaml"))
        trial15.preregister(Registry(settings.registry_path))  # written once, never rewritten
        print(f"trial 15 pre-registered in {settings.registry_path} (spec {trial15.spec_hash()})")
    elif job == "events":
        from stoa.newsbrain.alpaca_pull import NEWS_DIR
        from stoa.newsbrain.eventread import run_history

        run_history(NEWS_DIR)
    elif job == "feedpull":
        import os
        from pathlib import Path

        from stoa.config import load_settings
        from stoa.newsbrain import feed

        settings = load_settings(Path("config/default.yaml"))
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            print("HF_TOKEN is not set in this process; nothing read")
            return 2
        out = feed.Feed(ROOT / "feedcheck" / "hf")
        feed.pull_state(out, settings.big_paper.feed_repo, token, only=["*", "**"])
        print(json.dumps({"pulled": settings.big_paper.feed_repo, "to": str(out.root)}))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
