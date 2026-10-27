"""Run one background job: ``python -m stoa.newsbrain <job>``.

Jobs: news, minute, sec, score, panel, trial9, trial10, preregister11, trial11,
news_timing, harness, preregister12, trial12, preregister12h,
holdout12, coverage. Each logs to data/newsbrain/logs/
and can be stopped and restarted at any time.
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
    elif job == "coverage":
        from stoa.newsbrain.coverage import report

        print(report())
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
