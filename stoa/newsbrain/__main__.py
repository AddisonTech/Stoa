"""Run one background job: ``python -m stoa.newsbrain <job>``.

Jobs: news, minute, sec, score, panel, trial9, trial10, coverage. Each logs to data/newsbrain/logs/
and can be stopped and restarted at any time.
"""

from __future__ import annotations

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
    elif job == "coverage":
        from stoa.newsbrain.coverage import report

        print(report())
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
