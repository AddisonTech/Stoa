"""Trial 15's comparison: the new brain against trial 9's frozen brain, run once.

As registered (spec 557c95176a137379, ``comparison``): walk-forward on 2023-01-01 to
2026-06-30, the new brain refit every 63 sessions on every row labelled before the slice
(from 2016), recent rows weighted more; each session both brains rank the same rows and
their top 5 are graded on trial 9's label (open to open against SPY after costs). The new
brain goes live only if its average edge is higher and its hit rate at least as high.

The rows are trial 9's panel with the new inputs (``brain15``). A pick must be a liquid
company with a known label, the same for both brains.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa.experiments import Registry
from stoa.newsbrain import ROOT, brain15, eventread, trial15
from stoa.newsbrain import panel as pn
from stoa.newsbrain.sec_pull import SEC_DIR

log = logging.getLogger(__name__)

OUT = ROOT / "trial15"
NAME = "trial15:comparison"


def build_rows(edges: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[Any]]:
    """Trial 9's panel with every trial 15 input, and the session calendar."""
    rows = pd.read_parquet(pn.PANEL)
    days, cut = pn.calendar(pn.PANEL_START, pn.PANEL_END)
    labels = eventread.load()
    times = brain15.article_times(cut)
    events = brain15.day_events(labels, times)
    rows = brain15.with_events(rows, events, brain15.sectors())
    first = brain15.first_premarket(rows, times, days)
    catchup = pd.read_parquet(brain15.CATCHUP)
    rows = brain15.with_premarket(rows, first, brain15.open_times(rows, days), catchup)
    if edges is not None:
        from stoa.newsbrain import ripplemap

        ciks = pd.read_parquet(SEC_DIR / "ciks.parquet")
        rows = brain15.with_ripple(rows, events, ripplemap.symbol_edges(edges, ciks), cut)
    for col in brain15.NEW:
        if col not in rows:
            rows[col] = np.nan
    return rows, days


def companies() -> set[str]:
    sic = json.loads(brain15.SIC_PATH.read_text(encoding="utf-8"))
    return {s for s in sic if trial15.is_company(s, sic)}


def by_year(picks: pd.DataFrame, days: list[Any]) -> dict[int, dict[str, Any]]:
    year = pd.Series([days[int(i)].year for i in picks["sidx"]], index=picks.index)
    return {int(y): brain15.grade(g) for y, g in picks.groupby(year)}


def run(registry: Registry, edges: pd.DataFrame | None = None) -> dict[str, Any]:
    """The comparison, once. Writes trial15/comparison.json and logs one trial."""
    t0 = time.time()
    rows, days = build_rows(edges)
    test_mask = (rows["session"] >= trial15.TEST_START) & (rows["session"] <= trial15.TEST_END)
    ok = brain15.eligible(rows, companies()) & test_mask
    conv_new = brain15.walk_forward(rows, days, trial15.TEST_START, trial15.TEST_END)
    frozen = pickle.loads((ROOT / "trial9" / "model_h1.pkl").read_bytes())
    test = rows[ok & conv_new.notna()]
    new = brain15.top5(test, conv_new[test.index].to_numpy())
    old = brain15.top5(test, trial15.conviction(frozen, test))
    g_new, g_old = brain15.grade(new), brain15.grade(old)
    passed = (
        g_new["avg_edge"] is not None
        and g_old["avg_edge"] is not None
        and g_new["avg_edge"] > g_old["avg_edge"]
        and g_new["hit_rate"] >= g_old["hit_rate"]
    )
    coverage = {col: float(rows.loc[ok, col].notna().mean()) for col in brain15.NEW if col in rows}
    report = {
        "spec": trial15.spec_hash(),
        "window": [str(trial15.TEST_START), str(trial15.TEST_END)],
        "eligible_rows": int(ok.sum()),
        "base_rate": float((rows.loc[ok, "excess_1"] > 0).mean()),
        "new": g_new,
        "frozen": g_old,
        "new_by_year": by_year(new, days),
        "frozen_by_year": by_year(old, days),
        "new_goes_live": bool(passed),
        "input_coverage": coverage,
        "ripple_in": edges is not None,
        "seconds": round(time.time() - t0),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    daily = new.groupby("sidx")["excess_1"].mean()
    daily.index = pd.DatetimeIndex([pd.Timestamp(days[int(i)]) for i in daily.index])
    registry.append(
        kind="backtest",
        name=NAME,
        config_hash=trial15.spec_hash(),
        data_hash=eventread.VERSION,
        params={"features": list(brain15.FEATURES), "half_life_years": 2.0, "slice": 63},
        metrics={k: v for k, v in report.items() if k in ("new", "frozen", "new_goes_live")},
        returns=daily,
    )
    log.info("trial 15 comparison: %s", json.dumps(report))
    return report


def main() -> None:
    from stoa.config import load_settings

    settings = load_settings(Path("config/default.yaml"))
    edges_path = ROOT / "ripple" / "edges.parquet"
    edges = pd.read_parquet(edges_path) if edges_path.exists() else None
    print(json.dumps(run(Registry(settings.registry_path), edges), indent=1))
