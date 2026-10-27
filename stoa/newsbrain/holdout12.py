"""Trial 12 on the holdout: one run, as pre-registered (docs/DECISIONS.md).

Everything is trial 12's: trial 9's frozen models (no refit), the 13 features, the 1
session graded horizon, the SPY core, p of at least 0.55 and up to 10 a session, 9:45
am fills, the cost model and every risk rail. Only the window changes, to the holdout,
2026-07-01 to 2026-10-23. The wall opens for this run alone, through ``sealed.opened``,
and the holdout is marked spent at the end, whatever the result.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from stoa import clock
from stoa.backtest import run_backtest
from stoa.baselines import BuyAndHold
from stoa.config import Settings
from stoa.event_run import base_bars, source
from stoa.experiments import Registry, deflate
from stoa.newsbrain import ROOT, sealed
from stoa.newsbrain import spycore as sc
from stoa.newsbrain import trial as t9
from stoa.newsbrain.panel import HORIZONS, PANEL_START, build
from stoa.scoreboard import summarize
from stoa.universe import Universe

log = logging.getLogger(__name__)

SPEC = "trial12:holdout:v1"
START, END = sealed.HOLDOUT_START, sealed.HOLDOUT_END
TEST_TRADES = 193  # trial 12's graded 1 session SPY core run, 2023-01-01 to 2026-06-30
TEST_SESSIONS = 875
OUT = ROOT / "trial12_holdout"
At945 = Callable[[date, set[str]], dict[str, Decimal]]


def expected_trades() -> float:
    """Trial 12's test-window trade rate times the holdout's sessions."""
    return TEST_TRADES / TEST_SESSIONS * len(clock.sessions_between(START, END))


def spec() -> dict[str, Any]:
    exp = expected_trades()
    return {
        "spec": SPEC,
        "trial12_spec": sc.spec_hash(),
        "window": f"{START} to {END}, one run, never again",
        "same_as_trial_12": "trial 9's frozen models (no refit), the 13 features, the 1 session "
        "graded horizon, the SPY core, p >= 0.55 and up to 10 a session, 9:45 am fills (fetched "
        "by the same code as trial 12), the cost model, every risk rail",
        "reported_not_graded": "the cash version, the 5 and 20 session runs, MTUM, the deflated "
        "Sharpe (4 months is too short for MTUM or the deflated Sharpe to mean much)",
        "expected_trades": round(exp, 1),
        "starved": f"fewer than a quarter of the expected trades ({exp / 4:.2f}): STARVED",
        "pass_bar": "the 1 session SPY core run on the holdout: Sharpe at or above SPY's, return "
        "at or above SPY's, max drawdown at or below SPY's, max drawdown under 25%, not starved",
        "registry": "logged as kind holdout, not a new trial; the holdout is marked spent after",
    }


def spec_hash() -> str:
    return hashlib.sha256(json.dumps(spec(), sort_keys=True).encode()).hexdigest()[:16]


def preregister(registry: Registry) -> None:
    if any(e.kind == "preregistration" and e.name == "trial12:holdout" for e in registry.load()):
        return
    registry.append(
        kind="preregistration",
        name="trial12:holdout",
        config_hash=spec_hash(),
        data_hash="none",
        params=spec(),
        metrics={},
    )


# ---- grading -------------------------------------------------------------------------------


def grade(core: Any, spy: Any, trades: int) -> tuple[str, list[dict[str, Any]]]:
    """PASS, FAIL or STARVED on the pre-registered holdout bar."""
    exp = expected_trades()
    checks = [
        {
            "name": "sharpe_vs_spy",
            "passed": core.sharpe >= spy.sharpe,
            "detail": f"{core.sharpe:.2f} vs SPY {spy.sharpe:.2f}",
        },
        {
            "name": "return_vs_spy",
            "passed": core.total_return >= spy.total_return,
            "detail": f"{core.total_return:.2%} vs SPY {spy.total_return:.2%}",
        },
        {
            "name": "drawdown_vs_spy",
            "passed": core.max_drawdown <= spy.max_drawdown,
            "detail": f"{core.max_drawdown:.2%} vs SPY {spy.max_drawdown:.2%}",
        },
        {
            "name": "drawdown_cap",
            "passed": core.max_drawdown <= 0.25,
            "detail": f"{core.max_drawdown:.2%} (cap 25%)",
        },
        {
            "name": "not_starved",
            "passed": trades >= exp / 4,
            "detail": f"{trades} closed trades (expected {exp:.1f}; starved under {exp / 4:.2f})",
        },
    ]
    if all(c["passed"] for c in checks):
        return "PASS", checks
    return ("STARVED" if trades < exp / 4 else "FAIL"), checks


# ---- the run -----------------------------------------------------------------------------------


def replay(
    settings: Settings, bars: Any, sigs: list[Any], at_945: At945, name: str, *, spy_core: bool
) -> Any:
    src = source(settings, bars, END, signals=sigs, name=name)
    assert src.window_bars is not None
    return run_backtest(
        settings,
        src.window_bars(START, END),
        src.strategy,
        START,
        END,
        universe=src.universe,
        slippage=src.slippage,
        fill_prices=sc.core_fills(at_945) if spy_core else (lambda d, b, o: at_945(d, b | o)),
        sweep_symbol=sc.CORE if spy_core else None,
    )


def run(
    settings: Settings,
    at_945: At945,
    registry: Registry,
    out: Path = OUT,
    frozen_dir: Path = sc.FROZEN,
    *,
    owner_rerun: bool = False,
) -> dict[str, Any]:
    """The one holdout run. Opens the wall, runs, shuts it and marks the holdout spent."""
    out.mkdir(parents=True, exist_ok=True)
    with sealed.opened(sealed.HOLDOUT_KEY, owner_rerun=owner_rerun):
        panel = build(PANEL_START, END, out=out / "panel.parquet")
        rows = panel[(panel["session"] >= START) & (panel["session"] <= END)]
        for h in HORIZONS:
            t9.register_kind(h)
        frozen = {h: pickle.loads((frozen_dir / f"model_h{h}.pkl").read_bytes()) for h in HORIZONS}
        bars = base_bars(settings, END)
        adv = {
            (s, d): a for s, d, a in zip(rows["symbol"], rows["session"], rows["adv"], strict=True)
        }
        results: dict[str, dict[int, Any]] = {e: {} for e in sc.ENTRIES}
        trades: dict[str, dict[int, pd.DataFrame]] = {e: {} for e in sc.ENTRIES}
        for h in HORIZONS:
            sigs = sc.signals(rows, frozen[h].predict(rows), h)
            for e in sc.ENTRIES:
                res = replay(
                    settings, bars, sigs, at_945, f"holdout12{e}_h{h}", spy_core=e == "spycore"
                )
                results[e][h] = res
                trades[e][h] = t9.trade_table(res, bars["SPY"], adv)
        base = {}
        for name, sym in (("spy", settings.benchmark), ("mtum", "MTUM")):
            uni = Universe.assume_all_fractionable((*settings.watchlist, sym), settings.benchmark)
            res = run_backtest(settings, bars, BuyAndHold(sym, name=name), START, END, universe=uni)
            base[name] = summarize(res, settings, "holdout").metrics
    # The wall is shut again from here on.
    runs = {
        e: {h: summarize(r, settings, "holdout") for h, r in by.items()}
        for e, by in results.items()
    }
    core = runs["spycore"][sc.GRADED]
    n_trades = len(results["spycore"][sc.GRADED].trades)
    verdict, checks = grade(core.metrics, base["spy"], n_trades)
    dsr = {
        e: deflate(registry, list(runs[e][sc.GRADED].daily_returns), variance_kind="backtest")
        for e in runs
    }
    registry.append(
        kind="holdout",
        name="trial12:holdout:result",
        config_hash=f"{sc.spec_hash()}-holdout",
        data_hash=hashlib.sha256(json.dumps(list(core.daily_returns)).encode()).hexdigest()[:16],
        params={"trial": 12, "window": "holdout", "spec": spec_hash(), "verdict": verdict},
        metrics=core.metrics.to_dict(),
        returns=pd.Series(list(core.daily_returns)),
        now=datetime.now(UTC),
    )

    def row(m: Any, n: int | None) -> dict[str, Any]:
        return {
            "return": m.total_return,
            "sharpe": m.sharpe,
            "sortino": m.sortino,
            "max_drawdown": m.max_drawdown,
            "trades": n,
        }

    report = {
        "verdict": verdict,
        "checks": checks,
        "expected_trades": expected_trades(),
        "table": {
            "spy_core": row(core.metrics, n_trades),
            "cash": row(runs["cash"][sc.GRADED].metrics, len(results["cash"][sc.GRADED].trades)),
            "spy": row(base["spy"], None),
            "mtum": row(base["mtum"], None),
        },
        "runs": {
            e: {str(h): row(r.metrics, len(results[e][h].trades)) for h, r in by.items()}
            for e, by in runs.items()
        },
        "by_cap": {e: {str(h): t9.by_bucket(trades[e][h]) for h in HORIZONS} for e in sc.ENTRIES},
        "spy_core": sc.sweep_report(
            results["spycore"][sc.GRADED], float(results["spycore"][sc.GRADED].starting_equity)
        ),
        "deflated": dsr,
        "note": "MTUM and the deflated Sharpe are reported, not graded: 4 months is too short "
        "for either to mean much.",
        "spec_hash": spec_hash(),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    for e in sc.ENTRIES:
        for h in HORIZONS:
            trades[e][h].to_csv(out / f"trades_{e}_h{h}.csv", index=False)
    sealed.mark_spent({"verdict": verdict, "report": str(out / "report.json")})
    return report
