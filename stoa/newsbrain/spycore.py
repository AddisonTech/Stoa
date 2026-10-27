"""Trial 12: the news brain's 1 session signal with idle money in SPY (docs/DECISIONS.md).

Trial 9's frozen models, features, signals, replay engine, risk rails, pass bar and
holdout wall, unchanged. Two things differ: the graded horizon is 1 session, and any
money not in a pick sits in SPY (the SPY core, run through ``SweepBroker``). The same
signals with idle money in cash (trial 9's rule) run beside it as their own trial.
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

from stoa import event_signals
from stoa.backtest import run_backtest
from stoa.config import Settings
from stoa.data.base import Bars
from stoa.event_run import base_bars, source
from stoa.experiments import Registry, deflate, return_moments
from stoa.newsbrain import ROOT, sealed
from stoa.newsbrain import trial as t9
from stoa.newsbrain.panel import HORIZONS
from stoa.scoreboard import summarize

log = logging.getLogger(__name__)

SPEC = "trial12:newsbrain:spycore:v1"
GRADED = 1
CORE = "SPY"
OUT = ROOT / "trial12"
FROZEN = ROOT / "trial9"
ENTRIES = ("spycore", "cash")
At945 = Callable[[date, set[str]], dict[str, Decimal]]


def spec() -> dict[str, Any]:
    """The pre-registered rules, as written to the registry before any run."""
    return {
        "spec": SPEC,
        "question": "does the news brain's 1 session signal beat SPY when idle money sits in "
        "SPY instead of cash",
        "same_as_trial_9": "data, the 13 features, model settings, walk-forward 2016-01-01 to "
        "2022-12-31 with labels ending by 2022-12-30 (trial 9's frozen models, not refit), "
        "test 2023-01-01 to 2026-06-30, replay engine, every risk rail, pass bar, holdout wall",
        "graded_horizon": GRADED,
        "reported_horizons": list(HORIZONS),
        "signals": f"trial 9's: p >= {t9.THRESHOLD}, up to {t9.TOP_N} a session, best first; "
        "SPY itself is never a pick",
        "spy_core": "cash above the 2% reserve sits in SPY; at 9:45 am a buy that cash cannot "
        "cover sells SPY at the same 9:45 am fill; after the session's orders, cash above the "
        "reserve buys SPY; when a pick exits its money goes back to SPY; SPY is never bought and "
        "sold in the same session (the account's no same-day round trip rule), so spare cash on "
        "a session SPY was sold waits one session; SPY counts as idle money for the rails "
        "(no position slot, no size cap, no sector); the kill switch sells SPY too and "
        "nothing is swept while it is on",
        "spy_costs": "trial 9's cost model (25 x sqrt($10M / 20 session dollar volume), 5 to "
        "75 bps, 5 bps for SPY) on every SPY trade, plus the sell fee",
        "comparison": "the same signals with idle money in cash (trial 9's rule), logged as "
        "its own trial",
        "logged_trials": "SPY core and cash, horizons 1, 5 and 20 (6 configurations)",
        "graded": "the 1 session SPY core run, on trial 9's pass bar",
        "report": "trades, hit rate, return, Sharpe, Sortino, max drawdown, deflated Sharpe, "
        "vs SPY and MTUM, for both versions, by small, mid and large cap; share of sessions "
        "fully in SPY; total cost paid on SPY swaps",
        "starved": "fewer than 30 closed trades in the graded test window: STARVED, not FAIL",
        "known_limits": "the 1 session horizon was chosen after seeing trial 9's results on "
        "this same test window, so a pass is a lead, not proof; real proof is the sealed "
        "holdout and live paper trading",
        "holdout": "2026-07-01 to 2026-10-23 is never read",
    }


def spec_hash() -> str:
    return hashlib.sha256(json.dumps(spec(), sort_keys=True).encode()).hexdigest()[:16]


def preregister(registry: Registry) -> None:
    """The rules, in the registry before any run. Not a trial: it holds no returns."""
    if any(
        e.kind == "preregistration" and e.name == "trial12:preregistration" for e in registry.load()
    ):
        return
    registry.append(
        kind="preregistration",
        name="trial12:preregistration",
        config_hash=spec_hash(),
        data_hash="none",
        params=spec(),
        metrics={},
    )


# ---- the replays ------------------------------------------------------------------------------


def test_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """The test window's rows; the holdout is dropped before anything else."""
    rows = rows[rows["session"] < sealed.HOLDOUT_START]
    return rows[(rows["session"] >= t9.TEST_START) & (rows["session"] <= t9.TEST_END)]


def signals(test: pd.DataFrame, p: Any, h: int) -> list[event_signals.Signal]:
    return [s for s in t9.signals_for(test, p, h) if s.symbol != CORE]


def core_fills(at_945: At945) -> Callable[[date, set[str], set[str]], dict[str, Decimal]]:
    """9:45 am prices for every symbol that may trade, and always for SPY."""
    return lambda day, buys, others: at_945(day, buys | others | {CORE})


def replay(
    settings: Settings,
    bars: Bars,
    sigs: list[event_signals.Signal],
    h: int,
    at_945: At945,
    name: str,
    *,
    spy_core: bool,
) -> dict[str, t9.Replay]:
    out: dict[str, t9.Replay] = {}
    for w, (start, end) in t9.WINDOWS.items():
        sealed.check(end, f"the {w} replay")
        src = source(settings, bars, end, signals=sigs, name=name)
        assert src.window_bars is not None
        res = run_backtest(
            settings,
            src.window_bars(start, end),
            src.strategy,
            start,
            end,
            universe=src.universe,
            slippage=src.slippage,
            fill_prices=core_fills(at_945) if spy_core else (lambda d, b, o: at_945(d, b | o)),
            sweep_symbol=CORE if spy_core else None,
        )
        out[w] = t9.Replay(h, w, res, summarize(res, settings, w))
        log.info("replay %s h%d %s: %d trades", name, h, w, len(res.trades))
    return out


def sweep_report(res: Any, starting: float) -> dict[str, Any]:
    st = res.sweep
    if st is None:
        return {}
    return {
        "swaps": st.swaps,
        "bought": float(st.bought),
        "sold": float(st.sold),
        "cost": float(st.cost),
        "cost_pct_of_start": float(st.cost) / starting if starting else None,
        "sessions": st.sessions,
        "share_fully_in_spy": st.full_sessions / st.sessions if st.sessions else None,
    }


def log_trials(
    registry: Registry, runs: dict[str, dict[int, dict[str, t9.Replay]]]
) -> dict[str, Any]:
    now = datetime.now(UTC)
    for entry, by_h in runs.items():
        for h, rep in by_h.items():
            starved = len(rep["test"].result.trades) < t9.MIN_TRADES
            rets = list(rep["validation"].run.daily_returns)
            registry.append(
                kind="backtest",
                name=f"trial12:{entry}:h{h}:validation",
                config_hash=f"{spec_hash()}-{entry}-h{h}",
                data_hash=hashlib.sha256(json.dumps(rets).encode()).hexdigest()[:16],
                params={"trial": 12, "entry": entry, "horizon": h, "window": "validation"},
                metrics={
                    "sharpe_daily": return_moments(rets)[0],
                    **rep["validation"].run.metrics.to_dict(),
                },
                returns=pd.Series(rets),
                tags=["starved"] if starved else [],
                now=now,
            )
    return {
        entry: deflate(
            registry,
            list(runs[entry][GRADED]["validation"].run.daily_returns),
            variance_kind="backtest",
        )
        for entry in runs
    }


def run_trial12(
    settings: Settings,
    rows: pd.DataFrame,
    at_945: At945,
    registry: Registry,
    out: Path = OUT,
    frozen_dir: Path = FROZEN,
) -> dict[str, Any]:
    sealed.check(t9.TEST_END, "the test window")
    test = test_rows(rows)
    for h in HORIZONS:
        t9.register_kind(h)
    frozen = {h: pickle.loads((frozen_dir / f"model_h{h}.pkl").read_bytes()) for h in HORIZONS}
    bars = base_bars(settings, t9.TEST_END)
    adv = {(s, d): a for s, d, a in zip(test["symbol"], test["session"], test["adv"], strict=True)}
    runs: dict[str, dict[int, dict[str, t9.Replay]]] = {e: {} for e in ENTRIES}
    trades: dict[str, dict[int, pd.DataFrame]] = {e: {} for e in ENTRIES}
    for h in HORIZONS:
        sigs = signals(test, frozen[h].predict(test), h)
        for entry in ENTRIES:
            runs[entry][h] = replay(
                settings, bars, sigs, h, at_945, f"news12{entry}_h{h}", spy_core=entry == "spycore"
            )
            trades[entry][h] = t9.trade_table(runs[entry][h]["test"].result, bars["SPY"], adv)
    base = t9.baselines(settings, bars)
    deflated = log_trials(registry, runs)
    graded = runs["spycore"][GRADED]
    chk = t9.checks(graded, base, deflated["spycore"]["dsr"])
    passed = all(c["passed"] for c in chk)
    start = float(graded["test"].result.starting_equity)

    def summary(r: t9.Replay, tt: pd.DataFrame) -> dict[str, Any]:
        m = r.run.metrics
        return {
            "return": m.total_return,
            "sharpe": m.sharpe,
            "sortino": m.sortino,
            "max_drawdown": m.max_drawdown,
            "trades": m.trade_count,
            "hit_rate": float(tt["beat_spy"].mean()) if r.window == "test" and len(tt) else None,
        }

    report = {
        "trial": 12,
        "passed": passed,
        "verdict": t9.verdict(passed, len(graded["test"].result.trades)),
        "checks": chk,
        "deflated": deflated,
        "runs": {
            entry: {
                str(h): {w: summary(r, trades[entry][h]) for w, r in rep.items()}
                for h, rep in by_h.items()
            }
            for entry, by_h in runs.items()
        },
        "baselines": {
            k: {
                "return": v.metrics.total_return,
                "sharpe": v.metrics.sharpe,
                "sortino": v.metrics.sortino,
                "max_drawdown": v.metrics.max_drawdown,
            }
            for k, v in base.items()
        },
        "by_cap": {e: {str(h): t9.by_bucket(trades[e][h]) for h in HORIZONS} for e in ENTRIES},
        "spy_core": {
            str(h): sweep_report(runs["spycore"][h]["test"].result, start) for h in HORIZONS
        },
        "spec_hash": spec_hash(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    for e in ENTRIES:
        for h in HORIZONS:
            trades[e][h].to_csv(out / f"trades_{e}_h{h}.csv", index=False)
    return report
