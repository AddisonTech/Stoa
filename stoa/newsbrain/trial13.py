"""Trial 13: trial 12's picks, exactly, with a risk overlay on top (docs/DECISIONS.md).

Trial 9's frozen 1 session model, the 13 features, p of at least 0.55 and up to 10 a
session, 9:45 am fills, the SPY core, the cost model and every trial 12 rail, unchanged.
Only the overlay is new (stoa/overlay.py): volatility-scaled size, a 5% cap per
position, a lower exposure cap with the rest in cash while the account is in a
drawdown, the drawdown brake, and broker-side ATR stops. Its numbers are fixed below
and written to the registry before the run.

Trial 12 already ran on this window, and its drawdown was seen before the overlay was
set, so this run is in-sample for the overlay. The live big paper account is the grade.
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

import numpy as np
import pandas as pd

from stoa.backtest import BacktestResult, run_backtest
from stoa.config import Settings
from stoa.data.base import Bars
from stoa.event_run import base_bars, source
from stoa.event_strategy import EventStrategy
from stoa.experiments import Registry, deflate, return_moments
from stoa.models import Side
from stoa.newsbrain import ROOT, sealed
from stoa.newsbrain import spycore as sc
from stoa.newsbrain import trial as t9
from stoa.overlay import Overlay, OverlayConfig, account_settings
from stoa.scoreboard import summarize

log = logging.getLogger(__name__)

SPEC = "trial13:newsbrain:overlay:v1"
GRADED = sc.GRADED  # 1 session
OUT = ROOT / "trial13"
TRIAL12_REPORT = ROOT / "trial12" / "report.json"
STARTING_EQUITY = Decimal("100000")  # the big paper account
OVERLAY = OverlayConfig()  # the registered numbers; changing any of them is a new trial
NAME = "trial13:spycore-overlay:h1"
At945 = Callable[[date, set[str]], dict[str, Decimal]]


def spec() -> dict[str, Any]:
    """The pre-registered rules, as written to the registry before any run."""
    c = OVERLAY
    return {
        "spec": SPEC,
        "question": "does a risk overlay cut trial 12's drawdown while its picks stay the same",
        "same_as_trial_12": f"spec {sc.spec_hash()}: trial 9's frozen 1 session model (no "
        "refit), the 13 features, p >= 0.55 and up to 10 a session best first, SPY never a "
        "pick, buy at 9:45 am and sell at 9:45 am one session later, idle money in SPY (the "
        "SPY core), the cost model, and every trial 12 rail: 5 positions, 10 orders a day, "
        "the 2% cash reserve, the 3% daily loss stop, the 200 day regime filter, the 40% "
        "sector cap, the 0.8 correlation limit, the close-based ATR exits, the 25% kill switch",
        "account": "$100,000, the big paper account (trial 12 ran $100, where a whole-share "
        "stock costing more than its $20 slot could not be bought; at $100,000 every pick fits)",
        "overlay": {
            "a_volatility_size": {
                "value": f"{c.risk_per_position:.0%} of equity at risk per position",
                "rule": "money in a new position = 1% of equity / (2.5 x ATR(14) / price): if "
                "it falls to the engine's existing 2.5 ATR stop, the account loses 1%",
                "reason": "1% a position is the most common fixed-fraction rule in trading "
                "texts; 2.5 ATR is the stop the engine already uses. Nothing was tried and "
                "compared",
            },
            "b_position_cap": {
                "value": f"{c.max_position_pct:.0%} of equity",
                "reason": "the owner's number; with the 1% rule it binds for any stock whose "
                "ATR is under 8% of its price, so only very jumpy stocks get less",
            },
            "c_exposure_cap": {
                "value": f"picks plus SPY at most 100% of equity normally (the existing "
                f"no-borrowing rail); at most {c.drawdown_exposure_cap:.0%} while the account "
                f"is {c.brake_drawdown:.0%} or more off its high, or the brake is holding "
                "entries; money above it sits in cash, not SPY",
                "reason": "half, at the same line as the brake: one drawdown measure drives "
                "every lever, and halving matches the brake's step. Nothing was tuned",
            },
            "d_drawdown_brake": {
                "value": f"{c.brake_drawdown:.0%} off the high: new positions at half size; "
                f"{c.halt_drawdown:.0%} off: no new positions until back within "
                f"{c.resume_drawdown:.0%}",
                "reason": "the owner's numbers",
            },
            "e_broker_stops": {
                "value": "a broker-side ATR stop on every position, from the existing stop code "
                "(stops.py): 2.5 ATR(14) under entry, then 3 ATR under the highest high once "
                "higher; placed after the 9:45 am orders on each position held from a prior "
                "session",
                "reason": "the stop the engine already uses, held at the broker so a fall "
                "during the day is acted on. A position bought this morning gets none (no "
                "same-day round trip), and with a 1 session hold most positions are sold at "
                "9:45 am before a stop is placed: the stops guard a sale that fails",
                "backtest": "a stop already crossed at the 9:45 am price fills there, else at "
                "the stop when the day's low reaches it, less slippage and fees",
            },
        },
        "targets": "trial 12's worst walk-forward drop (2025-03-05 to 2025-04-08, 14.9%) was "
        "the SPY core falling with the market, with the picks net positive: (c) and (d) aim at "
        "it. Its holdout drop (2026-07-29, 5.8% in a day) was three chip and AI stocks bought "
        "the same morning at 20% each: (a) and (b) aim at it. The numbers were not changed "
        "for either",
        "windows": "trial 12's: tune_c 2023-01-01 to 2024-12-31, validation 2025-01-01 to "
        "2026-06-30, test 2023-01-01 to 2026-06-30",
        "graded": "the 1 session SPY core run with the overlay, on trial 9's pass bar, the same "
        "checks as trial 12",
        "logged_trials": "one configuration; its validation returns go in the registry as a "
        "backtest and count in the deflated Sharpe",
        "starved": "fewer than 30 closed trades in the test window: STARVED, not FAIL",
        "in_sample": "trial 12 already ran on this window and its drawdown was seen before the "
        "overlay was set: in-sample for the overlay, a lead, not proof; the live big paper "
        "account is the real grade",
        "report": "return, max drawdown, Sharpe, deflated Sharpe, trades and checks beside "
        "trial 12 and SPY; sessions braked, held and capped; stop-outs; the worst drop",
        "holdout": "2026-07-01 to 2026-10-23 was spent by trial 12 and is never read again",
    }


def spec_hash() -> str:
    return hashlib.sha256(json.dumps(spec(), sort_keys=True).encode()).hexdigest()[:16]


def preregister(registry: Registry) -> None:
    """The rules, in the registry before any run. Not a trial: it holds no returns."""
    if any(
        e.kind == "preregistration" and e.name == "trial13:preregistration" for e in registry.load()
    ):
        return
    registry.append(
        kind="preregistration",
        name="trial13:preregistration",
        config_hash=spec_hash(),
        data_hash="none",
        params=spec(),
        metrics={},
    )


def preregistered(registry: Registry) -> bool:
    return any(
        e.kind == "preregistration"
        and e.name == "trial13:preregistration"
        and e.config_hash == spec_hash()
        for e in registry.load()
    )


# ---- the run ---------------------------------------------------------------------------------


def replay(
    settings: Settings, bars: Bars, sigs: list[Any], at_945: At945
) -> dict[str, tuple[t9.Replay, Overlay]]:
    """Each window from a fresh $100,000 account with its own overlay."""
    out: dict[str, tuple[t9.Replay, Overlay]] = {}
    for w, (start, end) in t9.WINDOWS.items():
        sealed.check(end, f"the {w} replay")
        src = source(settings, bars, end, signals=sigs, name=f"news13_h{GRADED}")
        assert src.window_bars is not None and isinstance(src.strategy, EventStrategy)
        overlay = Overlay(OVERLAY)
        src.strategy.overlay = overlay
        res = run_backtest(
            settings,
            src.window_bars(start, end),
            src.strategy,
            start,
            end,
            universe=src.universe,
            slippage=src.slippage,
            fill_prices=sc.core_fills(at_945),
            sweep_symbol=sc.CORE,
            overlay=overlay,
        )
        out[w] = (t9.Replay(GRADED, w, res, summarize(res, settings, w)), overlay)
        log.info("replay trial 13 %s: %d trades, %d stop-outs", w, len(res.trades), res.stop_outs)
    return out


def worst_drop(res: BacktestResult) -> dict[str, Any]:
    eq = res.equity.to_numpy(dtype=float)
    days = [pd.Timestamp(x).date() for x in res.equity.index]
    peak = np.maximum.accumulate(eq)
    dd = 1 - eq / peak
    trough = int(dd.argmax())
    top = int(eq[: trough + 1].argmax())
    back = [i for i in range(trough, len(eq)) if eq[i] >= eq[top]]
    return {
        "depth": float(dd.max()),
        "peak": str(days[top]),
        "trough": str(days[trough]),
        "recovered": str(days[back[0]]) if back else None,
    }


def overlay_report(res: BacktestResult, overlay: Overlay) -> dict[str, Any]:
    h = overlay.history
    n = len(h) or 1
    weights = [
        float(f.qty * f.price) / float(res.equity.loc[: pd.Timestamp(f.session)].iloc[-1])
        for f in res.fills
        if f.side is Side.BUY and f.symbol != sc.CORE
    ]
    return {
        "decisions": len(h),
        "share_braked": sum(s.size_multiplier < 1 for s in h) / n,
        "share_held": sum(s.halted for s in h) / n,
        "share_capped": sum(s.in_drawdown for s in h) / n,
        "stop_outs": res.stop_outs,
        "kill_switch_trips": res.kill_switch_trips,
        "avg_pick_size": sum(weights) / len(weights) if weights else None,
        "max_pick_size": max(weights) if weights else None,
        "max_invested": float(res.exposure.max()),  # picks plus SPY, share of equity
    }


def log_trial(registry: Registry, rep: dict[str, t9.Replay]) -> dict[str, Any]:
    starved = len(rep["test"].result.trades) < t9.MIN_TRADES
    rets = list(rep["validation"].run.daily_returns)
    registry.append(
        kind="backtest",
        name=f"{NAME}:validation",
        config_hash=f"{spec_hash()}-h{GRADED}",
        data_hash=hashlib.sha256(json.dumps(rets).encode()).hexdigest()[:16],
        params={"trial": 13, "entry": "spycore-overlay", "horizon": GRADED, "window": "validation"},
        metrics={
            "sharpe_daily": return_moments(rets)[0],
            **rep["validation"].run.metrics.to_dict(),
        },
        returns=pd.Series(rets),
        tags=["starved"] if starved else [],
        now=datetime.now(UTC),
    )
    return deflate(registry, rets, variance_kind="backtest")


def row(m: Any, trades: int | None) -> dict[str, Any]:
    return {
        "return": m.total_return,
        "max_drawdown": m.max_drawdown,
        "sharpe": m.sharpe,
        "sortino": m.sortino,
        "trades": trades,
    }


def trial12_side(path: Path) -> dict[str, Any] | None:
    """Trial 12's graded run from its saved report (trial 12 is not run again)."""
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    return {
        "verdict": r["verdict"],
        "runs": r["runs"]["spycore"][str(GRADED)],
        "deflated_sharpe": r["deflated"]["spycore"]["dsr"],
        "checks": {c["name"]: c["passed"] for c in r["checks"]},
    }


def run_trial13(
    settings: Settings,
    rows: pd.DataFrame,
    at_945: At945,
    registry: Registry,
    out: Path = OUT,
    frozen_dir: Path = sc.FROZEN,
    trial12_report: Path = TRIAL12_REPORT,
) -> dict[str, Any]:
    if not preregistered(registry):
        raise RuntimeError("trial 13 is not pre-registered with this spec; nothing was run")
    sealed.check(t9.TEST_END, "the test window")
    s = account_settings(settings, OVERLAY, STARTING_EQUITY)
    test = sc.test_rows(rows)
    t9.register_kind(GRADED)
    model = pickle.loads((frozen_dir / f"model_h{GRADED}.pkl").read_bytes())
    bars = base_bars(s, t9.TEST_END)
    sigs = sc.signals(test, model.predict(test), GRADED)
    runs = replay(s, bars, sigs, at_945)
    rep = {w: r for w, (r, _) in runs.items()}
    adv = {(a, d): v for a, d, v in zip(test["symbol"], test["session"], test["adv"], strict=True)}
    trades = t9.trade_table(rep["test"].result, bars["SPY"], adv)
    base = t9.baselines(s, bars)
    deflated = log_trial(registry, rep)
    chk = t9.checks(rep, base, deflated["dsr"])
    passed = all(c["passed"] for c in chk)
    n_test = len(rep["test"].result.trades)
    report = {
        "trial": 13,
        "label": "IN-SAMPLE for the overlay: trial 12 already ran on this window",
        "passed": passed,
        "verdict": t9.verdict(passed, n_test),
        "checks": chk,
        "deflated": deflated,
        "runs": {w: row(r.run.metrics, len(r.result.trades)) for w, r in rep.items()},
        "hit_rate": float(trades["beat_spy"].mean()) if len(trades) else None,
        "overlay": {w: overlay_report(r.result, ov) for w, (r, ov) in runs.items()},
        "worst_drop": {w: worst_drop(r.result) for w, r in rep.items()},
        "spy_core": sc.sweep_report(rep["test"].result, float(STARTING_EQUITY)),
        "trial12": trial12_side(trial12_report),
        "baselines": {k: row(v.metrics, None) for k, v in base.items()},
        "by_cap": t9.by_bucket(trades),
        "spec_hash": spec_hash(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    trades.to_csv(out / "trades_h1.csv", index=False)
    pd.DataFrame(
        {"equity": rep["test"].result.equity, "exposure": rep["test"].result.exposure}
    ).to_csv(out / "equity_test.csv")
    return report
