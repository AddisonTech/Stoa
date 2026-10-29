"""Trial 14: trial 12 exactly, with idle money held as cash instead of the SPY core
(docs/DECISIONS.md).

Trial 9's frozen 1 session model, the 13 features, p of at least 0.55 and up to 10 a
session, 9:45 am fills, the sizing, the cost model and every trial 12 rail, unchanged.
Only idle money differs: it stays in cash, so the account's result is the brain's picks
alone. Its rules are written to the registry before the run.

Trial 12 already ran on this window, the same signals with idle money in cash included
(at $100), so this run is in-sample. The live big paper account is the grade.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from stoa.backtest import BacktestResult
from stoa.config import Settings
from stoa.event_run import base_bars
from stoa.experiments import Registry, deflate, return_moments
from stoa.newsbrain import ROOT, sealed
from stoa.newsbrain import spycore as sc
from stoa.newsbrain import trial as t9

SPEC = "trial14:newsbrain:cash:v1"
GRADED = sc.GRADED  # 1 session
OUT = ROOT / "trial14"
TRIAL12_REPORT = ROOT / "trial12" / "report.json"
STARTING_EQUITY = Decimal("100000")  # the big paper account
NAME = "trial14:cash:h1"
DECIDED = date(2026, 10, 29)


def spec() -> dict[str, Any]:
    """The pre-registered rules, as written to the registry before any run."""
    return {
        "spec": SPEC,
        "decided": f"{DECIDED} by the owner",
        "question": "what trial 12's picks earn alone, with idle money held as cash",
        "reason": "the owner wants the account's result to show the brain's picks only, not "
        "the market's move on the money parked in SPY",
        "same_as_trial_12": f"spec {sc.spec_hash()}: trial 9's frozen 1 session model (no "
        "refit), the 13 features, p >= 0.55 and up to 10 a session best first, SPY never a "
        "pick, buy at 9:45 am and sell at 9:45 am one session later, 20% of equity a position "
        "and 5 positions, the cost model, and every trial 12 rail: 10 orders a day, the 2% "
        "cash reserve, the 3% daily loss stop, the 200 day regime filter, the 40% sector cap, "
        "the 0.8 correlation limit, the close-based ATR exits, the 25% kill switch; no overlay "
        "and no broker-side stops",
        "idle_money": "cash: money not in a pick stays in cash and earns nothing; SPY is never "
        "bought or held (trial 9's rule, trial 12's cash comparison)",
        "account": "$100,000, the big paper account",
        "windows": "trial 12's: tune_c 2023-01-01 to 2024-12-31, validation 2025-01-01 to "
        "2026-06-30, test 2023-01-01 to 2026-06-30",
        "graded": "the 1 session cash run, on trial 9's pass bar, the same checks as trial 12",
        "reference": "trial 12's SPY core replayed at $100,000 beside it with the same code, "
        "so the only difference shown is where idle money sits; not logged as a trial. "
        "Trial 12's own saved report ($100) is shown as recorded and never rerun or changed",
        "logged_trials": "one configuration; its validation returns go in the registry as a "
        "backtest and count in the deflated Sharpe",
        "starved": "fewer than 30 closed trades in the test window: STARVED, not FAIL",
        "in_sample": "trial 12 already ran on this window, including these signals with idle "
        "money in cash at $100 (trial12:cash:h1): in-sample, a lead, not proof; the live big "
        "paper account is the real grade",
        "report": "total return, max drawdown, Sharpe, trade count, average share of money in "
        "cash and which checks pass, beside trial 12 and SPY",
        "live": "the big account sells its SPY at the next 9:45 am execute, alongside that "
        "day's picks, and never buys SPY again while big_paper.idle_money is cash; SPY stays "
        "the benchmark line",
        "holdout": "2026-07-01 to 2026-10-23 was spent by trial 12 and is never read again",
    }


def spec_hash() -> str:
    return hashlib.sha256(json.dumps(spec(), sort_keys=True).encode()).hexdigest()[:16]


def preregister(registry: Registry) -> None:
    """The rules, in the registry before any run. Not a trial: it holds no returns."""
    if any(
        e.kind == "preregistration" and e.name == "trial14:preregistration" for e in registry.load()
    ):
        return
    registry.append(
        kind="preregistration",
        name="trial14:preregistration",
        config_hash=spec_hash(),
        data_hash="none",
        params=spec(),
        metrics={},
    )


def preregistered(registry: Registry) -> bool:
    return any(
        e.kind == "preregistration"
        and e.name == "trial14:preregistration"
        and e.config_hash == spec_hash()
        for e in registry.load()
    )


# ---- the run ---------------------------------------------------------------------------------


def cash_share(res: BacktestResult) -> float:
    """The average share of the account in cash at each close (picks, and SPY when held, are
    invested)."""
    return float(1 - res.exposure.mean()) if len(res.exposure) else 1.0


def log_trial(registry: Registry, rep: dict[str, t9.Replay]) -> dict[str, Any]:
    starved = len(rep["test"].result.trades) < t9.MIN_TRADES
    rets = list(rep["validation"].run.daily_returns)
    registry.append(
        kind="backtest",
        name=f"{NAME}:validation",
        config_hash=f"{spec_hash()}-h{GRADED}",
        data_hash=hashlib.sha256(json.dumps(rets).encode()).hexdigest()[:16],
        params={"trial": 14, "entry": "cash", "horizon": GRADED, "window": "validation"},
        metrics={
            "sharpe_daily": return_moments(rets)[0],
            **rep["validation"].run.metrics.to_dict(),
        },
        returns=pd.Series(rets),
        tags=["starved"] if starved else [],
        now=datetime.now(UTC),
    )
    return deflate(registry, rets, variance_kind="backtest")


def row(m: Any, res: BacktestResult | None) -> dict[str, Any]:
    return {
        "return": m.total_return,
        "max_drawdown": m.max_drawdown,
        "sharpe": m.sharpe,
        "sortino": m.sortino,
        "trades": len(res.trades) if res is not None else None,
        "cash_share": cash_share(res) if res is not None else None,
    }


def trial12_side(path: Path) -> dict[str, Any] | None:
    """Trial 12's graded run and its cash comparison from its saved report ($100, not rerun)."""
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    return {
        "verdict": r["verdict"],
        "runs": r["runs"]["spycore"][str(GRADED)],
        "cash_runs": r["runs"]["cash"][str(GRADED)],
        "deflated_sharpe": r["deflated"]["spycore"]["dsr"],
        "checks": {c["name"]: c["passed"] for c in r["checks"]},
        "share_fully_in_spy": r["spy_core"][str(GRADED)].get("share_fully_in_spy"),
    }


def run_trial14(
    settings: Settings,
    rows: pd.DataFrame,
    at_945: sc.At945,
    registry: Registry,
    out: Path = OUT,
    frozen_dir: Path = sc.FROZEN,
    trial12_report: Path = TRIAL12_REPORT,
) -> dict[str, Any]:
    if not preregistered(registry):
        raise RuntimeError("trial 14 is not pre-registered with this spec; nothing was run")
    sealed.check(t9.TEST_END, "the test window")
    bt = settings.backtest.model_copy(update={"starting_equity": STARTING_EQUITY})
    s = settings.model_copy(update={"backtest": bt})
    test = sc.test_rows(rows)
    t9.register_kind(GRADED)
    model = pickle.loads((frozen_dir / f"model_h{GRADED}.pkl").read_bytes())
    bars = base_bars(s, t9.TEST_END)
    sigs = sc.signals(test, model.predict(test), GRADED)
    rep = sc.replay(s, bars, sigs, GRADED, at_945, f"news14cash_h{GRADED}", spy_core=False)
    ref = sc.replay(s, bars, sigs, GRADED, at_945, f"news14ref_h{GRADED}", spy_core=True)
    adv = {(a, d): v for a, d, v in zip(test["symbol"], test["session"], test["adv"], strict=True)}
    trades = t9.trade_table(rep["test"].result, bars["SPY"], adv)
    base = t9.baselines(s, bars)
    deflated = log_trial(registry, rep)
    chk = t9.checks(rep, base, deflated["dsr"])
    passed = all(c["passed"] for c in chk)
    ref_chk = [c for c in t9.checks(ref, base, float("nan")) if c["name"] != "deflated_sharpe"]
    report = {
        "trial": 14,
        "label": "IN-SAMPLE: trial 12 already ran on this window, these signals in cash included",
        "passed": passed,
        "verdict": t9.verdict(passed, len(rep["test"].result.trades)),
        "checks": chk,
        "deflated": deflated,
        "runs": {w: row(r.run.metrics, r.result) for w, r in rep.items()},
        "hit_rate": float(trades["beat_spy"].mean()) if len(trades) else None,
        "reference_spycore_100k": {
            "runs": {w: row(r.run.metrics, r.result) for w, r in ref.items()},
            "checks": ref_chk,
            "spy_core": sc.sweep_report(ref["test"].result, float(STARTING_EQUITY)),
        },
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
