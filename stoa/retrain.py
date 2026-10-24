"""`stoa model train` and the monthly retrain: walk-forward, report, gate, promote.

1. Load inputs and run the walk-forward with every candidate model.
2. Write the calibration report to results/ and log every candidate to the
   experiment registry.
3. Judge the walk-forward result (calibration and edge after the multiple
   testing penalty) and record it for the exam order.
4. Fit the final model on all labeled history and save it as a versioned
   artifact.
5. Promotion gate: the new model must be at least as good as the current
   model out of sample, on the same recent rows, or the current model stays.
6. Write the live expectation bands the drift fallback compares against.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from stoa.config import Settings
from stoa.dataset import Dataset
from stoa.exam import WalkForwardRecord, write_walkforward
from stoa.experiments import Registry, code_sha, deflate
from stoa.model import ModelStore
from stoa.pipeline import Inputs, feature_set, now_iso
from stoa.tracking import Expectation, expectation_from_returns, save_expectation
from stoa.training import (
    WalkForwardResult,
    calibration_report,
    dataset_for,
    fit_final,
    log_trials,
    long_short_returns,
    promotion_gate,
    render_calibration,
    run_walkforward,
    save_model,
)

log = logging.getLogger(__name__)


@dataclass
class TrainOutcome:
    wf: WalkForwardResult
    report_dir: Path
    passed: bool
    reasons: list[str]
    deflated: dict[str, float]
    version: str
    promoted: bool
    gate_reason: str
    expectation: Expectation | None
    experiment_ids: dict[str, str] = field(default_factory=dict)


def judge(
    settings: Settings, wf: WalkForwardResult, deflated: dict[str, float]
) -> tuple[bool, list[str]]:
    """Did the walk-forward pass? Calibrated, better than the base rate, and DSR high enough."""
    reasons: list[str] = []
    s = wf.pooled.get(wf.chosen)
    if s is None:
        return False, ["no pooled out-of-sample scores"]
    b = min(max(s.base_rate, 1e-9), 1 - 1e-9)
    base = -(b * math.log(b) + (1 - b) * math.log(1 - b))  # always predicting the base rate
    if s.log_loss >= base:
        reasons.append(f"log loss {s.log_loss:.4f} is not better than the base rate {base:.4f}")
    if s.ece > settings.model.max_ece:
        reasons.append(f"calibration error {s.ece:.3f} over {settings.model.max_ece}")
    dsr = deflated.get("dsr", math.nan)
    if not (math.isfinite(dsr) and dsr >= 0.95):
        reasons.append(f"deflated Sharpe {dsr:.3f} under 0.95 after the trial penalty")
    return not reasons, reasons


def recent_oos(wf: WalkForwardResult, kind: str, folds: int) -> pd.DataFrame:
    keep = {f.number for f in wf.folds[-folds:]}
    o = wf.oos(kind)
    return o.loc[o["fold"].isin(keep)] if len(o) else o


def incumbent_oos(
    store: ModelStore, settings: Settings, wf: WalkForwardResult, ds: Dataset
) -> pd.DataFrame | None:
    """The current model's predictions on the candidate's recent test rows."""
    cur = store.load_current()
    if cur is None:
        return None
    model, _ = cur
    rows = recent_oos(wf, wf.chosen, settings.model.gate_folds)
    if rows.empty:
        return None
    x = ds.panel.loc[rows.index]
    if any(c not in x.columns for c in model.features):
        return None
    return pd.DataFrame({"p": model.predict(x), "y": rows["y"]}, index=rows.index)


def train(
    settings: Settings,
    inputs: Inputs,
    *,
    cutoff: date,
    results_dir: Path,
    promote: bool = True,
    force_promote: bool = False,
    progress: Callable[[str], None] | None = None,
    tags: list[str] | None = None,
) -> TrainOutcome:
    s = settings
    fs = feature_set(s)
    ds = dataset_for(s, inputs.training, fs)
    wf = run_walkforward(s, inputs.training, fs, end=cutoff, dataset=ds, progress=progress)
    registry = Registry(s.registry_path)
    ids = log_trials(registry, s, wf, tags=tags)
    oos = wf.scored()
    oos.attrs["horizon"] = wf.horizon
    rets = long_short_returns(oos, s.model.max_candidates, s.model.threshold)
    deflated = deflate(registry, list(rets), period=int(rets.attrs.get("period", 1)))
    passed, reasons = judge(s, wf, deflated)

    created = now_iso()
    out = results_dir / f"{created[:19].replace(':', '').replace('-', '')}-model"
    rep = calibration_report(wf)
    rep["deflated"] = deflated
    rep["walkforward_passed"] = passed
    rep["reasons"] = reasons
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    md = render_calibration(rep) + render_verdict(passed, reasons, deflated)
    (out / "calibration.md").write_text(md, encoding="utf-8")

    final = fit_final(s, ds, wf.chosen, cutoff)
    store = ModelStore(s.artifact_dir)
    pooled = wf.pooled.get(wf.chosen)
    meta = save_model(
        store,
        final,
        settings=s,
        sha=code_sha(),
        data_hash=wf.data_hash,
        oos={**(pooled.to_dict() if pooled else {}), **deflated},
        experiment_id=ids.get(wf.chosen, ""),
        created=created,
    )
    gate = promotion_gate(
        recent_oos(wf, wf.chosen, s.model.gate_folds),
        incumbent_oos(store, s, wf, ds),
        tolerance=s.model.gate_tolerance,
        max_ece=s.model.max_ece,
    )
    promoted = False
    reason = gate.reason
    if promote and (gate.promote or force_promote):
        store.set_current(meta.version, "forced" if force_promote and not gate.promote else reason)
        promoted = True
        if force_promote and not gate.promote:
            reason = f"forced past the gate: {gate.reason}"
    registry.append(
        kind="promote_check",
        name=f"gate:{meta.version}",
        config_hash=s.strategy_hash(),
        data_hash=wf.data_hash,
        params={"version": meta.version, "promoted": promoted},
        metrics={"candidate": gate.candidate, "incumbent": gate.incumbent, "reason": reason},
    )
    expectation: Expectation | None = None
    daily = daily_book_returns(oos, s.model.max_candidates, s.model.threshold, wf.horizon)
    if len(daily) > s.fallback.sharpe_window + 2:
        expectation = expectation_from_returns(
            list(daily), s.fallback.sharpe_window, source=meta.version
        )
        if promoted:
            save_expectation(s.shared_dir, expectation)
    write_walkforward(
        s.shared_dir,
        WalkForwardRecord(
            created=created,
            passed=passed,
            reasons=reasons,
            checks={"pooled": pooled.to_dict() if pooled else {}, "deflated": deflated},
            report_path=str(out),
            model_version=meta.version,
        ),
    )
    return TrainOutcome(
        wf=wf,
        report_dir=out,
        passed=passed,
        reasons=reasons,
        deflated=deflated,
        version=meta.version,
        promoted=promoted,
        gate_reason=reason,
        expectation=expectation,
        experiment_ids=ids,
    )


def daily_book_returns(oos: pd.DataFrame, top: int, threshold: float, horizon: int) -> pd.Series:
    """Approximate daily returns of a book that buys the top names and holds them ``horizon`` days.

    Each day's book is the average of the last ``horizon`` daily cohorts, the
    standard way to turn overlapping forecasts into a tradable daily series.
    Each cohort's return is spread evenly across its days. Used only for the
    live expectation bands (rolling Sharpe and drawdown) the drift fallback
    compares against; total excess over SPY, before costs.
    """
    s = oos.loc[oos["fwd_excess"].notna()]
    if s.empty:
        return pd.Series(dtype=float)

    def day(g: pd.DataFrame) -> float:
        pick = g.loc[g["p"] > threshold].nlargest(top, "p")
        return float(pick["fwd_excess"].mean()) if len(pick) else 0.0

    cohort = s.groupby(level="date").apply(day).astype(float)
    per_day = (1.0 + cohort) ** (1.0 / horizon) - 1.0
    return per_day.rolling(horizon, min_periods=1).mean()


def render_verdict(passed: bool, reasons: list[str], deflated: dict[str, Any]) -> str:
    lines = [
        "",
        f"## Walk-forward verdict: {'PASS' if passed else 'FAIL'}",
        "",
        f"Deflated Sharpe {deflated.get('dsr', math.nan):.3f} after "
        f"{int(deflated.get('trials', 0))} recorded trials "
        f"(cost-free top-{int(deflated.get('top', 0)) or 'N'} basket, for ranking only; "
        "the trading backtest is in the scoreboard).",
    ]
    lines += [f"- {r}" for r in reasons]
    return "\n".join(lines) + "\n"
