"""Walk-forward training, the calibration report, the ablation and the promotion gate.

The protocol, in order:
1. Build the point-in-time dataset (survivorship-free universe, labels, features).
2. Cut purged, embargoed folds: train on the past, test on the next slice.
3. In every fold, fit each candidate model and its calibrator on that fold's
   training rows only, then predict the test slice. Those predictions are
   out of sample by construction.
4. Pool the out-of-sample predictions. Pick the candidate with the lowest
   pooled log loss (ties go to the lower Brier score), never by accuracy.
5. Log every candidate as a trial in the experiment registry.
6. A final model for live use is fit on every row whose label is complete by
   the cutoff, with the same calibration split.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa import calibration as cal
from stoa.config import Settings
from stoa.data.base import Bars
from stoa.dataset import Dataset, FeatureSet, build_dataset
from stoa.experiments import Registry, hash_frame_like
from stoa.membership import Membership
from stoa.model import ArtifactMeta, ModelStore, TrainedModel, fit, model_id
from stoa.walkforward import Fold, make_folds


@dataclass
class TrainingInputs:
    bars: Bars
    membership: Membership | None = None
    fundamentals: pd.DataFrame | None = None
    text_scores: pd.DataFrame | None = None
    extra_symbols: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


def data_hash(inputs: TrainingInputs) -> str:
    summary = {
        s: [len(df), str(df.index[0].date()), str(df.index[-1].date()), float(df["close"].sum())]
        for s, df in sorted(inputs.bars.items())
        if len(df)
    }
    parts = [hash_frame_like(summary)]
    for extra in (inputs.fundamentals, inputs.text_scores):
        parts.append(hash_frame_like(extra) if extra is not None and not extra.empty else "-")
    return hash_frame_like(parts)


def dataset_for(
    settings: Settings, inputs: TrainingInputs, features: FeatureSet, horizon: int | None = None
) -> Dataset:
    m = settings.model
    return build_dataset(
        inputs.bars,
        settings.benchmark,
        horizon or m.horizon,
        membership=inputs.membership,
        extra_symbols=inputs.extra_symbols,
        exclude=inputs.exclude,
        fundamentals=inputs.fundamentals,
        text_scores=inputs.text_scores,
        features=features,
        min_price=m.min_price,
        min_dollar_volume=m.min_dollar_volume_20,
        text_lookback_days=m.text_lookback_days,
    )


def folds_for(settings: Settings, ds: Dataset, end: date | None = None) -> list[Fold]:
    m = settings.model
    return make_folds(
        ds.sessions,
        horizon=ds.horizon,
        embargo=max(m.embargo, ds.horizon),
        test_sessions=m.test_sessions,
        min_train_sessions=m.min_train_sessions,
        window=m.train_window,
        rolling_sessions=m.rolling_sessions,
        end=end,
    )


def slice_train_rows(ds: Dataset, fold: Fold) -> pd.DataFrame:
    """Rows decided in the fold's training sessions whose labels end before the test slice."""
    p = ds.panel
    days = p.index.get_level_values("date")
    in_train = (days >= pd.Timestamp(fold.train_start)) & (days <= pd.Timestamp(fold.train_end))
    purged = p["label_end"].notna() & (p["label_end"] < pd.Timestamp(fold.test_start))
    return p.loc[in_train & purged & p["y"].notna()]


def slice_test_rows(ds: Dataset, fold: Fold) -> pd.DataFrame:
    p = ds.panel
    days = p.index.get_level_values("date")
    return p.loc[(days >= pd.Timestamp(fold.test_start)) & (days <= pd.Timestamp(fold.test_end))]


@dataclass
class FoldResult:
    fold: Fold
    kind: str
    model: TrainedModel
    predictions: pd.DataFrame  # index (date, symbol): p, y, fwd_excess
    scores: cal.Scores | None


@dataclass
class WalkForwardResult:
    feature_set: str
    horizon: int
    kinds: tuple[str, ...]
    folds: list[Fold]
    results: dict[str, list[FoldResult]]
    chosen: str
    data_hash: str
    pooled: dict[str, cal.Scores] = field(default_factory=dict)

    def oos(self, kind: str | None = None) -> pd.DataFrame:
        k = kind or self.chosen
        frames = [r.predictions.assign(fold=r.fold.number) for r in self.results[k]]
        return pd.concat(frames).sort_index() if frames else pd.DataFrame()

    def scored(self, kind: str | None = None) -> pd.DataFrame:
        o = self.oos(kind)
        return o.loc[o["y"].notna()] if not o.empty else o

    def model_for(self, session: date, kind: str | None = None) -> TrainedModel | None:
        for r in self.results[kind or self.chosen]:
            if r.fold.test_start <= session <= r.fold.test_end:
                return r.model
        return None


def _score_or_none(df: pd.DataFrame) -> cal.Scores | None:
    s = df.loc[df["y"].notna()]
    if len(s) < 20 or s["y"].nunique() < 2:
        return None
    return cal.score(s["p"].to_numpy(), s["y"].to_numpy())


def run_walkforward(
    settings: Settings,
    inputs: TrainingInputs,
    features: FeatureSet,
    *,
    kinds: tuple[str, ...] | None = None,
    horizon: int | None = None,
    end: date | None = None,
    dataset: Dataset | None = None,
    progress: Callable[[str], None] | None = None,
    select_before: date | None = None,
) -> WalkForwardResult:
    """Walk-forward over every candidate. ``select_before`` limits model choice to folds
    that end before that date, so a later window (validation) never influences the pick."""
    m = settings.model
    ds = dataset or dataset_for(settings, inputs, features, horizon)
    folds = folds_for(settings, ds, end)
    if not folds:
        raise ValueError(
            f"not enough history for one fold: {len(ds.sessions)} sessions, need "
            f"{m.min_train_sessions + m.embargo + 1}"
        )
    kinds = kinds or tuple(m.candidates)
    results: dict[str, list[FoldResult]] = {k: [] for k in kinds}
    for fold in folds:
        tr = slice_train_rows(ds, fold)
        te = slice_test_rows(ds, fold)
        for k in kinds:
            mdl = fit(
                tr,
                ds.features,
                kind=k,
                horizon=ds.horizon,
                embargo=max(m.embargo, ds.horizon),
                calibration=m.calibration,
                calibration_fraction=m.calibration_fraction,
                seed=m.seed,
            )
            pred = pd.DataFrame(
                {"p": mdl.predict(te), "y": te["y"], "fwd_excess": te["fwd_excess"]},
                index=te.index,
            )
            if end is not None:
                # A test label that finishes after ``end`` would use prices past the
                # cutoff (the sealed holdout, before it is scored). Never score it.
                late = te["label_end"].isna() | (te["label_end"] > pd.Timestamp(end))
                pred.loc[late.to_numpy(), ["y", "fwd_excess"]] = np.nan
            results[k].append(FoldResult(fold, k, mdl, pred, _score_or_none(pred)))
        if progress:
            progress(f"fold {fold.number}: test {fold.test_start} to {fold.test_end}")
    wf = WalkForwardResult(
        feature_set=features.name,
        horizon=ds.horizon,
        kinds=kinds,
        folds=folds,
        results=results,
        chosen=kinds[0],
        data_hash=data_hash(inputs),
    )
    for k in kinds:
        s = wf.scored(k)
        if len(s) and s["y"].nunique() == 2:
            wf.pooled[k] = cal.score(s["p"].to_numpy(), s["y"].to_numpy())
    select = wf.pooled
    if select_before is not None:
        select = {}
        for k in kinds:
            sc = wf.scored(k)
            sc = sc.loc[sc.index.get_level_values("date") < pd.Timestamp(select_before)]
            if len(sc) and sc["y"].nunique() == 2:
                select[k] = cal.score(sc["p"].to_numpy(), sc["y"].to_numpy())
    if select:
        wf.chosen = min(select, key=lambda k: (select[k].log_loss, select[k].brier))
    return wf


def fit_final(settings: Settings, ds: Dataset, kind: str, cutoff: date) -> TrainedModel:
    """The live model: every row whose label is complete by ``cutoff``."""
    m = settings.model
    rows = ds.rows_known_by(cutoff)
    return fit(
        rows,
        ds.features,
        kind=kind,
        horizon=ds.horizon,
        embargo=max(m.embargo, ds.horizon),
        calibration=m.calibration,
        calibration_fraction=m.calibration_fraction,
        seed=m.seed,
    )


# ---- strategy-level out-of-sample returns --------------------------------------------------


def long_short_returns(oos: pd.DataFrame, top: int, threshold: float) -> pd.Series:
    """Non-overlapping returns of the top-ranked names minus the average eligible name.

    Every ``horizon``-th decision day, take the names above ``threshold`` (at
    most ``top``) and record their average forward return minus the average
    forward return of every name the model scored that day. That isolates the
    skill of the ranking: if the average stock happens to beat SPY over the
    period, a random basket would too, and that is not edge. Days with no name
    above the threshold count as zero (the money sits out).

    Consecutive values never overlap, so the series has honest volatility and
    an honest observation count. (Averaging overlapping 10 day returns every
    day would shrink the volatility by about the square root of 10 and inflate
    the Sharpe by the same factor.)

    A cost-free summary used for the registry's Sharpe and the DSR. The
    trading backtest, with every rule and cost, against SPY, is the
    scoreboard. The series carries ``attrs["period"]``, the number of sessions
    each value spans.
    """
    h = max(1, int(oos.attrs.get("horizon", 1)))
    s = oos.loc[oos["fwd_excess"].notna()]
    if s.empty:
        out = pd.Series(dtype=float)
        out.attrs["period"] = h
        return out
    days = sorted(set(s.index.get_level_values("date")))[::h]
    s = s.loc[s.index.get_level_values("date").isin(days)]

    def day(g: pd.DataFrame) -> float:
        pick = g.loc[g["p"] > threshold].nlargest(top, "p")
        if not len(pick):
            return 0.0
        return float(pick["fwd_excess"].mean() - g["fwd_excess"].mean())

    out = s.groupby(level="date").apply(day).astype(float)
    out.attrs["period"] = h
    return out


# ---- registry -----------------------------------------------------------------------------


def log_trials(
    registry: Registry,
    settings: Settings,
    wf: WalkForwardResult,
    *,
    tags: list[str] | None = None,
    name_prefix: str = "walkforward",
) -> dict[str, str]:
    """One registry line per candidate. Returns experiment ids by kind."""
    ids = {}
    for k in wf.kinds:
        o = wf.scored(k)
        o.attrs["horizon"] = wf.horizon
        rets = long_short_returns(o, settings.model.max_candidates, settings.model.threshold)
        period = int(rets.attrs.get("period", 1))
        sr_p = float(rets.mean() / rets.std(ddof=1)) if len(rets) > 2 and rets.std() > 0 else 0.0
        sr = sr_p / math.sqrt(period)  # stored per session so every trial is comparable
        pooled = wf.pooled.get(k)
        metrics: dict[str, Any] = {"sharpe_daily": sr, "folds": len(wf.folds)}
        if pooled is not None:
            metrics.update(pooled.to_dict())
        exp = registry.append(
            kind="train",
            name=f"{name_prefix}:{wf.feature_set}:{k}:h{wf.horizon}",
            config_hash=settings.strategy_hash(),
            data_hash=wf.data_hash,
            params={
                "kind": k,
                "features": wf.feature_set,
                "horizon": wf.horizon,
                "test_sessions": settings.model.test_sessions,
                "embargo": settings.model.embargo,
                "calibration": settings.model.calibration,
                "window": settings.model.train_window,
            },
            metrics=metrics,
            returns=rets if len(rets) else None,
            tags=list(tags or []),
        )
        ids[k] = exp.id
    return ids


# ---- reports ------------------------------------------------------------------------------


def _fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def calibration_report(wf: WalkForwardResult) -> dict[str, Any]:
    out: dict[str, Any] = {
        "feature_set": wf.feature_set,
        "horizon": wf.horizon,
        "chosen": wf.chosen,
        "folds": [],
        "pooled": {k: v.to_dict() for k, v in wf.pooled.items()},
    }
    s = wf.scored()
    if len(s):
        p, y = s["p"].to_numpy(), s["y"].to_numpy()
        out["reliability"] = [b.__dict__ for b in cal.reliability(p, y)]
        out["buckets"] = [b.__dict__ for b in cal.bucket_table(p, y)]
    for r in wf.results[wf.chosen]:
        out["folds"].append(
            {
                "fold": r.fold.number,
                "train": [str(r.fold.train_start), str(r.fold.train_end)],
                "test": [str(r.fold.test_start), str(r.fold.test_end)],
                "scores": r.scores.to_dict() if r.scores else None,
            }
        )
    return out


def render_calibration(rep: dict[str, Any]) -> str:
    lines = [
        f"## Walk-forward calibration ({rep['feature_set']}, {rep['horizon']} sessions)",
        "",
        f"Chosen model: **{rep['chosen']}** (lowest pooled out-of-sample log loss).",
        "",
        "| Model | Rows | Base rate | Log loss | Brier | AUC | ECE | Hit rate when p > 0.5 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for k, s in rep["pooled"].items():
        lines.append(
            f"| {k} | {s['n']} | {_fmt(s['base_rate'])} | {_fmt(s['log_loss'], 4)} "
            f"| {_fmt(s['brier'], 4)} | {_fmt(s['auc'])} | {_fmt(s['ece'])} "
            f"| {_fmt(s['hit_rate_above_half'])} ({s['n_above_half']}) |"
        )
    if rep.get("buckets"):
        lines += [
            "",
            "Hit rate by probability bucket (chosen model):",
            "",
            "| Bucket | Rows | Mean p | Hit rate |",
            "| --- | --- | --- | --- |",
        ]
        for b in rep["buckets"]:
            lines.append(
                f"| {b['lo']:.2f} to {b['hi']:.2f} | {b['count']} | {_fmt(b['mean_p'])} "
                f"| {_fmt(b['hit_rate'])} |"
            )
    lines += [
        "",
        "Per fold (chosen model):",
        "",
        "| Fold | Test window | Rows | Log loss | Brier | AUC | ECE |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in rep["folds"]:
        s = f["scores"]
        if s is None:
            lines.append(
                f"| {f['fold']} | {f['test'][0]} to {f['test'][1]} | 0 | n/a | n/a | n/a | n/a |"
            )
            continue
        lines.append(
            f"| {f['fold']} | {f['test'][0]} to {f['test'][1]} | {s['n']} | "
            f"{_fmt(s['log_loss'], 4)} | {_fmt(s['brier'], 4)} | {_fmt(s['auc'])} "
            f"| {_fmt(s['ece'])} |"
        )
    return "\n".join(lines) + "\n"


def write_report(rep: dict[str, Any], out: Path, name: str = "calibration") -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    (out / f"{name}.md").write_text(render_calibration(rep), encoding="utf-8")
    return out


# ---- ablation -------------------------------------------------------------------------------


def ablation(
    settings: Settings,
    inputs: TrainingInputs,
    *,
    cutoff: date,
    kind: str | None = None,
) -> dict[str, Any]:
    """Out-of-sample scores with and without text features, before and after ``cutoff``.

    Before ``cutoff`` (the model's knowledge cutoff) the analyst may remember
    how events played out, so those scores are flagged as possibly contaminated.
    Only the after-cutoff comparison is clean evidence that text adds anything.
    """
    k = kind or settings.model.candidates[0]
    out: dict[str, Any] = {"cutoff": str(cutoff), "kind": k, "runs": {}}
    m = settings.model
    for label, text in (("without_text", False), ("with_text", True)):
        fs = FeatureSet(
            quality=m.use_quality, text=text, price_columns=m.price_features, normalize=m.normalize
        )
        wf = run_walkforward(settings, inputs, fs, kinds=(k,))
        s = wf.scored(k)
        days = s.index.get_level_values("date")
        parts = {}
        for name, mask in (
            ("pre_cutoff_contaminated", days < pd.Timestamp(cutoff)),
            ("post_cutoff_clean", days >= pd.Timestamp(cutoff)),
            ("all", np.ones(len(s), dtype=bool)),
        ):
            sub = s.loc[mask]
            parts[name] = (
                cal.score(sub["p"].to_numpy(), sub["y"].to_numpy()).to_dict()
                if len(sub) >= 20 and sub["y"].nunique() == 2
                else None
            )
        out["runs"][label] = parts
    return out


def render_ablation(rep: dict[str, Any]) -> str:
    lines = [
        f"## Text ablation ({rep['kind']}), knowledge cutoff {rep['cutoff']}",
        "",
        "Scores before the cutoff may be contaminated: the analyst may remember what happened.",
        "",
        "| Run | Period | Rows | Log loss | Brier | AUC |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run, parts in rep["runs"].items():
        for period, s in parts.items():
            if s is None:
                lines.append(f"| {run} | {period} | 0 | n/a | n/a | n/a |")
            else:
                lines.append(
                    f"| {run} | {period} | {s['n']} | {_fmt(s['log_loss'], 4)} "
                    f"| {_fmt(s['brier'], 4)} | {_fmt(s['auc'])} |"
                )
    return "\n".join(lines) + "\n"


# ---- promotion gate ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    promote: bool
    reason: str
    candidate: dict[str, float]
    incumbent: dict[str, float]


def promotion_gate(
    candidate: pd.DataFrame,
    incumbent: pd.DataFrame | None,
    *,
    tolerance: float,
    max_ece: float,
) -> GateResult:
    """Promote only if the new model is at least as good out of sample on the same rows.

    Both frames hold out-of-sample predictions (``p``) and labels (``y``) on the
    most recent folds. The candidate must have log loss no worse than the
    incumbent's (plus ``tolerance``), Brier no worse, and calibration error
    under ``max_ece``. With no incumbent, only the calibration and a
    better-than-coin-flip log loss are required.
    """
    c = candidate.loc[candidate["y"].notna()]
    if len(c) < 50 or c["y"].nunique() < 2:
        return GateResult(False, "not enough out-of-sample rows to judge", {}, {})
    cs = cal.score(c["p"].to_numpy(), c["y"].to_numpy())
    cd = {"log_loss": cs.log_loss, "brier": cs.brier, "ece": cs.ece, "n": float(cs.n)}
    coin = cal.log_loss(np.full(len(c), c["y"].mean()), c["y"].to_numpy())
    if cs.ece > max_ece:
        return GateResult(False, f"calibration error {cs.ece:.3f} over {max_ece}", cd, {})
    if incumbent is None:
        if cs.log_loss > coin:
            return GateResult(
                False, f"log loss {cs.log_loss:.4f} worse than the base rate {coin:.4f}", cd, {}
            )
        return GateResult(True, "first model; calibrated and better than the base rate", cd, {})
    old_p = incumbent[["p"]].rename(columns={"p": "p_old"})
    joined = c[["p", "y"]].join(old_p, how="inner").dropna()
    if len(joined) < 50:
        return GateResult(False, "too few shared rows with the current model", cd, {})
    new = cal.score(joined["p"].to_numpy(), joined["y"].to_numpy())
    old = cal.score(joined["p_old"].to_numpy(), joined["y"].to_numpy())
    cd = {"log_loss": new.log_loss, "brier": new.brier, "ece": new.ece, "n": float(new.n)}
    od = {"log_loss": old.log_loss, "brier": old.brier, "ece": old.ece, "n": float(old.n)}
    if new.log_loss > old.log_loss + tolerance:
        return GateResult(
            False, f"log loss {new.log_loss:.4f} worse than current {old.log_loss:.4f}", cd, od
        )
    if new.brier > old.brier + tolerance:
        return GateResult(
            False, f"Brier {new.brier:.4f} worse than current {old.brier:.4f}", cd, od
        )
    return GateResult(True, "no worse than the current model out of sample", cd, od)


def save_model(
    store: ModelStore,
    model: TrainedModel,
    *,
    settings: Settings,
    sha: str,
    data_hash: str,
    oos: dict[str, Any],
    experiment_id: str,
    created: str,
) -> ArtifactMeta:
    version = (
        created[:10].replace("-", "")
        + "-"
        + model_id({"k": model.kind, "d": data_hash, "c": created, "e": experiment_id})
    )
    meta = ArtifactMeta(
        version=version,
        created=created,
        kind=model.kind,
        features=model.features,
        horizon=model.horizon,
        train_start=str(model.train_start),
        train_end=str(model.train_end),
        payoff_ratio=model.payoff_ratio,
        code_sha=sha,
        config_hash=settings.strategy_hash(),
        data_hash=data_hash,
        oos=oos,
        experiment_id=experiment_id,
    )
    store.save(model, meta)
    return meta


def predictor_from_walkforward(wf: WalkForwardResult, kind: str | None = None) -> Any:
    """A backtest predictor that only ever shows out-of-sample probabilities."""
    from stoa.model_strategy import PanelPredictor

    k = kind or wf.chosen
    ratios: dict[date, float] = {}
    for r in wf.results[k]:
        for d in r.fold.test_sessions:
            ratios[d] = r.model.payoff_ratio
    return PanelPredictor(panel=wf.oos(k)[["p"]], ratios=ratios, name=f"model:{k}")
