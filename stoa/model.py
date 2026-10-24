"""The prediction model: a plain, honestly trained classifier of "beats SPY over H sessions".

Two candidates:
- ``logistic``: L2-regularized logistic regression on the standardized
  features. The baseline.
- ``hgb``: scikit-learn's HistGradientBoostingClassifier, small and
  regularized. The challenger.

Calibration happens inside each training set only. The training sessions are
split by time: the earlier part fits the classifier, the later
``calibration_fraction`` (with an embargo gap of H sessions between them)
fits an isotonic or Platt calibrator on the classifier's scores. Nothing from
the test slice ever touches either step.

A trained model is saved as a versioned artifact (pickle plus JSON metadata):
the code SHA, config hash, data hash, feature list, training window and the
out-of-sample metrics it was promoted on.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ARTIFACT_VERSION = 1
P_MIN, P_MAX = 0.01, 0.99


class Classifier(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> Any: ...

    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...


def make_classifier(kind: str, seed: int) -> Classifier:
    if kind == "logistic":
        return cast(Classifier, LogisticRegression(C=0.1, max_iter=2000))
    if kind == "hgb":
        return cast(
            Classifier,
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_depth=3,
                min_samples_leaf=200,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=seed,
            ),
        )
    raise ValueError(f"unknown model kind {kind!r}")


class Calibrator:
    """Isotonic regression or Platt scaling on raw scores, clipped to [0.01, 0.99]."""

    def __init__(self, method: str) -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"unknown calibration {method!r}")
        self.method = method
        self._iso: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> Calibrator:
        if self.method == "isotonic":
            self._iso = IsotonicRegression(y_min=P_MIN, y_max=P_MAX, out_of_bounds="clip")
            self._iso.fit(raw, y)
        else:
            logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
            self._platt = LogisticRegression(C=1e6, max_iter=1000)
            self._platt.fit(logit.reshape(-1, 1), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self._iso is not None:
            out = self._iso.predict(raw)
        elif self._platt is not None:
            logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
            out = self._platt.predict_proba(logit.reshape(-1, 1))[:, 1]
        else:
            raise RuntimeError("calibrator is not fitted")
        return np.clip(np.asarray(out, dtype=float), P_MIN, P_MAX)


@dataclass
class TrainedModel:
    kind: str
    features: list[str]
    horizon: int
    classifier: Any
    calibrator: Calibrator
    train_start: date
    train_end: date
    payoff_ratio: float  # average winning excess return over average losing one, from training
    meta: dict[str, Any] = field(default_factory=dict)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in x.columns]
        if missing:
            raise KeyError(f"features missing at prediction time: {missing}")
        arr = x[self.features].to_numpy(dtype=float)
        if not len(arr):
            return np.array([], dtype=float)
        raw = self.classifier.predict_proba(arr)[:, 1]
        return self.calibrator.transform(raw)


def _split_for_calibration(
    sessions: list[date], fraction: float, embargo: int
) -> tuple[list[date], list[date]]:
    n = len(sessions)
    n_cal = max(1, int(n * fraction))
    fit_end = n - n_cal - embargo
    if fit_end < max(20, embargo):
        raise ValueError("training window too short to split for calibration")
    return sessions[:fit_end], sessions[n - n_cal :]


def payoff_ratio(excess: pd.Series) -> float:
    wins = excess[excess > 0]
    losses = excess[excess <= 0]
    if wins.empty or losses.empty or losses.mean() == 0:
        return 1.0
    return float(wins.mean() / -losses.mean())


def fit(
    train: pd.DataFrame,
    features: list[str],
    *,
    kind: str,
    horizon: int,
    embargo: int,
    calibration: str,
    calibration_fraction: float,
    seed: int,
) -> TrainedModel:
    """Fit classifier then calibrator on a time split of ``train`` (indexed by date, symbol)."""
    if train.empty:
        raise ValueError("no training rows")
    dates = sorted({d.date() for d in train.index.get_level_values("date")})
    fit_days, cal_days = _split_for_calibration(dates, calibration_fraction, embargo)
    day_index = train.index.get_level_values("date")
    fit_rows = train.loc[day_index <= pd.Timestamp(fit_days[-1])]
    cal_rows = train.loc[day_index >= pd.Timestamp(cal_days[0])]
    if fit_rows["y"].nunique() < 2 or cal_rows["y"].nunique() < 2:
        raise ValueError("training labels are all one class")
    clf = make_classifier(kind, seed)
    clf.fit(fit_rows[features].to_numpy(dtype=float), fit_rows["y"].to_numpy(dtype=int))
    raw_cal = clf.predict_proba(cal_rows[features].to_numpy(dtype=float))[:, 1]
    cal = Calibrator(calibration).fit(raw_cal, cal_rows["y"].to_numpy(dtype=float))
    return TrainedModel(
        kind=kind,
        features=list(features),
        horizon=horizon,
        classifier=clf,
        calibrator=cal,
        train_start=dates[0],
        train_end=dates[-1],
        payoff_ratio=payoff_ratio(train["fwd_excess"].dropna()),
        meta={"fit_rows": len(fit_rows), "cal_rows": len(cal_rows)},
    )


# ---- artifacts ------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactMeta:
    version: str
    created: str
    kind: str
    features: list[str]
    horizon: int
    train_start: str
    train_end: str
    payoff_ratio: float
    code_sha: str
    config_hash: str
    data_hash: str
    oos: dict[str, Any]
    experiment_id: str = ""
    artifact_version: int = ARTIFACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_id(meta_seed: dict[str, Any]) -> str:
    blob = json.dumps(meta_seed, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


class ModelStore:
    """``<dir>/<version>/model.pkl`` and ``meta.json``, plus ``current.json`` naming the live one.

    Artifacts live on the state branch next to the rest of Stoa's durable
    state, so a scheduled run can always load the model it was promoted with.
    Pickles are only ever loaded from this directory, which only Stoa writes.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, model: TrainedModel, meta: ArtifactMeta) -> Path:
        d = self.root / meta.version
        d.mkdir(parents=True, exist_ok=True)
        with (d / "model.pkl").open("wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (d / "meta.json").write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
        return d

    def load(self, version: str) -> tuple[TrainedModel, ArtifactMeta]:
        d = self.root / version
        with (d / "model.pkl").open("rb") as fh:
            model = pickle.load(fh)
        meta = ArtifactMeta(**json.loads((d / "meta.json").read_text(encoding="utf-8")))
        if not isinstance(model, TrainedModel):
            raise TypeError(f"{d} does not hold a TrainedModel")
        return model, meta

    def current_version(self) -> str | None:
        p = self.root / "current.json"
        if not p.exists():
            return None
        return str(json.loads(p.read_text(encoding="utf-8"))["version"])

    def set_current(self, version: str, reason: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        hist = self.root / "promotions.jsonl"
        record = {"version": version, "at": datetime.now(UTC).isoformat(), "reason": reason}
        with hist.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        tmp = self.root / "current.json.tmp"
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        tmp.replace(self.root / "current.json")

    def load_current(self) -> tuple[TrainedModel, ArtifactMeta] | None:
        v = self.current_version()
        return None if v is None else self.load(v)

    def versions(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "meta.json").exists())
