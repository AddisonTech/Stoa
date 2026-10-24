"""Calibration and probabilistic scoring. When it says 60%, is it right about 60% of the time?

All functions take plain arrays: ``p`` (predicted probability of beating SPY)
and ``y`` (1 if it did, 0 if not).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

EPS = 1e-12


def _arrays(p: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    pa = np.asarray(p, dtype=float)
    ya = np.asarray(y, dtype=float)
    if pa.shape != ya.shape:
        raise ValueError("p and y must have the same shape")
    if pa.size and ((pa < 0).any() or (pa > 1).any()):
        raise ValueError("probabilities must be in [0, 1]")
    if ya.size and not np.isin(ya, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")
    return pa, ya


def brier(p: Any, y: Any) -> float:
    pa, ya = _arrays(p, y)
    return float(np.mean((pa - ya) ** 2)) if pa.size else math.nan


def log_loss(p: Any, y: Any) -> float:
    pa, ya = _arrays(p, y)
    if not pa.size:
        return math.nan
    q = np.clip(pa, EPS, 1 - EPS)
    return float(-np.mean(ya * np.log(q) + (1 - ya) * np.log(1 - q)))


def auc(p: Any, y: Any) -> float:
    """Area under the ROC curve via the rank-sum formula, ties counted as half."""
    pa, ya = _arrays(p, y)
    pos = int(ya.sum())
    neg = int(ya.size - pos)
    if pos == 0 or neg == 0:
        return math.nan
    order = np.argsort(pa, kind="mergesort")
    ranks = np.empty(pa.size)
    sorted_p = pa[order]
    i = 0
    while i < pa.size:
        j = i
        while j + 1 < pa.size and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = float(ranks[ya == 1].sum())
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    count: int
    mean_p: float
    hit_rate: float


def reliability(p: Any, y: Any, bins: int = 10) -> list[Bin]:
    """Equal-width bins on [0, 1]. The last bin includes 1.0."""
    pa, ya = _arrays(p, y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(pa, edges[1:-1], right=False), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        out.append(
            Bin(
                lo=float(edges[b]),
                hi=float(edges[b + 1]),
                count=n,
                mean_p=float(pa[m].mean()) if n else math.nan,
                hit_rate=float(ya[m].mean()) if n else math.nan,
            )
        )
    return out


def ece(p: Any, y: Any, bins: int = 10) -> float:
    """Expected calibration error: count-weighted mean |hit rate - mean p| over bins."""
    pa, _ = _arrays(p, y)
    if not pa.size:
        return math.nan
    total = 0.0
    for b in reliability(p, y, bins):
        if b.count:
            total += b.count * abs(b.hit_rate - b.mean_p)
    return total / pa.size


@dataclass(frozen=True)
class Scores:
    n: int
    base_rate: float
    log_loss: float
    brier: float
    auc: float
    ece: float
    hit_rate_above_half: float
    n_above_half: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def score(p: Any, y: Any, bins: int = 10) -> Scores:
    pa, ya = _arrays(p, y)
    above = pa > 0.5
    return Scores(
        n=int(pa.size),
        base_rate=float(ya.mean()) if pa.size else math.nan,
        log_loss=log_loss(pa, ya),
        brier=brier(pa, ya),
        auc=auc(pa, ya),
        ece=ece(pa, ya, bins),
        hit_rate_above_half=float(ya[above].mean()) if above.any() else math.nan,
        n_above_half=int(above.sum()),
    )


def bucket_table(
    p: Any, y: Any, edges: tuple[float, ...] = (0.0, 0.45, 0.5, 0.55, 0.6, 0.65, 1.0)
) -> list[Bin]:
    """Hit rate by probability bucket, for the 'hit rate by probability' report."""
    pa, ya = _arrays(p, y)
    out = []
    for lo, hi in itertools.pairwise(edges):
        m = (pa >= lo) & ((pa < hi) if hi < 1.0 else (pa <= hi))
        n = int(m.sum())
        out.append(
            Bin(
                lo=lo,
                hi=hi,
                count=n,
                mean_p=float(pa[m].mean()) if n else math.nan,
                hit_rate=float(ya[m].mean()) if n else math.nan,
            )
        )
    return out
