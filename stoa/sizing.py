"""Position sizing by confidence and volatility. Pure functions, no network, no model.

Bigger when the model is sure and the stock is calm, smaller when not. The
output is a target weight (a fraction of equity) per new position. The risk
engine then clips every weight by every hard cap: 20% per position, 5
positions, the 2% cash buffer, 100% gross, 40% per sector, and the pairwise
correlation check. Sizing can only ever ask for less than the caps allow.

Steps:
1. Edge. ``linear``: p - 0.5. ``kelly``: a fraction of the Kelly bet on the
   calibrated probability, f = p - (1 - p) / b, where b is the average win over
   the average loss (in excess return) measured on the training fold.
2. Raw weight = edge / annualized volatility (floored, so a quiet stock cannot
   take an absurd weight).
3. Scale every new weight by one factor so the whole book (existing positions
   held fixed plus the new ones) lands at the portfolio volatility target,
   using the sample covariance of daily returns.
4. Multiply by the event-day factor (FOMC and CPI days).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass(frozen=True)
class SizingInput:
    symbol: str
    probability: float  # calibrated probability of beating SPY over the horizon
    volatility: float  # annualized


def linear_edge(p: float) -> float:
    return max(0.0, p - 0.5)


def kelly_edge(p: float, payoff_ratio: float, fraction: float) -> float:
    """Fractional Kelly for a bet that wins ``payoff_ratio`` per unit lost. Never negative."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability {p} outside [0, 1]")
    if payoff_ratio <= 0:
        return 0.0
    full = p - (1.0 - p) / payoff_ratio
    return max(0.0, fraction * full)


def raw_weights(
    items: Sequence[SizingInput],
    *,
    edge: str,
    payoff_ratio: float,
    kelly_fraction: float,
    vol_floor: float,
    min_edge: float = 0.0,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for it in items:
        if edge == "linear":
            e = linear_edge(it.probability)
        elif edge == "kelly":
            e = kelly_edge(it.probability, payoff_ratio, kelly_fraction)
        else:
            raise ValueError(f"unknown edge {edge!r}")
        if e <= min_edge:
            out[it.symbol] = 0.0
            continue
        vol = it.volatility if math.isfinite(it.volatility) else vol_floor
        out[it.symbol] = e / max(vol, vol_floor)
    return out


def portfolio_vol(weights: Mapping[str, float], cov: Mapping[str, Mapping[str, float]]) -> float:
    """Annualized volatility of a weight vector given an annualized covariance."""
    syms = [s for s, w in weights.items() if w != 0]
    if not syms:
        return 0.0
    w = np.array([weights[s] for s in syms])
    m = np.array([[cov.get(a, {}).get(b, 0.0) for b in syms] for a in syms])
    var = float(w @ m @ w)
    return math.sqrt(max(var, 0.0))


def scale_to_target(
    new: Mapping[str, float],
    existing: Mapping[str, float],
    cov: Mapping[str, Mapping[str, float]],
    target: float,
    max_scale: float = 1e6,
) -> float:
    """The factor k such that existing + k * new has annualized volatility ``target``.

    Existing weights are held fixed. If the existing book alone is already at or
    above the target, k is 0. Volatility rises monotonically in k for k >= 0
    when the new weights are long only, so a bisection finds it.
    """
    if not any(v > 0 for v in new.values()):
        return 0.0

    def vol_at(k: float) -> float:
        combined = dict(existing)
        for s, w in new.items():
            combined[s] = combined.get(s, 0.0) + k * w
        return portfolio_vol(combined, cov)

    if vol_at(0.0) >= target:
        return 0.0
    hi = 1.0
    while vol_at(hi) < target and hi < max_scale:
        hi *= 2.0
    if vol_at(hi) < target:
        return hi
    lo = 0.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if vol_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return lo


def size_positions(
    items: Sequence[SizingInput],
    existing: Mapping[str, float],
    cov: Mapping[str, Mapping[str, float]],
    *,
    edge: str = "kelly",
    payoff_ratio: float = 1.0,
    kelly_fraction: float = 0.25,
    vol_floor: float = 0.10,
    vol_target: float = 0.15,
    min_edge: float = 0.0,
    multiplier: float = 1.0,
) -> dict[str, float]:
    """Target weights for new positions, before the risk engine's caps."""
    if not 0 < multiplier <= 1:
        raise ValueError("multiplier must be in (0, 1]")
    raw = raw_weights(
        items,
        edge=edge,
        payoff_ratio=payoff_ratio,
        kelly_fraction=kelly_fraction,
        vol_floor=vol_floor,
        min_edge=min_edge,
    )
    k = scale_to_target(raw, existing, cov, vol_target)
    return {s: w * k * multiplier for s, w in raw.items()}


def covariance(returns: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    """Annualized sample covariance from aligned daily returns (equal length, no NaN)."""
    syms = sorted(returns)
    if not syms:
        return {}
    m = np.array([np.asarray(returns[s], dtype=float) for s in syms])
    if m.shape[1] < 2:
        return {a: dict.fromkeys(syms, 0.0) for a in syms}
    c = np.cov(m, ddof=1) * TRADING_DAYS
    c = np.atleast_2d(c)
    return {a: {b: float(c[i, j]) for j, b in enumerate(syms)} for i, a in enumerate(syms)}


def correlation(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Pearson correlation of two aligned return series, None when undefined."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) != len(y) or len(x) < 3:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])
