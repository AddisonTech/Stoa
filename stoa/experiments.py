"""The experiment registry and the multiple-testing penalty.

Every training or backtest run appends one line to an append-only JSONL file:
when, the code SHA, a hash of the effective config, a hash of the data
snapshot, the parameters and the metrics. Nothing ever rewrites or deletes a
line. ``stoa experiments list`` reads it.

The penalty. If you try 100 versions, the best one looks good by luck alone.
The Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014) asks: given how
many versions were tried (N) and how much their Sharpe ratios varied, what is
the probability that the chosen version's true Sharpe is above zero, after
allowing for skew and fat tails in its returns?

    SR0  = sqrt(V) * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)))
    DSR  = Z( (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2) )

SR is the per-period (daily) Sharpe ratio, V the variance of the per-period
Sharpe ratios across trials, g the Euler-Mascheroni constant, T the number of
return observations, kurt the (non-excess) kurtosis. With N = 1 it reduces to
the Probabilistic Sharpe Ratio against zero.

Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado and Zhu,
2015) uses combinatorially symmetric cross-validation: split the shared
return history into S blocks, and for every way of picking half the blocks as
"in sample", find the best trial in sample and see where it ranks out of
sample. PBO is the share of splits where the in-sample winner lands in the
bottom half out of sample.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329
PERIODS_PER_YEAR = 252


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """SR0: the Sharpe the best of ``n_trials`` skill-less trials would show by luck."""
    if n_trials <= 1 or sharpe_variance <= 0:
        return 0.0
    g = EULER_GAMMA
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sharpe_variance) * ((1 - g) * a + g * b))


def probabilistic_sharpe(
    sharpe: float, benchmark: float, n_obs: int, skewness: float, kurt: float
) -> float:
    """PSR: probability the true per-period Sharpe exceeds ``benchmark``. ``kurt`` is not excess."""
    if n_obs < 2:
        return math.nan
    denom = 1.0 - skewness * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    if denom <= 0:
        return math.nan
    z = (sharpe - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(norm.cdf(z))


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    sharpe_variance: float,
    n_obs: int,
    skewness: float = 0.0,
    kurt: float = 3.0,
) -> tuple[float, float]:
    """(DSR, SR0) for a per-period Sharpe ratio."""
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    return probabilistic_sharpe(sharpe, sr0, n_obs, skewness, kurt), sr0


def return_moments(returns: Sequence[float]) -> tuple[float, int, float, float]:
    """(per-period Sharpe, n, skewness, non-excess kurtosis) of a return series."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 3 or float(np.std(r, ddof=1)) == 0:
        return 0.0, n, 0.0, 3.0
    sr = float(np.mean(r) / np.std(r, ddof=1))
    return sr, n, float(skew(r)), float(kurtosis(r, fisher=False))


def pbo(trial_returns: np.ndarray, blocks: int = 8) -> float:
    """Probability of Backtest Overfitting by CSCV.

    ``trial_returns`` is (T observations, N trials), all on the same dates.
    Returns NaN when there are fewer than 2 trials or too few observations.
    """
    m = np.asarray(trial_returns, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2 or m.shape[0] < blocks * 2 or blocks % 2:
        return math.nan
    chunks = np.array_split(np.arange(m.shape[0]), blocks)
    logits = []
    for ins in itertools.combinations(range(blocks), blocks // 2):
        is_idx = np.concatenate([chunks[i] for i in ins])
        oos_idx = np.concatenate([chunks[i] for i in range(blocks) if i not in ins])

        def sharpe_cols(rows: np.ndarray) -> np.ndarray:
            sub = m[rows]
            sd = sub.std(axis=0, ddof=1)
            return np.where(sd > 0, sub.mean(axis=0) / np.where(sd > 0, sd, 1), 0.0)

        best = int(np.argmax(sharpe_cols(is_idx)))
        oos = sharpe_cols(oos_idx)
        rank = (np.argsort(np.argsort(oos))[best] + 1) / (len(oos) + 1)
        logits.append(math.log(rank / (1 - rank)))
    return float(np.mean(np.array(logits) <= 0))


# ---- registry --------------------------------------------------------------------------


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def code_sha(repo: Path = PACKAGE_ROOT) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        sha = out.stdout.strip()
        if out.returncode != 0 or not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "stoa"],
            check=False,
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def hash_frame_like(obj: Any) -> str:
    """A stable hash of a pandas object or anything JSON-serializable."""
    try:
        import pandas as pd

        if isinstance(obj, (pd.DataFrame, pd.Series)):
            h = pd.util.hash_pandas_object(obj, index=True).to_numpy()
            return hashlib.sha256(h.tobytes()).hexdigest()[:16]
    except ImportError:  # pragma: no cover
        pass
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class Experiment:
    id: str
    created: str
    kind: str  # train | backtest | promote_check
    name: str
    code_sha: str
    config_hash: str
    data_hash: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    returns_path: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def sharpe_daily(self) -> float | None:
        v = self.metrics.get("sharpe_daily")
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


class Registry:
    """Append-only JSONL. Daily returns of each trial are stored beside it for PBO."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(
        self,
        *,
        kind: str,
        name: str,
        config_hash: str,
        data_hash: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        returns: Any | None = None,
        tags: list[str] | None = None,
        sha: str | None = None,
        now: datetime | None = None,
    ) -> Experiment:
        created = (now or datetime.now(UTC)).isoformat()
        sha = sha if sha is not None else code_sha()
        seed = {"c": created, "k": kind, "n": name, "h": config_hash, "d": data_hash, "p": params}
        eid = hashlib.sha256(json.dumps(seed, sort_keys=True, default=str).encode()).hexdigest()
        eid = eid[:12]
        returns_path = ""
        if returns is not None:
            rdir = self.path.parent / "returns"
            rdir.mkdir(parents=True, exist_ok=True)
            rp = rdir / f"{eid}.parquet"
            returns.to_frame("r").to_parquet(rp)
            returns_path = str(rp.relative_to(self.path.parent))
        exp = Experiment(
            id=eid,
            created=created,
            kind=kind,
            name=name,
            code_sha=sha,
            config_hash=config_hash,
            data_hash=data_hash,
            params=params,
            metrics=metrics,
            returns_path=returns_path,
            tags=list(tags or []),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(exp), sort_keys=True, default=str) + "\n")
        return exp

    def load(self) -> list[Experiment]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    out.append(Experiment(**json.loads(line)))
        return out

    def trials(self, kinds: tuple[str, ...] = ("train", "backtest")) -> list[Experiment]:
        return [e for e in self.load() if e.kind in kinds]

    @staticmethod
    def config_key(e: Experiment) -> str:
        """A trial is a configuration: the effective config plus the run's parameters.

        Re-running the same configuration (a monthly retrain on new data, or the
        same `stoa score` twice) is not a new try, so it does not add to the
        count. Changing anything that changes decisions is.
        """
        return json.dumps([e.config_hash, e.params], sort_keys=True, default=str)

    def unique_trials(self, kinds: tuple[str, ...] = ("train", "backtest")) -> list[Experiment]:
        latest: dict[str, Experiment] = {}
        for e in self.trials(kinds):
            latest[self.config_key(e)] = e
        return list(latest.values())

    def penalty_inputs(self, variance_kind: str = "train") -> tuple[int, float]:
        """(number of distinct configurations tried, variance of their daily Sharpe ratios).

        The count spans every kind of trial. The variance comes from trials of
        ``variance_kind`` only, so Sharpe ratios measured the same way are compared.
        """
        n = max(1, len(self.unique_trials()))
        vals = [
            s
            for s in (e.sharpe_daily for e in self.unique_trials((variance_kind,)))
            if s is not None
        ]
        if len(vals) < 2:
            # Too few trials of this kind to measure their spread: use every trial's
            # per-session Sharpe, so the penalty never quietly drops to zero.
            vals = [s for s in (e.sharpe_daily for e in self.unique_trials()) if s is not None]
        var = float(np.var(vals, ddof=1)) if len(vals) >= 2 else 0.0
        return n, var

    def trial_matrix(self) -> np.ndarray | None:
        """Aligned daily returns of every trial that stored them (for PBO)."""
        import pandas as pd

        cols = {}
        for e in self.unique_trials(("train",)):
            if not e.returns_path:
                continue
            p = self.path.parent / e.returns_path
            if p.exists():
                cols[e.id] = pd.read_parquet(p)["r"]
        if len(cols) < 2:
            return None
        df = pd.DataFrame(cols).dropna()
        return df.to_numpy() if len(df) else None


def deflate(
    registry: Registry,
    returns: Sequence[float],
    variance_kind: str = "train",
    period: int = 1,
) -> dict[str, float]:
    """DSR of a return series given every trial in the registry so far.

    ``period`` is how many sessions each return spans (1 for daily equity
    returns, H for non-overlapping H-session returns). The registry stores
    Sharpe ratios per session, so their variance is scaled to the series'
    period before computing the luck threshold.
    """
    sr, n, sk, ku = return_moments(returns)
    trials, var = registry.penalty_inputs(variance_kind)
    dsr, sr0 = deflated_sharpe(sr, trials, var * period, n, sk, ku)
    m = registry.trial_matrix() if period == 1 or variance_kind == "train" else None
    per_session = sr / math.sqrt(period)
    return {
        "sharpe_daily": per_session,
        "sharpe_annual": per_session * math.sqrt(PERIODS_PER_YEAR),
        "period": float(period),
        "observations": float(n),
        "trials": float(trials),
        "trial_sharpe_variance": var,
        "sr0_daily": sr0 / math.sqrt(period),
        "dsr": dsr,
        "psr": probabilistic_sharpe(sr, 0.0, n, sk, ku),
        "pbo": pbo(m) if m is not None else math.nan,
    }
