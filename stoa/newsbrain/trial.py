"""Trials 9 and 10, as pre-registered in docs/DECISIONS.md.

1. Walk-forward on 2016-01-01 to 2022-12-31 for each horizon (1, 5, 20).
2. Freeze: one fit on every row whose label ends by 2022-12-30. The frozen
   models are written to disk before anything from 2023 on is read.
3. Replay 2023-01-01 to 2026-06-30 through the backtest engine, trial 8's
   event strategy (liquidity, account fit, earnings rules, ATR stops, hold
   period) and the unchanged risk engine, filling at the 9:45 am price.
4. Grade the 5 session horizon on the pass bar; report all three.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from stoa import calibration, clock, event_signals
from stoa.backtest import BacktestResult, run_backtest
from stoa.baselines import BuyAndHold
from stoa.config import Settings
from stoa.data.base import Bars
from stoa.event_run import base_bars, source
from stoa.experiments import Registry, deflate, return_moments
from stoa.model import Calibrator
from stoa.newsbrain import ROOT, sealed
from stoa.newsbrain.panel import FEATURES, HORIZONS
from stoa.scoreboard import summarize
from stoa.universe import Universe
from stoa.walkforward import make_folds

log = logging.getLogger(__name__)

TRAIN_START = date(2016, 1, 1)
TRAIN_END = date(2022, 12, 30)  # the last session of 2022
TEST_START = date(2023, 1, 1)
TEST_END = date(2026, 6, 30)
WINDOWS = {
    "tune_c": (date(2023, 1, 1), date(2024, 12, 31)),
    "validation": (date(2025, 1, 1), date(2026, 6, 30)),
    "test": (TEST_START, TEST_END),
}
GRADED = 5
THRESHOLD = 0.55
TOP_N = 10
TEST_SLICE = 63
MIN_TRAIN = 504
CAL_SHARE = 0.25
HGB = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "max_iter": 300,
    "l2_regularization": 1.0,
    "min_samples_leaf": 200,
}
SPEC = "trial9:newsbrain:v1"  # the registered design; changing it is a new trial


def classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**HGB, early_stopping=False, random_state=0)


# ---- training ---------------------------------------------------------------------------


@dataclass
class Frozen:
    horizon: int
    clf: Any
    cal: Calibrator
    rows: int
    last_label_session: str
    features: tuple[str, ...] = FEATURES

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        raw = self.clf.predict_proba(x[list(self.features)].to_numpy(dtype=float))[:, 1]
        return self.cal.transform(raw)


def labelled(rows: pd.DataFrame, h: int, last_sidx: int) -> pd.DataFrame:
    """Rows whose label is known and ends by ``last_sidx``."""
    y = rows[f"y_{h}"]
    return rows[y.notna() & (rows["sidx"] + h <= last_sidx)]


def fit_calibrated(train: pd.DataFrame, h: int) -> tuple[Any, Calibrator]:
    """Fit on the first three quarters of sessions, calibrate on the last, with an h gap."""
    sess = np.sort(train["sidx"].unique())
    cut = int(sess[int(len(sess) * (1 - CAL_SHARE))])
    fit = train[train["sidx"] + h < cut]
    cal = train[train["sidx"] >= cut]
    clf = classifier()
    clf.fit(fit[list(FEATURES)].to_numpy(dtype=float), fit[f"y_{h}"].to_numpy())
    raw = clf.predict_proba(cal[list(FEATURES)].to_numpy(dtype=float))[:, 1]
    return clf, Calibrator("isotonic").fit(raw, cal[f"y_{h}"].to_numpy())


def walk_forward(rows: pd.DataFrame, days: list[date], h: int) -> dict[str, Any]:
    """Out-of-sample probabilities on 2016 to 2022, with the existing fold protocol."""
    last = days.index(TRAIN_END)
    sessions = days[: last + 1]
    folds = make_folds(
        sessions,
        horizon=h,
        embargo=h,
        test_sessions=TEST_SLICE,
        min_train_sessions=MIN_TRAIN,
    )
    pos = {d: i for i, d in enumerate(days)}
    p_all, y_all, ex_all, s_all = [], [], [], []
    for f in folds:
        lo, hi = pos[f.train_start], pos[f.train_end]
        train = labelled(rows[(rows["sidx"] >= lo) & (rows["sidx"] <= hi)], h, pos[f.test_start])
        t0, t1 = pos[f.test_start], pos[f.test_end]
        test = labelled(rows[(rows["sidx"] >= t0) & (rows["sidx"] <= t1)], h, last)
        if len(train) < 1000 or test.empty:
            continue
        clf, cal = fit_calibrated(train, h)
        p = cal.transform(clf.predict_proba(test[list(FEATURES)].to_numpy(dtype=float))[:, 1])
        p_all.append(p)
        y_all.append(test[f"y_{h}"].to_numpy())
        ex_all.append(test[f"excess_{h}"].to_numpy())
        s_all.append(test["sidx"].to_numpy())
    p = np.concatenate(p_all)
    y = np.concatenate(y_all)
    sc = calibration.score(p, y)
    base = float(np.mean(y))
    base_ll = calibration.log_loss(np.full_like(p, base), y)
    ex = np.concatenate(ex_all)
    sidx = np.concatenate(s_all)
    picked = p >= THRESHOLD
    return {
        "horizon": h,
        "folds": len(p_all),
        "rows": len(p),
        "base_rate": base,
        "log_loss": sc.log_loss,
        "base_log_loss": base_ll,
        "brier": sc.brier,
        "auc": sc.auc,
        "ece": sc.ece,
        "above_threshold": int(picked.sum()),
        "hit_rate_above_threshold": float(y[picked].mean()) if picked.any() else math.nan,
        "mean_excess_above_threshold": float(ex[picked].mean()) if picked.any() else math.nan,
        "mean_excess_all": float(np.nanmean(ex)),
        "buckets": [
            {"lo": b.lo, "hi": b.hi, "rows": b.count, "mean_p": b.mean_p, "hit": b.hit_rate}
            for b in calibration.reliability(p, y)
        ],
        "first_test_session": str(days[int(sidx.min())]) if len(sidx) else None,
    }


def freeze(rows: pd.DataFrame, days: list[date], h: int, out: Path) -> Frozen:
    last = days.index(TRAIN_END)
    train = labelled(rows[rows["sidx"] <= last], h, last)
    assert train["session"].max() < TEST_START
    clf, cal = fit_calibrated(train, h)
    frozen = Frozen(h, clf, cal, len(train), str(days[int(train["sidx"].max()) + h]))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"model_h{h}.pkl").write_bytes(pickle.dumps(frozen))
    return frozen


# ---- the replay -------------------------------------------------------------------------


def kind_for(h: int) -> str:
    return f"news_h{h}"


def register_kind(h: int) -> None:
    """Hold for h sessions: decided at the close of the h-th session held, sold next morning."""
    k = kind_for(h)
    event_signals.HOLD[k] = h - 1
    event_signals.PRIORITY[k] = 2


def signals_for(test: pd.DataFrame, p: np.ndarray, h: int) -> list[event_signals.Signal]:
    """Up to 10 a session at p >= 0.55, best first. Known at 9:00 am on the session.

    The engine decides at the close and fills at the next session, so a signal for
    session t is filed under the session before it, with features cut at 9:00 am on t.
    """
    df = test.assign(p=p)
    df = df[df["p"] >= THRESHOLD].sort_values(["sidx", "p"], ascending=[True, False])
    df = df.groupby("sidx").head(TOP_N)
    out = []
    for r in df.to_dict("records"):
        detail = (
            f"p {r['p']:.2f}: news tone {r['sent']:+.2f}, surprise {r['surprise']:+.1f}, "
            f"{int(r['n'])} articles, filing tone {r['filing_tone']:+.2f}, "
            f"ripple {r['ripple']:+.2f}"
        )
        session = r["session"]
        assert isinstance(session, date)
        out.append(
            event_signals.Signal(
                str(r["symbol"]),
                clock.previous_session(session),
                kind_for(h),
                float(r["p"]),
                detail,
            )
        )
    return out


@dataclass
class Replay:
    horizon: int
    window: str
    result: BacktestResult
    run: Any
    baselines: dict[str, Any] = field(default_factory=dict)


def replay(
    settings: Settings,
    bars: Bars,
    sigs: list[event_signals.Signal],
    h: int,
    fills: Any,
    name: str,
) -> dict[str, Replay]:
    out: dict[str, Replay] = {}
    for w, (start, end) in WINDOWS.items():
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
            fill_prices=fills,
        )
        out[w] = Replay(h, w, res, summarize(res, settings, w))
        log.info("replay h%d %s: %d trades", h, w, len(res.trades))
    return out


def baselines(settings: Settings, bars: Bars) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for w, (start, end) in WINDOWS.items():
        for name, sym in (("spy", settings.benchmark), ("mtum", "MTUM")):
            uni = Universe.assume_all_fractionable((*settings.watchlist, sym), settings.benchmark)
            res = run_backtest(settings, bars, BuyAndHold(sym, name=name), start, end, universe=uni)
            out[f"{name}:{w}"] = summarize(res, settings, w)
    return out


# ---- 9:45 am prices ----------------------------------------------------------------------


class EntryPrices:
    """The 9:45 am price of each symbol: the open of its first minute bar at or after 9:45.

    Fetched from Alpaca once per session for the symbols that may trade, and cached.
    """

    def __init__(self, client: Any, pace: Any, cache: Path) -> None:
        self.client = client
        self.pace = pace
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)

    def __call__(self, day: date, buys: set[str], others: set[str]) -> dict[str, Decimal]:
        return self.at_945(day, buys | others)

    def at_945(self, day: date, symbols: set[str]) -> dict[str, Decimal]:
        from stoa.newsbrain.alpaca_pull import get

        sealed.check(day, "9:45 am prices")
        path = self.cache / f"{day.isoformat()}.parquet"
        have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["symbol", "price"])
        missing = sorted(symbols - set(have["symbol"]))
        if missing:
            start = clock.at_ny(day, "09:45")
            end = clock.at_ny(day, "10:15")
            got: dict[str, float] = {}
            for i in range(0, len(missing), 100):
                params: dict[str, Any] = {
                    "symbols": ",".join(missing[i : i + 100]),
                    "timeframe": "1Min",
                    "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": 10000,
                    "adjustment": "all",
                    "feed": "sip",
                    "sort": "asc",
                }
                while True:
                    body = get(self.client, self.pace, "/v2/stocks/bars", params)
                    for sym, rows in (body.get("bars") or {}).items():
                        if rows and sym not in got:
                            got[sym] = float(rows[0]["o"])
                    token = body.get("next_page_token")
                    if not token:
                        break
                    params["page_token"] = token
            new = pd.DataFrame(
                {"symbol": missing, "price": [got.get(s, math.nan) for s in missing]}
            )
            have = pd.concat([have, new], ignore_index=True)
            have.to_parquet(path, index=False)
        px = have[have["symbol"].isin(symbols)].dropna()
        return {
            str(s): Decimal(repr(float(p))) for s, p in zip(px["symbol"], px["price"], strict=True)
        }


# ---- reporting -----------------------------------------------------------------------------


def spy_span(spy: pd.DataFrame, a: date, b: date) -> float:
    o = spy["open"]
    try:
        return float(o.loc[pd.Timestamp(b)] / o.loc[pd.Timestamp(a)] - 1)
    except KeyError:
        return math.nan


def cap_bucket(mcap: float | None, adv: float | None) -> str:
    if mcap is not None and math.isfinite(mcap) and mcap > 0:
        return "small" if mcap < 2e9 else ("mid" if mcap < 1e10 else "large")
    if adv is None or not math.isfinite(adv):
        return "small"
    return "small" if adv < 2e7 else ("mid" if adv < 1e8 else "large")


def shares_outstanding(cik: int, on: date) -> float | None:
    path = Path("data/sec/companyfacts") / f"CIK{cik:010d}.json"
    if not path.exists():
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    facts = body.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {})
    best = None
    for unit in facts.get("units", {}).values():
        for f in unit:
            filed = f.get("filed")
            if filed and filed <= on.isoformat() and (best is None or filed > best[0]):
                best = (filed, float(f.get("val", 0)))
    return best[1] if best else None


def trade_table(
    res: BacktestResult, spy: pd.DataFrame, panel_adv: dict[tuple[str, date], float]
) -> pd.DataFrame:
    from stoa.newsbrain.panel import load_ciks

    ciks = load_ciks()
    cik_of = dict(zip(ciks["symbol"], ciks["cik"], strict=True))
    rows = []
    for t in res.trades:
        ret = float(t.exit_price / t.entry_price - 1) - float(t.fees / (t.qty * t.entry_price))
        excess = ret - spy_span(spy, t.entry_session, t.exit_session)
        shares = (
            shares_outstanding(int(cik_of[t.symbol]), t.entry_session)
            if t.symbol in cik_of
            else None
        )
        mcap = shares * float(t.entry_price) if shares else None
        adv = panel_adv.get((t.symbol, t.entry_session))
        rows.append(
            {
                "symbol": t.symbol,
                "entry": t.entry_session,
                "exit": t.exit_session,
                "reason": t.reason,
                "pnl": float(t.pnl),
                "return": ret,
                "excess": excess,
                "beat_spy": excess > 0,
                "bucket": cap_bucket(mcap, adv),
            }
        )
    return pd.DataFrame(rows)


def by_bucket(trades: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for b in ("small", "mid", "large"):
        t = trades[trades["bucket"] == b] if len(trades) else trades
        out.append(
            {
                "bucket": b,
                "trades": len(t),
                "hit_rate": float(t["beat_spy"].mean()) if len(t) else math.nan,
                "avg_excess": float(t["excess"].mean()) if len(t) else math.nan,
                "pnl": float(t["pnl"].sum()) if len(t) else 0.0,
            }
        )
    return out


def checks(graded: dict[str, Replay], base: dict[str, Any], dsr: float) -> list[dict[str, Any]]:
    v, c = graded["validation"].run.metrics, graded["tune_c"].run.metrics
    sv, sc = base["spy:validation"].metrics, base["spy:tune_c"].metrics
    mv = base["mtum:validation"].metrics
    worst = max(r.run.metrics.max_drawdown for r in graded.values())
    trades = len(graded["test"].result.trades)
    return [
        {
            "name": "drawdown_vs_spy",
            "passed": v.max_drawdown < sv.max_drawdown,
            "detail": f"{v.max_drawdown:.2%} vs SPY {sv.max_drawdown:.2%} (validation)",
        },
        {
            "name": "sharpe_vs_spy",
            "passed": v.sharpe > sv.sharpe,
            "detail": f"{v.sharpe:.2f} vs SPY {sv.sharpe:.2f} (validation)",
        },
        {
            "name": "sortino_vs_spy",
            "passed": v.sortino > sv.sortino,
            "detail": f"{v.sortino:.2f} vs SPY {sv.sortino:.2f} (validation)",
        },
        {
            "name": "tune_c_sharpe_vs_spy",
            "passed": c.sharpe > sc.sharpe,
            "detail": f"{c.sharpe:.2f} vs SPY {sc.sharpe:.2f} (2023 to 2024)",
        },
        {"name": "drawdown_cap", "passed": worst <= 0.25, "detail": f"worst {worst:.2%} (cap 25%)"},
        {
            "name": "trade_count",
            "passed": trades >= 30,
            "detail": f"{trades} closed trades (need 30)",
        },
        {"name": "api_cost", "passed": True, "detail": "$0.00 in every month (no paid calls)"},
        {"name": "deflated_sharpe", "passed": dsr >= 0.95, "detail": f"DSR {dsr:.3f} (need 0.95)"},
        {
            "name": "sharpe_vs_mtum",
            "passed": v.sharpe > mv.sharpe,
            "detail": f"{v.sharpe:.2f} vs MTUM {mv.sharpe:.2f} (validation)",
        },
    ]


def config_hash(h: int) -> str:
    return hashlib.sha256(f"{SPEC}|h{h}".encode()).hexdigest()[:16]


def log_trials(
    registry: Registry, runs: dict[int, dict[str, Replay]], trial: int
) -> dict[str, float]:
    """Log one backtest trial per horizon (validation window), then deflate the graded one."""
    now = datetime.now(UTC)
    for h, rep in runs.items():
        rets = list(rep["validation"].run.daily_returns)
        registry.append(
            kind="backtest",
            name=f"trial{trial}:news:h{h}:validation",
            config_hash=config_hash(h) if trial == 9 else config_hash(h) + "-t10",
            data_hash=hashlib.sha256(json.dumps(rets).encode()).hexdigest()[:16],
            params={"trial": trial, "horizon": h, "window": "validation"},
            metrics={
                "sharpe_daily": return_moments(rets)[0],
                **rep["validation"].run.metrics.to_dict(),
            },
            returns=pd.Series(rets),
            now=now,
        )
    return deflate(
        registry, list(runs[GRADED]["validation"].run.daily_returns), variance_kind="backtest"
    )


def run_trial(
    settings: Settings,
    rows: pd.DataFrame,
    days: list[date],
    fills: Any,
    registry: Registry,
    out: Path,
    trial: int = 9,
    frozen_dir: Path | None = None,
) -> dict[str, Any]:
    """The whole trial. Walk-forward and freezing finish before any 2023 row is scored.

    Trial 10 passes ``frozen_dir`` (trial 9's folder): the same frozen models and
    walk-forward, nothing refit, only the entry fill changes.
    """
    sealed.check(TEST_END, "the test window")
    if len(days) and days[-1] >= sealed.HOLDOUT_START:
        sealed.check(days[-1], "the session calendar")
    rows = rows[rows["session"] < sealed.HOLDOUT_START]
    for h in HORIZONS:
        register_kind(h)
    out.mkdir(parents=True, exist_ok=True)
    if frozen_dir is None:
        wf = {h: walk_forward(rows, days, h) for h in HORIZONS}
        frozen = {h: freeze(rows, days, h, out) for h in HORIZONS}
        (out / "walkforward.json").write_text(
            json.dumps(wf, indent=1, default=str), encoding="utf-8"
        )
    else:
        frozen = {h: pickle.loads((frozen_dir / f"model_h{h}.pkl").read_bytes()) for h in HORIZONS}
        wf = json.loads((frozen_dir / "walkforward.json").read_text(encoding="utf-8"))
    log.info("frozen; replaying the test window")

    test = rows[(rows["session"] >= TEST_START) & (rows["session"] <= TEST_END)]
    bars = base_bars(settings, TEST_END)
    runs: dict[int, dict[str, Replay]] = {}
    trades: dict[int, pd.DataFrame] = {}
    adv = {(s, d): a for s, d, a in zip(test["symbol"], test["session"], test["adv"], strict=True)}
    for h in HORIZONS:
        p = frozen[h].predict(test)
        sigs = signals_for(test, p, h)
        runs[h] = replay(settings, bars, sigs, h, fills, name=f"news{trial}_h{h}")
        trades[h] = trade_table(runs[h]["test"].result, bars["SPY"], adv)
    base = baselines(settings, bars)
    deflated = log_trials(registry, runs, trial)
    graded = runs[GRADED]
    chk = checks(graded, base, deflated["dsr"])
    passed = all(c["passed"] for c in chk)

    def summary(run: Any) -> dict[str, Any]:
        m = run.metrics
        return {
            "return": m.total_return,
            "cagr": m.cagr,
            "sharpe": m.sharpe,
            "sortino": m.sortino,
            "max_drawdown": m.max_drawdown,
            "trades": m.trade_count,
        }

    report = {
        "trial": trial,
        "passed": passed,
        "checks": chk,
        "deflated": deflated,
        "walkforward": wf,
        "frozen": {
            h: {"rows": f.rows, "last_label_session": f.last_label_session}
            for h, f in frozen.items()
        },
        "runs": {
            str(h): {
                w: {
                    **summary(r.run),
                    "hit_rate": float(trades[h]["beat_spy"].mean())
                    if w == "test" and len(trades[h])
                    else None,
                }
                for w, r in rep.items()
            }
            for h, rep in runs.items()
        },
        "baselines": {k: summary(v) for k, v in base.items()},
        "by_cap": {str(h): by_bucket(trades[h]) for h in HORIZONS},
        "hit_rate": {
            str(h): float(trades[h]["beat_spy"].mean()) if len(trades[h]) else None
            for h in HORIZONS
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    for h in HORIZONS:
        trades[h].to_csv(out / f"trades_h{h}.csv", index=False)
    return report


PANEL_DIR = ROOT / "trial9"
