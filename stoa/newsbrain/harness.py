"""Known-answer test of the trial harness (docs/DECISIONS.md, "Harness known-answer test").

Three strategies whose answers are known go through exactly the code trial 9 was
graded with: trial 9's replay (the event strategy, the backtest engine, the
simulated broker, the risk engine and its rails, the liquidity cost model), 9:45 am
fills, trial 9's trade table and its pass bar. The test window is 2023-01-01 to
2026-06-30; the holdout is never read.

a. Buy and hold the S&P 500 through IVV (the event strategy never trades the benchmark
   symbol, SPY, by design): an IVV signal every session, held until a rail exits it.
   Known answer: daily returns equal to the book's exposure times IVV's, less costs, so
   its Sharpe sits near SPY's.
b. The cheater: each session's 10 rows with the best realised 5 session excess return
   (lookahead on purpose). Known answer: every check passes with a huge Sharpe.
c. Random picks, 5 seeds, 2 a session held 5 sessions (the book stays near its 5
   positions, as a normal run): known answer, the average trade beats SPY by what the
   average eligible row does, after the same costs.

These are harness checks, not trials: nothing is written to the experiment registry
and they never count in the deflated Sharpe. Results go to data/newsbrain/harness/.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa import clock, event_signals
from stoa.backtest import BacktestResult
from stoa.config import Settings
from stoa.event_run import base_bars
from stoa.experiments import Registry, deflate
from stoa.models import Side
from stoa.newsbrain import ROOT, sealed
from stoa.newsbrain import trial as t9
from stoa.newsbrain.panel import FEE
from stoa.newsbrain.timing import MinuteStore

log = logging.getLogger(__name__)

OUT = ROOT / "harness"
HOLD_KIND = "harness_hold"
INDEX_FUND = "IVV"  # tracks the S&P 500; SPY is the benchmark and never a pick
LEGS = ("spy", "cheater", "random")
SEEDS = (1, 2, 3, 4, 5)
RANDOM_PER_SESSION = 2
CHEATER_PER_SESSION = 10
H = t9.GRADED  # the 5 session horizon trial 9 was graded on
# Tolerances, fixed before the run.
SPY_SHARPE_GAP = 0.35  # the SPY book's Sharpe within this of SPY's
SPY_RETURN_GAP = 0.03  # its return within 3 points of exposure times SPY, after costs
CHEATER_MIN_SHARPE = 3.0
RANDOM_SE = 3.0  # the random book's mean trade within 3 standard errors of the known answer


# ---- 9:45 am prices, as trial 9 took them, with no network --------------------------------


class Prices:
    """The 9:45 am price: trial 9's cached Alpaca price if it has one, else the open of the
    symbol's first minute bar from 9:45 to 10:15 (the same rule, from the bars on disk)."""

    def __init__(self, cache: Path, store: MinuteStore) -> None:
        self.cache = cache
        self.store = store
        self.memo: dict[tuple[date, str], float | None] = {}
        self.cached: dict[date, dict[str, float]] = {}
        self.sources = {"cache": 0, "minute bars": 0, "none": 0}

    def _cache(self, day: date) -> dict[str, float]:
        if day not in self.cached:
            p = self.cache / f"{day.isoformat()}.parquet"
            df = (
                pd.read_parquet(p).dropna()
                if p.exists()
                else pd.DataFrame(columns=["symbol", "price"])
            )
            self.cached[day] = {
                str(s): float(v) for s, v in zip(df["symbol"], df["price"], strict=True)
            }
        return self.cached[day]

    def one(self, day: date, symbol: str) -> float | None:
        key = (day, symbol)
        if key not in self.memo:
            sealed.check(day, "9:45 am prices")
            got = self._cache(day).get(symbol)
            if got is not None:
                self.sources["cache"] += 1
            else:
                bars = self.store.day(symbol, day)
                lo = pd.Timestamp(clock.at_ny(day, "09:45"))
                hi = pd.Timestamp(clock.at_ny(day, "10:15"))
                after = bars[(bars["t"] >= lo) & (bars["t"] <= hi)] if len(bars) else bars
                got = float(after["o"].iloc[0]) if len(after) else None
                self.sources["minute bars" if got is not None else "none"] += 1
            self.memo[key] = got
        return self.memo[key]

    def at_945(self, day: date, symbols: set[str]) -> dict[str, Decimal]:
        out = {}
        for s in sorted(symbols):
            p = self.one(day, s)
            if p is not None and math.isfinite(p) and p > 0:
                out[s] = Decimal(repr(round(p, 6)))
        return out

    def __call__(self, day: date, buys: set[str], others: set[str]) -> dict[str, Decimal]:
        return self.at_945(day, buys | others)


# ---- the three strategies ------------------------------------------------------------------


def register() -> None:
    t9.register_kind(H)
    event_signals.HOLD[HOLD_KIND] = 100_000
    event_signals.PRIORITY[HOLD_KIND] = 2


def spy_signals(days: list[date], symbol: str = INDEX_FUND) -> list[event_signals.Signal]:
    return [
        event_signals.Signal(symbol, clock.previous_session(d), HOLD_KIND, 1.0, "buy and hold")
        for d in days
    ]


def cheater_signals(test: pd.DataFrame) -> list[event_signals.Signal]:
    """Each session's best rows by the realised 5 session excess return: lookahead on purpose."""
    df = test[test[f"excess_{H}"].notna() & (test[f"excess_{H}"] > 0)]
    df = df.sort_values(["sidx", f"excess_{H}"], ascending=[True, False])
    df = df.groupby("sidx").head(CHEATER_PER_SESSION)
    return [
        event_signals.Signal(
            str(sym), clock.previous_session(session), t9.kind_for(H), float(score), "cheater"
        )
        for sym, session, score in zip(df["symbol"], df["session"], df[f"excess_{H}"], strict=True)
    ]


def random_signals(test: pd.DataFrame, seed: int) -> list[event_signals.Signal]:
    rng = np.random.default_rng(seed)
    out = []
    for _, g in test.groupby("sidx", sort=True):
        take = g.iloc[rng.permutation(len(g))[:RANDOM_PER_SESSION]]
        for sym, session in zip(take["symbol"], take["session"], strict=True):
            out.append(
                event_signals.Signal(
                    str(sym),
                    clock.previous_session(session),
                    t9.kind_for(H),
                    float(rng.random()),
                    f"random seed {seed}",
                )
            )
    return out


# ---- what the engine charged ---------------------------------------------------------------


@dataclass
class Charged:
    buys: int
    sells: int
    buy_slip_bps: float
    sell_slip_bps: float
    fee_bps: float

    @property
    def round_trip_bps(self) -> float:
        return self.buy_slip_bps + self.sell_slip_bps + self.fee_bps


def charged(res: BacktestResult, prices: Prices) -> Charged:
    """Average cost per fill against the 9:45 am price handed to the engine."""
    b, s, fees = [], [], []
    for f in res.fills:
        ref = prices.memo.get((f.session, f.symbol))
        if ref is None or not ref:
            continue
        px = float(f.price)
        if f.side is Side.BUY:
            b.append((px / ref - 1) * 1e4)
        else:
            s.append((1 - px / ref) * 1e4)
            gross = float(f.gross)
            fees.append(float(f.fees) / gross * 1e4 if gross else 0.0)
    return Charged(
        len(b),
        len(s),
        float(np.mean(b)) if b else math.nan,
        float(np.mean(s)) if s else math.nan,
        float(np.mean(fees)) if fees else math.nan,
    )


# ---- known answers ---------------------------------------------------------------------------


def spy_expected(res: BacktestResult, spy_close: pd.Series) -> float:
    """Total return of a book holding yesterday's SPY exposure at SPY's daily return."""
    ret = spy_close.pct_change().reindex(res.equity.index).fillna(0.0)
    exp = res.exposure.shift(1).fillna(0.0)
    growth = np.prod(1.0 + exp.to_numpy(dtype=float) * ret.to_numpy(dtype=float))
    return float(growth) - 1.0


def summary(run: Any) -> dict[str, Any]:
    m = run.metrics
    return {
        "return": m.total_return,
        "sharpe": m.sharpe,
        "sortino": m.sortino,
        "max_drawdown": m.max_drawdown,
        "trades": m.trade_count,
    }


def run(
    settings: Settings,
    rows: pd.DataFrame,
    registry: Registry,
    out: Path = OUT,
    legs: tuple[str, ...] = LEGS,
) -> dict[str, Any]:
    """Run the given legs; legs run before (in results.json) are kept when only some run."""
    sealed.check(t9.TEST_END, "the test window")
    register()
    rows = rows[rows["session"] < sealed.HOLDOUT_START]
    test = rows[(rows["session"] >= t9.TEST_START) & (rows["session"] <= t9.TEST_END)]
    days = sorted(set(test["session"]))
    bars = base_bars(settings, t9.TEST_END)
    prices = Prices(ROOT / "entry", MinuteStore())
    base = t9.baselines(settings, bars)
    spy_test = base["spy:test"].metrics
    adv = {(s, d): a for s, d, a in zip(test["symbol"], test["session"], test["adv"], strict=True)}
    old = out / "results.json"
    keep = old.exists() and set(legs) != set(LEGS)
    results: dict[str, Any] = json.loads(old.read_text(encoding="utf-8")) if keep else {}
    if "spy" in legs:
        results["spy"] = index_leg(settings, bars, days, prices, spy_test)
    if "cheater" in legs:
        results["cheater"] = cheater_leg(settings, bars, test, base, registry, prices)
    if "random" in legs:
        results["random"] = random_leg(settings, bars, test, adv, prices, spy_test)
    honest = all(bool(results.get(k, {}).get("passed")) for k in LEGS)
    verdict = {
        "verdict": "honest" if honest else "bug",
        "passed": {k: results.get(k, {}).get("passed") for k in LEGS},
        "prices": dict(prices.sources),
    }
    out.mkdir(parents=True, exist_ok=True)
    old.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    return {"verdict": verdict, "results": results}


def costs(c: Charged) -> dict[str, Any]:
    return c.__dict__ | {"round_trip_bps": c.round_trip_bps}


def index_leg(
    settings: Settings, bars: Any, days: list[date], prices: Prices, spy_test: Any
) -> dict[str, Any]:
    """a. Buy and hold the S&P 500 through IVV."""
    from stoa.newsbrain.panel import bar_frame

    rep = t9.replay(settings, bars, spy_signals(days), H, prices, name="harness_index")
    res = rep["test"].result
    fund = bar_frame(INDEX_FUND, sorted(days))
    assert fund is not None, f"no daily bars for {INDEX_FUND}"
    expected = spy_expected(res, fund["close"])
    got = rep["test"].run.metrics
    ok = (
        len(res.trades) + len(res.final_positions) > 0
        and abs(got.sharpe - spy_test.sharpe) <= SPY_SHARPE_GAP
        and abs(got.total_return - expected) <= SPY_RETURN_GAP
    )
    out = {
        "passed": bool(ok),
        "fund": INDEX_FUND,
        "expected": f"Sharpe near SPY's {spy_test.sharpe:.2f} (within {SPY_SHARPE_GAP}); return "
        f"near {expected:.2%}, the book's exposure ({float(res.exposure.mean()):.0%} on "
        f"average) times {INDEX_FUND} day by day (within {SPY_RETURN_GAP:.0%}); SPY returned "
        f"{spy_test.total_return:.2%}",
        "actual": summary(rep["test"].run),
        "spy": {"return": spy_test.total_return, "sharpe": spy_test.sharpe},
        "expected_return": expected,
        "exposure": float(res.exposure.mean()),
        "charged": costs(charged(res, prices)),
    }
    log.info("harness index: %s", out)
    return out


def cheater_leg(
    settings: Settings,
    bars: Any,
    test: pd.DataFrame,
    base: dict[str, Any],
    registry: Registry,
    prices: Prices,
) -> dict[str, Any]:
    """b. The cheater, graded on trial 9's pass bar."""
    rep = t9.replay(settings, bars, cheater_signals(test), H, prices, name="harness_cheater")
    dsr = deflate(registry, list(rep["validation"].run.daily_returns), variance_kind="backtest")
    chk = t9.checks(rep, base, dsr["dsr"])
    sharpe = rep["test"].run.metrics.sharpe
    out = {
        "passed": bool(all(x["passed"] for x in chk) and sharpe >= CHEATER_MIN_SHARPE),
        "expected": f"every check passes, Sharpe at least {CHEATER_MIN_SHARPE}",
        "actual": summary(rep["test"].run),
        "checks": chk,
        "dsr": dsr["dsr"],
        "charged": costs(charged(rep["test"].result, prices)),
    }
    log.info("harness cheater: passed %s, Sharpe %.2f", out["passed"], sharpe)
    return out


def random_leg(
    settings: Settings,
    bars: Any,
    test: pd.DataFrame,
    adv: dict[tuple[str, date], float],
    prices: Prices,
    spy_test: Any,
) -> dict[str, Any]:
    """c. Random picks, 5 seeds, against the average eligible row after the same costs."""
    known = test[f"excess_{H}"].dropna()
    seeds = []
    pooled = []
    for seed in SEEDS:
        rep = t9.replay(
            settings, bars, random_signals(test, seed), H, prices, name=f"harness_random{seed}"
        )
        trades = t9.trade_table(rep["test"].result, bars["SPY"], adv)
        pooled.append(trades)
        seeds.append(
            {
                "seed": seed,
                **summary(rep["test"].run),
                "mean_trade_excess": float(trades["excess"].mean()) if len(trades) else math.nan,
                "charged": costs(charged(rep["test"].result, prices)),
            }
        )
    allt = pd.concat(pooled, ignore_index=True)
    mean = float(allt["excess"].mean())
    se = float(allt["excess"].std(ddof=1) / math.sqrt(len(allt)))
    target = float(known.mean())
    pre_cost = float((known + 2 * test.loc[known.index, "slip"] + FEE).mean())
    out = {
        "passed": bool(abs(mean - target) <= RANDOM_SE * se),
        "expected": f"mean trade beats SPY by {target:+.2%} after costs (the average eligible "
        f"row; {pre_cost:+.2%} before costs), within {RANDOM_SE:.0f} standard errors",
        "actual": {"mean_trade_excess": mean, "standard_error": se, "trades": len(allt)},
        "seeds": seeds,
        "known_answer": target,
        "pre_cost": pre_cost,
        "spy": {"return": spy_test.total_return, "sharpe": spy_test.sharpe},
    }
    log.info("harness random: %s", out["actual"])
    return out
