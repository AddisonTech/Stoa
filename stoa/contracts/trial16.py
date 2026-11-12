"""Trial 16: a big contract award to a small public company, bought at the next open.

``spec()`` is the rule book, written to the registry before any 2023 to 2026 result is
looked at. The order of work is fixed by ``run``: the known-answer check of this grader
(stop if it fails), the tuning on 2016 to 2022 only (each grid point logged as a trial),
then one test on 2023-01-01 to 2026-06-30 with the threshold frozen. The holdout (from
2026-07-01) stays sealed.

Grading is trial 15's rule, per trade: the stock's return from the entry open to the exit
close, minus SPY's over the same span, minus a round trip of slippage by dollar volume and
the fee. Hit rate is the share of trades above zero; edge is their mean.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa import clock, wide
from stoa.contracts import ROOT
from stoa.contracts.data import HOLDS, add_labels, cap_bucket, legs, load_bars, ratio_bucket
from stoa.experiments import Registry, deflate, return_moments
from stoa.newsbrain import sealed
from stoa.newsbrain.panel import FEE

log = logging.getLogger(__name__)

SPEC = "trial16:contracts:v1"
NAME = "trial16:preregistration"
DECIDED = date(2026, 11, 12)
TRAIN_START = date(2016, 1, 1)
TRAIN_END = date(2022, 12, 31)
TEST_START = date(2023, 1, 1)
TEST_END = date(2026, 6, 30)
GRID = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)  # award value / market cap at the prior close
MIN_TUNE_TRADES = 100
HIT_BAR = 0.54
DSR_BAR = 0.95
MIN_TRADES = 30
NEW_TYPES = ("new", "idiq", "order")
RESULTS = ROOT / "results"


def spec() -> dict[str, Any]:
    """The pre-registered rules, as written to the registry before the test is run."""
    return {
        "spec": SPEC,
        "decided": f"{DECIDED} by the owner",
        "hypothesis": "a new Department of War contract award, posted on war.gov at 5 pm New "
        "York time after the close, that is large next to the awardee's market value moves "
        "the awardee's stock up from the next open to the next close by more than SPY, after "
        "trial 15's costs, before the market has fully priced it in",
        "data": "every daily announcement page on war.gov/News/Contracts from 2016-01-01 "
        "(defense.gov addresses redirect there), cached raw, read into one row per award by "
        "fixed rules (stoa/contracts/parse.py); no language model reads a historical page",
        "matching": "awardee to company by EDGAR names in use on the announcement date and "
        "the subsidiary lists (10-K exhibits 21 and 22) filed before it, found by EDGAR "
        "full-text search for each awardee name and pulled in full for the most traded "
        "companies (stoa/contracts/tickers.py); whole names, the name before the first "
        "comma, a doing-business-as name, or a name shortened to no fewer than two words "
        "against EDGAR names only. The company must have filed a 10-K, 10-Q, 20-F or 40-F in "
        "the 450 days before and its stock must have a daily bar on the entry session. "
        "Delisted companies stay in while they traded. Private and unmatched awardees are "
        "dropped and counted",
        "universe": "awards typed new, idiq or order (new money) with a single awardee and a "
        "dollar value (for a single-award IDIQ, its announced ceiling), matched, with a "
        "market value at the prior close; one row per company and announcement date, values "
        "summed",
        "excluded": "multiple-award contracts (several awardees share one ceiling): the "
        "ceiling is a cap on all orders together, the split between awardees is not known at "
        "award and is usually decided by later competition, so dividing it equally would "
        "invent award sizes. They are counted, not traded. Modifications and option "
        "exercises are their own group, graded as a report beside the trial, never mixed "
        "with new awards",
        "known_at": "5 pm New York time on the announcement date; the entry is the open of "
        "the first session after it; the prior close is the last session's close at or "
        "before the announcement date; every SEC fact must be filed before that date",
        "signal": "value_to_mcap = award value / (shares outstanding from the latest filing "
        "before the date x the prior close). A row trades when value_to_mcap >= T, every "
        "qualifying row at equal weight",
        "tuning": f"T from the grid {list(GRID)}, chosen on 2016-01-01 to 2022-12-31 only: "
        f"the highest average net excess at the primary exit among grid points with at least "
        f"{MIN_TUNE_TRADES} trades in that window, ties to the lower T. Every grid point is "
        "logged as its own trial and counts in the deflated Sharpe",
        "entry": "the open of the first session after the 5 pm post",
        "exit": "primary: the close of the entry session. Secondary, reported and not "
        "graded: the close 1 and 5 sessions after the entry session",
        "benchmark": "SPY over the same span, open to close",
        "costs": "trial 15's: a round trip of slippage by the stock's 20 session dollar "
        "volume through the close before entry (stoa.wide.slippage_bps, each side) plus the "
        "0.3 bps fee",
        "test": f"{TEST_START} to {TEST_END}, run once with T frozen; the holdout from "
        f"{sealed.HOLDOUT_START} stays sealed",
        "pass": f"all of: average net excess above zero; hit rate above {HIT_BAR:.0%}; "
        f"deflated Sharpe of the daily series (each test session the mean net excess of the "
        f"trades entered, 0 on sessions with none) at least {DSR_BAR}, every trial in the "
        f"registry counted; at least {MIN_TRADES} trades, else STARVED",
        "gate": "the known-answer harness check passes first: the shared engine's "
        "(stoa/newsbrain/harness.py, rerun) and this grader's own three legs (SPY as every "
        "pick loses exactly its costs, a cheater with lookahead passes every check, random "
        "stock days land within 3 standard errors of the average stock day)",
        "reported": "the overnight gap (prior 4 pm close to the 9:30 am open) and 9:30 to "
        "9:45 am from minute bars, and the share of the prior close to exit move done before "
        "9:45 am; results by award size to company size bucket, new awards against "
        "modifications, and small (under $2B), mid and large (over $10B) companies; a 9:45 "
        "am entry variant as a diagnostic, not graded",
        "if_pass": "wired into trial 15 as a signal source: a nightly scrape after 5:15 pm "
        "pulls the day's page, the 8:15 am feed carries the qualifying awards, and at 9:00 am "
        "each qualifying stock takes one of the 5 slots ahead of the brain's picks (ranked by "
        "value_to_mcap), the reason saying it came from a contract award",
        "if_fail": "the failure is recorded in the registry with its numbers and nothing is "
        "wired in",
    }


def spec_hash() -> str:
    return hashlib.sha256(json.dumps(spec(), sort_keys=True).encode()).hexdigest()[:16]


def preregister(registry: Registry) -> bool:
    """The rules, in the registry once. Not a trial: no returns. True if written now."""
    if any(e.kind == "preregistration" and e.name == NAME for e in registry.load()):
        return False
    registry.append(
        kind="preregistration",
        name=NAME,
        config_hash=spec_hash(),
        data_hash="none",
        params=spec(),
        metrics={},
    )
    return True


def preregistered(registry: Registry) -> bool:
    return any(
        e.kind == "preregistration" and e.name == NAME and e.config_hash == spec_hash()
        for e in registry.load()
    )


# ---- the grader -------------------------------------------------------------------------------


def eligible(panel: pd.DataFrame, group: str = "new") -> pd.DataFrame:
    """The trial's universe: one awardee, a value, a market value, priced on the entry day."""
    df = panel[panel["group"] == group].copy()
    ok = (
        df["value"].notna()
        & (df["value"] > 0)
        & df["mcap"].notna()
        & (df["mcap"] > 0)
        & df["stk_open"].notna()
        & (df["stk_open"] > 0)
        & df["net_0"].notna()
    )
    return df[ok]


def window(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if end >= sealed.HOLDOUT_START:
        raise sealed.HoldoutBreach(f"{end} is inside the holdout, which trial 16 never reads")
    return df[(df["entry"] >= start) & (df["entry"] <= end)]


def daily_series(trades: pd.DataFrame, start: date, end: date, col: str = "net_0") -> pd.Series:
    """Each session's mean net excess of the trades entered, 0 when none."""
    days = clock.sessions_between(start, end)
    by_day = trades.groupby("entry")[col].mean() if len(trades) else pd.Series(dtype=float)
    return pd.Series([float(by_day.get(d, 0.0)) for d in days], index=pd.DatetimeIndex(days))


@dataclass
class Grade:
    trades: int
    hit_rate: float
    avg_net: float
    avg_ret: float
    avg_spy: float
    sharpe_daily: float
    sessions_with_trades: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def grade(trades: pd.DataFrame, start: date, end: date, col: str = "net_0") -> Grade:
    t = trades[trades[col].notna()]
    h = col.rsplit("_", maxsplit=1)[-1]
    series = daily_series(t, start, end, col)
    sr, _, _, _ = return_moments(list(series))
    return Grade(
        trades=len(t),
        hit_rate=float((t[col] > 0).mean()) if len(t) else math.nan,
        avg_net=float(t[col].mean()) if len(t) else math.nan,
        avg_ret=float(t[f"ret_{h}"].mean()) if len(t) and f"ret_{h}" in t else math.nan,
        avg_spy=float(t[f"spy_{h}"].mean()) if len(t) and f"spy_{h}" in t else math.nan,
        sharpe_daily=sr,
        sessions_with_trades=int(t["entry"].nunique()) if len(t) else 0,
    )


def checks(g: Grade, dsr: float) -> list[dict[str, Any]]:
    return [
        {"name": "edge", "passed": g.avg_net > 0, "detail": f"average net excess {g.avg_net:+.4%}"},
        {
            "name": "hit_rate",
            "passed": g.hit_rate > HIT_BAR,
            "detail": f"{g.hit_rate:.1%} of trades beat SPY after costs (need above {HIT_BAR:.0%})",
        },
        {
            "name": "deflated_sharpe",
            "passed": dsr >= DSR_BAR,
            "detail": f"DSR {dsr:.3f} (need {DSR_BAR})",
        },
        {
            "name": "trade_count",
            "passed": g.trades >= MIN_TRADES,
            "detail": f"{g.trades} trades (need {MIN_TRADES})",
        },
    ]


def verdict(chk: list[dict[str, Any]], trades: int) -> str:
    if trades < MIN_TRADES:
        return "STARVED"
    return "PASS" if all(c["passed"] for c in chk) else "FAIL"


def config_hash(t: float) -> str:
    return hashlib.sha256(f"{SPEC}|T={t}".encode()).hexdigest()[:16]


def rows_hash(rows: pd.DataFrame) -> str:
    """Which (company, entry) rows a run traded, as a short hash for the registry."""
    h = np.asarray(pd.util.hash_pandas_object(rows[["cik", "entry"]].astype(str), index=False))
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def select(rows: pd.DataFrame, t: float) -> pd.DataFrame:
    return rows[rows["value_to_mcap"] >= t]


# ---- known-answer legs of this grader ---------------------------------------------------------


def harness(
    panel: pd.DataFrame,
    registry: Registry,
    start: date = TEST_START,
    end: date = TEST_END,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    """Three strategies with known answers through this grader, on one window's announcement
    days. Not trials: nothing is written to the registry."""
    rows = window(eligible(panel), start, end)
    days = sorted(set(rows["entry"]))
    spy = load_bars("SPY")
    assert spy is not None
    out: dict[str, Any] = {}

    # a. SPY as every pick: its return equals the benchmark's, so each trade nets minus its costs.
    lg = [legs(spy, d) for d in days]
    a = pd.DataFrame(
        {
            "entry": days,
            **{f"stk_{k}": [x[k] for x in lg] for k in lg[0]},
            **{f"spy_{k}": [x[k] for x in lg] for k in lg[0]},
            "prior_close": [x["prev_close"] for x in lg],
        }
    )
    a = add_labels(a)
    expected = -(2 * a["slip"] + FEE)
    gap = float((a["net_0"] - expected).abs().max())
    ga = grade(a, start, end)
    out["spy"] = {
        "passed": bool(gap < 1e-12 and ga.hit_rate == 0.0 and ga.trades == len(days)),
        "expected": f"every trade nets minus its own costs exactly ({float(expected.mean()):+.4%} "
        "on average); hit rate 0",
        "actual": ga.as_dict() | {"largest_gap": gap},
    }

    # b. The cheater: the rows that went on to beat SPY after costs (lookahead on purpose).
    cheat = rows[rows["net_0"] > 0]
    gb = grade(cheat, start, end)
    dsr = deflate(registry, list(daily_series(cheat, start, end)), variance_kind="backtest")
    chk = checks(gb, dsr["dsr"])
    out["cheater"] = {
        "passed": verdict(chk, gb.trades) == "PASS",
        "expected": "every check passes",
        "actual": gb.as_dict() | {"dsr": dsr["dsr"]},
        "checks": chk,
    }

    # c. Random stock days: 2 random universe stocks on each announcement day, priced through
    # the same legs() and add_labels(), against the average of every universe stock day,
    # computed here on its own from the bars.
    syms = sorted(set(panel["symbol"]))
    loaded = {s: load_bars(s) for s in syms}
    bars: dict[str, pd.DataFrame] = {s: b for s, b in loaded.items() if b is not None}
    known_parts = []
    for b in bars.values():
        sub = b.loc[b.index.isin(pd.DatetimeIndex(days))]
        sp = spy.loc[sub.index]
        r = sub["close"] / sub["open"] - 1
        q = sp["close"] / sp["open"] - 1
        slip = np.array(
            [wide.slippage_bps(v if math.isfinite(v) else None) / 1e4 for v in sub["adv"]]
        )
        known_parts.append((r - q - 2 * slip - FEE).dropna())
    known = pd.concat(known_parts)
    pooled = []
    seeds_out = []
    trading = {d: [s for s in syms if s in bars and pd.Timestamp(d) in bars[s].index] for d in days}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        picks = []
        for d in days:
            have = trading[d]
            for s in rng.choice(have, size=min(2, len(have)), replace=False):
                picks.append((s, d))
        rr = pd.DataFrame(
            [
                {
                    "symbol": s,
                    "entry": d,
                    **{f"stk_{k}": v for k, v in legs(bars[s], d).items()},
                    **{f"spy_{k}": v for k, v in legs(spy, d).items()},
                }
                for s, d in picks
            ]
        )
        rr["prior_close"] = rr["stk_prev_close"]
        rr = add_labels(rr)
        rr = rr[rr["net_0"].notna()]
        pooled.append(rr)
        seeds_out.append({"seed": seed, **grade(rr, start, end).as_dict()})
    allr = pd.concat(pooled, ignore_index=True)
    mean = float(allr["net_0"].mean())
    se = float(allr["net_0"].std(ddof=1) / math.sqrt(len(allr)))
    target = float(known.mean())
    out["random"] = {
        "passed": bool(abs(mean - target) <= 3 * se),
        "expected": f"mean trade {target:+.4%} (every universe stock day), within 3 standard "
        "errors",
        "actual": {"mean": mean, "standard_error": se, "trades": len(allr)},
        "seeds": seeds_out,
    }
    out["verdict"] = (
        "honest" if all(out[k]["passed"] for k in ("spy", "cheater", "random")) else "bug"
    )
    return out


# ---- tuning, then the test ---------------------------------------------------------------------


def tune(panel: pd.DataFrame, registry: Registry, log_trials: bool = True) -> dict[str, Any]:
    """Each grid point on 2016 to 2022 only; the best average net excess with enough trades."""
    rows = window(eligible(panel), TRAIN_START, TRAIN_END)
    points = []
    for t in GRID:
        sel = select(rows, t)
        g = grade(sel, TRAIN_START, TRAIN_END)
        points.append({"T": t, **g.as_dict()})
        if log_trials:
            registry.append(
                kind="backtest",
                name=f"trial16:tune:T={t}",
                config_hash=config_hash(t),
                data_hash=rows_hash(sel),
                params={"spec": SPEC, "T": t},
                metrics={**g.as_dict(), "window": f"{TRAIN_START} to {TRAIN_END}"},
                returns=daily_series(sel, TRAIN_START, TRAIN_END),
                tags=["trial16", "tune"],
            )
    ok = [p for p in points if p["trades"] >= MIN_TUNE_TRADES and math.isfinite(p["avg_net"])]
    chosen = max(ok, key=lambda p: (p["avg_net"], -p["T"]))["T"] if ok else None
    return {"grid": points, "chosen": chosen, "rows": len(rows)}


def gaps(trades: pd.DataFrame) -> dict[str, Any]:
    """From minute bars: prior close to the 9:30 open, 9:30 to 9:45, 9:45 to the close; the
    share of the prior close to exit move done before 9:45; and the 9:45 entry variant."""
    from stoa.newsbrain.timing import MinuteStore

    store = MinuteStore()
    rec = []
    for sym, d, pc, o, c, slip, so, sc in zip(
        trades["symbol"], trades["entry"], trades["prior_close"], trades["stk_open"],
        trades["stk_close_0"], trades["slip"], trades["spy_open"], trades["spy_close_0"],
        strict=True,
    ):  # fmt: skip
        p945 = _at(store, str(sym), d, "09:45")
        s945 = _at(store, "SPY", d, "09:45")
        rec.append({"symbol": sym, "entry": d, "pc": pc, "open": o, "p945": p945, "close": c,
                    "slip": slip, "spy_open": so, "spy_945": s945, "spy_close": sc})  # fmt: skip
    df = pd.DataFrame(rec)
    df = df[df["p945"].notna() & df["pc"].notna() & (df["pc"] > 0)]
    if df.empty:
        return {"trades_with_minutes": 0}
    total = df["close"] / df["pc"] - 1
    overnight = df["open"] / df["pc"] - 1
    first15 = df["p945"] / df["open"] - 1
    rest = df["close"] / df["p945"] - 1
    before = df["p945"] / df["pc"] - 1
    net945 = rest - (df["spy_close"] / df["spy_945"] - 1) - 2 * df["slip"] - FEE
    return {
        "trades_with_minutes": len(df),
        "mean_close_to_close": float(total.mean()),
        "mean_overnight_gap": float(overnight.mean()),
        "mean_open_to_945": float(first15.mean()),
        "mean_945_to_close": float(rest.mean()),
        "share_before_945": float(before.mean() / total.mean()) if total.mean() else math.nan,
        "share_overnight": float(overnight.mean() / total.mean()) if total.mean() else math.nan,
        "median_overnight_gap": float(overnight.median()),
        "entry_945_avg_net": float(net945.mean()),
        "entry_945_hit_rate": float((net945 > 0).mean()),
    }


def _at(store: Any, symbol: str, day: date, hhmm: str) -> float | None:
    bars = store.day(symbol, day)
    if not len(bars):
        return None
    lo = pd.Timestamp(clock.at_ny(day, hhmm))
    hi = lo + pd.Timedelta(minutes=30)
    after = bars[(bars["t"] >= lo) & (bars["t"] <= hi)]
    return float(after["o"].iloc[0]) if len(after) else None


def breakdown(panel: pd.DataFrame, start: date, end: date) -> dict[str, Any]:
    """Every eligible row in the window (not only those over T), by bucket, for new awards and
    for modifications and options."""
    out: dict[str, Any] = {}
    for group in ("new", "mod"):
        rows = window(eligible(panel, group), start, end)
        rows = rows.assign(size=[ratio_bucket(x) for x in rows["value_to_mcap"]],
                           cap=[cap_bucket(x) for x in rows["mcap"]])  # fmt: skip
        part: dict[str, Any] = {"all": grade(rows, start, end).as_dict()}
        for col in ("size", "cap"):
            part[col] = {
                str(k): grade(g, start, end).as_dict() for k, g in rows.groupby(col, sort=True)
            }
        out[group] = part
    return out


def test(panel: pd.DataFrame, registry: Registry, t: float) -> dict[str, Any]:
    """The one run on 2023 to mid-2026 with T frozen."""
    rows = window(eligible(panel), TEST_START, TEST_END)
    sel = select(rows, t)
    g = grade(sel, TEST_START, TEST_END)
    series = daily_series(sel, TEST_START, TEST_END)
    dsr = deflate(registry, list(series), variance_kind="backtest")
    chk = checks(g, dsr["dsr"])
    v = verdict(chk, g.trades)
    secondary = {
        f"hold_{h}": grade(sel, TEST_START, TEST_END, f"net_{h}").as_dict() for h in HOLDS[1:]
    }
    years = {
        str(y): grade(
            sel[[d.year == y for d in sel["entry"]]], date(y, 1, 1), min(date(y, 12, 31), TEST_END)
        ).as_dict()
        for y in range(TEST_START.year, TEST_END.year + 1)
    }
    entry = registry.append(
        kind="backtest",
        name="trial16:contracts:test",
        config_hash=config_hash(t),
        data_hash=rows_hash(sel),
        params={"spec": SPEC, "T": t},
        metrics={
            **g.as_dict(),
            "dsr": dsr["dsr"],
            "verdict": v,
            "window": f"{TEST_START} to {TEST_END}",
        },
        returns=series,
        tags=["trial16", "test"],
    )
    return {
        "T": t,
        "verdict": v,
        "grade": g.as_dict(),
        "dsr": dsr,
        "checks": chk,
        "secondary": secondary,
        "years": years,
        "registry_id": entry.id,
        "selected": sel[
            ["symbol", "date", "entry", "value", "mcap", "value_to_mcap", "net_0"]
        ].to_dict("records"),
    }


def save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


class Stop(RuntimeError):
    """A registered stop condition: nothing after it runs."""


SHARED_HARNESS = Path("data/contracts/harness_rerun")


def shared_harness_ok(folder: Path = SHARED_HARNESS) -> dict[str, Any]:
    """The shared engine's known-answer rerun: the cheater and random legs pass and the index
    leg's return equals its known answer (exposure times IVV) within 3 points, as on
    2026-10-27 when the owner released it (its Sharpe rule assumed a fully invested book)."""
    res = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    idx = res["spy"]
    gap = abs(float(idx["actual"]["return"]) - float(idx["expected_return"]))
    ok = bool(res["cheater"]["passed"]) and bool(res["random"]["passed"]) and gap <= 0.03
    return {"passed": ok, "index_return_gap": gap, "cheater": res["cheater"]["passed"],
            "random": res["random"]["passed"]}  # fmt: skip


def run(panel: pd.DataFrame, registry: Registry, out: Path) -> dict[str, Any]:
    """The registered order of work. Raises Stop at a failed gate."""
    if not preregistered(registry):
        raise Stop("trial 16 is not in the registry: preregister before any result")
    report: dict[str, Any] = {"spec_hash": spec_hash()}
    report["shared_harness"] = shared_harness_ok()
    save(report, out / "report.json")
    if not report["shared_harness"]["passed"]:
        raise Stop(f"the shared known-answer harness failed: {report['shared_harness']}")
    report["harness_train"] = harness(panel, registry, TRAIN_START, TRAIN_END)
    save(report, out / "report.json")
    if report["harness_train"]["verdict"] != "honest":
        raise Stop("this grader's known-answer legs failed on 2016 to 2022")
    report["tune"] = tune(panel, registry)
    save(report, out / "report.json")
    chosen = report["tune"]["chosen"]
    if chosen is None:
        raise Stop(f"no grid point had {MIN_TUNE_TRADES} trades in 2016 to 2022")
    report["harness_test"] = harness(panel, registry, TEST_START, TEST_END)
    save(report, out / "report.json")
    if report["harness_test"]["verdict"] != "honest":
        raise Stop("this grader's known-answer legs failed on the test window")
    report["test"] = test(panel, registry, float(chosen))
    save(report, out / "report.json")
    sel = select(window(eligible(panel), TEST_START, TEST_END), float(chosen))
    report["gaps_selected"] = gaps(sel)
    report["gaps_all_new"] = gaps(window(eligible(panel), TEST_START, TEST_END))
    report["breakdown_test"] = breakdown(panel, TEST_START, TEST_END)
    save(report, out / "report.json")
    return report
