"""Trial 17's order of work: register, gate, run once, audit, report.

``python -m stoa.filings preregister a|b`` writes a spec to the registry (once).
``python -m stoa.filings trial a`` runs 17a: the known-answer harness first (stop if it
fails), then the one run, the benchmarks and diagnostics, the audit of the run's own
records, the report. ``trial todo`` lists the filings the 17b reader must read (17a's
ranked filings, quintile 5 first). ``trial b`` runs 17b once every quintile 5 filing is read.

The registry is the shared one on the state branch (``--registry``, default
``state-branch/shared/experiments/registry.jsonl``).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa import clock
from stoa.experiments import Registry
from stoa.filings import ROOT, known
from stoa.filings import trial17 as t17

log = logging.getLogger(__name__)

REGISTRY = Path("state-branch/shared/experiments/registry.jsonl")
RANKED = ROOT / "results" / "trial17a_ranked.parquet"
MEASURES = ("cos_full", "jac_full", "len_full", "cos_rf", "cos_mdna", "cos_legal")


class Stop(RuntimeError):
    """A registered stop condition: nothing after it runs."""


def registry_of(argv: list[str]) -> Registry:
    if "--registry" in argv:
        return Registry(Path(argv[argv.index("--registry") + 1]))
    return Registry(REGISTRY)


def tag_spec() -> dict[str, Any]:
    from stoa.filings import tag

    return {**tag.spec(), "checked": READER_CHECK}


# The reader's known-answer check (stoa/filings/tagcheck.py), measured 2026-11-16 before 17b
# was registered: the prompts were frozen on CASES (the new risk question reworded once), then
# FRESH was scored once.
READER_CHECK = {
    "made_up_passages": "40 (8 a question, half yes), the set the prompts were chosen on: 38 "
    "right; then 10 fresh ones written after the prompts were frozen: 9 right (one segment "
    "recast read as a restatement)",
    "threshold": 0.5,
}


def spec_b() -> dict[str, Any]:
    return t17.spec_b(tag_spec())


def prices_for(filings: pd.DataFrame) -> tuple[t17.Prices, list[date]]:
    sessions = clock.sessions_between(t17.DATA_START, t17.END)
    syms = set(filings.loc[filings["symbol"] != "", "symbol"]) | {"SPY"}
    return t17.load_prices(syms, sessions), sessions


def ranked_frame(ranked: dict[date, pd.DataFrame]) -> pd.DataFrame:
    parts = [df.assign(rebalance=pd.Timestamp(d)) for d, df in ranked.items() if len(df)]
    cols = [
        "rebalance",
        "cik",
        "accn",
        "prior_accn",
        "form",
        "accepted",
        "counts_from",
        "symbol",
        "quintile",
        *MEASURES,
    ]
    return pd.concat(parts, ignore_index=True)[cols]


def spreads(
    prices: t17.Prices, ranked: dict[date, pd.DataFrame], days: list[date], filings: pd.DataFrame
) -> dict[str, Any]:
    """Diagnostic, before costs: for each measure, the average monthly return of each
    quintile (open to next rebalance open, a stopped stock at its last close), and quintile 5
    less quintile 1 with its t statistic. Each measure is ranked on the same companies."""
    out: dict[str, Any] = {}
    nxt = {d: (days[k + 1] if k + 1 < len(days) else t17.END) for k, d in enumerate(days)}
    for m in MEASURES:
        rows = []
        for d, df in ranked.items():
            sub = df[df[m].notna()].sort_values([m, "cik"], kind="mergesort")
            if len(sub) < t17.QUINTILES:
                continue
            q = (np.arange(len(sub)) * t17.QUINTILES // len(sub)) + 1
            i0, i1 = prices.index[d], prices.index[nxt[d]]
            r = []
            for s in sub["symbol"]:
                j = prices.col[s]
                o = prices.open[i0, j]
                if not np.isfinite(o):
                    r.append(np.nan)
                    continue
                if prices.last[j] < i1:
                    e = prices.close[int(prices.last[j]), j]
                elif nxt[d] == t17.END:
                    e = prices.close_ff[i1, j]
                else:
                    e = (
                        prices.open[i1, j]
                        if np.isfinite(prices.open[i1, j])
                        else prices.close_ff[i1 - 1, j]
                    )
                r.append(e / o - 1)
            g = pd.DataFrame({"q": q, "r": r}).groupby("q")["r"].mean()
            rows.append(
                {"rebalance": str(d), **{f"q{k}": float(g.get(k, np.nan)) for k in range(1, 6)}}
            )
        t = pd.DataFrame(rows)
        if t.empty:
            continue
        spread = t["q5"] - t["q1"]
        out[m] = {
            "months": len(t),
            **{f"q{k}_monthly": float(t[f"q{k}"].mean()) for k in range(1, 6)},
            "q5_minus_q1_monthly": float(spread.mean()),
            "t_stat": float(spread.mean() / (spread.std(ddof=1) / math.sqrt(len(spread)))),
        }
    return out


def run_a(registry: Registry, out: Path) -> dict[str, Any]:
    spec = t17.spec_a()
    if not t17.preregistered(registry, t17.NAME_A, spec):
        raise Stop("trial 17a is not in the registry with this spec: preregister before any result")
    report: dict[str, Any] = {
        "spec_hash": t17.spec_hash(spec),
        "started": datetime.now(UTC).isoformat(),
    }
    report["known_answer"] = known.known_answer()
    t17.save(report, out / "report.json")
    if report["known_answer"]["verdict"] != "honest":
        raise Stop("the look-ahead known-answer harness failed")
    filings = t17.load_filings()
    prices, sessions = prices_for(filings)
    first = t17.first_rebalance(sessions)
    days = t17.rebalance_days(first, t17.END)
    ranked = t17.rank_all(filings, prices, days)
    rf = ranked_frame(ranked)
    rf.to_parquet(RANKED, index=False)
    report["universe"] = {
        "filings_scored": len(filings),
        "filings_with_pair": int(filings["cos_full"].notna().sum()),
        "filings_with_symbol": int((filings["symbol"] != "").sum()),
        "first_rebalance": str(first),
        "rebalances": len(days),
        "ranked_per_month": float(np.mean([len(v) for v in ranked.values()])),
        "ranked_min": int(min(len(v) for v in ranked.values())),
        "ranked_filings_unique": int(rf["accn"].nunique()),
        "companies_ever_ranked": int(rf["cik"].nunique()),
        "by_form": rf.groupby(rf["form"].str[:4])["accn"].nunique().to_dict(),
    }
    q5 = t17.pick(ranked, t17.QUINTILES)
    run = t17.simulate(prices, q5, t17.END, label="17a_q5")
    spy = t17.buy_and_hold(prices, "SPY", first, t17.END)
    ew = t17.simulate(prices, t17.pick(ranked, None), t17.END, label="every_filer")
    q1 = t17.simulate(prices, t17.pick(ranked, 1), t17.END, label="17a_q1_changers")
    small = t17.simulate(
        prices,
        t17.pick(ranked, t17.QUINTILES, cap=t17.SMALL_NAMES, prices=prices),
        t17.END,
        equity0=t17.SMALL_EQUITY,
        min_order=t17.MIN_ORDER,
        label="17a_q5_100_dollars",
    )
    report["audit"] = t17.audit(filings, ranked, run, prices)
    t17.save(report, out / "report.json")
    if not report["audit"]["passed"]:
        raise Stop(f"the look-ahead audit of the run's own records failed: {report['audit']}")
    report["graded"] = t17.report_for(run, spy, ew, prices, registry, spec, "trial17a:test")
    for r, name in (
        (q1, "trial17a:diagnostic:q1_changers"),
        (ew, "trial17a:diagnostic:every_filer"),
        (small, "trial17a:diagnostic:100_dollars"),
    ):
        t17.log_run(registry, name, spec, r, kind="diagnostic")
    report["changers_q1"] = {
        "stats": t17.stats(q1),
        "by_year": t17.by_year(q1),
        "trades": t17.trade_stats(t17.with_spy(q1.trades, prices)),
    }
    report["small_account"] = {
        "stats": t17.stats(small),
        "by_year": t17.by_year(small),
        "final_equity": float(small.equity.iloc[-1]),
        "names_per_month": float(np.mean([len(v) for v in small.holdings.values()])),
    }
    report["spy_100_dollars"] = float(spy.equity.iloc[-1] / spy.equity.iloc[0] * t17.SMALL_EQUITY)
    report["spreads_before_costs"] = spreads(prices, ranked, days, filings)
    t17.save(report, out / "report.json")
    for r in (run, spy, ew, q1, small):
        r.equity.to_frame("equity").to_csv(out / f"equity_{r.label}.csv")
    t17.with_spy(run.trades, prices).to_csv(out / "trades_17a.csv", index=False)
    return report


def todo() -> Path:
    """The filings the reader must read: every ranked filing, quintile 5 first."""
    rf = pd.read_parquet(RANKED)
    best = rf.groupby("accn")["quintile"].max().rename("best_q")
    f = rf.drop_duplicates("accn").merge(best, left_on="accn", right_index=True)
    f["first_q5"] = f["best_q"] == t17.QUINTILES
    f = f.sort_values(["first_q5", "accepted"], ascending=[False, True])
    path = ROOT / "results" / "tag_todo.parquet"
    f[["cik", "accn", "prior_accn", "best_q"]].to_parquet(path, index=False)
    return path


def run_b(registry: Registry, out: Path) -> dict[str, Any]:
    from stoa.filings.tag import load_tags

    spec = spec_b()
    if not t17.preregistered(registry, t17.NAME_B, spec):
        raise Stop("trial 17b is not in the registry with this spec: preregister before any result")
    report: dict[str, Any] = {
        "spec_hash": t17.spec_hash(spec),
        "started": datetime.now(UTC).isoformat(),
    }
    report["known_answer"] = known.known_answer()
    if report["known_answer"]["verdict"] != "honest":
        raise Stop("the look-ahead known-answer harness failed")
    filings = t17.load_filings()
    prices, sessions = prices_for(filings)
    first = t17.first_rebalance(sessions)
    days = t17.rebalance_days(first, t17.END)
    ranked = t17.rank_all(filings, prices, days)
    tags = load_tags()
    tags = tags[tags["version"] == spec["reader"]["version"]].drop_duplicates("accn", keep="last")
    neg = tags.set_index("accn")[list(t17.NEGATIVE_TAGS)].any(axis=1)
    need = {a for d, df in ranked.items() for a in df.loc[df["quintile"] == t17.QUINTILES, "accn"]}
    missing = sorted(need - set(tags["accn"]))
    report["reader_coverage"] = {"q5_filings": len(need), "missing": len(missing)}
    if missing:
        raise Stop(f"{len(missing)} quintile 5 filings are not read yet")
    bad = set(neg[neg].index)

    def drop(df: pd.DataFrame) -> pd.Series:
        return df["accn"].isin(bad)

    q5 = t17.pick(ranked, t17.QUINTILES)
    q5b = t17.pick(ranked, t17.QUINTILES, drop=drop)
    run_a_ = t17.simulate(prices, q5, t17.END, label="17a_q5")
    run = t17.simulate(prices, q5b, t17.END, label="17b")
    spy = t17.buy_and_hold(prices, "SPY", first, t17.END)
    ew = t17.simulate(prices, t17.pick(ranked, None), t17.END, label="every_filer")
    small = t17.simulate(
        prices,
        t17.pick(ranked, t17.QUINTILES, drop=drop, cap=t17.SMALL_NAMES, prices=prices),
        t17.END,
        equity0=t17.SMALL_EQUITY,
        min_order=t17.MIN_ORDER,
        label="17b_100_dollars",
    )
    report["audit"] = t17.audit(filings, ranked, run, prices)
    if not report["audit"]["passed"]:
        raise Stop(f"the look-ahead audit failed: {report['audit']}")
    report["graded"] = t17.report_for(run, spy, ew, prices, registry, spec, "trial17b:test")
    t17.log_run(registry, "trial17b:diagnostic:100_dollars", spec, small, kind="diagnostic")
    report["vs_17a"] = {
        "17a": t17.stats(run_a_),
        "17b": t17.stats(run),
        "17a_trades": t17.trade_stats(t17.with_spy(run_a_.trades, prices)),
    }
    dropped = {str(d): len(q5[d]) - len(q5b[d]) for d in days}
    report["dropped_per_month"] = {
        "mean": float(np.mean(list(dropped.values()))),
        "q5_mean": float(np.mean([len(v) for v in q5.values()])),
        "by_month": dropped,
    }
    rf = ranked_frame(ranked)
    tq = rf.drop_duplicates(["accn", "quintile"]).merge(tags, on="accn", how="inner")
    report["tag_rates_by_quintile"] = {
        str(q): {t: float(g[t].mean()) for t in (*t17.NEGATIVE_TAGS, "going_concern_removed")}
        for q, g in tq.groupby("quintile")
    }
    report["small_account"] = {
        "stats": t17.stats(small),
        "final_equity": float(small.equity.iloc[-1]),
    }
    t17.save(report, out / "report.json")
    run.equity.to_frame("equity").to_csv(out / "equity_17b.csv")
    t17.with_spy(run.trades, prices).to_csv(out / "trades_17b.csv", index=False)
    return report


def main(step: str, argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    reg = registry_of(argv)
    which = argv[0] if argv else ""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    if step == "preregister":
        spec = t17.spec_a() if which == "a" else spec_b()
        name = t17.NAME_A if which == "a" else t17.NAME_B
        wrote = t17.preregister(reg, name, spec)
        print(json.dumps({"name": name, "spec_hash": t17.spec_hash(spec), "written_now": wrote}))
        return 0
    if which == "todo":
        print(todo())
        return 0
    out = ROOT / "results" / f"trial17{which}-{stamp}"
    try:
        rep = run_a(reg, out) if which == "a" else run_b(reg, out)
    except Stop as exc:
        print(f"STOP: {exc}")
        return 3
    g = rep["graded"]
    print(
        json.dumps({"verdict": g["verdict"], "checks": g["checks"], "out": str(out)}, default=str)
    )
    return 0
