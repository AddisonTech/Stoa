"""Trial 18's rule books, its known-answer harness and its one run.

``python -m stoa.commod.run18 preregister a|b`` writes a spec to the registry (once).
``python -m stoa.commod.run18 trial`` runs the harness first (stop if it fails), then 18a and
18b once each, the benchmarks and the BOIL and KOLD diagnostics, and writes the report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa import clock
from stoa.commod import ROOT, eia
from stoa.commod import trial18 as t18
from stoa.experiments import Registry, deflate, return_moments

log = logging.getLogger(__name__)

REGISTRY = Path("state-branch/shared/experiments/registry.jsonl")
END = date(2026, 11, 13)  # the latest close at registration
MINUTE_TAIL = ROOT / "minute"


def common() -> dict[str, Any]:
    return {
        "decided": f"{t18.DECIDED} by the owner",
        "data": "EIA API v2: working gas in storage, Lower 48 (NG.NW2_EPG0_SWO_R48_BCF.W), and "
        "U.S. commercial crude oil stocks excluding the SPR (PET.WCESTUS1.W), weekly, the week's "
        "change = stocks less last week's (stoa/commod/eia.py)",
        "expected": "the 5-year average change for the same week, rebuilt from EIA's weekly "
        "history: the mean change of the week ending nearest 52 x k weeks earlier, k = 1 to 5 "
        "(within 3 days), all five needed; only earlier years, so no look-ahead; today's revised "
        "values, not the figure printed that week (the owner's choice, 2026-11-16: the API keeps "
        "no printed history)",
        "gas_weather": "for gas the expectation also moves with the week's population-weighted "
        "degree days against normal (NOAA CPC, week ending the Saturday after the storage week, "
        "used only if NOAA posted it before the release): expected = 5-year average + b_h x HDD "
        "departure + b_c x CDD departure, b fitted by least squares without intercept on earlier "
        f"reports only, once {t18.MIN_HISTORY} are available (b = 0 before)",
        "release": "EIA's API keeps no release times, so a conservative calendar (owner's choice, "
        "2026-11-16): oil Wednesday and gas Thursday 10:30 am New York time after the week ends; "
        "in a week with a federal holiday on or before the normal day, noon of the next business "
        "day after it (the latest EIA's shifts reach), marked as a holiday week. Late, never early",
        "surprise": "actual change less expected; scaled by the standard deviation of earlier "
        f"reports' surprises (at least {t18.MIN_HISTORY} earlier reports, else no signal). "
        f"Bullish: scaled surprise at or below -{t18.THRESHOLD} (a draw bigger than expected or a "
        "build smaller than expected); bearish at or above +1 (BOIL and KOLD diagnostic only)",
        "funds": "UNG for gas, USO for oil; long or flat, no leverage, no shorting",
        "account": f"${t18.EQUITY:,.0f}, two sleeves of equal money (gas, oil); a sleeve in a "
        f"trade holds {1 - t18.CASH_RESERVE:.0%} of its money in its fund, otherwise cash",
        "costs": "standard: slippage each side by the fund's 20 session dollar volume to the "
        "close before (stoa.wide.slippage_bps, 5 to 75 bps), and the 0.3 bps fee on sales",
        "window": f"{t18.START} (the first bar on disk; Alpaca has no bars before 2016) to {END}; "
        "EIA and NOAA history from earlier years feeds the 5-year averages, the weather fit and "
        "the scaling only",
        "benchmarks": "buy and hold of the same funds (half UNG, half USO, bought at the first "
        "open), and cash",
        "pass": "the standard bar of trials 1 to 16 with the funds' buy and hold in SPY's place, "
        "every line on the whole window: max drawdown strictly lower than buy and hold's; Sharpe "
        "and Sortino strictly higher; total return above cash (zero); deflated Sharpe of the "
        f"daily returns at least {t18.DSR_BAR}, every distinct trial in the registry counted, this "
        f"one included; max drawdown never past {t18.DRAWDOWN_CAP:.0%}; Sharpe above buy and "
        "hold's in at least 2 of the scoreboard's 3 tune windows; at least "
        f"{t18.MIN_TRADES} trades, else STARVED; plus, as the brief asks, hit rate (trades that "
        f"made money after costs) above {t18.HIT_BAR:.0%} (the per-trade bar of trials 15 and 16) "
        "and average return per trade after costs above zero; no Claude spend",
        "diagnostic": "BOIL (2x long gas) on bullish gas surprises and KOLD (2x short gas) on "
        "bearish ones, same-day as 18b: reported, not graded, logged as kind 'diagnostic'",
        "gate": "the known-answer harness (stoa/commod/known18.py, tests/test_trial18.py) passes "
        "first: no report number is used before its release time, the 5-year average uses only "
        "earlier years, and degree days count only if posted before the release",
        "run_once": "run once; no tuning after the run; a bug that invalidates the run is logged "
        "as void with its reason and the rerun is a new trial number",
        "rerun_of": f"trial {t18.VOID[0]}, void: {t18.VOID[1]}; same rules, the fix only skips "
        "reports before the window, which no rule traded",
        "claude": "Claude never reads a historical EIA report; the numbers go straight from the "
        "API into code",
    }


def spec_a() -> dict[str, Any]:
    return {
        "spec": t18.SPEC_A,
        **common(),
        "trade": f"buy the fund at the open of the first regular session after a bullish "
        f"release; sell at the open {t18.HOLD} sessions later; a bullish release "
        f"while a trade is open moves its exit to the open {t18.HOLD} sessions after "
        "the new entry day; Stoa's twice-a-day schedule can run it",
    }


def spec_b() -> dict[str, Any]:
    return {
        "spec": t18.SPEC_B,
        **common(),
        "trade": f"buy the fund at {t18.INTRADAY_ENTRY} am New York time on the release day "
        "(the open of the first minute bar at or after it, within 30 minutes), sell "
        "at that day's close; holiday weeks are skipped (the release time is not "
        "known exactly); a pass here means live Stoa would need an intraday "
        "execution path, which is not built",
    }


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


NAMES = {"a": (t18.NAME_A, spec_a), "b": (t18.NAME_B, spec_b)}


def preregister(registry: Registry, which: str) -> bool:
    name, fn = NAMES[which]
    if any(e.kind == "preregistration" and e.name == name for e in registry.load()):
        return False
    s = fn()
    registry.append(
        kind="preregistration",
        name=name,
        config_hash=spec_hash(s),
        data_hash="none",
        params=s,
        metrics={},
    )
    return True


def preregistered(registry: Registry, which: str) -> bool:
    name, fn = NAMES[which]
    h = spec_hash(fn())
    return any(
        e.kind == "preregistration" and e.name == name and e.config_hash == h
        for e in registry.load()
    )


# ---- prices ------------------------------------------------------------------------------------


def daily(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(
        Path("data/bars/wide") / f"{sym}.parquet", columns=["open", "close", "volume"]
    )
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


class Minutes:
    """The open of the first minute bar at or after a time, within 30 minutes: the compacted
    store through October 2026, the trial 18 tail after."""

    def __init__(self) -> None:
        from stoa.newsbrain.timing import MinuteStore

        self.store = MinuteStore()
        self.tail: dict[str, pd.DataFrame] = {}

    def _day(self, sym: str, day: date) -> pd.DataFrame:
        if day < date(2026, 11, 1):
            return self.store.day(sym, day)
        if sym not in self.tail:
            p = MINUTE_TAIL / f"{sym}.parquet"
            self.tail[sym] = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["t", "o"])
        t = self.tail[sym]
        if t.empty:
            return t
        ts = pd.to_datetime(t["t"], utc=True)
        ny = ts.dt.tz_convert("America/New_York").dt.date
        return t[ny == day].assign(t=ts[ny == day])

    def __call__(self, sym: str, day: date, hhmm: str) -> float | None:
        bars = self._day(sym, day)
        if not len(bars):
            return None
        lo = pd.Timestamp(clock.at_ny(day, hhmm))
        t = pd.to_datetime(bars["t"], utc=True)
        after = bars[(t >= lo) & (t <= lo + pd.Timedelta(minutes=30))]
        return float(after["o"].iloc[0]) if len(after) else None


# ---- the run -----------------------------------------------------------------------------------


def grade(
    name: str, which: str, eq: pd.Series, trades: list[t18.Trade], bh: pd.Series, registry: Registry
) -> dict[str, Any]:
    r = eq.pct_change().dropna()
    spec = spec_a() if which == "a" else spec_b()
    entry = registry.append(
        kind="backtest",
        name=name,
        config_hash=spec_hash(spec),
        data_hash=hashlib.sha256(
            json.dumps([round(x, 12) for x in r.tolist()]).encode()
        ).hexdigest()[:16],
        params={"spec": spec["spec"]},
        metrics={
            "sharpe_daily": return_moments(list(r))[0],
            **t18.stats(eq),
            **t18.trade_stats(trades),
        },
        returns=r.reset_index(drop=True),
        tags=[f"trial{t18.TRIAL}"],
        now=datetime.now(UTC),
    )
    dsr = deflate(registry, list(r), variance_kind="backtest")
    s, b = t18.stats(eq), t18.stats(bh)
    tune = {
        w: {"sharpe": t18.stats(eq, a, z)["sharpe"], "bh_sharpe": t18.stats(bh, a, z)["sharpe"]}
        for w, (a, z) in t18.TUNE_WINDOWS.items()
    }
    ts = t18.trade_stats(trades)
    chk = t18.checks(s, b, tune, dsr["dsr"], ts)
    per_fuel = {f: t18.trade_stats([t for t in trades if t.fuel == f]) for f in t18.FUNDS}
    by_year_trades: dict[str, Any] = {}
    for y in sorted({t.entry_day.year for t in trades}):
        by_year_trades[str(y)] = t18.trade_stats([t for t in trades if t.entry_day.year == y])
    return {
        "verdict": t18.verdict(chk, int(ts["trades"])),
        "checks": chk,
        "stats": s,
        "buy_and_hold": b,
        "trades": ts,
        "per_fuel": per_fuel,
        "tune_windows": tune,
        "by_year": t18.by_year(eq),
        "buy_and_hold_by_year": t18.by_year(bh),
        "trades_by_year": by_year_trades,
        "deflated": dsr,
        "registry_id": entry.id,
        "final_100": float(eq.iloc[-1] / eq.iloc[0] * 100),
    }


def run(registry: Registry, out: Path) -> dict[str, Any]:
    from stoa.commod.known18 import known_answer

    for w in ("a", "b"):
        if not preregistered(registry, w):
            raise SystemExit(f"trial 18{w} is not in the registry with this spec")
    report: dict[str, Any] = {"spec_a": spec_hash(spec_a()), "spec_b": spec_hash(spec_b())}
    report["known_answer"] = known_answer()
    if report["known_answer"]["verdict"] != "honest":
        raise SystemExit("trial 18's known-answer harness failed")
    reps = eia.load()
    bars = {s: daily(s) for s in ("UNG", "USO", "BOIL", "KOLD")}
    sig = {f: t18.surprises(reps[reps["fuel"] == f].drop(columns=["fuel"])) for f in t18.FUNDS}
    for f in t18.FUNDS:
        if f == "oil":
            sig[f] = sig[f].drop(columns=[c for c in ("hdd_dev", "cdd_dev") if c in sig[f]])
    report["signals"] = {
        f: {
            "reports": len(s),
            "scored": int(s["z"].notna().sum()),
            "bullish": int(s["bullish"].sum()),
            "bearish": int(s["bearish"].sum()),
            "holiday_weeks": int(s["holiday_week"].sum()),
        }
        for f, s in sig.items()
    }
    bh = t18.buy_and_hold(bars, END)
    ta = {f: t18.trades_next_open(sig[f], f, sym, bars[sym], END) for f, sym in t18.FUNDS.items()}
    eq_a = t18.equity_curve(ta, bars, END, intraday=False)
    mins = Minutes()
    no_hol = {f: s[~s["holiday_week"]] for f, s in sig.items()}
    tb = {
        f: t18.trades_intraday(no_hol[f], f, sym, bars[sym], mins, END)
        for f, sym in t18.FUNDS.items()
    }
    eq_b = t18.equity_curve(tb, bars, END, intraday=True)
    report["18a"] = grade(
        f"trial{t18.TRIAL}a:test", "a", eq_a, [t for v in ta.values() for t in v], bh, registry
    )
    report["18b"] = grade(
        f"trial{t18.TRIAL}b:test", "b", eq_b, [t for v in tb.values() for t in v], bh, registry
    )
    boil = t18.trades_intraday(no_hol["gas"], "gas", "BOIL", bars["BOIL"], mins, END, "bullish")
    kold = t18.trades_intraday(no_hol["gas"], "gas", "KOLD", bars["KOLD"], mins, END, "bearish")
    report["diagnostic"] = {
        "BOIL_on_bullish_gas": t18.trade_stats(boil),
        "KOLD_on_bearish_gas": t18.trade_stats(kold),
        "BOIL_total_compounded": float(np.prod([1 + t.ret for t in boil]) - 1),
        "KOLD_total_compounded": float(np.prod([1 + t.ret for t in kold]) - 1),
    }
    for nm, ts in (("boil", boil), ("kold", kold)):
        registry.append(
            kind="diagnostic",
            name=f"trial{t18.TRIAL}:diagnostic:{nm}",
            config_hash=spec_hash(spec_b()),
            data_hash="trades",
            params={"spec": t18.SPEC_B, "symbol": nm.upper()},
            metrics=t18.trade_stats(ts),
            now=datetime.now(UTC),
        )
    report["buy_and_hold_each"] = {
        s: t18.stats(t18.buy_and_hold({**bars, "UNG": bars[s], "USO": bars[s]}, END))
        for s in ("UNG", "USO")
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    pd.DataFrame({"18a": eq_a, "18b": eq_b, "buy_and_hold": bh}).to_csv(out / "equity.csv")
    for nm, tr in (("18a", ta), ("18b", tb)):
        pd.DataFrame([{**t.__dict__, "ret": t.ret} for v in tr.values() for t in v]).to_csv(
            out / f"trades_{nm}.csv", index=False
        )
    for f, s in sig.items():
        s.to_csv(out / f"signals_{f}.csv", index=False)
    return report


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO)
    reg = Registry(REGISTRY)
    if argv[0] == "preregister":
        wrote = preregister(reg, argv[1])
        s = spec_a() if argv[1] == "a" else spec_b()
        print(json.dumps({"which": argv[1], "spec_hash": spec_hash(s), "written_now": wrote}))
        return 0
    out = ROOT / "results" / f"trial{t18.TRIAL}-{datetime.now(UTC).strftime('%Y%m%d-%H%M')}"
    rep = run(reg, out)
    print(json.dumps({k: rep[k]["verdict"] for k in ("18a", "18b")} | {"out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
