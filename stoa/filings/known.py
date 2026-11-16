"""Trial 17's known-answer harness: made-up filings and prices whose right answers are known
before the engine runs, pushed through the trial's own code (``trial17.eligible``,
``trial17.simulate``).

Look-ahead. Six companies file around the 2020-03-02 rebalance (a Monday; its open is 9:30
am New York time):

- A, accepted Friday 2020-02-28 at 4:05 pm, after the close: counts from Monday's open, so
  it is ranked on 2020-03-02.
- C, accepted Monday 2020-03-02 at 9:29:59 am, one second before the open: ranked that day.
- B, accepted Monday 2020-03-02 at 9:31 am, one minute after the open: not ranked until the
  2020-04-01 rebalance. B has the highest similarity of all, and its stock triples during
  March, so an engine that used it early would both pick it and show the gain.
- D, E, F filed in February with lower similarity, so the quintiles are well defined.

Delisting. G files in March with the highest similarity, so it is bought at the 2020-04-01
open at $10. Its last bar is 2020-04-15, close $7.00. The engine must sell it at exactly that
close less one side's slippage and the fee, record the exit at 2020-04-15, and hold the cash.

A third leg checks the money: with every price flat at $10 and no names, equity never moves;
with one flat name, equity falls by exactly the entry slippage.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from stoa import clock, wide
from stoa.filings import trial17 as t17
from stoa.newsbrain.panel import FEE

VOLUME = 2_000_000  # shares a day: $20M a day at $10
START, END = date(2020, 1, 2), date(2020, 5, 29)


def _ny(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")


def fixture() -> tuple[pd.DataFrame, t17.Prices]:
    sessions = clock.sessions_between(START, END)
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in sessions])

    def flat(px: float = 10.0) -> pd.DataFrame:
        return pd.DataFrame({"open": px, "close": px, "volume": float(VOLUME)}, index=idx)

    bars = {s: flat() for s in ("A", "C", "D", "E", "F", "SPY")}
    b = flat()
    march = (idx >= pd.Timestamp("2020-03-02")) & (idx <= pd.Timestamp("2020-03-31"))
    b.loc[march, ["open", "close"]] = 30.0  # B triples from the 2020-03-02 open
    b.loc[idx > pd.Timestamp("2020-03-31"), ["open", "close"]] = 30.0
    bars["B"] = b
    g = flat()
    g = g[g.index <= pd.Timestamp("2020-04-15")].copy()
    g.loc[g.index >= pd.Timestamp("2020-04-02"), ["open", "close"]] = 7.0
    bars["G"] = g
    prices = t17.Prices(bars, sessions)
    filings = pd.DataFrame(
        [
            ("A", 1, "2020-02-28 16:05:00", 0.990),
            ("C", 3, "2020-03-02 09:29:59", 0.995),
            ("B", 2, "2020-03-02 09:31:00", 0.9999),
            ("D", 4, "2020-02-10 18:00:00", 0.900),
            ("E", 5, "2020-02-11 18:00:00", 0.910),
            ("F", 6, "2020-02-12 18:00:00", 0.920),
            ("G", 7, "2020-03-10 18:00:00", 0.99999),
        ],
        columns=["symbol", "cik", "accepted", "cos_full"],
    )
    filings["accepted"] = [_ny(a) for a in filings["accepted"]]
    filings["accn"] = [f"0000000000-20-{k:06d}" for k in filings["cik"]]
    filings["counts_from"] = pd.to_datetime(
        pd.Series(t17.counts_from(filings["accepted"]), index=filings.index, dtype=object)
    )
    return filings, prices


def known_answer() -> dict[str, Any]:
    filings, prices = fixture()
    out: dict[str, Any] = {}

    # 1. counts_from and the ranked set at 2020-03-02
    cf = dict(zip(filings["symbol"], [d.date() for d in filings["counts_from"]], strict=True))
    want_cf = {"A": date(2020, 3, 2), "C": date(2020, 3, 2), "B": date(2020, 3, 3)}
    mar = t17.eligible(filings, prices, date(2020, 3, 2))
    apr = t17.eligible(filings, prices, date(2020, 4, 1))
    out["counts_from"] = {
        "passed": all(cf[k] == v for k, v in want_cf.items()),
        "expected": {k: str(v) for k, v in want_cf.items()},
        "actual": {k: str(cf[k]) for k in want_cf},
    }
    out["no_early_filing"] = {
        "passed": "B" not in set(mar["symbol"])
        and {"A", "C"} <= set(mar["symbol"])
        and "B" in set(apr["symbol"]),
        "expected": "B (accepted 9:31 am) absent on 2020-03-02 and present on 2020-04-01; A "
        "(4:05 pm the trading day before) and C (9:29:59 am) present on 2020-03-02",
        "actual": {"2020-03-02": sorted(mar["symbol"]), "2020-04-01": sorted(apr["symbol"])},
    }
    top_mar = sorted(mar.loc[mar["quintile"] == t17.QUINTILES, "symbol"])
    out["march_pick"] = {
        "passed": top_mar == ["C"],
        "expected": ["C"],
        "actual": top_mar,
    }

    # 2. the run: B's tripling in March must not reach the equity curve
    targets = t17.pick(
        {
            d: t17.eligible(filings, prices, d)
            for d in (date(2020, 3, 2), date(2020, 4, 1), date(2020, 5, 1))
        },
        t17.QUINTILES,
    )
    run = t17.simulate(prices, targets, END, equity0=100_000.0, label="known")
    march_eq = run.equity[
        (run.equity.index >= pd.Timestamp("2020-03-02"))
        & (run.equity.index <= pd.Timestamp("2020-03-31"))
    ]
    slip = wide.slippage_bps(10.0 * VOLUME) / 1e4
    expected_march = 100_000.0 * (1 - t17.CASH_RESERVE) / (1 + slip) + 100_000.0 * t17.CASH_RESERVE
    out["no_lookahead_gain"] = {
        "passed": bool(np.allclose(march_eq.to_numpy(), expected_march, rtol=1e-12)),
        "expected": f"flat at {expected_march:.6f} all March (C is flat; B's tripling unseen)",
        "actual": [float(march_eq.min()), float(march_eq.max())],
    }

    # 3. G: bought 2020-04-01, last bar 2020-04-15 at $7.00, sold there
    g_exits = [e for e in run.exits if e["symbol"] == "G"]
    g_trades = run.trades[run.trades["symbol"] == "G"]
    ok = (
        len(g_exits) == 1
        and g_exits[0]["last_bar"] == date(2020, 4, 15)
        and g_exits[0]["price"] == 7.0
    )
    if ok:
        tr = g_trades.iloc[0]
        g = prices.close[:, prices.col["G"]]
        g = g[np.isfinite(g)]
        dv = float(np.mean(g[-20:] * VOLUME))  # G's 20 session dollar volume to its last bar
        want_net = 7.0 * (1 - wide.slippage_bps(dv) / 1e4) * (1 - FEE)
        ok = bool(
            tr["exit"] == date(2020, 4, 15)
            and abs(tr["exit_net"] - want_net) < 1e-12
            and tr["how"] == "delisted"
        )
    out["delisted_exit"] = {
        "passed": bool(ok),
        "expected": "G sold once at its last close, $7.00 on 2020-04-15, less slippage and fee",
        "actual": {"exits": g_exits, "trades": g_trades.to_dict("records")},
    }
    out["audit_on_fixture"] = t17.audit(
        filings, {d: t17.eligible(filings, prices, d) for d in targets}, run, prices
    )
    out["verdict"] = (
        "honest" if all(v["passed"] for v in out.values() if isinstance(v, dict)) else "bug"
    )
    return out
