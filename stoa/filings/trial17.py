"""Trial 17: hold the companies that did not change their 10-K and 10-Q text ("Lazy Prices").

Cohen, Malloy and Nguyen (Journal of Finance, 2020) found, on 1995 to 2014 filings, that
firms whose annual and quarterly reports change little from the year before ("non-changers")
earn more over the following months than firms that rewrite them ("changers"). Their data
ends in 2014, so 2016 to 2026 here is out of sample.

``spec_a()`` (17a: the plain rule) and ``spec_b()`` (17b: 17a minus filings a local model
reads as bad news) are the rule books, each written to the registry before its run. Each
runs once. Nothing is tuned: every number in the specs was set before any result.

The engine works on one calendar of NYSE sessions. At each monthly rebalance (the first
session of a month, trades at its open) the eligible filings are those that count by then:
a filing counts from the first regular session open after its EDGAR acceptance time. Of
the filings that started counting in the 3 months up to that open, each company's latest is
ranked by its full-document cosine similarity to the same report a year earlier, among the
companies that pass the liquidity floor at the close before. Quintile 5 (highest
similarity) is held equal weight until the next rebalance.

Costs are the standard cost model of the wide-universe trials (15 and 16): slippage on
each side by the stock's 20 session dollar volume through the close before the trade
(``stoa.wide.slippage_bps``, 5 to 75 bps) and the 0.3 bps fee on sales. A stock that stops
trading while held is sold at its last close less slippage (the backtest engine's rule),
and that cash waits for the next rebalance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa import clock, metrics, wide
from stoa.experiments import Registry, deflate, return_moments
from stoa.filings import ROOT
from stoa.newsbrain.panel import FEE

log = logging.getLogger(__name__)

SPEC_A = "trial17a:lazyprices:v1"
SPEC_B = "trial17b:lazyprices_read:v1"
NAME_A = "trial17a:preregistration"
NAME_B = "trial17b:preregistration"
DECIDED = date(2026, 11, 16)
DATA_START = date(2016, 1, 4)  # the first daily bar on disk (Alpaca's history starts here)
END = date(2026, 11, 13)  # the latest close when the trial was registered
LOOKBACK_MONTHS = 3
QUINTILES = 5
MIN_PRICE = wide.MIN_PRICE  # $5
MIN_ADV = wide.MIN_DOLLAR_VOLUME  # $5M, 20 sessions through the close before
ADV_SESSIONS = wide.ADV_SESSIONS
CASH_RESERVE = 0.02  # Stoa always keeps 2% in cash
EQUITY = 100_000.0  # the big paper account
SMALL_EQUITY = 100.0  # the $100 account
SMALL_NAMES = 50  # the $100 version holds at most this many (Alpaca's $1 order minimum)
MIN_ORDER = 1.0  # dollars, Alpaca's smallest fractional order
TUNE_WINDOWS = {  # the scoreboard's windows (config/scoreboard.yaml, docs/SCOREBOARD.md)
    "tune_a": (date(2019, 1, 1), date(2020, 12, 31)),
    "tune_b": (date(2021, 1, 1), date(2022, 12, 31)),
    "tune_c": (date(2023, 1, 1), date(2024, 12, 31)),
}
DSR_BAR = 0.95
DRAWDOWN_CAP = 0.25
MIN_TUNE_WINDOWS = 2
MIN_TRADES = 30
RESULTS = ROOT / "results"
NEGATIVE_TAGS = ("litigation", "exec_change", "new_risk", "going_concern_added", "restatement")


# ---- the rule books ---------------------------------------------------------------------------


def pass_rule() -> str:
    return (
        "the standard pass bar of trials 1 to 16 (config/scoreboard.yaml, docs/SCOREBOARD.md), "
        "every line must hold, on the whole window (nothing is tuned, so the whole window is "
        "out of sample): max drawdown strictly lower than SPY's; Sharpe strictly higher than "
        "SPY's; Sortino strictly higher than SPY's; deflated Sharpe of the daily returns at "
        f"least {DSR_BAR}, counting every distinct trial in the registry, this one included; "
        f"max drawdown never past {DRAWDOWN_CAP:.0%}; Sharpe above SPY's in at least "
        f"{MIN_TUNE_WINDOWS} of the scoreboard's 3 tune windows (2019-2020, 2021-2022, "
        f"2023-2024); at least {MIN_TRADES} closed trades, else STARVED; Claude spend under $25 "
        "a month (no Claude is used). Added for this trial, as the brief asks, so a small "
        "company effect cannot pass for an edge: Sharpe strictly higher than the equal "
        "weight portfolio of every filer in the same filters"
    )


def spec_a() -> dict[str, Any]:
    """Trial 17a, as written to the registry before its run."""
    return {
        "spec": SPEC_A,
        "decided": f"{DECIDED} by the owner",
        "paper": "Cohen, Malloy and Nguyen, Lazy Prices, Journal of Finance 2020 (1995 to "
        "2014 data); 2016 to 2026 is out of sample",
        "hypothesis": "companies whose 10-K or 10-Q text barely changed from the same report "
        "a year earlier (non-changers) beat the market over the following month, after "
        "costs, held long only in a cash account",
        "filings": "every original 10-K, 10-K405, 10-KT, 10-Q and 10-QT on EDGAR from "
        "2014-01-01 for the wide universe's filers (data/newsbrain/sec/ciks.parquet), primary "
        "document only, amendments excluded (stoa/filings/pull.py)",
        "text": "rules only (stoa/filings/parse.py, version p1): HTML to text with the hidden "
        "inline XBRL header removed; sections risk factors, MD&A and legal proceedings cut "
        "by their headings",
        "pairing": "each filing against the same filer's filing of the same base form "
        "(10-K with 10-K, 10-Q with 10-Q) whose period of report is 330 to 400 days earlier, "
        "nearest 365; filing dates by the same rule when no period pairs "
        "(stoa/filings/scores.py)",
        "measures": "cosine similarity of term counts, Jaccard similarity of term sets, and "
        "length change; terms are runs of 2 or more letters, lower case, no stemming, no "
        "stop list; for the full document and each section. Only the full-document cosine "
        "is traded; the rest are reported",
        "tickers": "each filing to its symbol point in time (stoa/filings/tickers.py): the "
        "SEC map and Form 4 symbols for the filer, and the symbol the filer's own filings "
        "state within 450 days; a symbol must have traded in the 30 days up to the filing "
        "and not be claimed by another filer within 450 days; the most traded share class "
        "wins. Delisted companies stay in while they traded",
        "known_at": "a filing counts from the first regular session open after its EDGAR "
        "acceptance time (Eastern time in EDGAR's index)",
        "rebalance": "monthly, at the open of the first session of each month, from the first "
        f"month start with {ADV_SESSIONS} sessions of bars behind it ({DATA_START} is the "
        f"first bar on disk) to the window's end",
        "ranking": f"filings that started counting in the {LOOKBACK_MONTHS} calendar months "
        "up to the rebalance open (after the open 3 months before, up to and including that "
        "open), with a symbol and a prior-year pair; each company's latest; ranked by "
        "full-document cosine similarity across all of them, ties by CIK; quintiles of equal "
        "count, 1 the lowest similarity (changers), 5 the highest (non-changers)",
        "filters": f"at the close before the rebalance: close at least ${MIN_PRICE:.0f} and "
        f"{ADV_SESSIONS} session average dollar volume at least ${MIN_ADV / 1e6:.0f}M "
        "(stoa.wide.liquid, the wide universe's floor), so the big paper account and the "
        "$100 account (fractional shares) can trade every name",
        "portfolio": f"quintile 5, equal weight, {1 - CASH_RESERVE:.0%} of equity at the "
        f"rebalance open ({CASH_RESERVE:.0%} always in cash, Stoa's rule); held to the next "
        "rebalance open; a name that does not trade at the rebalance open is not bought and "
        "its money waits in cash",
        "costs": "standard: slippage each side by 20 session dollar volume through the close "
        "before (stoa.wide.slippage_bps, 5 to 75 bps), and the 0.3 bps fee on sales; a stock "
        "that stops trading while held is sold at its last close less slippage",
        "account": f"${EQUITY:,.0f} (the big paper account); fractional shares",
        "window": f"{DATA_START} (the first bar on disk; Alpaca has no bars before 2016, so "
        f"2015 cannot be traded) to {END}, the latest close at registration. The months "
        "2026-07-01 to 2026-10-23 were trial 12's holdout, spent on 2026-10-27; by the "
        "owner's decision of 2026-11-19 (docs/DECISIONS.md, 'Trials 17 and 20 read the spent "
        "holdout months') they are ordinary out-of-sample months here. The trial reads daily "
        "bars and its own EDGAR pull, never the news brain's loaders, whose holdout wall "
        "(stoa/newsbrain/sealed.py) stays in place for the news brain",
        "benchmarks": "SPY bought with all equity at the first rebalance open and held; and "
        "the equal weight portfolio of every ranked filer in the same filters (all five "
        "quintiles), same rules and costs",
        "diagnostic": "quintile 1 (the changers, the avoid list), same rules and costs, "
        "reported and not graded; the $100 version (below); results by year; the other "
        "measures' quintile spreads",
        "small_account": f"diagnostic, not graded: ${SMALL_EQUITY:.0f} of equity, quintile 5 "
        f"capped at the {SMALL_NAMES} names with the most dollar volume, and no order under "
        f"${MIN_ORDER:.0f} (Alpaca's minimum): what the $100 account could actually have held",
        "trades": "a trade is one stock's continuous holding, from the open it was bought to "
        "the open it was sold (or its last close if it stopped trading); its return is the "
        "exit over the entry less both sides' slippage and the fee; it beats SPY when that "
        "beats SPY's return over the same span. Hit rate is the share that beat SPY",
        "pass": pass_rule(),
        "gate": "trial 17 results count only after the look-ahead check passes: a known-"
        "answer test (tests/test_trial17.py) that no filing is used before its acceptance "
        "time and that a delisted stock leaves at its real last close, and the same two "
        "checks run on the real run's own records",
        "registry": "the graded run's daily returns go in the registry as a backtest and count "
        "in the deflated Sharpe of every later trial; diagnostics are logged as kind "
        "'diagnostic', which the deflated Sharpe does not count (they cannot be chosen to "
        "trade)",
        "run_once": "run once; no tuning after the run; a bug that invalidates the run is "
        "logged as void with its reason and the rerun is a new trial number",
        "claude": "Claude never reads a historical filing",
    }


def spec_b(tag_spec: dict[str, Any]) -> dict[str, Any]:
    """Trial 17b, as written to the registry before its run. ``tag_spec`` is the reader's
    own rule book (stoa/filings/tag.py ``spec()``)."""
    return {
        "spec": SPEC_B,
        "decided": f"{DECIDED} by the owner",
        "hypothesis": "dropping the non-changers whose latest filing a local model reads as "
        "bad news (new or expanded litigation, a CEO or CFO change, a new risk factor, a "
        "going concern doubt added, a restatement) improves on 17a",
        "same_as_17a": f"spec {spec_hash(spec_a())}: every rule of 17a (filings, text, "
        "pairing, measures, tickers, known-at, rebalance, ranking, filters, costs, account, "
        "window, benchmarks, trades, pass rule, gate, run once)",
        "portfolio": "17a's quintile 5 at each rebalance, less every company whose ranked "
        "filing (its latest) carries any of the negative tags "
        f"{list(NEGATIVE_TAGS)}, even when its similarity is high; equal weight over the "
        "rest; if none is left the money waits in cash",
        "reader": tag_spec,
        "report": "how much 17b adds over 17a: return, Sharpe, drawdown and hit rate side by "
        "side; how many names the tags dropped each month; the tag rates by quintile",
        "registry": "17b's daily returns go in the registry as their own backtest; both 17a "
        "and 17b count in the deflated Sharpe",
        "claude": "Claude never reads a historical filing; only the local open-source model "
        "on the Dell's GPU reads them",
    }


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def preregister(registry: Registry, name: str, spec: dict[str, Any]) -> bool:
    """The rules, in the registry once. Not a trial: no returns. True if written now."""
    if any(e.kind == "preregistration" and e.name == name for e in registry.load()):
        return False
    registry.append(
        kind="preregistration",
        name=name,
        config_hash=spec_hash(spec),
        data_hash="none",
        params=spec,
        metrics={},
    )
    return True


def preregistered(registry: Registry, name: str, spec: dict[str, Any]) -> bool:
    return any(
        e.kind == "preregistration" and e.name == name and e.config_hash == spec_hash(spec)
        for e in registry.load()
    )


# ---- known at ----------------------------------------------------------------------------------


def counts_from(accepted: Iterable[pd.Timestamp]) -> list[date | None]:
    """For each acceptance time (UTC), the session whose open is the first regular session
    open strictly after it."""
    out: list[date | None] = []
    first: dict[date, date] = {}  # calendar day to its session on or after
    for ts in accepted:
        if ts is None or pd.isna(ts):
            out.append(None)
            continue
        t = pd.Timestamp(ts)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        day = t.tz_convert("America/New_York").date()
        if day not in first:
            first[day] = day if clock.is_session(day) else clock.next_session(day)
        d = first[day]
        if open_time(d) <= t:
            d = clock.next_session(d)
        out.append(d)
    return out


def open_time(day: date) -> pd.Timestamp:
    return pd.Timestamp(clock._calendar().session_open(pd.Timestamp(day)))


def rebalance_days(start: date, end: date) -> list[date]:
    """The first session of each month in [start, end]."""
    days = clock.sessions_between(start, end)
    out, seen = [], set()
    for d in days:
        if (d.year, d.month) not in seen:
            seen.add((d.year, d.month))
            out.append(d)
    return out


def first_rebalance(sessions: list[date]) -> date:
    """The first month start with ADV_SESSIONS sessions of bars before it."""
    for d in rebalance_days(sessions[0], sessions[-1]):
        if sessions.index(d) >= ADV_SESSIONS:
            return d
    raise ValueError("no month start has enough history")


# ---- prices ------------------------------------------------------------------------------------


class Prices:
    """Opens, closes and dollar volume of many symbols on one session calendar.

    ``adv[i, j]``: the average dollar volume of symbol j's last ADV_SESSIONS bars up to and
    including session i (its own bars, not padded). ``last[j]``: the index of j's last bar.
    """

    def __init__(self, bars: dict[str, pd.DataFrame], sessions: list[date]) -> None:
        self.sessions = sessions
        self.index = {d: i for i, d in enumerate(sessions)}
        self.symbols = sorted(bars)
        self.col = {s: j for j, s in enumerate(self.symbols)}
        n, m = len(sessions), len(self.symbols)
        self.open = np.full((n, m), np.nan)
        self.close = np.full((n, m), np.nan)
        self.adv = np.full((n, m), np.nan)
        self.last = np.full(m, -1, dtype=int)
        cal = pd.DatetimeIndex([pd.Timestamp(d) for d in sessions])
        for s, raw in bars.items():
            j = self.col[s]
            if raw.empty:
                continue
            idx = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
            df = raw.set_axis(idx)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df = df[(df.index >= cal[0]) & (df.index <= cal[-1])]
            if df.empty:
                continue
            adv = (
                (df["close"] * df["volume"]).rolling(ADV_SESSIONS, min_periods=ADV_SESSIONS).mean()
            )
            pos = cal.get_indexer(df.index)
            ok = pos >= 0
            self.open[pos[ok], j] = df["open"].to_numpy(dtype=float)[ok]
            self.close[pos[ok], j] = df["close"].to_numpy(dtype=float)[ok]
            self.adv[pos[ok], j] = adv.to_numpy(dtype=float)[ok]
            self.last[j] = int(pos[ok].max())
        # the latest known close and dollar volume at each session (carried over gaps)
        self.close_ff = pd.DataFrame(self.close).ffill().to_numpy()
        self.adv_ff = pd.DataFrame(self.adv).ffill().to_numpy()

    def slip(self, i: int, j: int) -> float:
        """One side's slippage for a trade at session i: dollar volume through the close
        before."""
        v = self.adv_ff[i - 1, j] if i > 0 else np.nan
        return wide.slippage_bps(float(v) if np.isfinite(v) else None) / 1e4


def load_prices(
    symbols: Iterable[str], sessions: list[date], root: Path = Path("data/bars/wide")
) -> Prices:
    bars = {}
    for s in sorted(set(symbols)):
        p = root / f"{s}.parquet"
        if p.exists():
            bars[s] = pd.read_parquet(p, columns=["open", "close", "volume"])
    return Prices(bars, sessions)


# ---- ranking -----------------------------------------------------------------------------------


def eligible(filings: pd.DataFrame, prices: Prices, day: date) -> pd.DataFrame:
    """The ranked set for the rebalance at ``day``'s open: filings that started counting in
    the LOOKBACK_MONTHS before, each company's latest, with a symbol, a full-document cosine
    and the liquidity floor at the close before. Adds ``quintile``."""
    lo = pd.Timestamp(day) - pd.DateOffset(months=LOOKBACK_MONTHS)
    cf = filings["counts_from"]
    f = filings[(cf > lo) & (cf <= pd.Timestamp(day))]
    f = f[(f["symbol"] != "") & f["cos_full"].notna()]
    f = f.sort_values(["accepted", "accn"]).groupby("cik", sort=False).tail(1)
    f = f.sort_values(["accepted", "accn"]).groupby("symbol", sort=False).tail(1)
    i = prices.index[day]
    keep = []
    for s in f["symbol"]:
        j = prices.col.get(s)
        if j is None or i == 0:
            keep.append(False)
            continue
        c, v = prices.close[i - 1, j], prices.adv[i - 1, j]
        keep.append(bool(np.isfinite(c) and c >= MIN_PRICE and np.isfinite(v) and v >= MIN_ADV))
    f = f[keep].copy()
    if f.empty:
        return f.assign(quintile=pd.Series(dtype=int))
    f = f.sort_values(["cos_full", "cik"], kind="mergesort").reset_index(drop=True)
    f["quintile"] = (np.arange(len(f)) * QUINTILES // len(f)) + 1
    return f


# ---- the simulation ----------------------------------------------------------------------------


@dataclass
class Run:
    label: str
    equity: pd.Series  # at each session's close
    trades: pd.DataFrame
    costs: float
    traded: float
    holdings: dict[date, list[str]] = field(default_factory=dict)
    exits: list[dict[str, Any]] = field(default_factory=list)  # delisting exits, for the audit

    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()


def simulate(
    prices: Prices,
    targets: dict[date, list[str]],
    end: date,
    *,
    equity0: float = EQUITY,
    reserve: float = CASH_RESERVE,
    min_order: float = 0.0,
    label: str = "",
) -> Run:
    """Hold each rebalance's names equal weight from its open to the next rebalance open."""
    days = sorted(targets)
    i0, i_end = prices.index[days[0]], prices.index[end]
    cash = equity0
    shares: dict[int, float] = {}
    spells: dict[int, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    exits: list[dict[str, Any]] = []
    costs = traded = 0.0
    eq = []
    sym = prices.symbols

    def close_spell(j: int, i_exit: int, px_net: float, how: str) -> None:
        sp = spells.pop(j)
        trades.append(
            {
                "symbol": sym[j],
                "entry": prices.sessions[sp["i"]],
                "exit": prices.sessions[i_exit],
                "entry_cost": sp["px"],
                "exit_net": px_net,
                "ret": px_net / sp["px"] - 1,
                "how": how,
            }
        )

    for i in range(i0, i_end + 1):
        day = prices.sessions[i]
        # a stock with no bar today and none after has stopped trading: sold at its last close
        for j in [j for j in shares if prices.last[j] < i]:
            li = int(prices.last[j])
            px = prices.close[li, j]
            s = prices.slip(li + 1, j)
            gross = shares[j] * px * (1 - s)
            fee = gross * FEE
            cash += gross - fee
            costs += shares[j] * px * s + fee
            traded += shares[j] * px
            exits.append(
                {
                    "symbol": sym[j],
                    "last_bar": prices.sessions[li],
                    "price": float(px),
                    "shares": shares[j],
                    "noticed": day,
                }
            )
            close_spell(j, li, px * (1 - s) * (1 - FEE), "delisted")
            del shares[j]
        if day in targets:
            names = [prices.col[s] for s in targets[day] if s in prices.col]
            names = [j for j in names if np.isfinite(prices.open[i, j])]
            px_now = {
                j: (
                    prices.open[i, j]
                    if np.isfinite(prices.open[i, j])
                    else prices.close_ff[i - 1, j]
                )
                for j in shares
            }
            equity_open = cash + sum(shares[j] * px_now[j] for j in shares)
            per = equity_open * (1 - reserve) / len(names) if names else 0.0
            want = set(names)
            # sells first: names leaving, then trims
            for j in sorted(shares):
                if not np.isfinite(prices.open[i, j]):
                    continue  # not trading today: kept until it trades or stops
                o = prices.open[i, j]
                tgt = per / o if j in want else 0.0
                sell = shares[j] - tgt
                if sell <= 0 or (sell * o < min_order and j in want):
                    continue
                s = prices.slip(i, j)
                gross = sell * o * (1 - s)
                fee = gross * FEE
                cash += gross - fee
                costs += sell * o * s + fee
                traded += sell * o
                shares[j] -= sell
                if j not in want or shares[j] <= 1e-12:
                    close_spell(j, i, o * (1 - s) * (1 - FEE), "sold")
                    shares.pop(j)
            for j in names:
                o = prices.open[i, j]
                have = shares.get(j, 0.0) * o
                spend = min(per - have, cash)
                if spend <= 0 or spend < min_order:
                    continue
                s = prices.slip(i, j)
                q = spend / (o * (1 + s))
                cash -= spend
                costs += q * o * s
                traded += q * o
                shares[j] = shares.get(j, 0.0) + q
                if j not in spells:
                    spells[j] = {"i": i, "px": o * (1 + s)}
        eq.append(cash + sum(q * prices.close_ff[i, j] for j, q in shares.items()))
    for j in list(spells):  # still held at the end: marked at the last close, no exit cost
        trades.append(
            {
                "symbol": sym[j],
                "entry": prices.sessions[spells[j]["i"]],
                "exit": prices.sessions[i_end],
                "entry_cost": spells[j]["px"],
                "exit_net": prices.close_ff[i_end, j],
                "ret": prices.close_ff[i_end, j] / spells[j]["px"] - 1,
                "how": "open",
            }
        )
    eqs = pd.Series(
        eq, index=pd.DatetimeIndex([pd.Timestamp(d) for d in prices.sessions[i0 : i_end + 1]])
    )
    return Run(
        label=label,
        equity=eqs,
        trades=pd.DataFrame(trades),
        costs=costs,
        traded=traded,
        holdings=dict(targets),
        exits=exits,
    )


def buy_and_hold(
    prices: Prices, symbol: str, start: date, end: date, equity0: float = EQUITY
) -> Run:
    """All equity into ``symbol`` at ``start``'s open (slippage on the way in), held."""
    return simulate(prices, {start: [symbol]}, end, equity0=equity0, reserve=0.0, label=symbol)


def with_spy(trades: pd.DataFrame, prices: Prices) -> pd.DataFrame:
    """SPY's return over each trade's span: entry open to exit open, or to the exit close for
    a stock that stopped trading or is still held."""
    if trades.empty:
        return trades.assign(spy=pd.Series(dtype=float), beat=pd.Series(dtype=bool))
    j = prices.col["SPY"]
    spy = []
    for e, x, how in zip(trades["entry"], trades["exit"], trades["how"], strict=True):
        a = prices.open[prices.index[e], j]
        xi = prices.index[x]
        b = prices.open[xi, j] if how == "sold" else prices.close_ff[xi, j]
        spy.append(b / a - 1)
    t = trades.copy()
    t["spy"] = spy
    t["beat"] = t["ret"] > t["spy"]
    return t


# ---- grading -----------------------------------------------------------------------------------


def stats(run: Run, start: date | None = None, end: date | None = None) -> dict[str, Any]:
    eq = run.equity
    if start is not None:
        eq = eq[eq.index >= pd.Timestamp(start)]
    if end is not None:
        eq = eq[eq.index <= pd.Timestamp(end)]
    r = eq.pct_change().dropna()
    dd, dd_len = metrics.max_drawdown(eq)
    return {
        "start": str(eq.index[0].date()) if len(eq) else "",
        "end": str(eq.index[-1].date()) if len(eq) else "",
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) else math.nan,
        "cagr": metrics.cagr(eq) if len(eq) > 1 else math.nan,
        "sharpe": metrics.sharpe(r),
        "sortino": metrics.sortino(r),
        "max_drawdown": dd,
        "max_drawdown_sessions": dd_len,
        "ann_vol": metrics.ann_vol(r),
    }


def by_year(run: Run) -> dict[str, float]:
    eq = run.equity
    out = {}
    years = pd.DatetimeIndex(eq.index).year
    for y in sorted(set(years)):
        prev = eq[years < y]
        cur = eq[years == y]
        base = float(prev.iloc[-1]) if len(prev) else float(cur.iloc[0])
        out[str(y)] = float(cur.iloc[-1] / base - 1)
    return out


def trade_stats(trades: pd.DataFrame) -> dict[str, Any]:
    closed = trades[trades["how"] != "open"] if len(trades) else trades
    if closed.empty:
        return {
            "trades": 0,
            "hit_rate": math.nan,
            "avg_net": math.nan,
            "avg_vs_spy": math.nan,
            "share_positive": math.nan,
        }
    return {
        "trades": len(closed),
        "hit_rate": float(closed["beat"].mean()),
        "avg_net": float(closed["ret"].mean()),
        "avg_vs_spy": float((closed["ret"] - closed["spy"]).mean()),
        "share_positive": float((closed["ret"] > 0).mean()),
        "median_days": float(
            np.median([(x - e).days for e, x in zip(closed["entry"], closed["exit"], strict=True)])
        ),
    }


def checks(
    s: dict[str, Any],
    spy: dict[str, Any],
    ew: dict[str, Any],
    tune: dict[str, Any],
    dsr: float,
    n_trades: int,
) -> list[dict[str, Any]]:
    beat = sum(1 for w in tune.values() if w["sharpe"] > w["spy_sharpe"])
    return [
        {
            "name": "drawdown_vs_spy",
            "passed": s["max_drawdown"] < spy["max_drawdown"],
            "detail": f"{s['max_drawdown']:.2%} vs SPY {spy['max_drawdown']:.2%}",
        },
        {
            "name": "sharpe_vs_spy",
            "passed": s["sharpe"] > spy["sharpe"],
            "detail": f"{s['sharpe']:.2f} vs SPY {spy['sharpe']:.2f}",
        },
        {
            "name": "sortino_vs_spy",
            "passed": s["sortino"] > spy["sortino"],
            "detail": f"{s['sortino']:.2f} vs SPY {spy['sortino']:.2f}",
        },
        {
            "name": "deflated_sharpe",
            "passed": dsr >= DSR_BAR,
            "detail": f"DSR {dsr:.3f} (need {DSR_BAR})",
        },
        {
            "name": "drawdown_cap",
            "passed": s["max_drawdown"] <= DRAWDOWN_CAP,
            "detail": f"worst {s['max_drawdown']:.2%} (cap {DRAWDOWN_CAP:.0%})",
        },
        {
            "name": "tune_windows",
            "passed": beat >= MIN_TUNE_WINDOWS,
            "detail": f"Sharpe above SPY in {beat} of 3 (need {MIN_TUNE_WINDOWS})",
        },
        {
            "name": "trade_count",
            "passed": n_trades >= MIN_TRADES,
            "detail": f"{n_trades} closed trades (need {MIN_TRADES})",
        },
        {"name": "api_cost", "passed": True, "detail": "no Claude spend"},
        {
            "name": "sharpe_vs_universe",
            "passed": s["sharpe"] > ew["sharpe"],
            "detail": f"{s['sharpe']:.2f} vs every filer equal weight {ew['sharpe']:.2f}",
        },
    ]


def verdict(chk: list[dict[str, Any]], n_trades: int) -> str:
    if all(c["passed"] for c in chk):
        return "PASS"
    return "STARVED" if n_trades < MIN_TRADES else "FAIL"


def tune_windows(run: Run, spy: Run) -> dict[str, Any]:
    out = {}
    for w, (a, b) in TUNE_WINDOWS.items():
        out[w] = {"sharpe": stats(run, a, b)["sharpe"], "spy_sharpe": stats(spy, a, b)["sharpe"]}
    return out


# ---- the look-ahead audit of a real run -----------------------------------------------------


def audit(
    filings: pd.DataFrame, ranked: dict[date, pd.DataFrame], run: Run, prices: Prices
) -> dict[str, Any]:
    """On the real run's own records: every ranked filing was accepted before its rebalance
    open, and every stock that stopped trading while held left at its last close on disk."""
    late = 0
    worst = None
    for day, df in ranked.items():
        t_open = open_time(day)
        acc = pd.to_datetime(df["accepted"], utc=True)
        bad = acc >= t_open
        late += int(bad.sum())
        gap = (t_open - acc).min() if len(acc) else None
        worst = gap if worst is None or (gap is not None and gap < worst) else worst
    wrong_exit = 0
    for e in run.exits:
        j = prices.col[e["symbol"]]
        li = int(prices.last[j])
        if prices.sessions[li] != e["last_bar"] or prices.close[li, j] != e["price"]:
            wrong_exit += 1
    return {
        "passed": late == 0 and wrong_exit == 0,
        "filings_after_open": late,
        "closest_acceptance_before_open": str(worst),
        "delisting_exits": len(run.exits),
        "delisting_exits_not_at_last_close": wrong_exit,
    }


# ---- assembling the inputs ---------------------------------------------------------------------


def load_filings() -> pd.DataFrame:
    from stoa.filings.scores import SCORES_FILE
    from stoa.filings.tickers import OUT as SYMBOLS

    sc = pd.read_parquet(SCORES_FILE)
    sy = pd.read_parquet(SYMBOLS)
    f = sc.merge(sy[["accn", "symbol", "map_rule"]], on="accn", how="left")
    f["symbol"] = f["symbol"].fillna("")
    f["accepted"] = pd.to_datetime(f["accepted"], utc=True)
    f["counts_from"] = pd.to_datetime(
        pd.Series(counts_from(f["accepted"]), index=f.index, dtype=object)
    )
    return f[f["counts_from"].notna()].reset_index(drop=True)


def rank_all(filings: pd.DataFrame, prices: Prices, days: list[date]) -> dict[date, pd.DataFrame]:
    return {d: eligible(filings, prices, d) for d in days}


def pick(
    ranked: dict[date, pd.DataFrame],
    q: int | None,
    drop: Callable[[pd.DataFrame], pd.Series] | None = None,
    cap: int | None = None,
    prices: Prices | None = None,
) -> dict[date, list[str]]:
    out = {}
    for d, df in ranked.items():
        sel = df if q is None else df[df["quintile"] == q]
        if drop is not None:
            sel = sel[~drop(sel)]
        if cap is not None and prices is not None and len(sel) > cap:
            i = prices.index[d]
            v = [prices.adv[i - 1, prices.col[s]] for s in sel["symbol"]]
            sel = sel.assign(_v=v).sort_values(["_v", "symbol"], ascending=[False, True]).head(cap)
        out[d] = sorted(sel["symbol"])
    return out


def log_run(
    registry: Registry,
    name: str,
    spec: dict[str, Any],
    run: Run,
    kind: str = "backtest",
    extra: dict[str, Any] | None = None,
) -> Any:
    r = run.returns()
    return registry.append(
        kind=kind,
        name=name,
        config_hash=spec_hash(spec),
        data_hash=hashlib.sha256(
            json.dumps([round(x, 12) for x in r.tolist()]).encode()
        ).hexdigest()[:16],
        params={"spec": spec["spec"], "label": run.label},
        metrics={"sharpe_daily": return_moments(list(r))[0], **stats(run), **(extra or {})},
        returns=r.reset_index(drop=True),
        tags=["trial17"],
        now=datetime.now(UTC),
    )


def save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def report_for(
    run: Run, spy: Run, ew: Run, prices: Prices, registry: Registry, spec: dict[str, Any], name: str
) -> dict[str, Any]:
    """Log the graded run, then grade it (the deflated Sharpe counts it)."""
    t = with_spy(run.trades, prices)
    ts = trade_stats(t)
    s, sp, e = stats(run), stats(spy), stats(ew)
    entry = log_run(registry, name, spec, run)
    dsr = deflate(registry, list(run.returns()), variance_kind="backtest")
    tw = tune_windows(run, spy)
    chk = checks(s, sp, e, tw, dsr["dsr"], int(ts["trades"]))
    return {
        "verdict": verdict(chk, int(ts["trades"])),
        "checks": chk,
        "stats": s,
        "spy": sp,
        "universe": e,
        "trades": ts,
        "by_year": by_year(run),
        "spy_by_year": by_year(spy),
        "universe_by_year": by_year(ew),
        "tune_windows": tw,
        "deflated": dsr,
        "costs": run.costs,
        "turnover_traded": run.traded,
        "names_per_month": float(np.mean([len(v) for v in run.holdings.values()])),
        "registry_id": entry.id,
    }


def main(step: str, argv: list[str]) -> int:
    from stoa.filings import runner

    return runner.main(step, argv)
