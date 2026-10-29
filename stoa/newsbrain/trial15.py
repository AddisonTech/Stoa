"""Trial 15: the owner's trading brain on the big paper account (docs/DECISIONS.md).

The brain reads what each news item means (a local open-source model, never Claude on
history), remembers how each kind of event moved stocks over the decade, follows ripples to
suppliers and customers, hunts news landing before the open on small, thinly covered
companies, and picks the 5 stocks it is most sure of every trading day. It reviews its own
picks every night and retrains every night, and a retrained version takes over only when it
grades better.

``spec()`` is the rule book, written to the registry before any trial 15 code. The pieces
live in their own modules; this one holds the rules and the grading.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import ROUND_DOWN, Decimal
from typing import Any

import numpy as np
import pandas as pd

from stoa import clock, event_signals, wide
from stoa.broker.base import BrokerPosition
from stoa.experiments import Registry
from stoa.models import PlannedOrder, Side

SPEC = "trial15:brain:v1"
NAME = "trial15:preregistration"
DECIDED = date(2026, 10, 29)
PICKS = 5
RESERVE = 0.02  # cash kept back; each pick gets (1 - RESERVE) / PICKS of equity
TARGET_HIT_RATE = 0.75  # the owner's target, reported beside the running hit rate
HALF_LIFE_YEARS = 2.0  # recent rows weigh more: a row's weight halves every 2 years
TEST_START = date(2023, 1, 1)
TEST_END = date(2026, 6, 30)
WINDOW = 20  # sessions in the rolling window a nightly retrain is graded over
MIN_SHADOW = 20  # graded shadow sessions before a retrain may take over
LESSON_MIN_COUNT = 5  # times a reason must recur before its rule is tested
LESSON_MIN_YEARS = 7  # of the 10 years a rule must help in, besides helping overall


def spec() -> dict[str, Any]:
    """The pre-registered rules, as written to the registry before any trial 15 code."""
    return {
        "spec": SPEC,
        "decided": f"{DECIDED} by the owner",
        "question": "can a brain that reads what news means, remembers how each kind of event "
        "moved stocks, follows ripples to partners and hunts thinly covered small companies "
        "before the open pick 5 stocks a day that beat SPY after costs",
        "target": "the owner's: picks beat SPY after costs 75% of the time; reported every "
        "day beside the running hit rate, never used to choose or tune anything",
        "account": "the big paper account (about $100,000); the $100 account and the old "
        "shadow scoreboard stay stopped",
        "trading": "every trading day the 9:00 am job ranks every stock the live brain scores "
        "and picks the top 5 by its score (conviction); no 0.55 line, no SPY, no regime filter. "
        "At 9:45 am the stocks held from the last plan are sold and the 5 bought at equal "
        "weight, (equity x 0.98) / 5 each, so 2% stays in cash. A stock picked again is kept "
        "and resized to its new weight instead of sold and bought back. Fewer than 5 scored "
        "stocks: the ones there are, same size each, the rest in cash. Funds and SPY are never "
        "picks",
        "rails": "kept: the 25% kill switch (everything sold, nothing bought until the owner "
        "resets it) and the price sanity check on every order. Off: the 0.55 line, the regime "
        "filter, the SPY core, the sector cap, the correlation limit, the 3% daily loss stop, "
        "the ATR exits and broker-side stops, which would change which 5 are held",
        "grading": "a pick beats SPY when its 9:45 am to 9:45 am return, one session later, "
        "minus SPY's over the same span, minus a round trip of slippage and the fee "
        "(stoa/shadowboard.py), is above zero. Hit rate is the share of picks that beat SPY; "
        "average edge is the mean of that net return; day hit rate is hits out of the day's "
        "picks",
        "brain_start": "trial 9's frozen 1 session model (13 features, trained 2016 to 2022), "
        "its p used only to rank; it trades from the first run until a new brain passes",
        "reader": "Qwen/Qwen2.5-1.5B-Instruct (Apache 2.0, revision 989aa798) on the Dell GPU "
        "for history and on CPU live, prompt version er1 (stoa/newsbrain/eventread.py): per "
        "article and tagged company, an event type from a fixed list of 20, a direction "
        "p(good) - p(bad) and an expected size 1 to 3. It reads the headline and the first "
        "120 characters of the summary, never a price or a date. Claude never reads "
        "historical news. Known limit: the open model was trained on text through about 2024 "
        "and may know some outcomes; the labels are what the text says and every outcome "
        "comes from prices",
        "event_memory": "for each event type by size (dollar volume tercile) and sector, the "
        "mean and hit rate of the next session's return against SPY after costs (open to "
        "open, trial 9's label), computed as of each session from earlier sessions only, "
        "shrunk toward the event type's mean with weight n / (n + 50)",
        "ripple_map": "supplier and customer links from the 10-K text already pulled: a "
        "company named in another's 10-K, typed customer, supplier or other by the words "
        "around the name. When news hits a company its partners are flagged, with the "
        "news's direction and how partners of that relation historically moved the next "
        "session after that event type (as of each session, earlier sessions only)",
        "premarket_hunter": "news known between the prior 4 pm close and the 9:00 am cutoff "
        "on small (bottom dollar volume tercile) and thinly covered (at most 2 articles in "
        "the 60 sessions before) companies gets its own inputs: the flag, the hours before "
        "the open, and from minute bars how much of the move to the next open was already "
        "done by 9:45 am for past such events of the same type (as of each session)",
        "retrain": "the same classifier as trial 9 (histogram gradient boosting, same "
        "settings, same 1 session label) on the 13 features plus the new inputs, every row "
        "weighted 0.5 ^ (age in years / 2)",
        "comparison": "walk-forward on 2023-01-01 to 2026-06-30: refit every 63 sessions on "
        "every row labelled before the slice (from 2016), same weights; each session both "
        "brains rank the same rows and their top 5 are graded on trial 9's label (open to "
        "open against SPY after costs). The new brain goes live only if its average edge is "
        "higher and its hit rate is at least as high, and only once its live inputs run; else "
        "the frozen brain stays. Run once, logged as one trial with its daily top 5 returns",
        "live_reading": "the 8:15 am feed and the 9:00 am sweep run the local reader on every "
        "new article. Claude (the existing analyst, its model and the monthly cap) reads the "
        "news behind the day's top 20 on top: a second opinion recorded beside each pick and "
        "graded nightly. It never changes the picks while it has no track record to test",
        "nightly_review": "after the 5:15 pm summary, for each graded pick: what it saw (its "
        "inputs and score), what happened against SPY after costs, and when wrong the most "
        "likely reason from a fixed list, by fixed rules. A running lessons file counts the "
        "reasons. A reason seen at least 5 times may turn into a candidate rule from a fixed "
        "menu; the rule is used only if over 2016 to the last full year it raises the top 5 "
        "average edge overall and in at least 7 of the 10 years, so one bad day never "
        "changes anything. Each rule is tested once and logged",
        "daily_learning": "every night the brain is refit on every row labelled through that "
        "day, same weights. The refit scores the next morning's rows beside the live brain, "
        "its top 5 locked before 9:30 am and graded like the picks. It takes over only after "
        "20 graded sessions when, over the last 20, its average edge is higher and its hit "
        "rate at least as high; else the live brain stays",
        "journal": "the 5:15 pm summary lists the day's 5 picks, each one's result against "
        "SPY, the day's hit rate and the running hit rate against the 75% target; the "
        "dashboard card shows trial 15 the same way",
        "ready_rule": "a piece goes live only when built, tested and checked; at 9:00 am the "
        "picks use whatever is ready, and nothing half-built trades",
        "holdout": "2026-07-01 to 2026-10-23 was spent by trial 12 and is never graded; its "
        "rows may be used to train later refits, as the retrain-all did",
    }


def spec_hash() -> str:
    return hashlib.sha256(json.dumps(spec(), sort_keys=True).encode()).hexdigest()[:16]


def preregister(registry: Registry) -> None:
    """The rules, in the registry before any trial 15 code. Not a trial: it holds no returns."""
    if any(e.kind == "preregistration" and e.name == NAME for e in registry.load()):
        return
    registry.append(
        kind="preregistration",
        name=NAME,
        config_hash=spec_hash(),
        data_hash="none",
        params=spec(),
        metrics={},
    )


def preregistered(registry: Registry) -> bool:
    return any(
        e.kind == "preregistration" and e.name == NAME and e.config_hash == spec_hash()
        for e in registry.load()
    )


# ---- the 9:00 am picks ------------------------------------------------------------------------

FUND_SIC = frozenset({"6221", "6722", "6726"})  # commodity trusts, funds, unit trusts
CORE = "SPY"
KIND = "trial15_top5"
MIN_RESIZE = Decimal("50")  # a repeat pick within $50 of its weight is left alone


def is_company(symbol: str, sic: Mapping[str, str | None]) -> bool:
    """An operating company: it has an industry code on file and is not a fund or trust.
    Funds (SPY, QQQ, sector and country funds) carry no code in the SEC map."""
    code = sic.get(symbol)
    return symbol != CORE and code is not None and str(code) not in FUND_SIC


def conviction(brain: Any, rows: pd.DataFrame) -> np.ndarray:
    """How sure the brain is, used only to rank. A calibrated model's p has flat steps
    (many stocks share one value), so the classifier's own probability before calibration
    is used: it ranks the same way and breaks the ties."""
    clf, feats = getattr(brain, "clf", None), getattr(brain, "features", None)
    if clf is not None and feats is not None:
        x = rows[list(feats)].to_numpy(dtype=float)
        return np.asarray(clf.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(brain.predict(rows), dtype=float)


def choose(rows: pd.DataFrame, sic: Mapping[str, str | None], n: int = PICKS) -> pd.DataFrame:
    """The top ``n`` companies by conviction, best first; ties by symbol. ``rows`` needs
    symbol and conviction."""
    if rows.empty:
        return rows
    ok = rows[[is_company(str(s), sic) for s in rows["symbol"]]]
    ok = ok[np.isfinite(ok["conviction"].to_numpy(dtype=float))]
    ranked = ok.sort_values(["conviction", "symbol"], ascending=[False, True], kind="stable")
    return ranked.drop_duplicates("symbol").head(n).reset_index(drop=True)


def detail(r: Mapping[str, Any]) -> str:
    """What the pick saw, in a line."""
    parts = [f"conviction {float(r['conviction']):.3f}"]
    for key, name, fmt in (
        ("sent", "news tone", "{:+.2f}"),
        ("n", "articles", "{:.0f}"),
        ("surprise", "surprise", "{:+.1f}"),
        ("filing_tone", "filing tone", "{:+.2f}"),
        ("ripple", "ripple", "{:+.2f}"),
        ("ret_1", "last day", "{:+.1%}"),
    ):
        v = r.get(key)
        if isinstance(v, int | float | np.floating) and math.isfinite(float(v)):
            parts.append(f"{name} {fmt.format(float(v))}")
    return ", ".join(parts)


def signals(chosen: pd.DataFrame, session: date) -> list[event_signals.Signal]:
    """The picks as engine signals, filed under the close before ``session``."""
    decided = clock.previous_session(session)
    return [
        event_signals.Signal(
            str(r["symbol"]),
            decided,
            KIND,
            float(r["conviction"]),
            detail({str(k): v for k, v in r.items()}),
        )
        for r in chosen.to_dict("records")
    ]


RECORD_KEYS = (
    "symbol",
    "conviction",
    "p",
    "adv",
    "sent",
    "n",
    "surprise",
    "filing_tone",
    "ripple",
    "ret_1",
    "ret_5",
    "vol_20",
    "log_adv",
    "coverage",
    "first_break",
)


def records(chosen: pd.DataFrame) -> list[dict[str, Any]]:
    """What each pick saw, kept for the nightly review: its inputs and scores."""
    out = []
    for r in chosen.to_dict("records"):
        rec: dict[str, Any] = {}
        for k in RECORD_KEYS:
            v = r.get(k)
            if isinstance(v, str):
                rec[k] = v
            elif isinstance(v, int | float | np.floating | np.integer) and math.isfinite(float(v)):
                rec[k] = float(v)
        out.append(rec)
    return out


def cents(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_DOWN)


def target_notional(equity: Decimal, n: int = PICKS) -> Decimal:
    """Each pick's money: (equity x 0.98) / 5."""
    return cents(equity * (Decimal(1) - Decimal(str(RESERVE))) / Decimal(n))


def plan_orders(
    picks: Sequence[event_signals.Signal],
    held: Mapping[str, BrokerPosition],
    equity: Decimal,
    ref_prices: Mapping[str, Decimal],
    fractionable: Mapping[str, bool],
) -> list[PlannedOrder]:
    """Sell what is held and not picked; buy each new pick at (equity x 0.98) / 5; resize a
    pick already held to that weight. Sells come first."""
    target = target_notional(equity)
    want = {x.symbol: x for x in picks}
    sells: list[PlannedOrder] = []
    buys: list[PlannedOrder] = []
    for sym, pos in sorted(held.items()):
        if pos.qty <= 0:
            continue
        if sym not in want:
            sells.append(
                PlannedOrder(
                    symbol=sym,
                    side=Side.SELL,
                    qty=pos.qty,
                    ref_price=pos.current_price,
                    reason="hold_period",
                    thesis="held one session; not in the new top 5",
                )
            )
            continue
        diff = target - pos.market_value
        if abs(diff) < MIN_RESIZE:
            continue
        price = ref_prices.get(sym) or pos.current_price
        if diff > 0:
            buys.append(
                PlannedOrder(
                    symbol=sym,
                    side=Side.BUY,
                    notional=cents(diff),
                    ref_price=price,
                    reason="resize",
                    probability=want[sym].score,
                    thesis="picked again; topped up to its weight. " + want[sym].detail,
                )
            )
        else:
            step = Decimal("0.000001") if fractionable.get(sym, False) else Decimal(1)
            qty = (-diff / price).quantize(step, rounding=ROUND_DOWN)
            if qty > 0:
                sells.append(
                    PlannedOrder(
                        symbol=sym,
                        side=Side.SELL,
                        qty=min(qty, pos.qty),
                        ref_price=price,
                        reason="resize",
                        thesis="picked again; trimmed to its weight",
                    )
                )
    for x in picks:
        if x.symbol in held and held[x.symbol].qty > 0:
            continue
        px = ref_prices.get(x.symbol)
        if px is None or px <= 0:
            continue  # no price to plan with; the alert lists it as not bought
        buys.append(
            PlannedOrder(
                symbol=x.symbol,
                side=Side.BUY,
                notional=target,
                ref_price=px,
                reason="trial15_pick",
                probability=x.score,
                thesis=x.detail,
            )
        )
    return sells + buys


# ---- grading -----------------------------------------------------------------------------------


def grade(
    picks: Sequence[str],
    entry: Mapping[str, float],
    exit_: Mapping[str, float],
    adv: Mapping[str, float | None],
) -> list[dict[str, Any]]:
    """Each pick's 9:45 am to 9:45 am return, SPY's over the same span, and the net after a
    round trip of slippage and the fee (the shadow scoreboard's rule). A pick with no price
    on either day is ungraded (hit None)."""
    from stoa.shadowboard import FEE

    def ret(sym: str) -> float | None:
        a, b = entry.get(sym), exit_.get(sym)
        if a is None or b is None or not a > 0 or not math.isfinite(b):
            return None
        return b / a - 1

    spy = ret(CORE)
    out: list[dict[str, Any]] = []
    for sym in picks:
        r = ret(sym)
        if r is None or spy is None:
            out.append({"symbol": sym, "r": None, "spy": spy, "net": None, "hit": None})
            continue
        v = adv.get(sym)
        slip = wide.slippage_bps(v if v is not None and math.isfinite(v) else None) / 1e4
        net = r - spy - 2 * slip - FEE
        out.append({"symbol": sym, "r": r, "spy": spy, "net": net, "hit": net > 0})
    return out


def tally(days: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Running totals over graded days: picks, hits, hit rate, average edge, and days whose
    picks together beat SPY."""
    graded = [g for d in days for g in d.get("picks", []) if g.get("hit") is not None]
    hits = sum(1 for g in graded if g["hit"])
    nets = [float(g["net"]) for g in graded]
    day_nets = []
    for d in days:
        ok = [float(g["net"]) for g in d.get("picks", []) if g.get("hit") is not None]
        if ok:
            day_nets.append(sum(ok) / len(ok))
    return {
        "days": len(day_nets),
        "picks": len(graded),
        "hits": hits,
        "hit_rate": hits / len(graded) if graded else None,
        "avg_edge": sum(nets) / len(nets) if nets else None,
        "days_beating_spy": sum(1 for x in day_nets if x > 0),
        "target": TARGET_HIT_RATE,
    }
