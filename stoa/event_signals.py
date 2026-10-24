"""Trial 8's two event signals, exactly as pre-registered in docs/DECISIONS.md.

Each signal is a row: the symbol, the session at whose close it becomes known
(the decision session; entry is at the next open), its kind, and the score used
to rank it when slots are short. Nothing here looks past that session.

- Earnings drift: excess return over SPY from the close before a release to the
  close of the first full session after it, at least +5%, with that session's
  volume at least 2 times its prior 20 session average. Hold 20 sessions.
- Insider cluster: 2 or more distinct officers or directors buying in the open
  market within 10 sessions, at least $100,000 combined, 10b5-1 purchases
  excluded. Known at the close of the session after the filing date of the
  purchase that completes the cluster. Hold 40 sessions. One per issuer per 40.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from stoa import clock
from stoa.features import cell
from stoa.insider import known_session

EARNINGS = "earnings_drift"
INSIDER = "insider_cluster"
HOLD = {EARNINGS: 20, INSIDER: 40}
PRIORITY = {INSIDER: 0, EARNINGS: 1}  # insider clusters first, then earnings drift

DRIFT_MIN_EXCESS = 0.05
DRIFT_VOLUME_MULTIPLE = 2.0
DRIFT_VOLUME_SESSIONS = 20
CLUSTER_SESSIONS = 10
CLUSTER_MIN_INSIDERS = 2
CLUSTER_MIN_VALUE = 100_000.0
CLUSTER_COOLDOWN = 40
TIMED_SOURCES = frozenset({"sec_8k_2.02"})  # releases with a real time; proxies have none

SIGNAL_COLUMNS = ("symbol", "session", "kind", "score", "detail")


@dataclass(frozen=True)
class Signal:
    symbol: str
    session: date  # known at this close; enter at the next open
    kind: str
    score: float
    detail: str


def to_frame(signals: list[Signal]) -> pd.DataFrame:
    return pd.DataFrame(
        [(s.symbol, s.session, s.kind, s.score, s.detail) for s in signals],
        columns=list(SIGNAL_COLUMNS),
    )


def ranked(signals: list[Signal]) -> list[Signal]:
    """The registered order: insider clusters (largest first), then drift (largest first)."""
    return sorted(signals, key=lambda s: (PRIORITY[s.kind], -s.score, s.symbol))


# ---- earnings drift -----------------------------------------------------------------------


def release_sessions(event_time: datetime) -> tuple[date, date]:
    """(the session whose close is the last before the release, the first full session after).

    Before the open: the day before, then that day. After the close: that day,
    then the next. During a session or on a closed day: the day before (or last
    session), then the next session.
    """
    d = clock.to_ny(event_time).date()
    if clock.is_session(d):
        s = clock.session(d)
        if event_time < s.open:
            return clock.previous_session(d), d
        if event_time >= s.close:
            return d, clock.next_session(d)
    return clock.previous_session(d), clock.next_session(d)


def earnings_drift(
    releases: pd.DataFrame, bars: Mapping[str, pd.DataFrame], benchmark: str
) -> list[Signal]:
    """Drift signals from timed earnings releases. ``bars`` must include the benchmark."""
    spy = bars[benchmark]["close"]
    out: list[Signal] = []
    timed = releases.loc[releases["source"].isin(TIMED_SOURCES)]
    for sym, group in timed.groupby("symbol"):
        df = bars.get(str(sym))
        if df is None or df.empty:
            continue
        idx = df.index
        if len(idx) <= DRIFT_VOLUME_SESSIONS:
            continue
        first_ok, last_ok = idx[DRIFT_VOLUME_SESSIONS].date(), idx[-1].date()
        for t in pd.to_datetime(group["event_time"], utc=True):
            when = t.to_pydatetime()
            if not first_ok <= when.date() <= last_ok:
                continue  # outside this stock's bar history (and the calendar is safe)
            before, first = release_sessions(when)
            b, f = pd.Timestamp(before), pd.Timestamp(first)
            if b not in idx or f not in idx or b not in spy.index or f not in spy.index:
                continue
            pos = idx.get_loc(f)
            assert isinstance(pos, int)
            if pos < DRIFT_VOLUME_SESSIONS:
                continue
            prior = df["volume"].iloc[pos - DRIFT_VOLUME_SESSIONS : pos]
            vol = float(df["volume"].iloc[pos])
            avg = float(prior.mean())
            if not avg > 0 or vol < DRIFT_VOLUME_MULTIPLE * avg:
                continue
            stock = cell(df, f, "close") / cell(df, b, "close") - 1.0
            bench = float(spy.loc[f]) / float(spy.loc[b]) - 1.0
            excess = stock - bench
            if excess >= DRIFT_MIN_EXCESS:
                out.append(
                    Signal(
                        symbol=str(sym),
                        session=first,
                        kind=EARNINGS,
                        score=excess,
                        detail=f"{excess:+.1%} vs SPY after the {t:%Y-%m-%d %H:%M} UTC release, "
                        f"volume {vol / avg:.1f}x",
                    )
                )
    return out


# ---- insider clusters ---------------------------------------------------------------------


def _ordinal(sessions: list[date], day: date) -> int:
    """Index of ``day`` among sessions, or of the next session when ``day`` is not one."""
    return bisect.bisect_left(sessions, day)


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("-", ".").replace("/", ".")


def insider_clusters(purchases: pd.DataFrame, sessions: list[date]) -> list[Signal]:
    """Cluster signals from officer and director open-market purchases, point in time.

    ``sessions`` is the trading calendar covering every purchase and filing date.
    """
    p = purchases.loc[purchases["plan_10b5_1"].map(lambda v: v is not True)].copy()
    if p.empty:
        return []
    p["t"] = [_ordinal(sessions, d) for d in p["trans_date"]]
    p["k"] = [_ordinal(sessions, known_session(d)) for d in p["filing_date"]]
    p = p.loc[p["t"] <= p["k"]]  # a purchase cannot be known before it happened
    out: list[Signal] = []
    for _cik, issuer in p.groupby("issuer_cik", sort=False):
        g = issuer.sort_values(["k", "t"])
        t = g["t"].to_numpy()
        k = g["k"].to_numpy()
        owner = g["owner_cik"].to_numpy()
        value = g["value"].to_numpy(dtype=float)
        symbol = g["symbol"].to_numpy()
        last_signal = -(10**9)
        for kk in sorted(set(k.tolist())):
            if kk - last_signal < CLUSTER_COOLDOWN or kk >= len(sessions):
                continue
            known = k <= kk
            best, best_n = 0.0, 0
            for ti in sorted({int(x) for x in t[k == kk]}):
                # Every 10 session window that contains this newly known purchase date.
                for lo in range(ti - CLUSTER_SESSIONS + 1, ti + 1):
                    inside = known & (t >= lo) & (t < lo + CLUSTER_SESSIONS)
                    n = len(set(owner[inside].tolist()))
                    total = float(value[inside].sum())
                    if n >= CLUSTER_MIN_INSIDERS and total > best:
                        best, best_n = total, n
            if best >= CLUSTER_MIN_VALUE:
                last_signal = kk
                sym = normalize_symbol(str(symbol[k == kk][-1]))
                out.append(
                    Signal(
                        symbol=sym,
                        session=sessions[kk],
                        kind=INSIDER,
                        score=best,
                        detail=f"{best_n} insiders, ${best:,.0f} bought within 10 sessions",
                    )
                )
    return out


def by_session(signals: list[Signal]) -> Mapping[date, list[Signal]]:
    out: dict[date, list[Signal]] = {}
    for s in signals:
        out.setdefault(s.session, []).append(s)
    return {d: ranked(v) for d, v in out.items()}
