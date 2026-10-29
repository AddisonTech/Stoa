"""How much of a small company's pre-market news move is done by 9:45 am, and how fast.

For news that lands before the open on small, thinly covered names, this measures from
minute bars how far the price had already moved by 9:45 am New York time (when the news
brain buys), how much was left from there to the next session's 9:45 am sale, and how
many minutes after the open the session's move took to get 80% done. It writes one row
per event; averaging the rows by event type happens elsewhere. Analysis only.

An event is a row of trial 9's panel (one per liquid symbol and session) where:

- pre-market news: at least one article was known after the previous session's close
  (4:00 pm, or the early close) and by 9:00 am on the session, the panel's own cutoff
  (``panel.load_news`` places each article on the first session whose 9:00 am cutoff is at
  or after its known time, the later of created and updated);
- small: its log_adv is in the bottom third of that session's panel rows;
- thin: at most 2 articles about it in the 60 sessions before. The panel's ``coverage`` is
  log1p of the mean daily article count over exactly those 60 sessions (sessions t-60 to
  t-1, zero-news days included, see ``panel.daily_news``), so the count is
  expm1(coverage) * 60. That is what the filter uses; ``prior_articles`` recounts it from
  the news files the same way as a check, and the build logs any disagreement.

Prices come from the compacted minute bars, regular-session minutes only (9:30 am to
4:00 pm New York time). A price at a time is the open of the bar at that minute, or of the
first bar after it within 30 minutes; with none it is NaN. ``prev_close`` is the close of
the previous session's last regular bar (the panel's daily close is kept beside it as
``prev_close_daily``). ``p_945`` is the same rule as the scoreboard's 9:45 am price (the
first bar from 9:45 to 10:15 am). ``minutes_to_80`` walks the session's bars in time, each
bar's open at its minute and its close one minute later, and gives the minutes after the
open until a price first comes within 20% of the move from prev_close to p_close; it is NaN
when that move is under 1%.

Sessions inside the holdout are never read (``sealed``): a price that would need one is
NaN. Each month is written as its own part, so a stopped run carries on where it was.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stoa import clock
from stoa.newsbrain import ROOT, sealed
from stoa.newsbrain import panel as pn
from stoa.newsbrain.alpaca_pull import MINUTE_STORE_DIR
from stoa.newsbrain.pace import write_parquet

log = logging.getLogger(__name__)

START = date(2016, 1, 1)  # the minute bars start in January 2016
SMALL_SHARE = 1 / 3  # bottom third of the session's log_adv
THIN_MAX = 2  # at most this many articles in the 60 sessions before
MOVE_MIN = 0.01  # minutes_to_80 needs a session move of at least 1%
DONE = 0.8  # share of the move that counts as caught up
BUY_AT = "09:45"
WITHIN = pd.Timedelta(minutes=30).value
MINUTE = pd.Timedelta(minutes=1).value
SPAN = 1 << 20  # more sessions than the calendar holds, for (symbol, session) keys

OUT = ROOT / "premarket" / "catchup.parquet"
PRICES = ("prev_close", "p_open", "p_945", "p_close", "p_next_945")
MEASURES = ("gap_945", "rest", "minutes_to_80")


class Minutes(NamedTuple):
    """One symbol's regular-session minute bars, sorted by time."""

    t: np.ndarray  # bar start, UTC nanoseconds
    o: np.ndarray
    c: np.ndarray


NO_BARS = Minutes(np.array([], dtype=np.int64), np.array([]), np.array([]))


# ---- sessions and the wall -------------------------------------------------------------------


@cache
def times(day: date) -> tuple[int, int, int]:
    """Open, 9:45 am and close of a session, as UTC nanoseconds."""
    s = clock.session(day)
    buy = pd.Timestamp(clock.at_ny(day, BUY_AT)).value
    return pd.Timestamp(s.open).value, buy, pd.Timestamp(s.close).value


def allowed(day: date) -> bool:
    """Whether minute bars of ``day`` may be read (the holdout wall says no inside it)."""
    try:
        sealed.check(day, "minute bars")
    except sealed.HoldoutBreach:
        return False
    return True


def stamp(day: date) -> str:
    return f"{day.year:04d}-{day.month:02d}"


# ---- which events ------------------------------------------------------------------------------


def premarket_news(news: pd.DataFrame, days: list[date]) -> pd.DataFrame:
    """Per (symbol, session): articles known after the previous session's close.

    ``news`` is ``panel.load_news`` output (symbol, sidx, known_at): every article is
    already on the session whose 9:00 am cutoff is the first at or after its known time.
    Returns symbol, sidx, first_known (the earliest such article) and n_pre (how many).
    """
    closes = np.array([times(d)[2] for d in days], dtype=np.int64)
    s = news["sidx"].to_numpy()
    known = pn.nanos(news["known_at"])
    before = np.where(s > 0, closes[np.clip(s - 1, 0, len(days) - 1)], np.iinfo(np.int64).max)
    pre = news.loc[(s > 0) & (known > before), ["symbol", "sidx", "known_at"]]
    g = pre.groupby(["symbol", "sidx"], sort=True)["known_at"]
    out = g.agg(first_known="min", n_pre="size").reset_index()
    out["first_known"] = pd.to_datetime(out["first_known"], utc=True)
    return pd.DataFrame(out)


def prior_articles(
    news: pd.DataFrame, keys: pd.DataFrame, window: int = pn.BASELINE_SESSIONS
) -> np.ndarray:
    """Articles about each key's symbol in the ``window`` sessions before its session.

    Counted as ``panel.daily_news`` counts them: one per article and tagged symbol, on the
    session ``panel.load_news`` gave it, over sessions sidx - window to sidx - 1.
    """
    cnt = news.groupby(["symbol", "sidx"], sort=True).size()
    syms = pd.Index(sorted(set(cnt.index.get_level_values(0)) | set(keys["symbol"])))
    code = syms.get_indexer(cnt.index.get_level_values(0)).astype(np.int64)
    pos = code * SPAN + cnt.index.get_level_values(1).to_numpy(dtype=np.int64)
    total = np.concatenate([[0], np.cumsum(cnt.to_numpy())])
    kc = syms.get_indexer(keys["symbol"]).astype(np.int64) * SPAN
    i = keys["sidx"].to_numpy(dtype=np.int64)

    def upto(x: np.ndarray) -> np.ndarray:
        out: np.ndarray = total[np.searchsorted(pos, kc + x, side="right")]
        return out

    got: np.ndarray = upto(i - 1) - upto(i - 1 - window)
    return got


def select_events(panel: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    """Panel rows that are small, thin and have pre-market news.

    ``panel``: symbol, sidx, log_adv, coverage and anything else to keep. Small is judged
    against all of the session's panel rows, before any other filter.
    """
    third = panel.groupby("sidx")["log_adv"].quantile(SMALL_SHARE)
    small = (panel["log_adv"] <= panel["sidx"].map(third)).to_numpy()
    prior = np.rint(np.expm1(panel["coverage"].to_numpy(dtype=float)) * pn.BASELINE_SESSIONS)
    rows = panel.assign(prior_60=prior)[small & (prior <= THIN_MAX)]
    out = rows.merge(pre, on=["symbol", "sidx"], how="inner")
    return out.sort_values(["sidx", "symbol"]).reset_index(drop=True)


def events() -> pd.DataFrame:
    """Every event from 2016 on, with its pre-market news and the thin check."""
    days, cut = pn.calendar(pn.PANEL_START, pn.PANEL_END)
    log.info("loading news")
    news = pn.load_news(cut)[["symbol", "sidx", "known_at"]]
    log.info("%d article-symbol rows", len(news))
    pre = premarket_news(news, days)
    log.info("%d symbol-sessions with pre-market news", len(pre))
    cols = ["symbol", "sidx", "session", "n", "log_adv", "prev_close", "coverage"]
    panel = pd.read_parquet(pn.PANEL, columns=cols)
    ev = select_events(panel, pre)
    del panel
    ev["prior_recount"] = prior_articles(news, ev)
    off = ev["prior_recount"] != ev["prior_60"]
    log.info(
        "thin check: coverage and the news recount disagree on %d of %d events",
        int(off.sum()),
        len(ev),
    )
    ev = ev[ev["session"] >= START].rename(columns={"prev_close": "prev_close_daily"})
    log.info("%d events from %s", len(ev), START)
    return ev.drop(columns=["coverage"]).reset_index(drop=True)


# ---- prices from minute bars ---------------------------------------------------------------------


def price(x: float) -> float:
    return float(x) if math.isfinite(x) and x > 0 else math.nan


def price_at(m: Minutes, target: int) -> float:
    """The open of the bar at ``target``, or of the first bar after it within 30 minutes."""
    j = int(np.searchsorted(m.t, target, side="left"))
    if j < len(m.t) and m.t[j] <= target + WITHIN:
        return price(float(m.o[j]))
    return math.nan


def last_close(m: Minutes, day: date) -> float:
    """The close of the last regular-session bar of ``day``."""
    lo, _, hi = times(day)
    j = int(np.searchsorted(m.t, hi, side="left")) - 1
    if j >= 0 and m.t[j] >= lo:
        return price(float(m.c[j]))
    return math.nan


def minutes_to_80(m: Minutes, day: date, prev_close: float, p_close: float) -> float:
    """Minutes after the open until a price first comes within 20% of the session's move.

    The path is each bar's open at its minute and its close one minute later. NaN when the
    move from ``prev_close`` to ``p_close`` is under 1% or a price is missing.
    """
    if not (math.isfinite(prev_close) and math.isfinite(p_close)) or prev_close <= 0:
        return math.nan
    move = p_close - prev_close
    if abs(move / prev_close) < MOVE_MIN:
        return math.nan
    lo, _, hi = times(day)
    a = int(np.searchsorted(m.t, lo, side="left"))
    b = int(np.searchsorted(m.t, hi, side="left"))
    if b <= a:
        return math.nan
    at = (m.t[a:b] - lo) / MINUTE
    band = (1 - DONE) * abs(move)
    hit_o = np.flatnonzero(np.abs(m.o[a:b] - p_close) <= band)
    hit_c = np.flatnonzero(np.abs(m.c[a:b] - p_close) <= band)
    first = [float(at[hit_o[0]])] if len(hit_o) else []
    first += [float(at[hit_c[0]]) + 1] if len(hit_c) else []
    return min(first) if first else math.nan


def measure(m: Minutes, day: date, prev_day: date, next_day: date) -> dict[str, float]:
    """Prices and measures for one event from its symbol's bars (sessions in the holdout
    are not read and give NaN)."""
    out = dict.fromkeys((*PRICES, *MEASURES), math.nan)
    if not allowed(day):
        return out
    open_ns, buy_ns, _ = times(day)
    if allowed(prev_day):
        out["prev_close"] = last_close(m, prev_day)
    out["p_open"] = price_at(m, open_ns)
    out["p_945"] = price_at(m, buy_ns)
    out["p_close"] = last_close(m, day)
    if allowed(next_day):
        out["p_next_945"] = price_at(m, times(next_day)[1])
    out["gap_945"] = out["p_945"] / out["prev_close"] - 1
    out["rest"] = out["p_next_945"] / out["p_945"] - 1
    out["minutes_to_80"] = minutes_to_80(m, day, out["prev_close"], out["p_close"])
    return out


def load_bars(
    root: Path, month: str, symbols: set[str], lo: pd.Timestamp, hi: pd.Timestamp
) -> pd.DataFrame:
    """Regular-session minute bars (symbol, t in UTC nanoseconds, o, c) of the given symbols
    in one month's files, from ``lo`` to ``hi``. Reads only what the filters let through."""
    cols = ["symbol", "t", "o", "c"]
    empty = pd.DataFrame({"symbol": [], "t": np.array([], dtype=np.int64), "o": [], "c": []})
    if not sealed.month_allowed(month) or not symbols:
        return empty
    want = sorted(symbols)
    parts = []
    for path in sorted((root / month).glob("*.parquet")):
        if pq.ParquetFile(path).metadata.num_rows == 0:
            continue  # an empty group file: no bars, and no text type to filter
        filters = [("symbol", "in", want), ("t", ">=", lo), ("t", "<=", hi)]
        df = pq.read_table(path, columns=cols, filters=filters).to_pandas()
        if len(df):
            parts.append(df)
    if not parts:
        return empty
    df = pd.concat(parts, ignore_index=True)
    t = pd.to_datetime(df["t"], utc=True)
    ny = t.dt.tz_convert(clock.NY)
    minute = ny.dt.hour * 60 + ny.dt.minute
    keep = ((minute >= 9 * 60 + 30) & (minute < 16 * 60)).to_numpy()
    out = pd.DataFrame(
        {
            "symbol": df["symbol"].astype(str).to_numpy()[keep],
            "t": t.dt.as_unit("ns").astype("int64").to_numpy()[keep],
            # float32 in the compacted files: back to the price as stored (as MinuteStore)
            "o": df["o"].to_numpy(dtype="float64")[keep].round(4),
            "c": df["c"].to_numpy(dtype="float64")[keep].round(4),
        }
    )
    return out


def by_symbol(bars: pd.DataFrame) -> dict[str, Minutes]:
    bars = bars.drop_duplicates(["symbol", "t"]).sort_values(["symbol", "t"], kind="stable")
    syms = bars["symbol"].to_numpy()
    t = bars["t"].to_numpy(dtype=np.int64)
    o = bars["o"].to_numpy(dtype=float)
    c = bars["c"].to_numpy(dtype=float)
    out: dict[str, Minutes] = {}
    if not len(syms):
        return out
    names, first = np.unique(syms, return_index=True)
    ends = np.append(first[1:], len(syms))
    for name, a, b in zip(names, first, ends, strict=True):
        out[str(name)] = Minutes(t[a:b], o[a:b], c[a:b])
    return out


def measure_month(ev: pd.DataFrame, root: Path = MINUTE_STORE_DIR) -> pd.DataFrame:
    """Prices and measures for the events of one month.

    Bars are read once per bar month for just the symbols that need them: the events'
    month whole, and from the months before and after only the edge sessions (the previous
    close of the month's first session, the next 9:45 am of its last).
    """
    sessions: list[date] = list(ev["session"])
    prev = [clock.previous_session(d) for d in sessions]
    nxt = [clock.next_session(d) for d in sessions]
    need: dict[str, set[str]] = defaultdict(set)
    span: dict[str, tuple[int, int]] = {}
    for sym, *ds in zip(ev["symbol"], prev, sessions, nxt, strict=True):
        for d in ds:
            if not allowed(d):
                continue
            k = stamp(d)
            need[k].add(str(sym))
            lo, _, hi = times(d)
            a, b = span.get(k, (lo, hi))
            span[k] = (min(a, lo), max(b, hi))
    frames = [
        load_bars(root, k, need[k], pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC"))
        for k, (lo, hi) in sorted(span.items())
    ]
    bars = by_symbol(pd.concat(frames, ignore_index=True))
    rows = [
        measure(bars.get(str(sym), NO_BARS), d, p, n)
        for sym, d, p, n in zip(ev["symbol"], sessions, prev, nxt, strict=True)
    ]
    got = pd.DataFrame(rows, index=ev.index, columns=[*PRICES, *MEASURES])
    out = pd.concat([ev, got], axis=1)
    opens = np.array([times(d)[0] for d in sessions], dtype=np.int64)
    out["hours_before_open"] = (opens - pn.nanos(out["first_known"])) / pd.Timedelta(hours=1).value
    return out


# ---- the build -----------------------------------------------------------------------------------


COLUMNS = [
    "symbol",
    "sidx",
    "session",
    "first_known",
    "hours_before_open",
    "n",
    "n_pre",
    "prior_60",
    "prior_recount",
    "log_adv",
    *PRICES,
    "prev_close_daily",
    *MEASURES,
]


def build(root: Path = MINUTE_STORE_DIR, out: Path = OUT) -> pd.DataFrame:
    """The per-event table, month by month; months already written are kept."""
    parts = out.parent / "catchup_parts"
    ev_path = out.parent / "catchup_events.parquet"
    if ev_path.exists():
        ev = pd.read_parquet(ev_path)
        log.info("%d events from %s", len(ev), ev_path)
    else:
        ev = events()
        write_parquet(ev, ev_path)
    months = ev["session"].map(stamp)
    done = []
    for k, grp in ev.groupby(months, sort=True):
        month = str(k)
        part = parts / f"{month}.parquet"
        if not sealed.month_allowed(month):
            log.info("%s is behind the holdout wall, skipped", month)
            continue
        if not part.exists():
            got = measure_month(grp, root)
            write_parquet(got[COLUMNS], part)
            log.info("%s: %d events, %d with gap and rest", month, len(got), rest_ok(got).sum())
        done.append(part)
    table = pd.concat([pd.read_parquet(p) for p in done], ignore_index=True)
    write_parquet(table, out)
    log.info("%d events written to %s", len(table), out)
    return table


def rest_ok(df: pd.DataFrame) -> pd.Series:
    ok: pd.Series = np.isfinite(df["gap_945"]) & np.isfinite(df["rest"])
    return ok


# ---- the summary ---------------------------------------------------------------------------------


def _f(x: Any) -> float | None:
    v = float(x)
    return v if math.isfinite(v) else None


def summary(df: pd.DataFrame) -> dict[str, Any]:
    """Events per year, gap and rest by the sign of the gap, the share done by 9:45 am, the
    median minutes to 80%, and counts of missing prices."""
    ok = rest_ok(df)
    by_sign: dict[str, Any] = {}
    for name, mask in (("up", df["gap_945"] > 0), ("down", df["gap_945"] < 0)):
        d = df[ok & mask]
        g, r = d["gap_945"], d["rest"]
        by_sign[name] = {
            "events": len(d),
            "gap_median": _f(g.median()),
            "gap_mean": _f(g.mean()),
            "rest_median": _f(r.median()),
            "rest_mean": _f(r.mean()),
            "share_done": _f(g.sum() / (g.sum() + r.sum())) if len(d) else None,
            "minutes_to_80_median": _f(d["minutes_to_80"].median()),
        }
    gap = df["prev_close"] / df["prev_close_daily"] - 1
    return {
        "events": len(df),
        "events_per_year": dict(sorted(Counter(d.year for d in df["session"]).items())),
        "with_gap_and_rest": int(ok.sum()),
        "by_sign": by_sign,
        "minutes_to_80_median": _f(df["minutes_to_80"].median()),
        "minutes_to_80_count": int(df["minutes_to_80"].notna().sum()),
        "missing": {c: int(df[c].isna().sum()) for c in (*PRICES, "minutes_to_80")},
        "prev_close_vs_daily_abs_median": _f(gap.abs().median()),
        "prev_close_vs_daily_over_5pct": int((gap.abs() > 0.05).sum()),
        "abs_gap_over_50pct": int((df["gap_945"].abs() > 0.5).sum()),
        "abs_rest_over_50pct": int((df["rest"].abs() > 0.5).sum()),
        "thin_check_disagree": int((df["prior_recount"] != df["prior_60"]).sum()),
    }


def main() -> None:
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT / "logs" / "catchup.log", encoding="utf-8"),
        ],
    )
    table = build()
    s = summary(table)
    OUT.with_name("catchup_summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    log.info("summary: %s", json.dumps(s, indent=1))
