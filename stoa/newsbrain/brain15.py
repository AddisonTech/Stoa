"""Trial 15's new brain: what the news means, the decade's memory of each kind of event,
ripples to partners and the pre-market hunt, on top of trial 9's 13 inputs.

Every input for a session is known by 9:00 am that session. The event memory for a session
reads only rows whose label was known before that 9:00 am cutoff: a row's 1 session label
ends at the next open, so a session's memory stops two sessions back.

Inputs added to trial 9's 13 (``NEW``):

- ``ev_dir``: the day's news direction for the company, the reader's p(good) - p(bad)
  averaged over its items, weighted by how sure the reader was and how big it said;
- ``ev_size``, ``ev_conf``: the main event's size (1 to 3) and the reader's probability for
  its type; ``ev_count``: how many items were a real event (not "other");
- ``mem_edge``, ``mem_hit``, ``mem_n``: the event memory. For the main event's type and
  sign, the company's size (dollar volume third within the session) and sector (SIC
  division), the mean next session return against SPY after costs and the share that beat
  SPY, over every earlier session, shrunk toward the type and sign's own mean by
  n / (n + 50), and that toward the overall mean the same way; ``mem_n`` is log(1 + n);
- ``rip_cust``, ``rip_supp``, ``rip_mem``: ripples (``stoa/newsbrain/ripplemap.py``). The
  news direction at the company's customers and at its suppliers since the session before,
  and how companies in that position historically moved the next session after that kind
  of news at a partner;
- ``pm_flag``, ``pm_hours``, ``pm_done``: the pre-market hunter. News known between the
  4 pm close before and 9:00 am on a small (bottom third by dollar volume), thinly covered
  (at most 2 articles in the 60 sessions before) company; how many hours before the open it
  landed; and, for past such events of the same type, the share of the move to the next
  9:45 am already done by 9:45 am (``stoa/newsbrain/catchup.py``).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa.newsbrain import eventread, sealed
from stoa.newsbrain import panel as pn
from stoa.newsbrain.alpaca_pull import NEWS_DIR

log = logging.getLogger(__name__)

SHRINK = 50.0  # n / (n + 50): a cell needs about 50 past rows to count as much as its parent
LAG = 2  # a session's memory reads rows at least 2 sessions back (labels end at the next open)
DIR_CUT = 0.2  # |direction| above this is a signed event
EVENT_FEATURES = ("ev_dir", "ev_size", "ev_conf", "ev_count")
MEMORY_FEATURES = ("mem_edge", "mem_hit", "mem_n")
RIPPLE_FEATURES = ("rip_cust", "rip_supp", "rip_mem")
PM_FEATURES = ("pm_flag", "pm_hours", "pm_done")
NEW = (*EVENT_FEATURES, *MEMORY_FEATURES, *RIPPLE_FEATURES, *PM_FEATURES)
FEATURES = (*pn.FEATURES, *NEW)
OTHER = eventread.CODES.index("other")
SIC_PATH = Path("data/events/sic.json")


def division(sic: str | None) -> int:
    """The SIC division (0 to 9), a broad sector: 0 unknown."""
    if not sic or not str(sic)[:2].isdigit():
        return 0
    major = int(str(sic)[:2])
    for div, (lo, hi) in enumerate(
        ((1, 9), (10, 14), (15, 17), (20, 39), (40, 49), (50, 51), (52, 59), (60, 67), (70, 89)),
        start=1,
    ):
        if lo <= major <= hi:
            return div
    return 0


def sectors(path: Path = SIC_PATH) -> dict[str, int]:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return {k: division(v) for k, v in raw.items()}


# ---- the reader's labels on the session calendar -------------------------------------------


def article_times(cut: np.ndarray, news_dir: Path = NEWS_DIR) -> pd.DataFrame:
    """id, symbol, known_at, sidx for every (article, tagged company) pair the reader reads,
    on months the wall allows."""
    parts = []
    for path in sorted(news_dir.glob("*.parquet")):
        if not sealed.month_allowed(path.stem):
            continue
        df = pd.read_parquet(path, columns=["id", "created_at", "updated_at", "symbols"])
        df = df[df["symbols"].map(len).between(1, eventread.MAX_TAGS)]
        df["known_at"] = df[["created_at", "updated_at"]].max(axis=1)
        parts.append(df[["id", "symbols", "known_at"]])
    news = pd.concat(parts, ignore_index=True).explode("symbols")
    news = news.rename(columns={"symbols": "symbol"})
    news["symbol"] = news["symbol"].astype(str)
    news["sidx"] = pn.to_session(news["known_at"], cut)
    return news[news["sidx"] < len(cut)].reset_index(drop=True)


def day_events(labels: pd.DataFrame, times: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, sidx): the day's news direction and its main event."""
    df = labels.merge(times, on=["id", "symbol"], how="inner")
    if df.empty:
        return pd.DataFrame(
            columns=["symbol", "sidx", *EVENT_FEATURES, "ev_type", "ev_sign", "first_known"]
        )
    df["code"] = df["event"].cat.codes.astype(np.int16)
    df["w"] = (df["p_event"] * df["size"]).astype(float)
    df["wd"] = df["w"] * df["direction"].astype(float)
    df["real"] = df["code"] != OTHER
    # The main event: the surest, biggest real event of the day; "other" when there is none.
    df["rank"] = df["w"] + df["real"].astype(float) * 10.0
    main = df.sort_values(["symbol", "sidx", "rank"], ascending=[True, True, False])
    main = main.drop_duplicates(["symbol", "sidx"])
    g = df.groupby(["symbol", "sidx"])
    out = pd.DataFrame(
        {
            "ev_dir": g["wd"].sum() / g["w"].sum().clip(lower=1e-9),
            "ev_count": g["real"].sum().astype(float),
            "first_known": g["known_at"].min(),
        }
    ).reset_index()
    m = main[["symbol", "sidx", "code", "size", "p_event", "direction"]].rename(
        columns={"code": "ev_type", "size": "ev_size", "p_event": "ev_conf"}
    )
    out = out.merge(m, on=["symbol", "sidx"], how="left")
    d = out.pop("direction").astype(float)
    out["ev_sign"] = np.where(d > DIR_CUT, 1, np.where(d < -DIR_CUT, -1, 0)).astype(np.int8)
    return out


# ---- as-of statistics -----------------------------------------------------------------------


def asof(hist: pd.DataFrame, query: pd.DataFrame, key: str, lag: int = LAG) -> pd.DataFrame:
    """For each query row (key, sidx): the sum and count of ``x`` and ``hit`` over history
    rows with the same key and sidx at most query sidx - lag. Columns n, s, h, in query order.
    """
    agg = (
        hist.groupby([key, "sidx"])
        .agg(s=("x", "sum"), h=("hit", "sum"), n=("x", "count"))
        .reset_index()
        .sort_values([key, "sidx"])
    )
    agg[["s", "h", "n"]] = agg.groupby(key)[["s", "h", "n"]].cumsum()
    q = query[[key, "sidx"]].copy()
    q["pos"] = np.arange(len(q))
    q["at"] = q["sidx"] - lag
    q = q.sort_values("at")
    right = agg.rename(columns={"sidx": "at"}).sort_values("at")
    m = pd.merge_asof(q, right, on="at", by=key, direction="backward")
    m = m.sort_values("pos")
    return m[["n", "s", "h"]].fillna(0.0).reset_index(drop=True)


def shrunk(n: np.ndarray, mean: np.ndarray, parent: np.ndarray, k: float = SHRINK) -> np.ndarray:
    """n / (n + k) of the cell's own mean, the rest the parent's."""
    w = n / (n + k)
    own = np.where(n > 0, mean, 0.0)
    return np.asarray(w * own + (1 - w) * parent, dtype=float)


def memory_key(d: pd.DataFrame, level: str) -> pd.Series:
    """The memory's cells: everything, the event type and sign, or type, sign, size third
    and sector."""
    if level == "all":
        return pd.Series(0, index=d.index)
    kind = d["ev_type"].astype(int) * 3 + (d["ev_sign"].astype(int) + 1)
    if level == "type":
        return kind
    return kind * 100 + d["third"].astype(int) * 10 + d["div"].astype(int)


def memory(rows: pd.DataFrame, hist: pd.DataFrame | None = None) -> pd.DataFrame:
    """The event memory for ``rows`` (symbol, sidx, ev_type, ev_sign, size third, division),
    from ``hist`` (the same keys with x = excess_1 and hit = y_1, labelled rows only; ``rows``
    itself when None). Columns mem_edge, mem_hit, mem_n in ``rows`` order."""
    if hist is None:
        hist = rows[rows["excess_1"].notna()].assign(
            x=lambda d: d["excess_1"].astype(float), hit=lambda d: d["y_1"].astype(float)
        )
    stats = {}
    for name in ("all", "type", "cell"):
        h = hist.assign(k=memory_key(hist, name))
        q = rows.assign(k=memory_key(rows, name))
        stats[name] = asof(h[["k", "sidx", "x", "hit"]], q[["k", "sidx"]], "k")
    a, t, c = stats["all"], stats["type"], stats["cell"]

    def mean(st: pd.DataFrame, col: str) -> np.ndarray:
        n = st["n"].to_numpy(dtype=float)
        return np.where(n > 0, st[col].to_numpy(dtype=float) / np.maximum(n, 1), np.nan)

    out = {}
    for col, name in (("s", "mem_edge"), ("h", "mem_hit")):
        top = np.nan_to_num(mean(a, col))
        mid = shrunk(t["n"].to_numpy(dtype=float), mean(t, col), top)
        out[name] = shrunk(c["n"].to_numpy(dtype=float), mean(c, col), mid)
    out["mem_n"] = np.log1p(c["n"].to_numpy(dtype=float))
    res = pd.DataFrame(out, index=rows.index)
    no_event = rows["ev_type"].isna().to_numpy()
    res.loc[no_event, list(MEMORY_FEATURES)] = math.nan
    return res


def thirds(rows: pd.DataFrame) -> pd.Series:
    """The company's size third within its session by dollar volume: 0 small, 2 large."""
    pct = rows.groupby("sidx")["log_adv"].rank(pct=True)
    return pd.Series(np.minimum((pct.fillna(0.5) * 3).astype(int), 2), index=rows.index)


def with_events(rows: pd.DataFrame, events: pd.DataFrame, sic: dict[str, int]) -> pd.DataFrame:
    """``rows`` (panel rows: symbol, sidx, log_adv, excess_1, y_1, ...) with the event inputs
    and the event memory."""
    out = rows.merge(
        events[["symbol", "sidx", *EVENT_FEATURES, "ev_type", "ev_sign", "first_known"]],
        on=["symbol", "sidx"],
        how="left",
    )
    out["third"] = thirds(out)
    out["div"] = out["symbol"].map(sic).fillna(0).astype(int)
    has = out["ev_type"].notna()
    mem = memory(out[has].assign(ev_type=out.loc[has, "ev_type"].astype(int)))
    for col in MEMORY_FEATURES:
        out[col] = math.nan
        out.loc[has, col] = mem[col].to_numpy()
    return out


# ---- training with recent rows weighted more -------------------------------------------------

SESSIONS_A_YEAR = 252


def recency(sidx: np.ndarray, end: int, half_life_years: float = 2.0) -> np.ndarray:
    """0.5 ^ (age in years / half life): a row from two years before ``end`` counts half."""
    age = np.maximum(end - sidx, 0) / SESSIONS_A_YEAR
    return np.asarray(0.5 ** (age / half_life_years), dtype=float)


class Brain:
    """The classifier trial 9 uses, on ``features``, fit with recency weights. Ranks only:
    its conviction is the classifier's probability."""

    def __init__(self, features: tuple[str, ...] = FEATURES) -> None:
        from stoa.newsbrain import trial as t9

        self.features = features
        self.clf = t9.classifier()
        self.rows = 0
        self.last_sidx = -1

    def fit(self, train: pd.DataFrame, half_life_years: float = 2.0) -> Brain:
        end = int(train["sidx"].max())
        w = recency(train["sidx"].to_numpy(), end, half_life_years)
        x = train[list(self.features)].to_numpy(dtype=float)
        self.clf.fit(x, train["y_1"].to_numpy(dtype=float), sample_weight=w)
        self.rows, self.last_sidx = len(train), end
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        x = rows[list(self.features)].to_numpy(dtype=float)
        return np.asarray(self.clf.predict_proba(x)[:, 1], dtype=float)


def eligible(rows: pd.DataFrame, companies: set[str]) -> pd.Series:
    """Rows a trial 15 pick could come from: liquid companies with a known label."""
    liquid = rows["liquid"].fillna(False).astype(bool) if "liquid" in rows else True
    return liquid & rows["symbol"].isin(companies) & rows["excess_1"].notna()


def top5(rows: pd.DataFrame, score: np.ndarray, n: int = 5) -> pd.DataFrame:
    """Each session's top ``n`` by score, ties by symbol."""
    df = rows[["sidx", "symbol", "excess_1", "y_1"]].assign(score=score)
    df = df.sort_values(["sidx", "score", "symbol"], ascending=[True, False, True], kind="stable")
    return df.groupby("sidx").head(n).reset_index(drop=True)


def grade(picks: pd.DataFrame) -> dict[str, Any]:
    """Hit rate and average edge of the picks, and the share of days whose picks together
    beat SPY after costs."""
    if picks.empty:
        return {"picks": 0, "days": 0, "hit_rate": None, "avg_edge": None, "day_hit_rate": None}
    day = picks.groupby("sidx")["excess_1"].mean()
    return {
        "picks": len(picks),
        "days": int(day.size),
        "hit_rate": float((picks["excess_1"] > 0).mean()),
        "avg_edge": float(picks["excess_1"].mean()),
        "day_hit_rate": float((day > 0).mean()),
    }


def walk_forward(
    rows: pd.DataFrame,
    days: list[Any],
    first: Any,
    last: Any,
    features: tuple[str, ...] = FEATURES,
    slice_len: int = 63,
    train_from: Any = None,
) -> pd.Series:
    """Out-of-sample conviction for every row from ``first`` to ``last``: refit every
    ``slice_len`` sessions on every row labelled before the slice (from ``train_from``),
    recent rows weighted more."""
    from stoa.newsbrain import trial as t9

    pos = {d: i for i, d in enumerate(days)}
    lo = min(i for d, i in pos.items() if d >= first)
    hi = max(i for d, i in pos.items() if d <= last)
    start = min(i for d, i in pos.items() if d >= (train_from or t9.TRAIN_START))
    out = pd.Series(np.nan, index=rows.index)
    for t0 in range(lo, hi + 1, slice_len):
        t1 = min(t0 + slice_len - 1, hi)
        train = t9.labelled(rows[rows["sidx"] >= start], 1, t0 - 1)
        test = rows[(rows["sidx"] >= t0) & (rows["sidx"] <= t1)]
        if test.empty:
            continue
        brain = Brain(features).fit(train)
        out.loc[test.index] = brain.predict(test)
        log.info("walk-forward slice %s to %s: %d train rows", days[t0], days[t1], len(train))
    return out
