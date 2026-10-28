"""The news brain's rolling state: trial 9's 13 features one session at a time.

The full rebuild (``panel.build`` from 2015) needs the Dell's 2 GB of history and about
16 minutes. This state holds only what the 13 features read from that history, and a
daily update adds the newest session. What each feature reads, and how far back:

- sent, first_break: the session's articles that tag 1 to 3 stocks (FinBERT tone, weight
  1 for a first break, 0.25 for a rewrite). A rewrite is a headline sharing half its
  words with one of the same stock's headlines of the 3 days before (by publish time),
  so the call needs those headlines too, and an article updated late can change the
  call for articles published after it.
- surprise: the stock's tone against its last 60 days with news (at least 5). For a quiet
  stock those 60 days reach back years, and pandas' running mean and deviation carry
  every earlier value, so the last digits depend on the whole history.
- quiet_spike, coverage: article counts over the previous 60 sessions.
- ripple: the tone times first-break share of larger linked companies over the previous
  3 sessions. Links come from 10-K customer paragraphs (a 10-K naming a company in the
  top 1,500 by dollar volume that month), good for 315 sessions or until the company's
  next 10-K; size is the 20 session dollar volume at the close before.
- filing_tone, guidance, beat_miss: the newest earnings 8-K (item 2.02) accepted in the
  last 5 sessions: FinBERT tone of the press release, raised or cut guidance, beat or
  miss, all read from the filing text.
- ret_1, ret_5, vol_20, log_adv: the daily bars to the close before, 1 and 5 session
  returns, the 20 session deviation of returns and the 20 session average dollar
  volume. A stock needs at least 30 closes since 2015 and today's liquidity ($1 price,
  20 session dollar volume) for a row at all. The deviation and the average are pandas
  running sums carried from 2015.

So the state keeps:

- ``news/``: every scored article (headline, times, tone, stocks), a seed file and one
  file a session. Trial 9's own news code (``panel.daily_news``) runs on it each
  morning, about a minute, so a late update is handled exactly as the rebuild would.
- ``filings.parquet``: each earnings 8-K's tone, guidance and beat or miss, computed
  once when it arrives, kept 10 sessions.
- ``links.parquet``, ``tenk.parquet``: the supplier links and each company's latest 10-K.
- ``bars.npz``: per stock, the last 21 closes and 20 volumes and pandas' own running
  sums for the deviation and the average (``RollVar``, ``RollMean``, ported from pandas
  3.1 and checked bit for bit). A split or dividend changes a stock's past prices; the
  feed then rebuilds that stock's sums from its full history.
- ``static/``: the CIK map, company names, the top 1,500 by month and the stock list,
  copied at the seed. They are fixed inputs of the rebuild too.

``rows`` gives the same rows as ``panel.build`` for the session, exactly (proved by
``check``: every row and all 13 features bit for bit against the full rebuild).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa import clock, wide
from stoa.newsbrain import ROOT, panel, sealed
from stoa.newsbrain.pace import write_parquet

log = logging.getLogger(__name__)

STATE_DIR = ROOT / "rolling"
VERSION = 1
CLOSES = 21  # closes kept: 20 returns, and the close 5 sessions back
WINDOW = 20  # the deviation and the average dollar volume
VOL_MIN = 15  # returns needed for the deviation
MIN_CLOSES = 30  # closes since 2015 for a stock to have rows at all
FILING_KEEP = 10  # sessions of filings kept (5 are read)
INV_COND_TOL = float(np.finfo(np.float64).eps) * 1e3  # pandas' cancellation test
ROW_KEYS = ["symbol", "sidx", *panel.FEATURES]


# ---- pandas' running sums ----------------------------------------------------------------


@dataclass
class RollVar:
    """pandas 3.1 ``roll_var`` for a fixed window, one step a session, for every stock at once.

    Welford's method with Kahan compensation; when a step loses too many digits the
    window is summed again from scratch, as pandas does. Missing values are skipped.
    """

    n: int
    window: int = WINDOW
    minp: int = VOL_MIN
    nobs: np.ndarray = field(init=False)
    mean: np.ndarray = field(init=False)
    ssq: np.ndarray = field(init=False)
    ca: np.ndarray = field(init=False)
    cr: np.ndarray = field(init=False)
    unstable: np.ndarray = field(init=False)
    buf: np.ndarray = field(init=False)  # the window, oldest first, NaN where empty
    steps: int = 0

    def __post_init__(self) -> None:
        self.nobs, self.mean, self.ssq = (np.zeros(self.n) for _ in range(3))
        self.ca, self.cr = np.zeros(self.n), np.zeros(self.n)
        self.unstable = np.zeros(self.n, dtype=bool)
        self.buf = np.full((self.n, self.window), np.nan)

    def _add(self, v: np.ndarray, rows: np.ndarray) -> None:
        ok = rows & ~np.isnan(v)
        if not ok.any():
            return
        prev_m2 = self.ssq[ok]
        val = v[ok]
        nobs = self.nobs[ok] + 1
        mean, ca = self.mean[ok], self.ca[ok]
        prev_mean = mean - ca
        y = val - ca
        t = y - mean
        ca = t + mean - y
        mean = np.where(nobs != 0, mean + t / np.where(nobs != 0, nobs, 1.0), 0.0)
        ssq = prev_m2 + (val - prev_mean) * (val - mean)
        self.nobs[ok], self.mean[ok], self.ca[ok], self.ssq[ok] = nobs, mean, ca, ssq
        self.unstable[ok] |= prev_m2 * INV_COND_TOL > ssq

    def _remove(self, v: np.ndarray, rows: np.ndarray) -> None:
        ok = rows & ~np.isnan(v)
        if not ok.any():
            return
        prev_m2 = self.ssq[ok]
        val = v[ok]
        nobs = self.nobs[ok] - 1
        mean, cr = self.mean[ok], self.cr[ok]
        live = nobs != 0
        prev_mean = mean - cr
        y = val - cr
        t = y - mean
        new_cr = t + mean - y
        new_mean = mean - t / np.where(live, nobs, 1.0)
        new_ssq = prev_m2 - (val - prev_mean) * (val - new_mean)
        self.nobs[ok] = nobs
        self.cr[ok] = np.where(live, new_cr, cr)
        self.mean[ok] = np.where(live, new_mean, 0.0)
        self.ssq[ok] = np.where(live, new_ssq, 0.0)
        self.unstable[ok] = np.where(
            live, self.unstable[ok] | (prev_m2 * INV_COND_TOL > new_ssq), False
        )

    def push(self, v: np.ndarray) -> None:
        everyone = np.ones(self.n, dtype=bool)
        leaving = self.buf[:, 0].copy()
        full = self.steps >= self.window
        self.buf = np.concatenate([self.buf[:, 1:], v[:, None]], axis=1)
        if self.steps > 0:
            if full:
                self._remove(leaving, everyone)
            self._add(v, everyone)
            redo = self.unstable.copy()
        else:
            redo = everyone
        if redo.any():
            self.nobs[redo] = self.mean[redo] = self.ssq[redo] = 0.0
            self.ca[redo] = self.cr[redo] = 0.0
            for j in range(self.window):
                self._add(self.buf[:, j], redo)
            self.unstable[redo] = False
        self.steps += 1

    def std(self) -> np.ndarray:
        ok = (self.nobs >= max(self.minp, 1)) & (self.nobs > 1)
        var = np.where(ok, self.ssq / np.where(ok, self.nobs - 1, 1.0), np.nan)
        with np.errstate(invalid="ignore"):
            out = np.sqrt(var)
        out[var < 0] = 0.0
        return out


@dataclass
class RollMean:
    """pandas 3.1 ``roll_mean`` for a fixed window: Kahan sums, one step a session."""

    n: int
    window: int = WINDOW
    minp: int = WINDOW
    nobs: np.ndarray = field(init=False)
    neg: np.ndarray = field(init=False)
    sum: np.ndarray = field(init=False)
    ca: np.ndarray = field(init=False)
    cr: np.ndarray = field(init=False)
    same: np.ndarray = field(init=False)  # consecutive equal values added
    prev: np.ndarray = field(init=False)
    buf: np.ndarray = field(init=False)
    steps: int = 0

    def __post_init__(self) -> None:
        self.nobs, self.neg, self.same = (np.zeros(self.n, dtype=np.int64) for _ in range(3))
        self.sum, self.ca, self.cr = (np.zeros(self.n) for _ in range(3))
        self.prev = np.full(self.n, np.nan)
        self.buf = np.full((self.n, self.window), np.nan)

    def push(self, v: np.ndarray) -> None:
        if self.steps == 0:
            self.prev = v.copy()
            self.same[:] = 0
        if self.steps >= self.window:
            out = self.buf[:, 0]
            ok = ~np.isnan(out)
            y = -out[ok] - self.cr[ok]
            t = self.sum[ok] + y
            self.cr[ok] = t - self.sum[ok] - y
            self.sum[ok] = t
            self.nobs[ok] -= 1
            self.neg[ok] -= np.signbit(out[ok]).astype(np.int64)
        self.buf = np.concatenate([self.buf[:, 1:], v[:, None]], axis=1)
        ok = ~np.isnan(v)
        y = v[ok] - self.ca[ok]
        t = self.sum[ok] + y
        self.ca[ok] = t - self.sum[ok] - y
        self.sum[ok] = t
        self.nobs[ok] += 1
        self.neg[ok] += np.signbit(v[ok]).astype(np.int64)
        self.same[ok] = np.where(v[ok] == self.prev[ok], self.same[ok] + 1, 1)
        self.prev[ok] = v[ok]
        self.steps += 1

    def value(self) -> np.ndarray:
        ok = (self.nobs >= self.minp) & (self.nobs > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = self.sum / np.where(ok, self.nobs, 1)
        r = np.where(self.same >= self.nobs, self.prev, r)
        r = np.where((self.same < self.nobs) & (self.neg == 0) & (r < 0), 0.0, r)
        r = np.where((self.same < self.nobs) & (self.neg == self.nobs) & (r > 0), 0.0, r)
        return np.where(ok, r, np.nan)


@dataclass
class BarState:
    """Per stock: the closes and volumes the bar features read, and the running sums."""

    symbols: list[str]
    close: np.ndarray  # (n, CLOSES) raw closes, NaN on a session with no bar
    volume: np.ndarray  # (n, WINDOW)
    filled: np.ndarray  # (n, CLOSES) closes carried forward over gaps
    closes: np.ndarray  # closes since 2015
    var: RollVar
    mean: RollMean
    pushed: int = 0  # sessions pushed, from the panel's first

    @classmethod
    def empty(cls, symbols: list[str]) -> BarState:
        n = len(symbols)
        return cls(
            symbols=list(symbols),
            close=np.full((n, CLOSES), np.nan),
            volume=np.full((n, WINDOW), np.nan),
            filled=np.full((n, CLOSES), np.nan),
            closes=np.zeros(n, dtype=np.int64),
            var=RollVar(n),
            mean=RollMean(n),
        )

    def push(self, close: np.ndarray, volume: np.ndarray) -> None:
        """One session's bars, NaN where a stock has none (``panel.bar_features``' inputs)."""
        prev = self.filled[:, -1]
        filled = np.where(np.isnan(close), prev, close)
        with np.errstate(invalid="ignore", divide="ignore"):
            ret = filled / prev - 1  # pct_change on the carried closes
            dv = close * volume
        self.var.push(ret)
        self.mean.push(dv)
        self.close = np.concatenate([self.close[:, 1:], close[:, None]], axis=1)
        self.volume = np.concatenate([self.volume[:, 1:], volume[:, None]], axis=1)
        self.filled = np.concatenate([self.filled[:, 1:], filled[:, None]], axis=1)
        self.closes += ~np.isnan(close)
        self.pushed += 1

    def features(self) -> pd.DataFrame:
        """``panel.bar_features`` for the session after the last one pushed."""
        with np.errstate(invalid="ignore", divide="ignore"):
            ret_1 = self.filled[:, -1] / self.filled[:, -2] - 1
            ret_5 = self.filled[:, -1] / self.filled[:, -6] - 1
        adv = self.mean.value()
        out = pd.DataFrame(
            {
                "symbol": self.symbols,
                "ret_1": ret_1,
                "ret_5": ret_5,
                "vol_20": self.var.std(),
                "adv": adv,
                "prev_close": self.close[:, -1],
                "closes": self.closes,
            }
        )
        out["log_adv"] = np.log(out["adv"].clip(lower=1.0))
        out["liquid"] = (out["prev_close"] >= wide.MIN_PRICE) & (
            out["adv"] >= wide.MIN_DOLLAR_VOLUME
        )
        return out

    def replace(self, i: int, other: BarState, j: int) -> None:
        """Stock ``i`` takes stock ``j``'s state from ``other`` (a rebuild from full history)."""
        for name in ("close", "volume", "filled", "closes"):
            getattr(self, name)[i] = getattr(other, name)[j]
        for acc, src in ((self.var, other.var), (self.mean, other.mean)):
            for k in vars(acc):
                arr = getattr(acc, k)
                if isinstance(arr, np.ndarray):
                    arr[i] = getattr(src, k)[j]

    def save(self, path: Path) -> None:
        arrays: dict[str, np.ndarray] = {
            "symbols": np.array(self.symbols, dtype=object),
            "close": self.close,
            "volume": self.volume,
            "filled": self.filled,
            "closes": self.closes,
            "meta": np.array([self.pushed, self.var.steps, self.mean.steps]),
        }
        for prefix, acc in (("var_", self.var), ("mean_", self.mean)):
            for k, v in vars(acc).items():
                if isinstance(v, np.ndarray):
                    arrays[prefix + k] = v
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)  # type: ignore[arg-type]
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> BarState:
        z = np.load(path, allow_pickle=True)
        symbols = [str(s) for s in z["symbols"]]
        st = cls.empty(symbols)
        st.close, st.volume, st.filled, st.closes = (
            z["close"],
            z["volume"],
            z["filled"],
            z["closes"],
        )
        st.pushed, st.var.steps, st.mean.steps = (int(x) for x in z["meta"])
        for prefix, acc in (("var_", st.var), ("mean_", st.mean)):
            for k, v in vars(acc).items():
                if isinstance(v, np.ndarray):
                    setattr(acc, k, z[prefix + k])
        return st


def bars_from_history(symbols: list[str], close: np.ndarray, volume: np.ndarray) -> BarState:
    """A bar state from full histories: sessions in columns, from the panel's first."""
    st = BarState.empty(symbols)
    for t in range(close.shape[1]):
        st.push(close[:, t], volume[:, t])
    return st


# ---- filings and links -----------------------------------------------------------------


def filing_row(f: pd.DataFrame) -> dict[str, Any] | None:
    """One earnings 8-K's features, exactly as ``panel.filing_features`` computes them."""
    ex = f[(f["kind"] == "ex99") & (f["text"].str.len() > 0)]
    use = ex if len(ex) else f[(f["kind"] == "main") & (f["text"].str.len() > 0)]
    if use.empty or pd.isna(f["accepted"].iloc[0]):
        return None
    text = " ".join(use["text"].tolist())[:40_000]
    return {
        "cik": int(f["cik"].iloc[0]),
        "accepted": f["accepted"].iloc[0],
        "filing_tone": float((use["positive"] - use["negative"]).mean()),
        "guidance": panel.direction(text, panel._RAISED, panel._CUT),
        "beat_miss": panel.direction(text, panel._BEAT, panel._MISS),
    }


def file_key(cik: int) -> str:
    """The rebuild reads a company's SEC file in file-name order, as text: its sort key."""
    return f"{cik}.parquet"


FILING_COLS = ["cik", "accn", "accepted", "filing_tone", "guidance", "beat_miss", "fkey", "j"]
TENK_COLS = ["cik", "accn", "accepted", "fkey", "j", "others"]
LINK_COLS = ["cik", "other", "start", "end", "fkey", "j", "order"]


@dataclass
class Static:
    """Fixed inputs of the rebuild, copied at the seed."""

    ciks: pd.DataFrame  # cik, symbol
    names: dict[int, str]  # cik to normalized company name
    top: dict[str, list[int]]  # month to the top 1,500 companies by dollar volume
    universe: list[str]  # every stock with daily bars (the news pull keeps these)

    def save(self, root: Path) -> None:
        d = root / "static"
        d.mkdir(parents=True, exist_ok=True)
        write_parquet(self.ciks, d / "ciks.parquet")
        (d / "names.json").write_text(json.dumps(self.names), encoding="utf-8")
        (d / "top.json").write_text(json.dumps(self.top), encoding="utf-8")
        (d / "universe.json").write_text(json.dumps(self.universe), encoding="utf-8")

    @classmethod
    def load(cls, root: Path) -> Static:
        d = root / "static"
        return cls(
            ciks=pd.read_parquet(d / "ciks.parquet"),
            names={int(k): v for k, v in json.loads((d / "names.json").read_text()).items()},
            top=json.loads((d / "top.json").read_text()),
            universe=json.loads((d / "universe.json").read_text()),
        )

    @classmethod
    def from_dell(cls) -> Static:
        from stoa.newsbrain.alpaca_pull import PLAN, universe

        ciks = panel.load_ciks()
        plan = pd.read_parquet(PLAN, columns=["symbol", "month", "dollars"])
        sym_cik = dict(zip(ciks["symbol"], ciks["cik"], strict=True))
        top: dict[str, list[int]] = {}
        for month, g in plan.groupby("month"):
            names = g.nlargest(panel.TOP_NAMES, "dollars")["symbol"]
            top[str(month)] = sorted({int(sym_cik[s]) for s in names if s in sym_cik})
        return cls(ciks=ciks, names=panel.name_book(ciks), top=top, universe=universe())


def tenk_links(
    cik: int, accepted: pd.Timestamp, text: str, static: Static, by_name: dict[str, int]
) -> list[int]:
    """The companies a 10-K's customer paragraphs name, as ``panel.links`` finds them."""
    longest = max((len(n.split()) for n in by_name), default=1)
    allowed = set(static.top.get(pd.Timestamp(accepted).strftime("%Y-%m"), []))
    return [
        o
        for o in panel.mentioned(panel.norm_name(str(text)), by_name, longest)
        if o != cik and o in allowed
    ]


def name_index(static: Static) -> dict[str, int]:
    by_name: dict[str, int] = {}
    for c, n in static.names.items():
        by_name.setdefault(n, c)
    return by_name


# ---- the state -----------------------------------------------------------------------------


NEWS_COLS = ["id", "pos", "created_at", "known_at", "symbols", "headline", "net", "keep"]


@dataclass
class State:
    """Everything known by the 9:00 am cutoff of ``session``."""

    root: Path
    session: date
    static: Static
    bars: BarState
    filings: pd.DataFrame  # cik, accepted, filing_tone, guidance, beat_miss, pos
    tenk: pd.DataFrame  # per company: start of its latest 10-K, pos
    links: pd.DataFrame  # cik, other, start, end, pos
    news_files: list[str] = field(default_factory=list)

    # ---- storage

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.static.save(self.root)
        self.bars.save(self.root / "bars.npz")
        write_parquet(self.filings, self.root / "filings.parquet")
        write_parquet(self.tenk, self.root / "tenk.parquet")
        write_parquet(self.links, self.root / "links.parquet")
        meta = {
            "version": VERSION,
            "session": self.session.isoformat(),
            "pandas": pd.__version__,
            "news_files": self.news_files,
        }
        (self.root / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, root: Path = STATE_DIR) -> State:
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        if meta["version"] != VERSION:
            raise ValueError(f"rolling state version {meta['version']}, this code reads {VERSION}")
        return cls(
            root=root,
            session=date.fromisoformat(meta["session"]),
            static=Static.load(root),
            bars=BarState.load(root / "bars.npz"),
            filings=pd.read_parquet(root / "filings.parquet"),
            tenk=pd.read_parquet(root / "tenk.parquet"),
            links=pd.read_parquet(root / "links.parquet"),
            news_files=list(meta["news_files"]),
        )

    def add_news(self, name: str, articles: pd.DataFrame) -> None:
        """One file of articles: id, pos, created_at, known_at, symbols, headline, net, keep."""
        d = self.root / "news"
        d.mkdir(parents=True, exist_ok=True)
        write_parquet(articles[NEWS_COLS].reset_index(drop=True), d / f"{name}.parquet")
        if name not in self.news_files:
            self.news_files.append(name)

    def archive(self) -> pd.DataFrame:
        """Every article's latest version, in the rebuild's reading order."""
        parts = [pd.read_parquet(self.root / "news" / f"{n}.parquet") for n in self.news_files]
        a = pd.concat(parts, ignore_index=True).drop_duplicates("id", keep="last")
        a = a[a["keep"]]
        return a.sort_values("pos", kind="stable").reset_index(drop=True)

    # ---- the daily update

    def update(
        self,
        session: date,
        *,
        news: pd.DataFrame,
        filings: pd.DataFrame,
        tenks: pd.DataFrame,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> None:
        """Add one session: its news to 9:00 am, filings and 10-Ks accepted by then, and the
        bars of the close before (aligned with ``bars.symbols``, NaN where a stock has none).

        ``news``: ``NEWS_COLS``. ``filings``: ``FILING_COLS`` (``filing_row``). ``tenks``:
        ``TENK_COLS`` (``tenk_links``), in accepted order.
        """
        if session != clock.next_session(self.session):
            raise ValueError(f"the state is at {self.session}; {session} is not the next session")
        days, cut = panel.calendar(panel.PANEL_START, session)
        sidx = len(days) - 1
        if self.bars.pushed != sidx - 1:
            raise ValueError(f"bars pushed through {self.bars.pushed} sessions, need {sidx - 1}")
        self.bars.push(close, volume)
        self.add_news(session.isoformat(), news)
        keep = pd.concat([self.filings, filings[FILING_COLS]], ignore_index=True)
        if len(keep):
            keep = keep[panel.to_session(keep["accepted"], cut) >= sidx - FILING_KEEP]
        self.filings = keep.reset_index(drop=True)
        if len(tenks):
            starts = panel.to_session(tenks["accepted"], cut)
            for t, start in zip(tenks.to_dict("records"), starts, strict=True):
                self.add_tenk(int(t["cik"]), int(start), str(t["fkey"]), int(t["j"]), t["others"])
        self.links = self.links[self.links["end"] > sidx].reset_index(drop=True)
        self.session = session

    def add_tenk(self, cik: int, start: int, fkey: str, j: int, others: Any) -> None:
        """A new 10-K: the company's earlier links end at its start; its own links begin."""
        others = [int(o) for o in others]
        prev = self.links["cik"] == cik
        self.links.loc[prev, "end"] = np.minimum(self.links.loc[prev, "end"], start)
        new = pd.DataFrame(
            {
                "cik": np.full(len(others), cik, dtype=np.int64),
                "other": np.array(others, dtype=np.int64),
                "start": np.full(len(others), start, dtype=np.int64),
                "end": np.full(len(others), start + panel.LINK_SESSIONS, dtype=np.int64),
                "fkey": [fkey] * len(others),
                "j": np.full(len(others), j, dtype=np.int64),
                "order": np.arange(len(others), dtype=np.int64),
            }
        )
        if len(new):
            self.links = pd.concat([self.links, new], ignore_index=True)
        self.tenk = pd.concat(
            [
                self.tenk[self.tenk["cik"] != cik],
                pd.DataFrame({"cik": [cik], "start": [start], "fkey": [fkey], "j": [j]}),
            ],
            ignore_index=True,
        )

    # ---- the session's rows

    def rows(self, today_bars: Iterable[str] = ()) -> pd.DataFrame:
        """``panel.build``'s rows for ``session``: keys, liquidity and the 13 features.

        ``today_bars``: stocks with a bar on the session itself. Live, at 9:00 am, there are
        none; a rebuild of a past session counts that bar toward the 30 closes a stock needs.
        """
        session = self.session
        days, cut = panel.calendar(panel.PANEL_START, session)
        sidx = len(days) - 1
        ciks = self.static.ciks
        # News: trial 9's own code on every article of the stocks with news lately.
        a = self.archive()
        news = (
            a.drop(columns=["pos", "keep"]).explode("symbols").rename(columns={"symbols": "symbol"})
        )
        news["sidx"] = panel.to_session(news["known_at"], cut)
        news = news[news["sidx"] < len(cut)]
        recent = set(news.loc[news["sidx"] >= sidx - max(panel.RIPPLE_LAGS), "symbol"])
        news = news[news["symbol"].isin(recent)]
        news = news.sort_values(["symbol", "created_at"]).reset_index(drop=True)
        day = panel.daily_news(news, len(days)) if len(news) else _empty_day()
        # Filings: the newest earnings 8-K of the last 5 sessions.
        fil = self.filing_rows(cut, sidx)
        # Links valid today, larger company to smaller.
        lk = self.links.sort_values(["fkey", "j", "order"], kind="stable")
        lk = lk[["cik", "other", "start", "end"]].drop_duplicates().reset_index(drop=True)
        bf = self.bars.features()
        counted = bf["closes"] + bf["symbol"].isin(set(today_bars)).astype(int)
        bars = bf[bf["liquid"].fillna(False) & (counted >= MIN_CLOSES)].copy()
        bars["sidx"] = sidx
        adv = bars[["symbol", "sidx", "adv"]]
        lagged = day[(day["sidx"] < sidx) & (day["sidx"] >= sidx - max(panel.RIPPLE_LAGS))]
        rip = panel.ripple(lagged, lk, ciks, adv)
        today = day[day["sidx"] == sidx]
        fil = fil[fil["sidx"] == sidx]
        rip = rip[rip["sidx"] == sidx]
        keys = pd.concat(
            [today[["symbol", "sidx"]], fil[["symbol", "sidx"]], rip[["symbol", "sidx"]]]
        ).drop_duplicates()
        rows = keys.merge(bars, on=["symbol", "sidx"], how="inner")
        rows = rows.merge(today, on=["symbol", "sidx"], how="left")
        rows = rows.merge(fil, on=["symbol", "sidx"], how="left")
        rows = rows.merge(rip, on=["symbol", "sidx"], how="left")
        for c in ("n", "sent", "surprise", "first_break", "quiet_spike", "ripple", "filing_tone"):
            rows[c] = rows[c].fillna(0.0)
        for c in ("guidance", "beat_miss"):
            rows[c] = rows[c].fillna(0).astype("int8")
        rows["coverage"] = rows["coverage"].fillna(0.0)
        rows["session"] = session
        return rows.sort_values("symbol").reset_index(drop=True)

    def filing_rows(self, cut: np.ndarray, sidx: int) -> pd.DataFrame:
        cols = ["symbol", "sidx", "filing_tone", "guidance", "beat_miss"]
        fl = self.filings.sort_values(["fkey", "j"], kind="stable")
        if fl.empty:
            return pd.DataFrame(columns=cols)
        fl = fl.assign(sidx0=panel.to_session(fl["accepted"], cut))
        fl = fl[(fl["sidx0"] <= sidx) & (fl["sidx0"] > sidx - panel.FILING_SESSIONS)]
        fl = fl.merge(self.static.ciks, on="cik")
        spread = [fl.assign(sidx=fl["sidx0"] + k, age=k) for k in range(panel.FILING_SESSIONS)]
        out = pd.concat(spread, ignore_index=True)
        out = out[out["sidx"] == sidx]
        out = out.sort_values(["symbol", "sidx", "age"]).drop_duplicates(["symbol", "sidx"])
        return out[cols]


def _empty_day() -> pd.DataFrame:
    cols = ["symbol", "sidx", "n", "sent", "surprise", "first_break", "quiet_spike", "coverage"]
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols}).astype(
        {"symbol": object, "sidx": np.int64}
    )


# ---- reading the Dell's history (the seed, and the replay that proves the match) -------


def news_source(months: Callable[[str], bool]) -> pd.DataFrame:
    """Every scored article in the news files, in the rebuild's order (``pos``)."""
    from stoa.newsbrain.alpaca_pull import NEWS_DIR
    from stoa.newsbrain.score import SCORE_DIR

    parts = []
    for k, path in enumerate(sorted(NEWS_DIR.glob("*.parquet"))):
        if not months(path.stem):
            continue
        df = pd.read_parquet(
            path, columns=["id", "created_at", "updated_at", "symbols", "headline"]
        )
        df["pos"] = np.int64(k) * 10_000_000 + np.arange(len(df), dtype=np.int64)
        sc = pd.read_parquet(SCORE_DIR / "news" / path.name)
        df = df.merge(sc[["id", "positive", "negative"]], on="id", how="inner")
        df["keep"] = df["symbols"].map(len).between(1, panel.MAX_TAGS)
        df["known_at"] = df[["created_at", "updated_at"]].max(axis=1)
        df["net"] = (df["positive"] - df["negative"]).astype("float32")
        parts.append(df[NEWS_COLS])
    return pd.concat(parts, ignore_index=True)


def filings_source(before: Callable[[pd.DataFrame, str], pd.DataFrame]) -> pd.DataFrame:
    """Every earnings 8-K's features, in the rebuild's order (``pos``)."""
    from stoa.newsbrain.score import SCORE_DIR
    from stoa.newsbrain.sec_pull import SEC_DIR

    rows = []
    for path in sorted((SEC_DIR / "earnings").glob("*.parquet")):
        scores = SCORE_DIR / "sec" / "earnings" / path.name
        if not scores.exists():
            continue
        df = pd.read_parquet(path, columns=["cik", "accn", "accepted", "kind", "doc", "text"])
        df = before(df, "accepted")
        if df.empty:
            continue
        df = df.merge(pd.read_parquet(scores), on=["accn", "kind", "doc"], how="left")
        for j, (accn, f) in enumerate(df.groupby("accn", sort=False)):
            r = filing_row(f)
            if r is not None:
                rows.append({**r, "accn": str(accn), "fkey": path.name, "j": j})
    return pd.DataFrame(rows, columns=FILING_COLS)


def tenk_source(
    static: Static, before: Callable[[pd.DataFrame, str], pd.DataFrame]
) -> pd.DataFrame:
    """Every 10-K with customer paragraphs: cik, accepted, pos, the companies it names."""
    from stoa.newsbrain.sec_pull import SEC_DIR

    by_name = name_index(static)
    rows = []
    for path in sorted((SEC_DIR / "tenk").glob("*.parquet")):
        df = pd.read_parquet(path, columns=["cik", "accn", "accepted", "text"])
        df = before(df, "accepted")
        df = df[df["text"].str.len() > 0].sort_values("accepted")
        for j, (_, r) in enumerate(df.iterrows()):
            if pd.isna(r["accepted"]):
                continue
            cik = int(r["cik"])
            rows.append(
                {
                    "cik": cik,
                    "accn": str(r["accn"]),
                    "accepted": r["accepted"],
                    "fkey": path.name,
                    "j": j,
                    "others": tenk_links(cik, r["accepted"], r["text"], static, by_name),
                }
            )
    return pd.DataFrame(rows, columns=TENK_COLS)


def bar_matrix(symbols: list[str], days: list[date]) -> tuple[np.ndarray, np.ndarray]:
    """Closes and volumes, stocks by sessions, read as ``panel.bar_frame`` reads them."""
    close = np.full((len(symbols), len(days)), np.nan)
    volume = np.full((len(symbols), len(days)), np.nan)
    for i, sym in enumerate(symbols):
        df = panel.bar_frame(sym, days)
        if df is None:
            continue
        close[i] = df["close"].to_numpy(dtype=float)
        volume[i] = df["volume"].to_numpy(dtype=float)
    return close, volume


@dataclass
class History:
    """The Dell's inputs, read once, for seeding and replaying the state."""

    static: Static
    news: pd.DataFrame
    filings: pd.DataFrame
    tenks: pd.DataFrame
    days: list[date]
    close: np.ndarray
    volume: np.ndarray

    @classmethod
    def read(cls, through: date, live: bool) -> History:
        """Everything the rebuild for a session up to ``through`` may read.

        ``live``: the sessions are after the holdout, read as ``sealed.live`` reads them;
        otherwise the holdout's months and filings are left out, as the rebuild leaves them.
        """
        static = Static.from_dell()

        def months(m: str) -> bool:
            return live or sealed.month_allowed(m)

        def before(df: pd.DataFrame, col: str) -> pd.DataFrame:
            if live:
                ts = pd.to_datetime(df[col], utc=True)
                nxt = pd.Timestamp(through.isoformat(), tz="America/New_York") + pd.Timedelta(
                    days=1
                )
                return df[ts < nxt.tz_convert("UTC")]
            return sealed.before_holdout(df, col)

        days = clock.sessions_between(panel.PANEL_START, through)
        log.info("reading news")
        news = news_source(months)
        log.info("reading filings")
        fil = filings_source(before)
        log.info("reading 10-Ks")
        tenks = tenk_source(static, before)
        log.info("reading bars")
        close, volume = bar_matrix(list(static.universe), days)
        return cls(static, news, fil, tenks, days, close, volume)

    def cutoff(self, d: date) -> int:
        return int(pd.Timestamp(clock.at_ny(d, "09:00")).value)

    def seed(self, root: Path, session: date) -> State:
        """The state as of ``session``'s 9:00 am cutoff, from the full history."""
        k = self.days.index(session)
        cut = self.cutoff(session)
        st = empty_state(root, session, self.static, self.static.universe)
        st.bars = bars_from_history(self.static.universe, self.close[:, :k], self.volume[:, :k])
        st.add_news("seed", self.news[panel.nanos(self.news["known_at"]) <= cut])
        cuts = np.array([self.cutoff(d) for d in self.days[: k + 1]], dtype=np.int64)
        fs = self.filings
        if len(fs):
            s0 = panel.to_session(fs["accepted"], cuts)
            st.filings = fs[(s0 <= k) & (s0 >= k - FILING_KEEP)].reset_index(drop=True)
        tk = self.tenks
        if len(tk):
            tk = tk.assign(start=panel.to_session(tk["accepted"], cuts))
            tk = tk[tk["start"] <= k].sort_values(["fkey", "j"], kind="stable")
            for t in tk.to_dict("records"):
                st.add_tenk(
                    int(t["cik"]), int(t["start"]), str(t["fkey"]), int(t["j"]), t["others"]
                )
            st.links = st.links[st.links["end"] > k].reset_index(drop=True)
        return st

    def day(self, session: date) -> dict[str, Any]:
        """What the daily update adds for ``session``: what arrived after the cutoff before."""
        k = self.days.index(session)
        lo, hi = self.cutoff(self.days[k - 1]), self.cutoff(session)

        def arrived(df: pd.DataFrame, col: str) -> pd.DataFrame:
            t = panel.nanos(df[col]) if len(df) else np.zeros(0, dtype=np.int64)
            return df[(t > lo) & (t <= hi)]

        return {
            "news": arrived(self.news, "known_at"),
            "filings": arrived(self.filings, "accepted"),
            "tenks": arrived(self.tenks, "accepted").sort_values(["fkey", "j"], kind="stable"),
            "close": self.close[:, k - 1],
            "volume": self.volume[:, k - 1],
        }

    def bars_on(self, session: date) -> list[str]:
        """Stocks with a bar on the session itself (the rebuild counts it toward 30 closes)."""
        k = self.days.index(session)
        if k >= self.close.shape[1]:
            return []
        have = ~np.isnan(self.close[:, k])
        return [s for s, h in zip(self.static.universe, have, strict=True) if h]


def empty_state(root: Path, session: date, static: Static, symbols: list[str]) -> State:
    return State(
        root=root,
        session=session,
        static=static,
        bars=BarState.empty(symbols),
        filings=pd.DataFrame(
            {
                "cik": pd.Series(dtype=np.int64),
                "accn": pd.Series(dtype=object),
                "accepted": pd.Series(dtype="datetime64[ns, UTC]"),
                "filing_tone": pd.Series(dtype=np.float64),
                "guidance": pd.Series(dtype=np.int64),
                "beat_miss": pd.Series(dtype=np.int64),
                "fkey": pd.Series(dtype=object),
                "j": pd.Series(dtype=np.int64),
            }
        ),
        tenk=pd.DataFrame(
            {
                "cik": pd.Series(dtype=np.int64),
                "start": pd.Series(dtype=np.int64),
                "fkey": pd.Series(dtype=object),
                "j": pd.Series(dtype=np.int64),
            }
        ),
        links=pd.DataFrame(
            {c: pd.Series(dtype=object if c == "fkey" else np.int64) for c in LINK_COLS}
        ),
    )
