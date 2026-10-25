"""Trial 9's rows: one per (symbol, session) with news, a recent filing or a ripple signal.

Everything is known by 9:00 am New York time on the session (see DECISIONS.md,
trial 9). Daily-bar inputs end at the close of the session before. Labels look
forward from the open of the session and are only ever read by training on
rows whose label ends before the cut a caller gives.
"""

from __future__ import annotations

import logging
import math
import re
from collections import deque
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from stoa import clock, wide
from stoa.newsbrain import ROOT
from stoa.newsbrain.alpaca_pull import NEWS_DIR, PLAN, WIDE_BARS
from stoa.newsbrain.score import SCORE_DIR
from stoa.newsbrain.sec_pull import SEC_DIR

log = logging.getLogger(__name__)

HORIZONS = (1, 5, 20)
MAX_TAGS = 3
REWRITE_DAYS = 3
REWRITE_JACCARD = 0.5
REWRITE_WEIGHT = 0.25
BASELINE_SESSIONS = 60
BASELINE_MIN = 5
FILING_SESSIONS = 5
RIPPLE_LAGS = (1, 2, 3)
LINK_SESSIONS = 315  # about 15 months
TOP_NAMES = 1500
FEATURES = (
    "sent",
    "surprise",
    "first_break",
    "quiet_spike",
    "coverage",
    "ripple",
    "filing_tone",
    "guidance",
    "beat_miss",
    "ret_1",
    "ret_5",
    "vol_20",
    "log_adv",
)
FEE = 0.3e-4

_WORD = re.compile(r"[a-z]{3,}")


# ---- the session calendar --------------------------------------------------------------


def calendar(start: date, end: date) -> tuple[list[date], np.ndarray]:
    """Sessions and their 9:00 am New York cutoffs as UTC nanoseconds."""
    days = clock.sessions_between(start, end)
    cut = np.array([pd.Timestamp(clock.at_ny(d, "09:00")).value for d in days], dtype=np.int64)
    return days, cut


def nanos(ts: pd.Series) -> np.ndarray:
    """UTC nanoseconds, whatever unit pandas stored the times in."""
    out: np.ndarray = pd.to_datetime(ts, utc=True).dt.as_unit("ns").astype("int64").to_numpy()
    return out


def to_session(ts: pd.Series, cut: np.ndarray) -> np.ndarray:
    """Index of the first session whose 9:00 am cutoff is at or after each time."""
    return np.searchsorted(cut, nanos(ts), side="left")


# ---- news ------------------------------------------------------------------------------


def load_news(cut: np.ndarray) -> pd.DataFrame:
    """One row per (article, tagged symbol) for articles tagging at most 3 symbols."""
    parts = []
    for path in sorted(NEWS_DIR.glob("*.parquet")):
        scores = SCORE_DIR / "news" / path.name
        if not scores.exists():
            raise FileNotFoundError(f"{path.name} is not scored yet")
        df = pd.read_parquet(
            path, columns=["id", "created_at", "updated_at", "symbols", "headline"]
        )
        sc = pd.read_parquet(scores)
        df = df.merge(sc[["id", "positive", "negative"]], on="id", how="inner")
        df = df[df["symbols"].map(len).between(1, MAX_TAGS)]
        df["known_at"] = df[["created_at", "updated_at"]].max(axis=1)
        df["net"] = (df["positive"] - df["negative"]).astype("float32")
        parts.append(df.drop(columns=["updated_at", "positive", "negative"]))
    news = pd.concat(parts, ignore_index=True).explode("symbols")
    news = news.rename(columns={"symbols": "symbol"})
    news["sidx"] = to_session(news["known_at"], cut)
    news = news[news["sidx"] < len(cut)]
    return news.sort_values(["symbol", "created_at"]).reset_index(drop=True)


def words(headline: str) -> frozenset[str]:
    return frozenset(_WORD.findall(headline.lower()))


def first_breaks(news: pd.DataFrame) -> np.ndarray:
    """True for an article that is first on its story for its symbol (sorted by symbol, time)."""
    out = np.ones(len(news), dtype=bool)
    window = pd.Timedelta(days=REWRITE_DAYS).value
    syms = news["symbol"].to_numpy()
    times = nanos(news["created_at"])
    heads = news["headline"].to_numpy()
    recent: deque[tuple[int, frozenset[str]]] = deque()
    last = None
    for i in range(len(news)):
        if syms[i] != last:
            recent.clear()
            last = syms[i]
        while recent and times[i] - recent[0][0] > window:
            recent.popleft()
        w = words(str(heads[i]))
        if w:
            for _, prev in recent:
                inter = len(w & prev)
                if inter and inter / len(w | prev) >= REWRITE_JACCARD:
                    out[i] = False
                    break
        recent.append((times[i], w))
    return out


def daily_news(news: pd.DataFrame, n_sessions: int) -> pd.DataFrame:
    """Per (symbol, session): counts, weighted tone, first-break share, and the baselines."""
    news = news.assign(first=first_breaks(news))
    news["w"] = np.where(news["first"], 1.0, REWRITE_WEIGHT)
    news["wnet"] = news["w"] * news["net"]
    g = news.groupby(["symbol", "sidx"], sort=True)
    day = g.agg(n=("net", "size"), wsum=("w", "sum"), wnet=("wnet", "sum"), first=("first", "mean"))
    day = day.reset_index()
    day["sent"] = day["wnet"] / day["wsum"]
    day["first_break"] = day["first"].astype("float32")
    out = []
    for _, grp in day.groupby("symbol", sort=False):
        d = grp.copy()
        prior = d["sent"].shift(1).rolling(BASELINE_SESSIONS, min_periods=BASELINE_MIN)
        mean, std = prior.mean(), prior.std()
        d["surprise"] = ((d["sent"] - mean) / std).where(std > 1e-6, 0.0).fillna(0.0)
        # Average daily articles over the previous 60 sessions, zero-news days included.
        counts = np.zeros(n_sessions + 1)
        np.add.at(counts, d["sidx"].to_numpy() + 1, d["n"].to_numpy())
        cum = np.cumsum(counts)
        i = d["sidx"].to_numpy()
        lo = np.maximum(i - BASELINE_SESSIONS, 0)
        avg = (cum[i] - cum[lo]) / BASELINE_SESSIONS
        d["quiet_spike"] = np.log((1 + d["n"]) / (1 + avg))
        d["coverage"] = np.log1p(avg)
        out.append(d)
    keep = ["symbol", "sidx", "n", "sent", "surprise", "first_break", "quiet_spike", "coverage"]
    return pd.concat(out, ignore_index=True)[keep]


# ---- filings ---------------------------------------------------------------------------

_RAISED = re.compile(
    r"\b(rais(e|es|ed|ing)|increas(e|es|ed|ing)|boost(s|ed)?|lift(s|ed)?)\b[^.]{0,60}"
    r"\b(guidance|outlook|forecast)\b|\b(guidance|outlook|forecast)\b[^.]{0,40}"
    r"\b(raised|increased)\b",
    re.I,
)
_CUT = re.compile(
    r"\b(lower(s|ed|ing)?|cut(s|ting)?|reduc(e|es|ed|ing)|withdr(aw|ew|awn|aws))\b[^.]{0,60}"
    r"\b(guidance|outlook|forecast)\b|\b(guidance|outlook|forecast)\b[^.]{0,40}"
    r"\b(lowered|reduced|cut|withdrawn)\b",
    re.I,
)
_BEAT = re.compile(
    r"\b(exceed(s|ed|ing)?|above|beat(s|ing)?|surpass(es|ed)?|ahead of)\b[^.]{0,40}"
    r"\b(expectations|consensus|estimates|guidance)\b",
    re.I,
)
_MISS = re.compile(
    r"\b(below|miss(es|ed)?|short of|fell short)\b[^.]{0,40}"
    r"\b(expectations|consensus|estimates|guidance)\b",
    re.I,
)


def direction(text: str, up: re.Pattern[str], down: re.Pattern[str]) -> int:
    u, d = bool(up.search(text)), bool(down.search(text))
    return int(u) - int(d)


def load_ciks() -> pd.DataFrame:
    return pd.read_parquet(SEC_DIR / "ciks.parquet")


def filing_features(cut: np.ndarray, ciks: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, session): tone, guidance and beat or miss of an 8-K 2.02 in the last 5."""
    rows = []
    for path in sorted((SEC_DIR / "earnings").glob("*.parquet")):
        scores = SCORE_DIR / "sec" / "earnings" / path.name
        if not scores.exists():
            continue
        df = pd.read_parquet(path, columns=["cik", "accn", "accepted", "kind", "doc", "text"])
        if df.empty:
            continue
        sc = pd.read_parquet(scores)
        df = df.merge(sc, on=["accn", "kind", "doc"], how="left")
        for accn, f in df.groupby("accn", sort=False):
            ex = f[(f["kind"] == "ex99") & (f["text"].str.len() > 0)]
            use = ex if len(ex) else f[(f["kind"] == "main") & (f["text"].str.len() > 0)]
            if use.empty or pd.isna(f["accepted"].iloc[0]):
                continue
            text = " ".join(use["text"].tolist())[:40_000]
            rows.append(
                {
                    "cik": int(f["cik"].iloc[0]),
                    "accn": accn,
                    "accepted": f["accepted"].iloc[0],
                    "filing_tone": float((use["positive"] - use["negative"]).mean()),
                    "guidance": direction(text, _RAISED, _CUT),
                    "beat_miss": direction(text, _BEAT, _MISS),
                }
            )
    fl = pd.DataFrame(rows)
    if fl.empty:
        return pd.DataFrame(columns=["symbol", "sidx", "filing_tone", "guidance", "beat_miss"])
    fl["sidx0"] = to_session(fl["accepted"], cut)
    fl = fl.merge(ciks, on="cik")
    spread = []
    for k in range(FILING_SESSIONS):
        spread.append(fl.assign(sidx=fl["sidx0"] + k, age=k))
    out = pd.concat(spread, ignore_index=True)
    out = out[out["sidx"] < len(cut)]
    # The newest filing wins when two overlap.
    out = out.sort_values(["symbol", "sidx", "age"]).drop_duplicates(["symbol", "sidx"])
    return out[["symbol", "sidx", "filing_tone", "guidance", "beat_miss"]]


# ---- ripple ----------------------------------------------------------------------------

_SUFFIX = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|companies|ltd|limited|plc|holdings?|"
    r"group|sa|nv|ag|se|llc|lp|the|class [a-c]|common stock|new)\b"
)
_PUNCT = re.compile(r"[^a-z0-9 ]+")


def norm_name(name: str) -> str:
    t = _PUNCT.sub(" ", name.lower())
    t = _SUFFIX.sub(" ", t)
    return " ".join(t.split())


def name_book(ciks: pd.DataFrame) -> dict[int, str]:
    """CIK to a normalized company name from the SEC ticker map."""
    import json

    body = json.loads(Path("data/sec/company_tickers.json").read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for row in body.values():
        n = norm_name(str(row.get("title", "")))
        if len(n) >= 5:
            out.setdefault(int(row["cik_str"]), n)
    return {c: out[c] for c in set(ciks["cik"]) if c in out}


def mentioned(text: str, by_name: dict[str, int], longest: int) -> set[int]:
    """Companies whose whole normalized name appears in ``text`` as a run of words."""
    toks = text.split()
    found: set[int] = set()
    for i in range(len(toks)):
        for k in range(1, longest + 1):
            if i + k > len(toks):
                break
            hit = by_name.get(" ".join(toks[i : i + k]))
            if hit is not None:
                found.add(hit)
    return found


def links(cut: np.ndarray, ciks: pd.DataFrame) -> pd.DataFrame:
    """Customer and supplier links from 10-K text: (cik, partner symbol, valid from, valid to)."""
    names = name_book(ciks)
    plan = pd.read_parquet(PLAN, columns=["symbol", "month", "dollars"])
    sym_cik = dict(zip(ciks["symbol"], ciks["cik"], strict=True))
    top_by_month: dict[str, set[int]] = {}
    for month, g in plan.groupby("month"):
        top = g.nlargest(TOP_NAMES, "dollars")["symbol"]
        top_by_month[str(month)] = {sym_cik[s] for s in top if s in sym_cik}
    by_name: dict[str, int] = {}
    for c, n in names.items():
        by_name.setdefault(n, c)
    longest = max((len(n.split()) for n in by_name), default=1)
    rows = []
    for path in sorted((SEC_DIR / "tenk").glob("*.parquet")):
        df = pd.read_parquet(path, columns=["cik", "accn", "accepted", "text"])
        df = df[df["text"].str.len() > 0].sort_values("accepted")
        if df.empty:
            continue
        starts = to_session(df["accepted"], cut)
        ends = np.append(starts[1:], len(cut))
        for (_, r), s0, s1 in zip(df.iterrows(), starts, ends, strict=True):
            if pd.isna(r["accepted"]):
                continue
            month = pd.Timestamp(r["accepted"]).strftime("%Y-%m")
            allowed = top_by_month.get(month, set())
            for other in mentioned(norm_name(str(r["text"])), by_name, longest):
                if other != r["cik"] and other in allowed:
                    rows.append((int(r["cik"]), other, int(s0), int(min(s1, s0 + LINK_SESSIONS))))
    lk = pd.DataFrame(rows, columns=["cik", "other", "start", "end"])
    return lk.drop_duplicates()


def ripple(
    day: pd.DataFrame, lk: pd.DataFrame, ciks: pd.DataFrame, adv: pd.DataFrame
) -> pd.DataFrame:
    """Per (symbol, session): mean news of larger linked companies over the previous 3 sessions.

    ``adv`` has columns symbol, sidx, adv (20 session dollar volume at the close before).
    """
    if lk.empty:
        return pd.DataFrame(columns=["symbol", "sidx", "ripple"])
    by_cik = ciks.groupby("cik")["symbol"].apply(list).to_dict()
    pairs = []
    for r in lk.itertuples(index=False):
        for a in by_cik.get(r.cik, []):
            for b in by_cik.get(r.other, []):
                pairs.append((a, b, r.start, r.end))
                pairs.append((b, a, r.start, r.end))
    edges = pd.DataFrame(pairs, columns=["target", "partner", "start", "end"]).drop_duplicates()
    sig = day.assign(signal=day["sent"] * day["first_break"])[["symbol", "sidx", "signal"]]
    lagged = pd.concat(
        [sig.assign(sidx=sig["sidx"] + lag) for lag in RIPPLE_LAGS], ignore_index=True
    )
    partner_mean = lagged.groupby(["symbol", "sidx"])["signal"].mean().reset_index()
    partner_mean = partner_mean.rename(columns={"symbol": "partner"})
    m = edges.merge(partner_mean, on="partner")
    m = m[(m["sidx"] >= m["start"]) & (m["sidx"] < m["end"])]
    a = adv.rename(columns={"symbol": "target", "adv": "adv_t"})
    m = m.merge(a, on=["target", "sidx"], how="inner")
    b = adv.rename(columns={"symbol": "partner", "adv": "adv_p"})
    m = m.merge(b, on=["partner", "sidx"], how="inner")
    m = m[m["adv_p"] > m["adv_t"]]  # from the larger company to the smaller only
    out = m.groupby(["target", "sidx"])["signal"].mean().reset_index()
    return out.rename(columns={"target": "symbol", "signal": "ripple"})


# ---- daily bars: controls, liquidity and labels ------------------------------------------


def bar_frame(symbol: str, days: list[date]) -> pd.DataFrame | None:
    path = WIDE_BARS / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["open", "close", "volume"])
    if df.empty:
        return None
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated()].reindex(pd.DatetimeIndex([pd.Timestamp(d) for d in days]))


def bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per session t, from the close of t-1: controls, liquidity and the cost of a fill."""
    close = df["close"].ffill()
    dv = (df["close"] * df["volume"]).rolling(wide.ADV_SESSIONS, min_periods=wide.ADV_SESSIONS)
    adv = dv.mean()
    ret = close.pct_change(fill_method=None)
    out = pd.DataFrame(
        {
            "ret_1": ret.shift(1),
            "ret_5": close.pct_change(5, fill_method=None).shift(1),
            "vol_20": ret.rolling(20, min_periods=15).std().shift(1),
            "adv": adv.shift(1),
            "prev_close": df["close"].shift(1),
        },
        index=df.index,
    )
    out["log_adv"] = np.log(out["adv"].clip(lower=1.0))
    out["liquid"] = (out["prev_close"] >= wide.MIN_PRICE) & (out["adv"] >= wide.MIN_DOLLAR_VOLUME)
    out["slip"] = [wide.slippage_bps(v if math.isfinite(v) else None) / 1e4 for v in out["adv"]]
    return out


def forward_returns(df: pd.DataFrame, h: int) -> pd.Series:
    """Open of t to the open of t+h; if that open is missing, the last close by then.

    The last rows, whose t+h is past the data, use the last close too, so a stock
    that stopped trading inside the horizon is labelled with its last price.
    """
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].ffill().to_numpy(dtype=float)
    n = len(df)
    i = np.arange(n)
    j = np.minimum(i + h, n - 1)
    exit_open = np.where(i + h < n, opens[j], np.nan)
    exit_px = np.where(np.isfinite(exit_open) & (exit_open > 0), exit_open, closes[j])
    ok = np.isfinite(opens) & (opens > 0) & np.isfinite(exit_px) & (j > i)
    out = np.where(ok, exit_px / np.where(ok, opens, 1.0) - 1, np.nan)
    return pd.Series(out, index=df.index)


def label_frame(df: pd.DataFrame, spy: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for h in HORIZONS:
        r = forward_returns(df, h)
        b = forward_returns(spy, h)
        net = r - b - 2 * feats["slip"] - FEE
        out[f"excess_{h}"] = net
        out[f"y_{h}"] = (net > 0).astype("float32").where(net.notna())
    return pd.DataFrame(out, index=df.index)


# ---- assembly --------------------------------------------------------------------------


def build(start: date, end: date, out: Path | None = None) -> pd.DataFrame:
    """Every row trial 9 scores, with features, liquidity, cost and labels."""
    days, cut = calendar(start, end)
    ciks = load_ciks()
    log.info("loading news")
    news = load_news(cut)
    log.info("%d article-symbol rows", len(news))
    day = daily_news(news, len(days))
    del news
    log.info("%d symbol-sessions with news", len(day))
    fil = filing_features(cut, ciks)
    log.info("%d symbol-sessions with a recent filing", len(fil))
    lk = links(cut, ciks)
    log.info("%d customer and supplier links", len(lk))

    spy = bar_frame("SPY", days)
    assert spy is not None
    linked = set(lk["cik"]) | set(lk["other"]) if len(lk) else set()
    syms = sorted(
        set(day["symbol"]) | set(fil["symbol"]) | set(ciks[ciks["cik"].isin(linked)]["symbol"])
    )
    feats: list[pd.DataFrame] = []
    for k, sym in enumerate(syms):
        df = bar_frame(sym, days)
        if df is None or df["close"].notna().sum() < 30:
            continue
        bf = bar_features(df)
        lab = label_frame(df, spy, bf)
        both = pd.concat([bf, lab], axis=1)
        both["symbol"] = sym
        both["sidx"] = np.arange(len(days))
        both = both[both["liquid"].fillna(False)]
        feats.append(both)
        if k % 1000 == 0:
            log.info("bars %d of %d", k, len(syms))
    bars = pd.concat(feats, ignore_index=True)
    adv = bars[["symbol", "sidx", "adv"]]
    rip = ripple(day, lk, ciks, adv)
    keys = pd.concat(
        [day[["symbol", "sidx"]], fil[["symbol", "sidx"]], rip[["symbol", "sidx"]]]
    ).drop_duplicates()
    rows = keys.merge(bars, on=["symbol", "sidx"], how="inner")
    rows = rows.merge(day, on=["symbol", "sidx"], how="left")
    rows = rows.merge(fil, on=["symbol", "sidx"], how="left")
    rows = rows.merge(rip, on=["symbol", "sidx"], how="left")
    for c in ("n", "sent", "surprise", "first_break", "quiet_spike", "ripple", "filing_tone"):
        rows[c] = rows[c].fillna(0.0)
    for c in ("guidance", "beat_miss"):
        rows[c] = rows[c].fillna(0).astype("int8")
    rows["coverage"] = rows["coverage"].fillna(0.0)
    rows["session"] = [days[i] for i in rows["sidx"]]
    rows = rows.sort_values(["sidx", "symbol"]).reset_index(drop=True)
    if out is not None:
        from stoa.newsbrain.pace import write_parquet

        write_parquet(rows, out)
    return rows


PANEL = ROOT / "trial9" / "panel.parquet"
# News from 2015 feeds the baselines; nothing is read after the test window, so no
# label, feature or price ever touches the holdout (2026-07-01 to 2026-10-23).
PANEL_START = date(2015, 1, 2)
PANEL_END = date(2026, 6, 30)
