"""The big paper account's daily news brain feed: GitHub Actions, the rolling state, no Dell.

Twice each market day, on the big account's own keys:

- 8:15 am FEED: news since the last pull (Alpaca), scored by FinBERT on the CPU at the
  pinned revision; earnings 8-Ks and 10-Ks of the universe's companies since the last
  pull (EDGAR); the daily bars of the close before for every stock, with the 25 sessions
  before it to catch splits and dividends. Everything waits in ``pending/``.
- 9:00 am, the first step of PICKS: the same pulls again for whatever arrived since,
  then the rolling state (``rolling.State``) advances to the session, with exactly what
  was known by the 9:00 am cutoff, as the model was trained.

``problems`` says what is stale or failed; the picks job then makes no buys, the account
sits in SPY, and the BIG PAPER alert says why.

The state keeps the pandas version it was built with; a different pandas would change
the last digits of the running sums, so the feed refuses to run on one.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa import clock, event_signals
from stoa.newsbrain import panel, rolling, sealed
from stoa.newsbrain import spycore as sc
from stoa.newsbrain import trial as t9
from stoa.newsbrain.pace import write_parquet

log = logging.getLogger(__name__)

BAR_LOOKBACK = 25  # sessions re-read each morning to catch splits and dividends
NEWS_OVERLAP = timedelta(minutes=30)  # each news pull re-reads this much, for late listings
NEWS_SEEN = timedelta(days=3)  # articles already in the state, remembered to skip repeats
SEC_SEEN = timedelta(days=30)  # filings already taken, by accession number
STATIC_FILES = ("model_h1.pkl", "sic.json")  # copied at the seed with the other fixed inputs

NewsSource = Callable[[datetime, datetime], pd.DataFrame]
FilingSource = Callable[[datetime, datetime], tuple[pd.DataFrame, pd.DataFrame]]
BarSource = Callable[[list[str], date, date], dict[str, pd.DataFrame]]
Scorer = Callable[[list[str], int], np.ndarray]


class FeedError(RuntimeError):
    """The feed cannot bring the state to the session."""


@dataclass
class Sources:
    """Where the feed reads from: Alpaca and EDGAR live, the Dell's files in the checks.

    ``news``: articles updated in a window (id, created_at, updated_at, symbols, headline,
    summary). ``filings``: earnings 8-K documents and 10-K customer paragraphs accepted in
    a window, in ``sec_pull``'s columns. ``bars``: daily bars by symbol for a date range.
    ``scorer``: FinBERT probabilities (positive, negative, neutral) at a token limit.
    """

    news: NewsSource
    filings: FilingSource
    bars: BarSource
    scorer: Scorer


# ---- the feed's own record ----------------------------------------------------------------


@dataclass
class Feed:
    root: Path

    @property
    def state_dir(self) -> Path:
        return self.root / "state"

    @property
    def pending(self) -> Path:
        return self.root / "pending"

    def record(self) -> dict[str, Any]:
        path = self.root / "feed.json"
        out: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return out

    def write_record(self, rec: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "feed.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")

    def read(self, name: str) -> pd.DataFrame | None:
        path = self.pending / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else None

    def write(self, name: str, df: pd.DataFrame) -> None:
        write_parquet(df.reset_index(drop=True), self.pending / f"{name}.parquet")

    def state(self) -> rolling.State:
        st = rolling.State.load(self.state_dir)
        built = json.loads((self.state_dir / "meta.json").read_text(encoding="utf-8"))["pandas"]
        if built != pd.__version__:
            raise FeedError(
                f"the state was built with pandas {built}, this is {pd.__version__}; the "
                "running sums would differ in their last digits"
            )
        return st


def cutoff(d: date) -> datetime:
    return clock.at_ny(d, "09:00").astimezone(UTC)


# ---- news ------------------------------------------------------------------------------------


def score_news(raw: pd.DataFrame, scorer: Scorer) -> pd.Series:
    """Positive minus negative, as ``score.score_news_file`` and ``panel.load_news`` make it."""
    from stoa.reader import text_of

    texts = [text_of(h, s) for h, s in zip(raw["headline"], raw["summary"], strict=True)]
    probs = scorer(texts, 128)
    pos, neg = probs[:, 0].astype(np.float32), probs[:, 1].astype(np.float32)
    return pd.Series((pos - neg).astype("float32"), index=raw.index)


def month_index(ts: pd.Timestamp) -> int:
    """The news file a month's articles belong to, as the rebuild counts them from 2015-01."""
    return (ts.year - 2015) * 12 + ts.month - 1


def news_rows(raw: pd.DataFrame, net: pd.Series, counts: dict[str, int]) -> pd.DataFrame:
    """Articles in the archive's columns; ``counts`` numbers each month's rows in arrival order."""
    df = raw.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
    df["known_at"] = df[["created_at", "updated_at"]].max(axis=1)
    df["net"] = net.astype("float32")
    df["keep"] = df["symbols"].map(len).between(1, panel.MAX_TAGS)
    pos = []
    for ts in df["updated_at"]:
        key = f"{ts.year:04d}-{ts.month:02d}"
        n = counts.get(key, 0)
        counts[key] = n + 1
        pos.append(np.int64(month_index(ts)) * 10_000_000 + n)
    df["pos"] = np.array(pos, dtype=np.int64)
    return df[rolling.NEWS_COLS]


def pull_news(feed: Feed, src: Sources, until: datetime) -> int:
    rec = feed.record()
    since = datetime.fromisoformat(rec["news_to"]) - NEWS_OVERLAP
    raw = src.news(since, until)
    old = feed.read("news")
    if len(raw):
        # Already pulled or in the state, unchanged: skip. Updated since: the new version
        # comes in and replaces the old one.
        seen = {int(k): pd.Timestamp(v) for k, v in rec.get("news_seen", {}).items()}
        if old is not None:
            seen.update(
                {int(i): pd.Timestamp(k) for i, k in zip(old["id"], old["known_at"], strict=True)}
            )
        known = pd.to_datetime(raw[["created_at", "updated_at"]].max(axis=1), utc=True)
        fresh = [
            int(i) not in seen or k > seen[int(i)] for i, k in zip(raw["id"], known, strict=True)
        ]
        raw = raw[fresh]
    if len(raw):
        counts: dict[str, int] = rec.get("news_rows", {})
        rows = news_rows(raw, score_news(raw, src.scorer), counts)
        rec["news_rows"] = counts
        merged = rows if old is None else pd.concat([old, rows], ignore_index=True)
        feed.write("news", merged)
    rec["news_to"] = until.isoformat()
    feed.write_record(rec)
    return len(raw)


# ---- filings ---------------------------------------------------------------------------------


def score_filings(docs: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    """Each document's FinBERT scores on its first section, as ``score.score_sec_file``."""
    from stoa.newsbrain.score import first_section

    probs = scorer([first_section(t) for t in docs["text"]], 256)
    out = docs.copy()
    out["positive"] = probs[:, 0].astype(np.float32)
    out["negative"] = probs[:, 1].astype(np.float32)
    return out


def filing_rows(docs: pd.DataFrame) -> pd.DataFrame:
    """Earnings 8-K documents to ``rolling.FILING_COLS``: one row per filing."""
    rows = []
    for accn, f in docs.groupby("accn", sort=False):
        r = rolling.filing_row(f)
        if r is not None:
            cik = int(r["cik"])
            # Newest first within a company, as EDGAR lists them and the rebuild reads them.
            j = -int(pd.Timestamp(r["accepted"]).value // 1_000_000_000)
            rows.append({**r, "accn": str(accn), "fkey": rolling.file_key(cik), "j": j})
    return pd.DataFrame(rows, columns=rolling.FILING_COLS)


def tenk_rows(docs: pd.DataFrame, static: rolling.Static) -> pd.DataFrame:
    """10-K customer paragraphs to ``rolling.TENK_COLS``, in accepted order."""
    by_name = rolling.name_index(static)
    rows = []
    docs = docs[docs["text"].str.len() > 0].sort_values("accepted", kind="stable")
    for _, r in docs.iterrows():
        cik = int(r["cik"])
        rows.append(
            {
                "cik": cik,
                "accn": str(r["accn"]),
                "accepted": r["accepted"],
                "fkey": rolling.file_key(cik),
                "j": int(pd.Timestamp(r["accepted"]).value // 1_000_000_000),
                "others": rolling.tenk_links(cik, r["accepted"], r["text"], static, by_name),
            }
        )
    return pd.DataFrame(rows, columns=rolling.TENK_COLS)


def pull_filings(feed: Feed, src: Sources, static: rolling.Static, until: datetime) -> int:
    rec = feed.record()
    since = datetime.fromisoformat(rec["sec_to"])
    earn, tenk = src.filings(since, until)
    seen = set(rec.get("sec_seen", {}))
    earn = earn[~earn["accn"].astype(str).isin(seen)] if len(earn) else earn
    tenk = tenk[~tenk["accn"].astype(str).isin(seen)] if len(tenk) else tenk
    taken = {
        str(a): str(t)
        for df in (earn, tenk)
        if len(df)
        for a, t in zip(df["accn"], df["accepted"], strict=True)
    }
    rec["sec_seen"] = recent_only({**rec.get("sec_seen", {}), **taken}, until - SEC_SEEN)
    n = 0
    if len(earn):
        rows = filing_rows(score_filings(earn, src.scorer))
        old = feed.read("filings")
        feed.write("filings", rows if old is None else pd.concat([old, rows], ignore_index=True))
        n += len(rows)
    if len(tenk):
        rows = tenk_rows(tenk, static)
        old = feed.read("tenks")
        feed.write("tenks", rows if old is None else pd.concat([old, rows], ignore_index=True))
        n += len(rows)
    rec["sec_to"] = until.isoformat()
    feed.write_record(rec)
    return n


# ---- bars ------------------------------------------------------------------------------------


def pull_bars(feed: Feed, src: Sources, symbols: list[str], through: date) -> int:
    """Daily bars of the last ``BAR_LOOKBACK`` sessions through ``through``, every stock."""
    days = clock.sessions_between(panel.PANEL_START, through)[-BAR_LOOKBACK:]
    got = src.bars(symbols, days[0], through)
    rows = []
    for sym, df in got.items():
        if df is None or df.empty:
            continue
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        d = pd.DataFrame(
            {"close": df["close"].to_numpy(float), "volume": df["volume"].to_numpy(float)},
            index=idx.normalize(),
        )
        d = d[~d.index.duplicated()]
        d = d[d.index.isin(pd.DatetimeIndex([pd.Timestamp(x) for x in days]))]
        rows.append(d.assign(symbol=sym).rename_axis("date").reset_index())
    out = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["date", "close", "volume", "symbol"])
    )
    feed.write("bars", out)
    rec = feed.record()
    rec["bars_through"] = through.isoformat()
    feed.write_record(rec)
    return int(out["symbol"].nunique()) if len(out) else 0


def bar_columns(bars: pd.DataFrame, symbols: list[str], day: date) -> tuple[np.ndarray, np.ndarray]:
    """One session's closes and volumes aligned with the state's stocks, NaN where none."""
    d = bars[bars["date"] == pd.Timestamp(day)].drop_duplicates("symbol").set_index("symbol")
    close = d["close"].reindex(symbols).to_numpy(dtype=float)
    volume = d["volume"].reindex(symbols).to_numpy(dtype=float)
    return close, volume


def adjusted(st: rolling.State, bars: pd.DataFrame, before: date) -> list[str]:
    """Stocks whose re-read past bars differ from the state's: a split or a dividend."""
    days = clock.sessions_between(panel.PANEL_START, before)[-rolling.WINDOW :]
    out = set()
    for k, day in enumerate(days):
        close, volume = bar_columns(bars, st.bars.symbols, day)
        col = rolling.WINDOW - len(days) + k
        old_c, old_v = st.bars.close[:, -rolling.WINDOW :][:, col], st.bars.volume[:, col]
        have = ~np.isnan(close)
        diff = have & ((close != old_c) | (volume != old_v))
        out |= {st.bars.symbols[i] for i in np.flatnonzero(diff)}
    return sorted(out)


def rebuild_bars(st: rolling.State, src: Sources, symbols: list[str], through: date) -> None:
    """Rebuild these stocks' running sums from their full history, read fresh."""
    days = clock.sessions_between(panel.PANEL_START, through)
    got = src.bars(symbols, days[0], through)
    close = np.full((len(symbols), len(days)), np.nan)
    volume = np.full((len(symbols), len(days)), np.nan)
    cal = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    for i, sym in enumerate(symbols):
        df = got.get(sym)
        if df is None or df.empty:
            continue
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        df = df.set_axis(idx.normalize())
        df = df[~df.index.duplicated()].reindex(cal)
        close[i], volume[i] = df["close"].to_numpy(float), df["volume"].to_numpy(float)
    fresh = rolling.bars_from_history(symbols, close, volume)
    at = {s: i for i, s in enumerate(st.bars.symbols)}
    for j, sym in enumerate(symbols):
        st.bars.replace(at[sym], fresh, j)


# ---- advancing the state ---------------------------------------------------------------------


def advance(feed: Feed, src: Sources, session: date) -> rolling.State:
    """Bring the state to ``session`` with what pending holds up to each 9:00 am cutoff."""
    st = feed.state()
    news, filings, tenks = feed.read("news"), feed.read("filings"), feed.read("tenks")
    bars = feed.read("bars")
    while st.session < session:
        day = clock.next_session(st.session)
        prev = clock.previous_session(day)
        if bars is None or not (bars["date"] == pd.Timestamp(prev)).any():
            raise FeedError(f"no daily bars for {prev} in the feed")
        cut = cutoff(day)
        if st.bars.pushed >= rolling.WINDOW:
            moved = adjusted(st, bars, clock.previous_session(prev))
            if moved:
                log.info("rebuilding %d stocks after splits or dividends", len(moved))
                rebuild_bars(st, src, moved, clock.previous_session(prev))
        close, volume = bar_columns(bars, st.bars.symbols, prev)

        n_now, news = due(news, "known_at", cut)
        remember(feed, n_now)
        f_now, filings = due(filings, "accepted", cut)
        k_now, tenks = due(tenks, "accepted", cut)
        with wall(day):
            st.update(
                day,
                news=n_now if len(n_now) else pd.DataFrame(columns=rolling.NEWS_COLS),
                filings=f_now if len(f_now) else pd.DataFrame(columns=rolling.FILING_COLS),
                tenks=k_now if len(k_now) else pd.DataFrame(columns=rolling.TENK_COLS),
                close=close,
                volume=volume,
            )
    st.save()
    for name, df in (("news", news), ("filings", filings), ("tenks", tenks)):
        if df is not None:
            feed.write(name, df)
    return st


def recent_only(stamps: dict[str, str], since: datetime) -> dict[str, str]:
    return {k: v for k, v in stamps.items() if pd.Timestamp(v) >= pd.Timestamp(since)}


def remember(feed: Feed, news: pd.DataFrame) -> None:
    """Articles now in the state, kept a few days so a re-read of the same window skips them."""
    if news.empty:
        return
    rec = feed.record()
    seen = dict(rec.get("news_seen", {}))
    seen.update(
        {
            str(int(i)): pd.Timestamp(k).isoformat()
            for i, k in zip(news["id"], news["known_at"], strict=True)
        }
    )
    newest = max(pd.Timestamp(v) for v in seen.values())
    rec["news_seen"] = recent_only(seen, newest - NEWS_SEEN)
    feed.write_record(rec)


def due(
    df: pd.DataFrame | None, col: str, cut: datetime
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Rows known by the cutoff, and the rest, which wait for the next session."""
    if df is None or df.empty:
        return pd.DataFrame(), df
    t = pd.to_datetime(df[col], utc=True)
    return df[t <= cut], df[t > cut]


def wall(session: date) -> AbstractContextManager[None]:
    """The holdout wall's live mode for a session after the holdout, as ``live.rows_for``."""
    return sealed.live(session) if session > sealed.HOLDOUT_END else nullcontext()


# ---- the runs -------------------------------------------------------------------------------


# ---- where the state lives: a private Hugging Face dataset -------------------------------


def pull_state(feed: Feed, repo: str, token: str) -> None:
    """Download the feed's folder from the private dataset. The token is used for this call
    only: it is not saved, and nothing is logged in."""
    from huggingface_hub import HfApi, snapshot_download

    private_or_refuse(HfApi(token=token), repo)
    snapshot_download(repo_id=repo, repo_type="dataset", token=token, local_dir=feed.root)


def push_state(feed: Feed, repo: str, token: str, message: str) -> None:
    """Upload the feed's folder to the private dataset (only changed files travel)."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    private_or_refuse(api, repo)
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=feed.root,
        commit_message=message,
        ignore_patterns=[".cache/**", "**/*.tmp", "**/*.tmp.npz"],
    )


def private_or_refuse(api: Any, repo: str) -> None:
    """The state holds the frozen model and every headline: it never goes to a public repo."""
    info = api.repo_info(repo, repo_type="dataset")
    if not getattr(info, "private", False):
        raise FeedError(f"the Hugging Face dataset {repo} is not private; nothing sent or read")


def run(feed: Feed, src: Sources, now: datetime) -> dict[str, Any]:
    """One feed run: pull what is new, and advance the state through every session whose
    9:00 am cutoff has passed (at 8:15 am, the session before)."""
    st = feed.state()
    day = clock.to_ny(now).date()
    session = day if clock.is_session(day) else clock.previous_session(day)
    out: dict[str, Any] = {"at": now.isoformat(), "session": session.isoformat()}
    out["news"] = pull_news(feed, src, now)
    out["filings"] = pull_filings(feed, src, st.static, now)
    prev = clock.previous_session(session)
    if feed.record().get("bars_through") != prev.isoformat():
        out["bars"] = pull_bars(feed, src, st.bars.symbols, prev)
    target = session if now >= cutoff(session) else clock.previous_session(session)
    if st.session < target:
        st = advance(feed, src, target)
    out["state"] = st.session.isoformat()
    rec = feed.record()
    rec["last_run"] = out
    feed.write_record(rec)
    return out


def problems(feed: Feed, session: date, now: datetime) -> list[str]:
    """What is stale for picks on ``session``: empty when the state is ready."""
    rec = feed.record()
    if not rec:
        return ["the feed has never run"]
    out = []
    meta = json.loads((feed.state_dir / "meta.json").read_text(encoding="utf-8"))
    if meta["session"] != session.isoformat():
        out.append(f"the news brain state is at {meta['session']}, not {session}")
    cut = cutoff(session)
    for key, what in (("news_to", "news"), ("sec_to", "SEC filings")):
        if key not in rec or datetime.fromisoformat(rec[key]) < cut:
            out.append(f"{what} pulled only to {rec.get(key, 'never')}, not past 9:00 am")
    prev = clock.previous_session(session)
    if rec.get("bars_through") != prev.isoformat():
        out.append(f"daily bars through {rec.get('bars_through', 'never')}, not {prev}")
    return out


def signals(feed: Feed, session: date) -> list[event_signals.Signal]:
    """Trial 12's picks for ``session`` from the rolling state: p >= 0.55, up to 10, best first."""
    st = feed.state()
    if st.session != session:
        raise FeedError(f"the news brain state is at {st.session}, not {session}")
    with wall(session):
        rows = st.rows()
    if rows.empty:
        return []
    model = pickle.loads((st.root / "static" / "model_h1.pkl").read_bytes())
    t9.register_kind(sc.GRADED)
    return sc.signals(rows, model.predict(rows), sc.GRADED)


def seed_from_dell(feed: Feed, session: date, sec_pulled: datetime) -> rolling.State:
    """The one seed build, on the Dell: the state as of ``session``'s 9:00 am cutoff.

    ``sec_pulled``: when the Dell's last SEC pull began (its companies were read over a day
    or more); the feed reads the SEC again from a day before it."""
    with wall(session):
        hist = rolling.History.read(session, live=session > sealed.HOLDOUT_END)
    return seed(feed, hist, session, sec_pulled)


def seed(feed: Feed, hist: rolling.History, session: date, sec_pulled: datetime) -> rolling.State:
    """The feed's state and record from the Dell's history, read once."""
    import shutil

    from stoa.newsbrain import ROOT

    with wall(session):
        st = hist.seed(feed.state_dir, session)
    st.save()
    for name, src in (
        ("model_h1.pkl", ROOT / "trial9" / "model_h1.pkl"),
        ("sic.json", Path("data/events/sic.json")),
    ):
        shutil.copyfile(src, st.root / "static" / name)
    feed.write_record(seed_record(hist, session, sec_pulled))
    return st


def seed_record(hist: rolling.History, session: date, sec_pulled: datetime) -> dict[str, Any]:
    """Where the feed takes over from the seed: each source from where the Dell's pull ended
    (the news at its newest article, the SEC from the start of its last pass), with what is
    already in the state remembered so nothing is taken twice."""
    cut = cutoff(session)
    news = hist.news[pd.to_datetime(hist.news["known_at"], utc=True) <= cut]
    newest = pd.to_datetime(news["known_at"], utc=True).max().to_pydatetime()
    recent = news[pd.to_datetime(news["known_at"], utc=True) >= newest - NEWS_SEEN]
    sec_to = min(cut, sec_pulled - timedelta(days=1))
    filed = pd.concat(
        [hist.filings[["accn", "accepted"]], hist.tenks[["accn", "accepted"]]], ignore_index=True
    )
    filed = filed[pd.to_datetime(filed["accepted"], utc=True) <= cut]
    return {
        "news_to": min(cut, newest).isoformat(),
        "sec_to": sec_to.isoformat(),
        "bars_through": clock.previous_session(session).isoformat(),
        "news_rows": month_rows(hist.news),
        "news_seen": {
            str(int(i)): pd.Timestamp(k).isoformat()
            for i, k in zip(recent["id"], recent["known_at"], strict=True)
        },
        "sec_seen": recent_only(
            {str(a): str(t) for a, t in zip(filed["accn"], filed["accepted"], strict=True)},
            sec_to - SEC_SEEN,
        ),
        "seeded": session.isoformat(),
    }


def month_rows(news: pd.DataFrame) -> dict[str, int]:
    """Rows already in each month's news file, so new articles are numbered after them."""
    k = (news["pos"] // 10_000_000).astype(int)
    n = (news["pos"] % 10_000_000).astype(int)
    last = pd.DataFrame({"k": k, "n": n}).groupby("k")["n"].max() + 1
    return {
        f"{2015 + int(str(i)) // 12:04d}-{int(str(i)) % 12 + 1:02d}": int(v)
        for i, v in last.items()
    }


# ---- the live sources: Alpaca on the big account's keys, EDGAR, FinBERT on the CPU ---------

EDGAR = "https://www.sec.gov"
EDGAR_FORMS = frozenset({"8-K", "8-K/A", "10-K", "10-K405", "10-KT", "10-K/A"})
_IDX = re.compile(r"^(?P<form>\S+(?: \S+)*?)\s{2,}.+?\s{2,}(?P<cik>\d+)\s{2,}\d{8}\s{2,}\S+\s*$")
_ENTRY = re.compile(r"<entry>.*?</entry>", re.S)
_TITLE = re.compile(r"<title>(?P<form>[^<]+?) - .*?\((?P<cik>\d{10})\)")
_UPDATED = re.compile(r"<updated>([^<]+)</updated>")


def alpaca_news(key: str, secret: str, wanted: set[str]) -> NewsSource:
    """Every article updated in the window that tags a universe stock (``alpaca_pull``'s rule)."""
    import httpx

    from stoa.newsbrain import alpaca_pull
    from stoa.newsbrain.pace import Pace

    def fetch(since: datetime, until: datetime) -> pd.DataFrame:
        client = httpx.Client(
            base_url=alpaca_pull.DATA_URL,
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=60.0,
        )
        pace = Pace("feed-news", 150, alpaca_pull.LIMITS, respect_live=False)
        params: dict[str, Any] = {
            "start": since.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": until.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 50,
            "sort": "asc",
            "include_content": "false",
            "exclude_contentless": "false",
        }
        rows: list[dict[str, Any]] = []
        while True:
            body = alpaca_pull.get(client, pace, "/v1beta1/news", params)
            rows.extend(alpaca_pull.news_rows(list(body.get("news") or []), wanted))
            token = body.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        cols = ["id", "created_at", "updated_at", "symbols", "headline", "summary"]
        df = pd.DataFrame(rows, columns=cols)
        if len(df):
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
            df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
            df = df.drop_duplicates("id", keep="last").reset_index(drop=True)
        return df

    return fetch


def alpaca_bars(provider: Any) -> BarSource:
    """Daily bars as the wide universe was fetched (``stoa universe wide``), 100 a request."""
    from stoa import wide

    def fetch(symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), 100):
            got = wide.fetch_robust(lambda s: provider.fetch(s, start, end), symbols[i : i + 100])
            out.update(got)
        return out

    return fetch


def edgar_ciks(get: Callable[[str], str | None], since: datetime, until: datetime) -> set[int]:
    """Companies with an 8-K or 10-K in the window: EDGAR's nightly form index for past
    days, and its latest-filings feed for today (the nightly index comes out at night)."""
    out: set[int] = set()
    d = clock.to_ny(since).date()
    while d < clock.to_ny(until).date():
        if d.weekday() < 5:
            q = (d.month - 1) // 3 + 1
            body = get(f"{EDGAR}/Archives/edgar/daily-index/{d.year}/QTR{q}/form.{d:%Y%m%d}.idx")
            for line in (body or "").splitlines():
                m = _IDX.match(line)
                if m and m["form"] in EDGAR_FORMS:
                    out.add(int(m["cik"]))
        d += timedelta(days=1)
    for form in ("8-K", "10-K"):
        start = 0
        while True:
            body = get(
                f"{EDGAR}/cgi-bin/browse-edgar?action=getcurrent&type={form}&company=&dateb="
                f"&owner=include&start={start}&count=100&output=atom"
            )
            entries = _ENTRY.findall(body or "")
            oldest: datetime | None = None
            for e in entries:
                t, u = _TITLE.search(e), _UPDATED.search(e)
                if t and t["form"].strip() in EDGAR_FORMS:
                    out.add(int(t["cik"]))
                if u:
                    oldest = datetime.fromisoformat(u.group(1))
            if len(entries) < 100 or oldest is None or oldest < since:
                break
            start += 100
    return out


def edgar_filings(ciks: set[int]) -> FilingSource:
    """Earnings 8-K documents and 10-K customer paragraphs accepted in the window, read by
    ``sec_pull``'s own code, for the universe's companies."""
    from stoa.data.sec import DATA_URL, make_sec_client
    from stoa.newsbrain import sec_pull

    def fetch(since: datetime, until: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
        sec = make_sec_client(sec_pull.USER_AGENT, None, 8.0)
        fetcher = sec_pull.Fetcher(sec.client, sec.limiter)

        def text(url: str) -> str | None:
            resp = fetcher.get(url)
            return resp.text if resp is not None else None

        earn: list[dict[str, Any]] = []
        tenk: list[dict[str, Any]] = []
        for cik in sorted(edgar_ciks(text, since, until) & ciks):
            body = sec.get_json(f"{DATA_URL}/submissions/CIK{cik:010d}.json", None, True)
            recent = (body or {}).get("filings", {}).get("recent", {})
            fl = sec_pull.filings(recent)
            acc = pd.to_datetime(fl["accepted"], utc=True)
            fl = fl[(acc > pd.Timestamp(since)) & (acc <= pd.Timestamp(until))]
            if fl.empty:
                continue
            earn += sec_pull.earnings_docs(fetcher, cik, fl)
            tenk += sec_pull.tenk_docs(fetcher, cik, fl)
        cols = ["cik", "accn", "form", "items", "filed", "accepted", "kind", "doc", "text"]
        return pd.DataFrame(earn, columns=cols), pd.DataFrame(tenk, columns=cols)

    return fetch


def finbert() -> Scorer:
    """FinBERT on the CPU at the revision that scored the history (``score.REVISION``)."""
    from stoa.newsbrain.score import MODEL, REVISION
    from stoa.reader import FinBert

    readers: dict[int, Any] = {}

    def score(texts: list[str], max_length: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, 3), dtype=np.float32)
        if max_length not in readers:
            readers[max_length] = FinBert(
                MODEL, REVISION, device="cpu", batch_size=32, max_length=max_length
            )
        out: np.ndarray = readers[max_length].score(texts)
        return out

    return score
