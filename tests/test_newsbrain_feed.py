"""The big paper account's news brain feed: pulls, the 9:00 am cutoff, stale data, splits.

The sources are fakes reading the small world of ``test_newsbrain_rolling``; the feed's
picks must equal those from trial 9's full rebuild of the same world, bit for bit.
"""

from __future__ import annotations

import json
import pickle
import shutil
from datetime import UTC, date, datetime, timedelta
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stoa import clock
from stoa.newsbrain import feed, panel, rolling
from stoa.newsbrain import spycore as sc
from stoa.newsbrain import trial as t9
from stoa.newsbrain.score import first_section
from tests.test_newsbrain_rolling import CHECKS, LAST, SEED, world


class FixedModel:
    """A stand-in for trial 9's frozen model: a fixed function of the features."""

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        x = 0.6 + 0.3 * rows["sent"] + 0.2 * rows["ripple"] + 0.1 * rows["filing_tone"]
        return np.asarray(x.clip(0, 1), dtype=float)


def at(d: date, hhmm: str, seconds: int = 0) -> datetime:
    return clock.at_ny(d, hhmm).astimezone(UTC) + pd.Timedelta(seconds=seconds).to_pytimedelta()


class Dell:
    """Fake Alpaca, EDGAR and FinBERT reading the world's files, as the live sources would."""

    def __init__(self, root: Path) -> None:
        self.root = root
        news = pd.concat(
            [pd.read_parquet(p) for p in sorted((root / "data/newsbrain/news").glob("*.parquet"))]
        )
        self.news = news.reset_index(drop=True)
        scores = pd.concat(
            [
                pd.read_parquet(p)
                for p in sorted((root / "data/newsbrain/scores/news").glob("*.parquet"))
            ]
        )
        self.by_story = {
            f"story {i}": (p, n, u)
            for i, p, n, u in zip(
                scores["id"], scores["positive"], scores["negative"], scores["neutral"], strict=True
            )
        }
        self.by_text: dict[str, tuple[float, float, float]] = {}
        sec = root / "data/newsbrain/sec"
        earn = [pd.read_parquet(p) for p in sorted((sec / "earnings").glob("*.parquet"))]
        self.earn = pd.concat(earn, ignore_index=True)
        for p in sorted((root / "data/newsbrain/scores/sec/earnings").glob("*.parquet")):
            s = pd.read_parquet(p)
            docs = self.earn.merge(s, on=["accn", "kind", "doc"])
            for t, a, b, c in zip(
                docs["text"], docs["positive"], docs["negative"], docs["neutral"], strict=True
            ):
                self.by_text[first_section(t)] = (a, b, c)
        tenk = pd.concat([pd.read_parquet(p) for p in sorted((sec / "tenk").glob("*.parquet"))])
        self.tenk = tenk.assign(kind="customers", doc="k.htm", form="10-K", items="", filed="")
        self.edits: dict[str, float] = {}  # symbol -> factor applied to past closes (a split)
        self.full_reads: list[list[str]] = []

    def news_source(self, since: datetime, until: datetime) -> pd.DataFrame:
        known = self.news[["created_at", "updated_at"]].max(axis=1)
        got = self.news[(known > pd.Timestamp(since)) & (known <= pd.Timestamp(until))]
        return got.sort_values("created_at").reset_index(drop=True)

    def filings(self, since: datetime, until: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
        def window(df: pd.DataFrame) -> pd.DataFrame:
            t = pd.to_datetime(df["accepted"], utc=True)
            return df[(t > pd.Timestamp(since)) & (t <= pd.Timestamp(until))]

        return window(self.earn), window(self.tenk)

    def bars(self, symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        if start == panel.PANEL_START:
            self.full_reads.append(sorted(symbols))
        out = {}
        for sym in symbols:
            path = self.root / "data/bars/wide" / f"{sym}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            df = df[
                (df.index >= pd.Timestamp(start, tz="UTC"))
                & (df.index <= pd.Timestamp(end, tz="UTC"))
            ]
            if sym in self.edits:
                df = df.assign(close=df["close"] * self.edits[sym])
            out[sym] = df
        return out

    def scorer(self, texts: list[str], max_length: int) -> np.ndarray:
        rows = []
        for t in texts:
            key = t.split(". ")[-1] if max_length == 128 else t
            rows.append(self.by_story[key] if max_length == 128 else self.by_text[key])
        return np.array(rows, dtype=np.float32)

    def sources(self) -> feed.Sources:
        return feed.Sources(
            news=self.news_source, filings=self.filings, bars=self.bars, scorer=self.scorer
        )


@pytest.fixture()
def dell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Dell]:
    world(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path, Dell(tmp_path)


def seeded(root: Path, sec_pulled: datetime | None = None) -> feed.Feed:
    """The seed, as ``feed.seed_from_dell`` writes it, with the stand-in model."""
    fd = feed.Feed(root / "feed")
    hist = rolling.History.read(LAST, live=False)
    st = hist.seed(fd.state_dir, SEED)
    st.save()
    (st.root / "static" / "model_h1.pkl").write_bytes(pickle.dumps(FixedModel()))
    # The SEC pull ended at the seed's cutoff: the feed reads the day before again and
    # must skip what the seed already holds, as it must for re-read news.
    fd.write_record(feed.seed_record(hist, SEED, sec_pulled or feed.cutoff(SEED)))
    return fd


def test_the_feed_gives_the_rebuilds_picks_exactly(dell: tuple[Path, Dell]) -> None:
    root, fake = dell
    fd = seeded(root)
    src = fake.sources()
    day = SEED
    checked = 0
    while day < LAST:
        day = clock.next_session(day)
        feed.run(fd, src, at(day, "08:15"))
        assert feed.problems(fd, day, at(day, "08:15"))  # not past 9:00 am yet
        feed.run(fd, src, at(day, "09:00", 30))
        assert feed.problems(fd, day, at(day, "09:00", 30)) == []
        if day not in CHECKS:
            continue
        got = feed.signals(fd, day)
        ref = panel.build(panel.PANEL_START, day)
        ref = ref[ref["session"] == day].sort_values(["sidx", "symbol"]).reset_index(drop=True)
        t9.register_kind(sc.GRADED)
        want = sc.signals(ref, FixedModel().predict(ref), sc.GRADED)
        assert [(s.symbol, s.score) for s in got] == [(s.symbol, s.score) for s in want], day
        checked += len(want)
    assert checked > 0


def test_the_first_run_takes_nothing_the_seed_holds(dell: tuple[Path, Dell]) -> None:
    root, fake = dell
    fd = seeded(root, sec_pulled=feed.cutoff(SEED) - timedelta(days=2))
    day = clock.next_session(SEED)
    feed.run(fd, fake.sources(), at(day, "08:15"))
    archive = set(fd.state().archive()["id"])
    news = fd.read("news")
    assert news is None or not set(news["id"]) & archive
    filings = fd.read("filings")
    assert filings is None or "d-1" not in set(filings["accn"])  # 4/30's 8-K, in the seed
    feed.run(fd, fake.sources(), at(day, "09:00", 30))
    assert feed.problems(fd, day, at(day, "09:00", 30)) == []
    assert fd.state().filings["accn"].tolist().count("d-1") == 1


def test_stale_data_is_named(dell: tuple[Path, Dell]) -> None:
    root, _ = dell
    assert feed.problems(feed.Feed(root / "none"), CHECKS[0], at(CHECKS[0], "09:01"))
    fd = seeded(root)
    day = clock.next_session(SEED)
    stale = feed.problems(fd, day, at(day, "09:01"))
    assert any("state is at" in p for p in stale)
    assert any(p.startswith("news pulled only to") for p in stale)
    assert any(p.startswith("daily bars through") for p in stale)
    with pytest.raises(feed.FeedError, match="not"):
        feed.signals(fd, day)


def test_a_split_rebuilds_that_stock_from_its_full_history(dell: tuple[Path, Dell]) -> None:
    root, fake = dell
    fd = seeded(root)
    src = fake.sources()
    day = clock.next_session(SEED)
    fake.edits["BBB"] = 0.5  # a 2 for 1 split: every past close halves
    for f in (root / "data/bars/wide").glob("BBB.parquet"):
        df = pd.read_parquet(f)
        df.assign(close=df["close"] * 0.5).to_parquet(f)
    fake.edits.clear()
    feed.run(fd, src, at(day, "08:15"))
    feed.run(fd, src, at(day, "09:00", 30))
    assert fake.full_reads == [["BBB"]]
    hist = rolling.History.read(LAST, live=False)  # the whole history, split included
    k = hist.days.index(day)
    fresh = rolling.bars_from_history(hist.static.universe, hist.close[:, :k], hist.volume[:, :k])
    pd.testing.assert_frame_equal(fd.state().bars.features(), fresh.features())


def test_the_feed_refuses_a_different_pandas(dell: tuple[Path, Dell]) -> None:
    root, _ = dell
    fd = seeded(root)
    meta = fd.state_dir / "meta.json"
    meta.write_text(meta.read_text().replace(pd.__version__, "0.0.0"))
    with pytest.raises(feed.FeedError, match="pandas"):
        fd.state()
    shutil.rmtree(fd.state_dir)


class FakeHub:
    """huggingface_hub's calls the feed makes, recording what each one was given."""

    def __init__(self, private: bool, store: Path) -> None:
        self.private, self.store, self.calls = private, store, []
        self.commits: list[tuple[str, str]] = []  # (message, id), oldest first

    def api(self, token: str | None = None) -> FakeHub:
        self.calls.append(("api", token))
        return self

    def create_repo(self, repo: str, **kw: object) -> None:
        self.calls.append(("create", repo, kw.get("private")))

    def repo_info(self, repo: str, **kw: object) -> object:
        return type("Info", (), {"private": self.private})()

    def upload_folder(self, **kw: object) -> None:
        self.calls.append(("upload", kw["repo_id"], kw["ignore_patterns"]))
        allow = kw.get("allow_patterns")
        self.calls.append(("upload_only", allow))
        src = Path(str(kw["folder_path"]))
        for f in list(self.store.rglob("*")):
            rel = f.relative_to(self.store).as_posix()
            if f.is_file() and any(fnmatch(rel, p) for p in kw.get("delete_patterns") or []):  # type: ignore[attr-defined]
                f.unlink()
        for f in src.rglob("*"):
            rel = f.relative_to(src).as_posix()
            if f.is_file() and (allow is None or any(fnmatch(rel, p) for p in allow)):  # type: ignore[attr-defined]
                (self.store / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, self.store / rel)
        cid = commit_id(str(kw["commit_message"]))
        self.commits.append((str(kw["commit_message"]), cid))
        shutil.copytree(self.store, self.store.parent / f"hub-{cid}")

    def snapshot_download(self, **kw: object) -> None:
        self.calls.append(("download", kw["repo_id"], kw["token"]))
        self.calls.append(("download_only", kw.get("allow_patterns")))
        rev = kw.get("revision")
        self.calls.append(("download_rev", rev))
        src = self.store if rev is None else self.store.parent / f"hub-{rev}"
        shutil.copytree(src, Path(str(kw["local_dir"])), dirs_exist_ok=True)

    def list_repo_commits(self, repo: str, **kw: object) -> list[object]:
        return [
            type("Commit", (), {"title": t, "commit_id": c})() for t, c in reversed(self.commits)
        ]


def commit_id(message: str) -> str:
    import hashlib

    return hashlib.sha1(message.encode()).hexdigest()


def hub(monkeypatch: pytest.MonkeyPatch, private: bool, store: Path) -> FakeHub:
    import sys
    import types

    fake = FakeHub(private, store)
    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = fake.api  # type: ignore[attr-defined]
    mod.snapshot_download = fake.snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return fake


def test_the_state_goes_to_the_private_dataset_and_back(
    dell: tuple[Path, Dell], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = dell
    fd = seeded(root)
    fake = hub(monkeypatch, True, root / "hub")
    feed.push_state(fd, "ASmith91/stoa-feed", "tok-123", "seed")
    assert ("create", "ASmith91/stoa-feed", True) in fake.calls
    back = feed.Feed(root / "runner")
    feed.pull_state(back, "ASmith91/stoa-feed", "tok-123")
    assert back.state().session == SEED and back.record() == fd.record()
    for f in [*(root / "hub").rglob("*"), *(root / "runner").rglob("*")]:
        if f.is_file():
            assert b"tok-123" not in f.read_bytes(), f  # the token is never written


def test_a_dry_run_keeps_its_pulls_and_rows_for_the_dell(
    dell: tuple[Path, Dell], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, fake = dell
    fd = seeded(root)
    src = fake.sources()
    day = SEED
    for k in range(3):
        day = clock.next_session(day)
        for hhmm, tag in (("08:15", f"t{k}a-feed"), ("09:00", f"t{k}b-picks")):
            fd.raw_tag = tag
            feed.run(fd, src, at(day, hhmm, 30))
    rows, sigs = feed.scored(fd, day)
    feed.save_check(fd, day, rows, sigs)
    assert [(s.symbol, s.score) for s in sigs] == [
        (s.symbol, s.score) for s in feed.signals(fd, day)
    ]
    raw = root / "feed" / "raw"
    assert (raw / "t0a-feed" / "bars.parquet").exists()
    kept = pd.concat([pd.read_parquet(p) for p in raw.glob("*/news.parquet")])
    pending = fd.read("news")
    later = set() if pending is None else set(pending["id"])
    assert len(kept) and set(kept["id"]) <= set(fd.state().archive()["id"]) | later
    check = root / "feed" / "checks" / "t2b-picks"
    body = json.loads((check / "picks.json").read_text(encoding="utf-8"))
    assert body["session"] == day.isoformat() and body["rows"] == len(rows)
    assert body["picks"] == [{"symbol": s.symbol, "p": s.score} for s in sigs]
    got = pd.read_parquet(check / "rows.parquet")
    assert list(got["symbol"]) == list(rows["symbol"]) and "p" in got.columns
    # A real run downloads the state only, never the dry runs' pulls and checks.
    hubbed = hub(monkeypatch, True, root / "hub")
    feed.push_state(fd, "ASmith91/stoa-feed", "tok", "dry", only=["raw/t0a-feed/**"])
    feed.pull_state(feed.Feed(root / "runner"), "ASmith91/stoa-feed", "tok")
    assert ("upload_only", ["raw/t0a-feed/**"]) in hubbed.calls
    assert ("download_only", feed.STATE_FILES) in hubbed.calls


def test_a_replay_starts_from_a_past_commit_and_never_moves_the_live_state(
    dell: tuple[Path, Dell], monkeypatch: pytest.MonkeyPatch
) -> None:
    from stoa import alerts, bigpaper
    from stoa.config import Secrets
    from tests.conftest import make_settings

    root, fake = dell
    fd = seeded(root)
    hubbed = hub(monkeypatch, True, root / "hub")
    feed.push_state(fd, "ASmith91/stoa-feed", "tok", "seed")
    live = (root / "hub" / "feed.json").read_text(encoding="utf-8")
    run = root / "run"
    rt = bigpaper.BigRuntime(
        base=make_settings(big_paper__feed_dir=str(run)),
        secrets=Secrets.from_env({}),
        notifier=alerts.RecordingNotifier(),
        dry=True,
        replay_from="seed",
    )
    day = SEED
    for k in range(2):
        day = clock.next_session(day)
        shutil.rmtree(run, ignore_errors=True)  # each step on a fresh runner
        step = feed.Feed(run, f"t{k}-feed")
        bigpaper.load_feed(rt, step, "tok")
        feed.run(step, fake.sources(), at(day, "08:15", 30))
        bigpaper.save_feed(rt, step, "tok", "feed")
        assert step.state().session == clock.previous_session(day)
    seed = commit_id("seed")
    assert [c for c in hubbed.calls if c[0] == "download_rev"] == [("download_rev", seed)]
    assert ("upload_only", ["raw/t1-feed/**"]) in hubbed.calls
    assert (root / "hub" / "raw" / "t1-feed").exists()
    assert (root / "hub" / "feed.json").read_text(encoding="utf-8") == live  # untouched


def test_a_reset_puts_back_the_seed_and_keeps_the_dry_runs(
    dell: tuple[Path, Dell], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, fake = dell
    fd = seeded(root)
    hubbed = hub(monkeypatch, True, root / "hub")
    feed.push_state(fd, "ASmith91/stoa-feed", "tok", "seed")
    seed = (root / "hub" / "feed.json").read_text(encoding="utf-8")
    fd.raw_tag = "t0-feed"
    feed.run(fd, fake.sources(), at(clock.next_session(clock.next_session(SEED)), "08:15", 30))
    (fd.pending / "extra.parquet").write_bytes(b"later")  # a file the seed never had
    feed.push_state(fd, "ASmith91/stoa-feed", "tok", "feed")
    assert (root / "hub" / "feed.json").read_text(encoding="utf-8") != seed
    rev = feed.reset_state(feed.Feed(root / "empty"), "ASmith91/stoa-feed", "tok", "seed")
    assert rev == commit_id("seed") and ("download_rev", rev) in hubbed.calls
    assert (root / "hub" / "feed.json").read_text(encoding="utf-8") == seed
    assert not (root / "hub" / "pending" / "extra.parquet").exists()
    assert (root / "hub" / "raw" / "t0-feed").exists()  # the dry runs' pulls stay


def test_a_public_dataset_is_refused(
    dell: tuple[Path, Dell], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = dell
    fd = seeded(root)
    fake = hub(monkeypatch, False, root / "hub")
    with pytest.raises(feed.FeedError, match="not private"):
        feed.push_state(fd, "ASmith91/stoa-feed", "tok", "seed")
    with pytest.raises(feed.FeedError, match="not private"):
        feed.pull_state(feed.Feed(root / "runner"), "ASmith91/stoa-feed", "tok")
    assert not any(c[0] in ("upload", "download") for c in fake.calls)


class Always:
    """A stand-in model that gives every stock the same p."""

    def __init__(self, p: float) -> None:
        self.p = p

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.p)


def test_the_picks_model_can_be_swapped_and_rolled_back(dell: tuple[Path, Dell]) -> None:
    import hashlib

    root, fake = dell
    fd = seeded(root)
    src = fake.sources()
    day = SEED
    for _ in range(2):
        day = clock.next_session(day)
        feed.run(fd, src, at(day, "09:00", 30))
    base, _ = feed.scored(fd, day)  # the feed state's own model
    path = root / "models" / "v2" / "model_h1.pkl"
    path.parent.mkdir(parents=True)
    blob = pickle.dumps(Always(0.9))
    path.write_bytes(blob)
    rows, sigs = feed.scored(fd, day, path, hashlib.sha256(blob).hexdigest())
    assert len(rows) and (rows["p"] == 0.9).all() and len(sigs) > 0
    with pytest.raises(feed.FeedError, match="sha256"):
        feed.scored(fd, day, path, "0" * 64)
    back, _ = feed.scored(fd, day)  # rolled back: the same p as before the swap
    assert back["p"].tolist() == base["p"].tolist()


def test_a_shadow_model_scores_the_picks_rows_and_locks_beside_them(
    dell: tuple[Path, Dell], monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from stoa import alerts, bigpaper, shadowboard
    from stoa.config import Secrets
    from tests.conftest import make_settings
    from tests.test_shadowboard import FakeHub as ScoreHub

    root, fake = dell
    fd = seeded(root)
    src = fake.sources()
    day = SEED
    for _ in range(2):
        day = clock.next_session(day)
        feed.run(fd, src, at(day, "09:00", 30))
    rows, _ = feed.scored(fd, day)  # what the picks scored, with the frozen model
    blob = pickle.dumps(Always(0.9))
    (root / "hub" / "models" / "v2").mkdir(parents=True)
    (root / "hub" / "models" / "v2" / "model_h1.pkl").write_bytes(blob)
    hub(monkeypatch, True, root / "hub")

    def runtime(name: str, pin: str) -> bigpaper.BigRuntime:
        base = make_settings(
            big_paper__enabled="true",
            big_paper__feed_dir=str(fd.root),
            big_paper__state_dir=str(root / name),
        )
        bp = base.big_paper.model_copy(update={"shadow_models": {"v2": pin}})
        rt = bigpaper.BigRuntime(
            base=base.model_copy(update={"big_paper": bp}),
            secrets=Secrets.from_env(
                {"ALPACA_BIG_PAPER_KEY_ID": "k", "ALPACA_BIG_PAPER_SECRET": "s", "HF_TOKEN": "t"}
            ),
            notifier=alerts.RecordingNotifier(),
        )
        rt.hub = ScoreHub(root / f"{name}-hub")
        rt.now = lambda: at(day, "09:05")
        return rt

    pin = hashlib.sha256(blob).hexdigest()
    rt = runtime("state", pin)
    assert bigpaper.shadow_lock(rt).status == "no_picks_lock"  # the picks locked nothing
    bigpaper.lock_scoreboard(rt, day, rows, bigpaper.FROZEN_VERSION, at(day, "09:03"))
    assert bigpaper.shadow_lock(rt).status == "locked"
    got = root / "state-hub" / shadowboard.folder("locked", day, "v2")
    locked = pd.read_parquet(got / "predictions.parquet")
    assert set(locked["symbol"]) == set(rows["symbol"]) and (locked["p"] == 0.9).all()
    assert set(locked["model_version"]) == {"v2"} and locked["above"].all()
    logged = shadowboard.logged_hashes(root / "state")
    assert set(logged) == {
        shadowboard.folder("locked", day),
        shadowboard.folder("locked", day, "v2"),
    }
    assert bigpaper.shadow_lock(rt).status == "already_locked"
    # The picks' own p must come back from the rows read again, to the last bit.
    off = runtime("state-off", pin)
    bigpaper.lock_scoreboard(
        off, day, rows.assign(p=rows["p"] + 1e-12), "t9-frozen", at(day, "09:03")
    )
    with pytest.raises(feed.FeedError, match="picks' own p"):
        bigpaper.shadow_lock(off)
    assert not (root / "state-off-hub" / "scoreboard" / "shadow").exists()
    # A model that is not the pinned one, or a lock after 9:30 am, is not locked.
    wrong = runtime("state-pin", "0" * 64)
    bigpaper.lock_scoreboard(wrong, day, rows, "t9-frozen", at(day, "09:03"))
    assert bigpaper.shadow_lock(wrong).status == "failed"
    late = runtime("state-late", pin)
    bigpaper.lock_scoreboard(late, day, rows, "t9-frozen", at(day, "09:03"))
    late.now = lambda: at(day, "09:30")
    assert bigpaper.shadow_lock(late).status == "failed"
    for name in ("state-pin", "state-late"):
        assert not (root / f"{name}-hub" / "scoreboard" / "shadow").exists()
