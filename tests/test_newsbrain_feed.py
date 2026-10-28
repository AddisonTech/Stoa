"""The big paper account's news brain feed: pulls, the 9:00 am cutoff, stale data, splits.

The sources are fakes reading the small world of ``test_newsbrain_rolling``; the feed's
picks must equal those from trial 9's full rebuild of the same world, bit for bit.
"""

from __future__ import annotations

import json
import pickle
import shutil
from datetime import UTC, date, datetime, timedelta
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

    def api(self, token: str | None = None) -> FakeHub:
        self.calls.append(("api", token))
        return self

    def create_repo(self, repo: str, **kw: object) -> None:
        self.calls.append(("create", repo, kw.get("private")))

    def repo_info(self, repo: str, **kw: object) -> object:
        return type("Info", (), {"private": self.private})()

    def upload_folder(self, **kw: object) -> None:
        self.calls.append(("upload", kw["repo_id"], kw["ignore_patterns"]))
        self.calls.append(("upload_only", kw.get("allow_patterns")))
        shutil.copytree(Path(str(kw["folder_path"])), self.store, dirs_exist_ok=True)

    def snapshot_download(self, **kw: object) -> None:
        self.calls.append(("download", kw["repo_id"], kw["token"]))
        self.calls.append(("download_only", kw.get("allow_patterns")))
        shutil.copytree(self.store, Path(str(kw["local_dir"])), dirs_exist_ok=True)


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
