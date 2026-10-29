"""The shadow scoreboard (stoa/shadowboard.py): the lock, the hash, the buckets, the
grader's trade rule, a shadow model beside the picks' model, and the whole round trip
against a fake dataset."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, date, datetime, timedelta
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from stoa import clock, shadowboard, wide

SESSION = date(2026, 10, 29)  # a Thursday
EXIT = date(2026, 10, 30)


def ny(d: date, hhmm: str) -> datetime:
    return clock.at_ny(d, hhmm).astimezone(UTC)


def scored() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "p": [0.40, 0.52, 0.57, 0.66],
            "adv": [5e7, 5e7, 2e8, 1e9],
        }
    )


class FakeHub:
    """The huggingface_hub calls the scoreboard makes, on a local folder, with the server's
    commit time set by the test."""

    def __init__(self, store: Path) -> None:
        self.store = store
        self.when: dict[str, datetime] = {}
        self.clock = ny(SESSION, "09:03")

    def file_exists(self, repo: str, path: str, **kw: Any) -> bool:
        return (self.store / path).exists()

    def upload_folder(self, **kw: Any) -> None:
        src = Path(kw["folder_path"])
        for f in src.rglob("*"):
            rel = f.relative_to(src).as_posix()
            if f.is_file() and any(fnmatch(rel, p) for p in kw["allow_patterns"]):
                (self.store / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, self.store / rel)
                self.when[rel] = self.clock

    def list_repo_tree(self, repo: str, path_in_repo: str, **kw: Any) -> list[Any]:
        base = self.store / path_in_repo
        if not base.exists():
            raise FileNotFoundError(path_in_repo)
        return [type("F", (), {"path": f"{path_in_repo}/{p.name}"})() for p in base.iterdir()]

    def get_paths_info(self, repo: str, paths: list[str], **kw: Any) -> list[Any]:
        commit = type("C", (), {"date": self.when[paths[0]]})()
        return [type("I", (), {"last_commit": commit})()]

    def hf_hub_download(self, repo: str, path: str, **kw: Any) -> str:
        out = Path(kw["local_dir"]) / path
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.store / path, out)
        return str(out)


def prices_945(book: dict[date, dict[str, float]], calls: list[date]) -> shadowboard.Prices:
    def at(day: date, symbols: set[str]) -> dict[str, float]:
        calls.append(day)
        return {s: v for s, v in book[day].items() if s in symbols}

    return at


BOOK = {
    SESSION: {"SPY": 500.0, "AAA": 10.0, "BBB": 20.0, "CCC": 30.0, "DDD": 40.0},
    EXIT: {"SPY": 505.0, "AAA": 10.0, "BBB": 20.6, "CCC": 30.9, "DDD": 40.0},
}


# ---- the lock ------------------------------------------------------------------------------


def test_the_lock_refuses_a_late_file_and_a_second_lock(tmp_path: Path) -> None:
    api = FakeHub(tmp_path / "hub")
    df = shadowboard.predictions(scored(), "v1")
    state, work = tmp_path / "state", tmp_path / "work"
    with pytest.raises(shadowboard.LockRefused, match="9:30"):
        shadowboard.lock(api, "r", state, work, SESSION, df, ny(SESSION, "09:30"))
    assert not (tmp_path / "hub").exists() and not shadowboard.locks_path(state).exists()
    meta = shadowboard.lock(api, "r", state, work, SESSION, df, ny(SESSION, "09:04"))
    assert meta["rows"] == 4 and meta["above"] == 2 and meta["kind"] == "locked"
    assert shadowboard.logged_hashes(state) == {
        shadowboard.folder("locked", SESSION): meta["sha256"]
    }
    with pytest.raises(shadowboard.LockRefused, match="already locked"):
        shadowboard.lock(api, "r", state, work, SESSION, df, ny(SESSION, "09:05"))


def test_the_grader_refuses_a_file_written_or_uploaded_after_the_open(tmp_path: Path) -> None:
    df = shadowboard.predictions(scored(), "v1")
    meta = shadowboard.write(tmp_path, "locked", SESSION, df, ny(SESSION, "09:31"))
    blob = (tmp_path / shadowboard.folder("locked", SESSION) / "predictions.parquet").read_bytes()
    why = shadowboard.problems(meta, blob, meta["sha256"], ny(SESSION, "09:05"))
    assert any("not before 9:30" in w for w in why)
    meta = shadowboard.write(tmp_path, "locked", SESSION, df, ny(SESSION, "09:05"))
    assert shadowboard.problems(meta, blob, meta["sha256"], ny(SESSION, "09:06")) == []
    late = shadowboard.problems(meta, blob, meta["sha256"], ny(SESSION, "09:30"))
    assert late and "uploaded at" in late[0]
    assert shadowboard.problems(meta, blob, meta["sha256"], None) == [
        "its upload time could not be read"
    ]


# ---- the hash ------------------------------------------------------------------------------


def test_the_hash_must_match_the_file_and_the_state_branch(tmp_path: Path) -> None:
    df = shadowboard.predictions(scored(), "v1")
    meta = shadowboard.write(tmp_path, "locked", SESSION, df, ny(SESSION, "09:05"))
    path = tmp_path / shadowboard.folder("locked", SESSION) / "predictions.parquet"
    blob = path.read_bytes()
    up = ny(SESSION, "09:06")
    assert meta["sha256"] == shadowboard.sha256(blob)
    assert shadowboard.problems(meta, blob, meta["sha256"], up) == []
    edited = shadowboard.to_bytes(df.assign(p=df["p"].where(df["symbol"] != "AAA", 0.9)))
    assert "does not match" in shadowboard.problems(meta, edited, meta["sha256"], up)[0]
    assert "differs" in shadowboard.problems(meta, blob, "0" * 64, up)[0]
    assert "no hash" in shadowboard.problems(meta, blob, None, up)[0]
    back = shadowboard.write(tmp_path, "backfill", SESSION, df, ny(SESSION, "20:00"))
    assert shadowboard.problems(back, blob, None, None) == []  # no lock to check


# ---- the buckets ---------------------------------------------------------------------------


def test_the_buckets_add_up_by_day_and_in_total() -> None:
    p = [0.49, 0.50, 0.549, 0.55, 0.5999, 0.60, 0.6499, 0.65, 0.91]
    hit = [False, True, False, True, True, False, True, True, False]
    rows = {b["bucket"]: b for b in shadowboard.buckets(p, hit)}
    assert {k: v["n"] for k, v in rows.items()} == {
        "under 0.50": 1,
        "0.50-0.55": 2,
        "0.55-0.60": 2,
        "0.60-0.65": 2,
        "0.65+": 2,
    }
    assert rows["0.50-0.55"]["hits"] == 1 and rows["0.65+"]["hits"] == 1
    assert rows["0.55-0.60"]["p_sum"] == pytest.approx(0.55 + 0.5999)
    day = dict.fromkeys(shadowboard.SUMS, 0) | {
        "session": "2026-10-29",
        "buckets": list(rows.values()),
    }
    day.update(above_graded=5, above_hits=3, above_vs_spy_sum=0.05, top_n=4, top_vs_spy_sum=0.02)
    two = shadowboard.totals([day, {**day, "session": "2026-10-30"}])
    assert two["days"] == 2 and two["above_graded"] == 10 and two["above_hits"] == 6
    assert two["hit_rate"] == pytest.approx(0.6)
    assert two["above_avg_vs_spy"] == pytest.approx(0.01)
    assert two["top_avg_vs_spy"] == pytest.approx(0.005)
    cal = {c["bucket"]: c for c in two["calibration"]}
    assert cal["0.60-0.65"]["n"] == 4 and cal["0.60-0.65"]["hit_rate"] == pytest.approx(0.5)
    assert cal["0.60-0.65"]["mean_p"] == pytest.approx((0.60 + 0.6499) / 2)
    assert shadowboard.totals([])["hit_rate"] is None


# ---- the grader ----------------------------------------------------------------------------


def test_the_grader_uses_the_trades_945_to_945_rule() -> None:
    df = shadowboard.predictions(scored(), "v1")
    day, g = shadowboard.grade(df, BOOK[SESSION], BOOK[EXIT])
    spy = 505 / 500 - 1
    r = g.set_index("symbol")
    assert r.loc["CCC", "r"] == pytest.approx(0.03) and r.loc["CCC", "vs_spy"] == pytest.approx(
        0.03 - spy
    )
    cost = 2 * wide.slippage_bps(5e7) / 1e4 + shadowboard.FEE
    assert r.loc["BBB", "net"] == pytest.approx(0.03 - spy - cost)
    assert bool(r.loc["CCC", "hit"]) and not bool(r.loc["DDD", "hit"])  # flat lost to SPY
    assert day["above"] == 2 and day["above_hits"] == 1  # CCC and DDD are above 0.55
    assert day["above_vs_spy_sum"] == pytest.approx((0.03 - spy) + (0.0 - spy))
    assert day["top_n"] == 4
    with pytest.raises(shadowboard.GradeError, match="SPY"):
        shadowboard.grade(df, {"AAA": 1.0}, {"AAA": 1.0})


def test_the_945_price_is_the_first_minute_bar_from_945_to_1015() -> None:
    seen: list[dict[str, Any]] = []

    class Resp:
        status_code = 200

        def __init__(self, body: dict[str, Any]) -> None:
            self.body = body

        def json(self) -> dict[str, Any]:
            return self.body

    class Client:
        def request(self, method: str, url: str, **kw: Any) -> Resp:
            seen.append(kw["params"])
            return Resp({"bars": {"AAA": [{"o": 10.5}, {"o": 11.0}], "BBB": []}})

    got = shadowboard.alpaca_945(Client())(SESSION, {"AAA", "BBB"})
    assert got == {"AAA": 10.5}
    q = seen[0]
    assert q["timeframe"] == "1Min" and q["adjustment"] == "all" and q["feed"] == "sip"
    assert q["start"] == "2026-10-29T13:45:00Z" and q["end"] == "2026-10-29T14:15:00Z"


def test_lock_then_grade_the_next_morning(tmp_path: Path) -> None:
    api = FakeHub(tmp_path / "hub")
    state, work = tmp_path / "state", tmp_path / "work"
    df = shadowboard.predictions(scored(), "v1")
    shadowboard.lock(api, "r", state, work, SESSION, df, ny(SESSION, "09:03"))
    calls: list[date] = []
    prices = prices_945(BOOK, calls)
    # Before the exit price exists, nothing is graded.
    assert shadowboard.grade_pending(api, "r", state, work, prices, ny(SESSION, "17:15")) == []
    got = shadowboard.grade_pending(api, "r", state, work, prices, ny(EXIT, "17:15"))
    assert [d["session"] for d in got] == [SESSION.isoformat()] and "refused" not in got[0]
    assert calls == [SESSION, EXIT]  # in at 9:45 am of the session, out at 9:45 am the next
    board = shadowboard.read_board(state)
    assert board is not None and [m["version"] for m in board["models"]] == ["v1"]
    col = board["models"][0]
    assert col["role"] == "picks" and col["locked"]["days"] == 1 and col["backfill"]["days"] == 0
    assert col["locked"]["scored"] == 4 and col["locked"]["above"] == 2
    line = shadowboard.shadow_line(shadowboard.latest_locked(state))
    assert line.startswith("Shadow: 4 scored, 2 above line, above-line avg vs SPY ")
    assert line.endswith("% (2026-10-29)")
    # Graded once only.
    assert shadowboard.grade_pending(api, "r", state, work, prices, ny(EXIT, "17:20")) == []


def test_a_tampered_or_late_lock_is_refused_and_backfill_kept_apart(tmp_path: Path) -> None:
    api = FakeHub(tmp_path / "hub")
    state, work = tmp_path / "state", tmp_path / "work"
    df = shadowboard.predictions(scored(), "v1")
    shadowboard.lock(api, "r", state, work, SESSION, df, ny(SESSION, "09:03"))
    # Someone replaces the file in the dataset after the open.
    path = tmp_path / "hub" / shadowboard.folder("locked", SESSION) / "predictions.parquet"
    path.write_bytes(shadowboard.to_bytes(df.assign(p=0.99)))
    api.when[f"{shadowboard.folder('locked', SESSION)}/predictions.parquet"] = ny(SESSION, "11:00")
    # A backfilled morning, written after the fact.
    prev = clock.previous_session(SESSION)
    shadowboard.write(work, "backfill", prev, df, ny(SESSION, "20:00"))
    api.clock = ny(SESSION, "20:00")
    api.upload_folder(folder_path=work, allow_patterns=["scoreboard/backfill/**"])
    book = {**BOOK, prev: BOOK[SESSION]}
    got = shadowboard.grade_pending(api, "r", state, work, prices_945(book, []), ny(EXIT, "17:15"))
    by = {(d["kind"], d["session"]): d for d in got}
    bad = by[("locked", SESSION.isoformat())]["refused"]
    assert any("does not match" in w for w in bad) and any("uploaded at" in w for w in bad)
    assert "refused" not in by[("backfill", prev.isoformat())]
    board = shadowboard.read_board(state)
    assert board is not None
    col = board["models"][0]
    assert col["locked"]["days"] == 0 and col["backfill"]["days"] == 1
    assert board["refused"][0]["session"] == SESSION.isoformat()
    assert shadowboard.latest_locked(state) is None
    assert shadowboard.shadow_line(None, "first locked day 2026-10-29 is graded later.") == (
        "Shadow: first locked day 2026-10-29 is graded later."
    )


# ---- a shadow model -------------------------------------------------------------------------


def shadow_scored() -> pd.DataFrame:
    return scored().assign(p=[0.60, 0.61, 0.30, 0.30])  # AAA and BBB above the line


def test_a_shadow_model_is_locked_and_graded_in_its_own_column(tmp_path: Path) -> None:
    api = FakeHub(tmp_path / "hub")
    state, work = tmp_path / "state", tmp_path / "work"
    picks = shadowboard.predictions(scored(), "v1")
    shadow = shadowboard.predictions(shadow_scored(), "v2")
    shadowboard.lock(api, "r", state, work, SESSION, picks, ny(SESSION, "09:03"))
    meta = shadowboard.lock(api, "r", state, work, SESSION, shadow, ny(SESSION, "09:04"), "v2")
    path = shadowboard.folder("locked", SESSION, "v2")
    assert path == "scoreboard/shadow/v2/locked/2026-10-29"
    assert shadowboard.logged_hashes(state)[path] == meta["sha256"] and meta["kind"] == "locked"
    assert shadowboard.locked_sessions(state) == [SESSION]
    assert shadowboard.locked_sessions(state, "v2") == [SESSION]
    with pytest.raises(shadowboard.LockRefused, match="already locked"):
        shadowboard.lock(api, "r", state, work, SESSION, shadow, ny(SESSION, "09:05"), "v2")
    with pytest.raises(shadowboard.LockRefused, match="9:30"):
        shadowboard.lock(api, "r", state, work, SESSION, shadow, ny(SESSION, "09:30"), "v3")
    # A file in one model's folder that names another model is never counted for it.
    shadowboard.lock(api, "r", state, work, SESSION, picks, ny(SESSION, "09:06"), "v4")
    lineup = [("v1", "picks"), ("v2", "shadow"), ("v4", "shadow")]
    got = shadowboard.grade_pending(
        api, "r", state, work, prices_945(BOOK, []), ny(EXIT, "17:15"), lineup
    )
    assert sorted(d["path"] for d in got) == [
        "scoreboard/locked/2026-10-29",
        "scoreboard/shadow/v2/locked/2026-10-29",
        "scoreboard/shadow/v4/locked/2026-10-29",
    ]
    board = shadowboard.read_board(state)
    assert board is not None
    cols = {m["version"]: m for m in board["models"]}
    assert [m["version"] for m in board["models"]] == ["v1", "v2", "v4"]
    assert cols["v1"]["role"] == "picks" and cols["v2"]["role"] == "shadow"
    assert cols["v1"]["locked"]["above"] == 2 and cols["v1"]["locked"]["above_hits"] == 1
    v2 = cols["v2"]["locked"]
    assert v2["days"] == 1 and v2["scored"] == 4 and v2["above"] == 2 and v2["above_hits"] == 1
    spy = 505 / 500 - 1
    assert v2["above_avg_vs_spy"] == pytest.approx(((0.0 - spy) + (0.03 - spy)) / 2)
    assert cols["v4"]["locked"]["days"] == 0
    assert "its model is v1, not v4" in board["refused"][0]["why"]
    assert shadowboard.latest_locked(state)["model_version"] == "v1"  # type: ignore[index]
    line = shadowboard.shadow_line(shadowboard.latest_locked(state, "v2"), shadow="v2")
    assert line.startswith("Shadow v2: 4 scored, 2 above line, above-line avg vs SPY +0.50%")
    assert ", top 10 vs SPY " in line and line.endswith("(2026-10-29)")
    # Graded once only, in each column.
    assert (
        shadowboard.grade_pending(api, "r", state, work, prices_945(BOOK, []), ny(EXIT, "18:00"))
        == []
    )


def test_the_card_shows_the_running_numbers(tmp_path: Path) -> None:
    from stoa import dashboard

    api = FakeHub(tmp_path / "hub")
    state, work = tmp_path / "state", tmp_path / "work"
    shadowboard.lock(
        api, "r", state, work, SESSION, shadowboard.predictions(scored(), "v1"), ny(SESSION, "9:03")
    )
    shadowboard.lock(
        api,
        "r",
        state,
        work,
        SESSION,
        shadowboard.predictions(shadow_scored(), "v2"),
        ny(SESSION, "9:04"),
        "v2",
    )
    lineup = [("v1", "picks"), ("v2", "shadow"), ("v3", "shadow")]
    shadowboard.grade_pending(
        api, "r", state, work, prices_945(BOOK, []), ny(EXIT, "17:15"), lineup
    )
    html = dashboard.render_shadow({"shadow_board": shadowboard.read_board(state)})
    assert "v1 (picks model)" in html and "v2 (shadow, never trades)" in html
    assert "2026-10-29 (1 mornings)" in html and "Above the 0.55 line" in html
    assert "waiting" in html  # v3 has no graded morning yet
    assert "0.55-0.60" in html and "0.65+" in html and "Calibration" in html
    assert "No locked morning" in dashboard.render_shadow({"shadow_board": None})
    old = {"line": 0.55, "locked": {"days": 0}, "backfill": {"days": 0}}  # before the columns
    assert "No locked morning" in dashboard.render_shadow({"shadow_board": old})
    assert json.loads(json.dumps(shadowboard.read_board(state)))  # plain JSON for the page
    assert timedelta(0) <= ny(EXIT, "10:15") - clock.at_ny(EXIT, "10:15")
