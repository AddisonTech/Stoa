from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pandas as pd
import pytest
import zstandard

from stoa import backup as bk


def write(root: Path, rel: str, data: bytes | str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode()
    p.write_bytes(data)
    return p


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("bars/wide/AAPL.parquet", None),
        ("filings/logs/pull.jsonl", None),
        ("filings/logs/pull.err", "log"),
        ("filings/logs/pids.txt", "log"),
        ("newsbrain/logs/feedpush.log", "log"),
        ("commod/logs/eia.jsonl", "log"),
        ("contracts_docs_0.out", "log"),
        ("dispatch.log", "log"),
        ("filings/__pycache__/x.cpython-312.pyc", "cache"),
        ("newsbrain/minute_c/2018-11/part-000.parquet", None),
        ("x/y.tmp", "temp"),
    ],
)
def test_skip_reason(path: str, reason: str | None) -> None:
    assert bk.skip_reason(path) == reason


def test_group_and_chunk() -> None:
    assert bk.group_of("a.json") == "_top"
    assert bk.group_of("bars/AAPL.parquet") == "bars"
    assert bk.group_of("newsbrain/minute_c/2018-11/p.parquet") == "newsbrain/minute_c"
    files = [(f"bars/wide/{i}.parquet", 40, 0) for i in range(5)] + [("a.json", 5, 0)]
    out = bk.chunk(files, max_bytes=100)
    assert [(g, len(f)) for g, f in out] == [
        ("_top", 1),
        ("bars/wide", 2),
        ("bars/wide", 2),
        ("bars/wide", 1),
    ]


def test_select_skips_and_refuses_state_branch(tmp_path: Path) -> None:
    write(tmp_path, "bars/wide/A.parquet", b"x")
    write(tmp_path, "filings/logs/pull.jsonl", "{}")
    write(tmp_path, "filings/logs/pull.err", "")
    sel = bk.select(tmp_path)
    assert [f for f, _, _ in sel.files] == ["bars/wide/A.parquet", "filings/logs/pull.jsonl"]
    assert [f for f, _ in sel.skipped["log"]] == ["filings/logs/pull.err"]
    write(tmp_path, "state-branch/shared/x.json", "{}")
    with pytest.raises(SystemExit):
        bk.select(tmp_path)


def test_scanner_finds_secrets_and_passes_clean_files(tmp_path: Path) -> None:
    fake_hf = "hf_" + "Ab3" * 12
    env_value = "S3cretValueFromDotEnv42"
    write(tmp_path, "clean.json", '{"symbol": "AAPL", "close": 1.5}')
    # an identifier that merely contains "hf_" inside a longer word is not a token
    write(tmp_path, "ids.json", '{"id": "Bxhf_' + "Ab3" * 12 + '"}')
    pd.DataFrame({"raw": ["<td id='Bxhf_" + "Ab3" * 12 + "'>"]}).to_parquet(
        tmp_path / "ids.parquet"
    )
    write(tmp_path, "hf.jsonl", '{"msg": "using ' + fake_hf + '"}')
    write(tmp_path, "assign.py", 'API_KEY = "abcdefghijklmnop1234"\n')
    write(tmp_path, "page.html", '<meta name="csrf-token" content="abcdefghijklmnopqrstuv">')
    write(tmp_path, "blob.pkl", b"\x80\x04" + env_value.encode() + b"\x00")
    pd.DataFrame({"text": ["plain words", "has " + env_value], "n": [1, 2]}).to_parquet(
        tmp_path / "t.parquet"
    )
    pd.DataFrame({"text": ["plain words"], "n": [1]}).to_parquet(tmp_path / "ok.parquet")
    write(tmp_path, "my_token.json", "{}")
    scanner = bk.Scanner([env_value])
    files = sorted(p.name for p in tmp_path.iterdir())
    hits = {(h.path, h.pattern) for h in bk.scan(tmp_path, files, scanner, workers=2)}
    assert hits == {
        ("hf.jsonl", "hugging face token"),
        ("assign.py", "key = value assignment"),
        ("blob.pkl", ".env value"),
        ("t.parquet", ".env value"),
        ("my_token.json", "secret-looking file name"),
    }


def test_secret_values_reads_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = write(tmp_path, ".env", "EIA_API_KEY=abcdef123456\nSHORT=x\n")
    monkeypatch.setenv("SOME_TOKEN", "zzzzzzzzzzzz")
    values = bk.secret_values(env)
    assert "abcdef123456" in values
    assert "zzzzzzzzzzzz" in values
    assert "x" not in values


def test_pack_plan_restore_round_trip(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write(data, "bars/wide/A.parquet", os.urandom(3000))
    write(data, "filings/raw/1.parquet", os.urandom(2000))
    write(data, "bars/wide/C.parquet", os.urandom(500))
    write(data, "notes.md", "hello")
    stage = tmp_path / "stage"

    sel = bk.select(data)
    p1 = bk.plan(sel.files, bk.Manifest())
    assert len(p1.pack) == 4
    entries, shards = bk.pack(data, p1.pack, stage, "r1", say=lambda _: None)
    m1 = bk.Manifest(files=entries, shards=shards, created="r1")
    m1 = bk.Manifest.from_json(m1.to_json())
    assert set(m1.shards) == m1.live_shards()

    # change one file, delete one, add one
    write(data, "bars/wide/A.parquet", os.urandom(3100))
    (data / "notes.md").unlink()
    write(data, "bars/wide/B.parquet", os.urandom(10))
    sel2 = bk.select(data)
    p2 = bk.plan(sel2.files, m1)
    assert sorted(f for f, _, _ in p2.pack) == ["bars/wide/A.parquet", "bars/wide/B.parquet"]
    assert p2.deleted == ["notes.md"]
    assert sorted(p2.keep) == ["bars/wide/C.parquet", "filings/raw/1.parquet"]
    e2, s2 = bk.pack(data, p2.pack, stage, "r2", say=lambda _: None)
    m2 = bk.Manifest(files={**p2.keep, **e2}, created="r2")
    live = m2.live_shards()
    m2.shards = {k: v for k, v in {**m1.shards, **s2}.items() if k in live}
    assert sorted(set(m1.shards) - live) == ["shards/_top/r1-000.tar.zst"]

    dest = tmp_path / "restored"
    ok = 0
    for shard in sorted(live):
        res = bk.extract_shard(stage / shard, shard, m2, dest, m2.shards[shard]["sha256"])
        assert res.bad == []
        assert res.shard_ok
        ok += res.ok
    assert ok == 4
    # the older copy of A in r1's bars/wide shard was not used
    assert (dest / "bars/wide/A.parquet").read_bytes() == (
        data / "bars/wide/A.parquet"
    ).read_bytes()
    assert bk.verify_tree(dest, m2) == []
    assert bk.count_files(dest) == 4
    assert (dest / "bars/wide/A.parquet").stat().st_mtime_ns == int(
        m2.files["bars/wide/A.parquet"]["mtime_ns"]
    )


def test_extract_rejects_bad_hash_and_unsafe_paths(tmp_path: Path) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, body in (("../evil.txt", b"x"), ("ok.txt", b"hello")):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    shard = tmp_path / "s.tar.zst"
    shard.write_bytes(zstandard.ZstdCompressor().compress(raw.getvalue()))
    m = bk.Manifest(
        files={
            "../evil.txt": {"size": 1, "sha256": "0", "mtime_ns": 0, "shard": "s"},
            "ok.txt": {"size": 5, "sha256": "f" * 64, "mtime_ns": 0, "shard": "s"},
        }
    )
    dest = tmp_path / "dest"
    res = bk.extract_shard(shard, "s", m, dest)
    assert res.ok == 0
    assert sorted(why for _, why in res.bad) == [
        "size or SHA-256 differs from the manifest",
        "unsafe path in archive",
    ]
    assert not (tmp_path / "evil.txt").exists()
    assert not (dest / "ok.txt").exists()
