"""Restore data/ from the private Hugging Face dataset repo on a fresh clone.

    python scripts/data_restore.py                  # into data/
    python scripts/data_restore.py --dest D:/check  # somewhere else, for a test
    python scripts/data_restore.py --verify         # re-hash every file after unpacking

Downloads manifest.json, then each shard in turn: unpacks the files the manifest assigns to
it, checks every file's size and SHA-256 against the manifest, sets its modification time,
and deletes the downloaded shard before the next one. Exits 1 if any file is missing or
differs. Needs a Hugging Face token with read access (see docs/RUNBOOK.md, "Moving to a new
machine").
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stoa import backup as bk  # noqa: E402

DEFAULT_CACHE = Path(tempfile.gettempdir()) / "stoa-data-restore"


def say(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=bk.REPO_ID)
    ap.add_argument("--dest", type=Path, default=ROOT / "data")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--force", action="store_true", help="restore into a folder with files")
    ap.add_argument("--verify", action="store_true", help="re-hash every file at the end")
    ap.add_argument("--keep-downloads", action="store_true")
    args = ap.parse_args()
    dest: Path = args.dest
    cache: Path = args.cache

    api = bk.hf_api()
    manifest, _ = bk.remote_manifest(api, args.repo, cache)
    if not manifest.files:
        raise SystemExit(f"no manifest in {args.repo}; nothing to restore")
    if dest.exists() and any(dest.iterdir()) and not args.force:
        raise SystemExit(f"{dest} already has files; pass --force to restore over them")
    dest.mkdir(parents=True, exist_ok=True)

    shards = sorted(manifest.live_shards())
    total = sum(int(manifest.shards[s]["bytes"]) for s in shards)
    say(
        f"{args.repo}: {len(manifest.files)} files, "
        f"{bk.human(sum(int(e['size']) for e in manifest.files.values()))} unpacked, "
        f"{len(shards)} shards, {bk.human(total)} to download"
    )
    ok = 0
    bad: list[tuple[str, str]] = []
    bad_shards: list[str] = []
    t0 = time.monotonic()
    for i, shard in enumerate(shards, 1):
        path = bk.download(args.repo, shard, cache)
        res = bk.extract_shard(path, shard, manifest, dest, str(manifest.shards[shard]["sha256"]))
        ok += res.ok
        bad.extend(res.bad)
        if not res.shard_ok:
            bad_shards.append(shard)
        if not args.keep_downloads:
            path.unlink()
        say(f"  [{i}/{len(shards)}] {shard}: {res.ok} files ({time.monotonic() - t0:.0f}s)")

    restored = {p for p in manifest.files if (dest / p).is_file()}
    missing = sorted(set(manifest.files) - restored)
    if args.verify:
        say("re-hashing every restored file")
        bad.extend(bk.verify_tree(dest, manifest))
    if not args.keep_downloads:
        bk.rmtree(cache)

    say(f"restored and verified: {ok} of {len(manifest.files)} files")
    say(f"files now under {dest}: {bk.count_files(dest)}")
    for s in bad_shards:
        say(f"  shard hash differs: {s}")
    for p, why in sorted(set(bad)):
        say(f"  {p}: {why}")
    for p in missing[:50]:
        say(f"  missing: {p}")
    if bad or missing or bad_shards or ok != len(manifest.files):
        say("RESTORE FAILED")
        return 1
    say("restore OK: every SHA-256 matches the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
