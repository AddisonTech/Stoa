"""Back up the git-ignored data/ folder to the private Hugging Face dataset repo.

    python scripts/data_backup.py              # scan, pack what changed, upload
    python scripts/data_backup.py --plan       # sizes and what would go up; writes nothing
    python scripts/data_backup.py --pack-only  # scan and pack into the stage folder only
    python scripts/data_backup.py --full       # repack every file; old shards are deleted

Only files that are new, or whose size or modification time changed since the repo's
manifest.json, are packed and uploaded. Packed shards wait in the stage folder until they are
up, so an interrupted run picks up where it stopped (upload_large_folder resumes too). Every
file to upload is checked for keys and tokens first; any match stops the run. If more than
10% of the backed-up files are gone from data/, the run stops unless --allow-deletes is given.
See stoa/backup.py for the format and docs/RUNBOOK.md, "Moving to a new machine".
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stoa import backup as bk  # noqa: E402

DATA = ROOT / "data"
ENV = ROOT / ".env"
DEFAULT_STAGE = Path(tempfile.gettempdir()) / "stoa-data-backup"
README = """---
license: other
---
Private backup of a git-ignored research data folder. Restore with scripts/data_restore.py
from the project repository; manifest.json lists every file with its size and SHA-256.
"""


def say(msg: str) -> None:
    print(msg, flush=True)


def table(sel: bk.Selection) -> None:
    sizes: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for path, size, _ in sel.files:
        g = sizes[bk.group_of(path)]
        g[0] += 1
        g[1] += size
    for files in sel.skipped.values():
        for path, size in files:
            g = sizes[bk.group_of(path)]
            g[2] += 1
            g[3] += size
    rows = [("folder", "files up", "size up", "files skipped", "size skipped")]
    for name in sorted(sizes, key=lambda k: -sizes[k][1]):
        n, s, ns, ss = sizes[name]
        rows.append((name, str(n), bk.human(s), str(ns), bk.human(ss)))
    total_up = sum(s for _, s, _ in sel.files)
    total_skip = sum(s for v in sel.skipped.values() for _, s in v)
    rows.append(
        (
            "total",
            str(len(sel.files)),
            bk.human(total_up),
            str(sum(len(v) for v in sel.skipped.values())),
            bk.human(total_skip),
        )
    )
    for line in bk.iter_lines(rows):
        say(line)
    for reason, files in sorted(sel.skipped.items()):
        say(f"skipped as {reason}: {len(files)} files, {bk.human(sum(s for _, s in files))}")


def upload_pending(api: Any, repo: str, stage: Path) -> bk.Manifest:
    """Put the packed shards up, then the manifest, then delete shards nothing uses."""
    pending = json.loads((stage / "pending.json").read_text(encoding="utf-8"))
    upload = stage / "upload"
    bk.ensure_private_repo(api, repo)
    say(f"{repo} is private; uploading shards from {upload}")
    readme = upload / "README.md"
    if not readme.exists():
        readme.write_text(README, encoding="utf-8")
    if hasattr(api, "upload_large_folder"):  # huggingface_hub 0.x and 1.x
        api.upload_large_folder(
            repo_id=repo, folder_path=str(upload), repo_type=bk.REPO_TYPE, private=True
        )
    else:
        # In huggingface_hub 2.x upload_large_folder became part of upload_folder: it commits
        # in batches and a rerun after an interruption skips what is already up.
        api.upload_folder(
            repo_id=repo,
            folder_path=str(upload),
            repo_type=bk.REPO_TYPE,
            commit_message=f"Shards {pending['run_id']}",
        )
    api.upload_file(
        path_or_fileobj=str(stage / "pending_manifest.json"),
        path_in_repo=bk.MANIFEST,
        repo_id=repo,
        repo_type=bk.REPO_TYPE,
        commit_message=f"Manifest {pending['run_id']}",
    )
    stale = list(pending["stale"])
    if stale:
        api.delete_files(
            repo_id=repo,
            delete_patterns=stale,
            repo_type=bk.REPO_TYPE,
            commit_message=f"Drop {len(stale)} shards replaced by {pending['run_id']}",
        )
    manifest_text = (stage / "pending_manifest.json").read_text(encoding="utf-8")
    (stage / "manifest.json").write_text(manifest_text, encoding="utf-8")
    bk.rmtree(upload)
    (stage / "pending_manifest.json").unlink()
    (stage / "pending.json").unlink()
    say(f"uploaded run {pending['run_id']}; removed {len(stale)} stale shards")
    return bk.Manifest.from_json(manifest_text)


def report_storage(api: Any, repo: str) -> None:
    owner = repo.split("/", maxsplit=1)[0]
    try:
        repos = [("dataset", r.id) for r in api.list_datasets(author=owner)]
        repos += [("model", r.id) for r in api.list_models(author=owner)]
        repos += [("space", r.id) for r in api.list_spaces(author=owner)]
        used = 0
        for kind, rid in repos:
            # the list endpoints do not return usedStorage; repo_info does
            info = api.repo_info(rid, repo_type=kind, expand=["usedStorage", "private"])
            n = int(getattr(info, "used_storage", 0) or 0)
            used += n
            say(f"  {rid}: {bk.human(n)}{'' if info.private else ' (PUBLIC)'}")
        say(f"private storage used: {bk.human(used)} of 100 GB")
    except Exception as exc:  # reporting only; the backup itself is done
        say(f"could not read storage use: {type(exc).__name__}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=bk.REPO_ID)
    ap.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    ap.add_argument("--plan", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--pack-only", action="store_true", help="pack but do not upload")
    ap.add_argument("--full", action="store_true", help="repack every file")
    ap.add_argument("--max-shard-gb", type=float, default=bk.MAX_SHARD_BYTES / 1024**3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--allow-deletes",
        action="store_true",
        help="go ahead even if more than 10%% of the backed-up files are gone from data/",
    )
    args = ap.parse_args()
    stage: Path = args.stage
    if not DATA.is_dir():
        raise SystemExit(f"no data folder at {DATA}")

    sel = bk.select(DATA)
    table(sel)
    if args.plan:
        return 0

    api = None if args.pack_only else bk.hf_api()
    cache = stage / "remote"
    if api is not None:
        prev, base = bk.remote_manifest(api, args.repo, cache)
    elif (stage / "manifest.json").exists():
        prev = bk.Manifest.from_json((stage / "manifest.json").read_text(encoding="utf-8"))
        base = "local"
    else:
        prev, base = bk.Manifest(), ""

    if (stage / "pending.json").exists():
        pending = json.loads((stage / "pending.json").read_text(encoding="utf-8"))
        if api is None:
            say(f"run {pending['run_id']} is packed in {stage} and waiting to be uploaded")
            return 0
        if pending["base"] not in (base, "local"):
            raise SystemExit(
                f"the repo's manifest changed since run {pending['run_id']} was packed; "
                f"delete {stage} and run again"
            )
        prev = upload_pending(api, args.repo, stage)
        report_storage(api, args.repo)

    p = bk.plan(sel.files, prev, full=args.full)
    say(
        f"plan: {len(p.pack)} files to pack ({bk.human(sum(s for _, s, _ in p.pack))}), "
        f"{len(p.keep)} unchanged, {len(p.deleted)} gone since the last backup"
    )
    if prev.files and len(p.deleted) > 0.1 * len(prev.files) and not args.allow_deletes:
        # data/ was probably moved, emptied after a restore elsewhere, or not restored yet;
        # going on would drop those files and their shards from the backup
        raise SystemExit(
            f"STOP: {len(p.deleted)} of {len(prev.files)} backed-up files are gone from "
            f"{DATA}. Restore first (scripts/data_restore.py), or rerun with --allow-deletes "
            "if they were removed on purpose."
        )
    if not p.pack and not p.deleted:
        say("nothing changed since the last backup")
        return 0

    scanner = bk.Scanner(bk.secret_values(ENV))
    t0 = time.monotonic()

    def progress(i: int, n: int) -> None:
        if i % 2000 == 0 or i == n:
            say(f"  scanned {i}/{n} files ({time.monotonic() - t0:.0f}s)")

    hits = bk.scan(DATA, [f for f, _, _ in p.pack], scanner, args.workers, progress)
    if hits:
        say("STOP: possible secrets in the upload list (path: pattern). Nothing was packed.")
        for h in hits:
            say(f"  {h.path}: {h.pattern}")
        return 2
    say("secret scan clean")

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    upload = stage / "upload"
    bk.rmtree(upload)
    entries, shards = bk.pack(
        DATA, p.pack, upload, run_id, int(args.max_shard_gb * 1024**3), say=say
    )
    files = {**p.keep, **entries}
    new = bk.Manifest(files=files, created=run_id)
    live = new.live_shards()
    new.shards = {k: v for k, v in {**prev.shards, **shards}.items() if k in live}
    stale = sorted(set(prev.shards) - live)
    (stage / "pending_manifest.json").write_text(new.to_json(), encoding="utf-8")
    (stage / "pending.json").write_text(
        json.dumps({"run_id": run_id, "base": base, "stale": stale}), encoding="utf-8"
    )
    packed = sum(int(v["bytes"]) for v in shards.values())
    say(
        f"packed run {run_id}: {len(entries)} files into {len(shards)} shards, "
        f"{bk.human(packed)}; manifest lists {len(files)} files"
    )
    if api is None:
        say(f"pack only: shards wait in {upload}; run without --pack-only to upload")
        return 0
    upload_pending(api, args.repo, stage)
    report_storage(api, args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
