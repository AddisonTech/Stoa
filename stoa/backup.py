"""Back up the git-ignored ``data/`` folder to a private Hugging Face dataset repo and restore it.

``scripts/data_backup.py`` and ``scripts/data_restore.py`` are thin wrappers around this module.

Layout of the dataset repo:

- ``manifest.json``: every backed-up file's path (relative to ``data/``, forward slashes),
  size, SHA-256, modification time and the shard that holds it, plus every shard's size and
  SHA-256.
- ``shards/<group>/<run>-<n>.tar.zst``: tar archives compressed with zstd, each under
  ``MAX_SHARD_BYTES`` of input. A group is the first two levels of the folder (for example
  ``newsbrain/minute_c``), so the folder layout is kept inside every tar.

Files go in byte for byte. Most of the folder is already Parquet with zstd (the minute bars,
the filings), and re-encoding it would change every hash, so tabular files are not rewritten.

A backup only packs files that are new or whose size or modification time changed since the
manifest. Their new shards go up first, then the new manifest, then shards nothing points at
any more are deleted. A shard can hold an older copy of a file that has since moved to a newer
shard; a restore takes each file only from the shard the manifest names.

Nothing that looks like a secret leaves the machine: before packing, every file name and every
file's content (text, Parquet string columns, pickles, zip members) is checked against key
and token patterns and against the values in ``.env``. Any match stops the run and prints the
path and the pattern's name, never the matched text.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import time
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO, Any, BinaryIO, cast

import zstandard

from stoa import envfile

REPO_ID = "ASmith91/stoa-data"
REPO_TYPE = "dataset"
MANIFEST = "manifest.json"
FORMAT = 1
MAX_SHARD_BYTES = 4 * 1024**3  # input bytes per shard; packed shards stay well under 10 GB
MAX_FILE_BYTES = 9 * 1024**3  # one file this large would break the 10 GB shard limit
CHUNK = 8 * 1024 * 1024

# ---------------------------------------------------------------------------------------------
# What goes up
# ---------------------------------------------------------------------------------------------

SKIP_DIRS = {"__pycache__", ".cache", ".ipynb_checkpoints", "tmp", "temp"}
SKIP_SUFFIXES = {
    ".out",
    ".err",
    ".log",
    ".tmp",
    ".temp",
    ".part",
    ".partial",
    ".lock",
    ".pyc",
    ".swp",
    ".restore-tmp",
}
SKIP_NAMES = {"pids.txt", "thumbs.db", "desktop.ini", ".ds_store"}
# Progress logs the filings pipeline resumes from (pull.jsonl is the EDGAR pull log).
KEEP_LOGS = ("filings/logs/",)
KEEP_LOG_SUFFIX = ".jsonl"
# Never inside a backup, whatever folder they turn up in.
FORBIDDEN_PARTS = {"state-branch", ".git"}
ALREADY_COMPRESSED = {".parquet", ".zip", ".npz", ".gz", ".zst", ".7z", ".png", ".jpg"}


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def skip_reason(relpath: str) -> str | None:
    """Why a file under ``data/`` stays out of the backup, or None if it goes up."""
    parts = relpath.split("/")
    name = parts[-1].lower()
    dirs = [p.lower() for p in parts[:-1]]
    if any(d in SKIP_DIRS for d in dirs):
        return "cache"
    if "logs" in dirs:
        if relpath.startswith(KEEP_LOGS) and name.endswith(KEEP_LOG_SUFFIX):
            return None
        return "log"
    if name in SKIP_NAMES:
        return "temp"
    suffix = Path(name).suffix
    if suffix in {".out", ".err", ".log"}:
        return "log"
    if suffix in SKIP_SUFFIXES:
        return "temp"
    return None


@dataclass
class Selection:
    files: list[tuple[str, int, int]] = field(default_factory=list)  # path, size, mtime_ns
    skipped: dict[str, list[tuple[str, int]]] = field(default_factory=dict)  # reason -> files


def select(root: Path) -> Selection:
    """Every regular file under ``root`` with the reason it is skipped, if it is."""
    out = Selection()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        here = Path(dirpath)
        for name in sorted(filenames):
            p = here / name
            if p.is_symlink() or not p.is_file():
                continue
            r = rel(p, root)
            if FORBIDDEN_PARTS & set(r.split("/")):
                raise SystemExit(f"refusing: {r} is inside a forbidden folder")
            st = p.stat()
            reason = skip_reason(r)
            if reason is None:
                out.files.append((r, st.st_size, st.st_mtime_ns))
            else:
                out.skipped.setdefault(reason, []).append((r, st.st_size))
    return out


# ---------------------------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------------------------

# Token shapes that are never data. Each is specific enough not to fire on filing text.
STRICT_PATTERNS: dict[str, str] = {
    "hugging face token": r"\bhf_[A-Za-z0-9]{30,}",
    "anthropic key": r"sk-ant-[A-Za-z0-9_\-]{20,}",
    "openai-style key": r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}",
    "aws access key": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "github token": r"\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,})",
    "alpaca key id": r"\b(?:PK|AK)[A-Z0-9]{18}\b",
    "private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "slack token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    "ntfy token": r"\btk_[a-z0-9]{29}\b",
}
# key = value assignments in our own text files (logs, json, scripts, notes).
ASSIGN_PATTERN = (
    r"(?i)(?:api[_-]?key|secret|token|passw(?:or)?d|authorization)[\"']?\s*[:=]\s*"
    r"[\"']?(?:bearer\s+)?[A-Za-z0-9_\-/+=.]{16,}"
)
SECRET_NAME = re.compile(
    r"(?i)(^\.env(\..*)?$|\.pem$|\.key$|\.p12$|\.pfx$|^id_(rsa|ed25519|ecdsa)|"
    r"secret|token|credential|password|\.netrc$)"
)
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".py",
    ".ps1",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".xml",
}
# Fetched web pages: checked for token shapes and .env values, not for key = value text,
# since public pages carry their own form tokens.
PAGE_SUFFIXES = {".html", ".htm"}
SECRET_ENV_NAME = re.compile(r"(?i)(key|secret|token|passw|pat$|agent)")


def secret_values(env_path: Path) -> list[str]:
    """Values that must never appear in an upload: all of ``.env`` plus secret-named env vars."""
    values: set[str] = set()
    if env_path.exists():
        values.update(v for v in envfile.parse(env_path.read_text(encoding="utf-8")).values())
    for k, v in os.environ.items():
        if SECRET_ENV_NAME.search(k):
            values.add(v)
    return sorted(v for v in values if len(v) >= 8)


@dataclass(frozen=True)
class Hit:
    path: str
    pattern: str


class Scanner:
    def __init__(self, values: Iterable[str]) -> None:
        self.values = [v for v in values if v]
        self.strict = {k: re.compile(v.encode()) for k, v in STRICT_PATTERNS.items()}
        self.assign = re.compile(ASSIGN_PATTERN.encode())
        self.value_bytes = [v.encode() for v in self.values]

    def scan_bytes(self, data: bytes, assignments: bool) -> list[str]:
        found = [name for name, rx in self.strict.items() if rx.search(data)]
        if assignments and self.assign.search(data):
            found.append("key = value assignment")
        if any(v in data for v in self.value_bytes):
            found.append(".env value")
        return found

    def scan_file(self, path: Path, relpath: str) -> list[Hit]:
        hits: list[Hit] = []
        if SECRET_NAME.search(path.name):
            hits.append(Hit(relpath, "secret-looking file name"))
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            found = self._scan_parquet(path)
        elif suffix == ".zip":
            found = self._scan_zip(path)
        else:
            found = self._scan_stream(path, assignments=suffix in TEXT_SUFFIXES)
        hits.extend(Hit(relpath, f) for f in sorted(set(found)))
        return hits

    def _scan_stream(self, path: Path, assignments: bool) -> list[str]:
        found: set[str] = set()
        overlap = 4096
        tail = b""
        with path.open("rb") as fh:
            while block := fh.read(64 * 1024 * 1024):
                found.update(self.scan_bytes(tail + block, assignments))
                tail = block[-overlap:]
        return sorted(found)

    def _scan_zip(self, path: Path) -> list[str]:
        found: set[str] = set()
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if SECRET_NAME.search(PurePosixPath(info.filename).name):
                    found.add("secret-looking file name in zip")
                with zf.open(info) as fh:
                    tail = b""
                    while block := fh.read(64 * 1024 * 1024):
                        found.update(self.scan_bytes(tail + block, assignments=False))
                        tail = block[-4096:]
        return sorted(found)

    def _scan_parquet(self, path: Path) -> list[str]:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        found: set[str] = set()
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        meta = b" ".join(k + b" " + v for k, v in (schema.metadata or {}).items() if k != b"pandas")
        found.update(self.scan_bytes(" ".join(schema.names).encode() + b" " + meta, False))
        cols = [
            f.name
            for f in schema
            if pa.types.is_string(f.type)
            or pa.types.is_large_string(f.type)
            or pa.types.is_binary(f.type)
            or pa.types.is_large_binary(f.type)
            or (pa.types.is_dictionary(f.type) and pa.types.is_string(f.type.value_type))
        ]
        if not cols:
            return sorted(found)
        strict = dict(STRICT_PATTERNS)
        for batch in pf.iter_batches(batch_size=20_000, columns=cols):
            for col in batch.columns:
                arr = col
                if pa.types.is_dictionary(arr.type):
                    arr = arr.dictionary
                if pa.types.is_binary(arr.type) or pa.types.is_large_binary(arr.type):
                    arr = pc.cast(arr, pa.large_string(), safe=False)
                for name, rx in strict.items():
                    if name not in found and pc.any(pc.match_substring_regex(arr, rx)).as_py():
                        found.add(name)
                for v in self.values:
                    if ".env value" not in found and pc.any(pc.match_substring(arr, v)).as_py():
                        found.add(".env value")
        return sorted(found)


def scan(
    root: Path,
    files: list[str],
    scanner: Scanner,
    workers: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> list[Hit]:
    hits: list[Hit] = []
    with cf.ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(scanner.scan_file, root / f, f): f for f in files}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            hits.extend(fut.result())
            if progress:
                progress(i, len(files))
    return sorted(hits, key=lambda h: (h.path, h.pattern))


# ---------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------


@dataclass
class Manifest:
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    shards: dict[str, dict[str, Any]] = field(default_factory=dict)
    created: str = ""

    def to_json(self) -> str:
        body = {
            "format": FORMAT,
            "created": self.created,
            "file_count": len(self.files),
            "total_bytes": sum(int(e["size"]) for e in self.files.values()),
            "shards": {k: self.shards[k] for k in sorted(self.shards)},
            "files": [{"path": p, **self.files[p]} for p in sorted(self.files)],
        }
        return json.dumps(body, indent=0, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        body = json.loads(text)
        if body.get("format") != FORMAT:
            raise SystemExit(f"unknown manifest format {body.get('format')}")
        files = {e["path"]: {k: v for k, v in e.items() if k != "path"} for e in body["files"]}
        return cls(files=files, shards=dict(body["shards"]), created=body.get("created", ""))

    def live_shards(self) -> set[str]:
        return {str(e["shard"]) for e in self.files.values()}


@dataclass
class Plan:
    keep: dict[str, dict[str, Any]]
    pack: list[tuple[str, int, int]]
    deleted: list[str]


def plan(selected: list[tuple[str, int, int]], prev: Manifest, full: bool = False) -> Plan:
    keep: dict[str, dict[str, Any]] = {}
    pack: list[tuple[str, int, int]] = []
    for path, size, mtime in selected:
        old = prev.files.get(path)
        if not full and old and int(old["size"]) == size and int(old["mtime_ns"]) == mtime:
            keep[path] = old
        else:
            pack.append((path, size, mtime))
    names = {p for p, _, _ in selected}
    return Plan(keep=keep, pack=pack, deleted=sorted(set(prev.files) - names))


def group_of(relpath: str) -> str:
    parts = relpath.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else (parts[0] if len(parts) == 2 else "_top")


def chunk(
    files: list[tuple[str, int, int]], max_bytes: int = MAX_SHARD_BYTES
) -> list[tuple[str, list[tuple[str, int, int]]]]:
    """Files grouped by folder, then cut into runs of at most ``max_bytes``."""
    by_group: dict[str, list[tuple[str, int, int]]] = {}
    for f in sorted(files):
        by_group.setdefault(group_of(f[0]), []).append(f)
    out: list[tuple[str, list[tuple[str, int, int]]]] = []
    for g in sorted(by_group):
        cur: list[tuple[str, int, int]] = []
        total = 0
        for f in by_group[g]:
            if f[1] > MAX_FILE_BYTES:
                raise SystemExit(f"{f[0]} is {f[1] / 1e9:.1f} GB, over the shard limit")
            if cur and total + f[1] > max_bytes:
                out.append((g, cur))
                cur, total = [], 0
            cur.append(f)
            total += f[1]
        if cur:
            out.append((g, cur))
    return out


# ---------------------------------------------------------------------------------------------
# Packing and unpacking
# ---------------------------------------------------------------------------------------------


class _HashReader(io.RawIOBase):
    def __init__(self, fh: BinaryIO) -> None:
        self.fh = fh
        self.sha = hashlib.sha256()
        self.n = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        b = self.fh.read(size)
        self.sha.update(b)
        self.n += len(b)
        return b


class _HashWriter(io.RawIOBase):
    def __init__(self, fh: BinaryIO) -> None:
        self.fh = fh
        self.sha = hashlib.sha256()
        self.n = 0

    def writable(self) -> bool:
        return True

    def write(self, b: Any) -> int:
        data = bytes(b)
        self.fh.write(data)
        self.sha.update(data)
        self.n += len(data)
        return len(data)

    def flush(self) -> None:
        self.fh.flush()


def level_for(files: list[tuple[str, int, int]]) -> int:
    """Fast zstd for already-compressed files, a stronger level for text."""
    packed = sum(s for p, s, _ in files if Path(p).suffix.lower() in ALREADY_COMPRESSED)
    return 1 if packed * 2 > sum(s for _, s, _ in files) else 9


def shard_name(group: str, run_id: str, n: int) -> str:
    return f"shards/{group.replace('/', '__')}/{run_id}-{n:03d}.tar.zst"


def pack_shard(
    root: Path, files: list[tuple[str, int, int]], out: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Write one tar.zst; return the manifest entries (shard left blank) and the shard's info."""
    out.parent.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}
    cctx = zstandard.ZstdCompressor(level=level_for(files), threads=-1)
    with out.open("wb") as raw:
        hw = _HashWriter(raw)
        with (
            cctx.stream_writer(cast(IO[bytes], hw), closefd=False) as zw,
            tarfile.open(fileobj=zw, mode="w|", format=tarfile.PAX_FORMAT) as tar,
        ):
            for path, _, _ in files:
                src = root / path
                st = src.stat()
                info = tarfile.TarInfo(path)
                info.size = st.st_size
                info.mtime = int(st.st_mtime)
                info.mode = 0o644
                with src.open("rb") as fh:
                    hr = _HashReader(fh)
                    tar.addfile(info, hr)
                entries[path] = {
                    "size": st.st_size,
                    "sha256": hr.sha.hexdigest(),
                    "mtime_ns": st.st_mtime_ns,
                }
    return entries, {"bytes": hw.n, "sha256": hw.sha.hexdigest(), "files": len(files)}


def pack(
    root: Path,
    todo: list[tuple[str, int, int]],
    stage: Path,
    run_id: str,
    max_bytes: int = MAX_SHARD_BYTES,
    say: Callable[[str], None] = print,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    entries: dict[str, dict[str, Any]] = {}
    shards: dict[str, dict[str, Any]] = {}
    counters: dict[str, int] = {}
    for group, files in chunk(todo, max_bytes):
        n = counters.get(group, 0)
        counters[group] = n + 1
        name = shard_name(group, run_id, n)
        t0 = time.monotonic()
        got, info = pack_shard(root, files, stage / name)
        for e in got.values():
            e["shard"] = name
        entries.update(got)
        shards[name] = info
        raw = sum(s for _, s, _ in files)
        say(
            f"  packed {name}: {len(files)} files, {raw / 1e9:.2f} GB in, "
            f"{info['bytes'] / 1e9:.2f} GB out, {time.monotonic() - t0:.0f}s"
        )
    return entries, shards


def safe_member(name: str) -> bool:
    if not name or "\\" in name or name.startswith("/"):
        return False
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts and ":" not in p.parts[0]


@dataclass
class Restored:
    ok: int = 0
    bad: list[tuple[str, str]] = field(default_factory=list)
    shard_ok: bool = True


def extract_shard(
    shard_path: Path, shard: str, manifest: Manifest, dest: Path, expect_sha: str | None = None
) -> Restored:
    """Unpack the files the manifest assigns to ``shard`` into ``dest`` and check each hash."""
    res = Restored()
    dctx = zstandard.ZstdDecompressor()
    dest_root = dest.resolve()
    with shard_path.open("rb") as raw:
        hr = _HashReader(raw)
        with (
            dctx.stream_reader(cast(IO[bytes], hr), closefd=False) as zr,
            tarfile.open(fileobj=zr, mode="r|") as tar,
        ):
            for m in tar:
                if not m.isfile():
                    continue
                name = m.name
                entry = manifest.files.get(name)
                if entry is None or entry["shard"] != shard:
                    continue  # an older copy, or not in the backup any more
                if not safe_member(name):
                    res.bad.append((name, "unsafe path in archive"))
                    continue
                target = (dest / name).resolve()
                if dest_root not in target.parents:
                    res.bad.append((name, "path escapes the destination"))
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".restore-tmp")
                sha = hashlib.sha256()
                n = 0
                src = tar.extractfile(m)
                if src is None:
                    res.bad.append((name, "unreadable member"))
                    continue
                with tmp.open("wb") as out:
                    while block := src.read(CHUNK):
                        sha.update(block)
                        n += len(block)
                        out.write(block)
                if n != int(entry["size"]) or sha.hexdigest() != entry["sha256"]:
                    tmp.unlink()
                    res.bad.append((name, "size or SHA-256 differs from the manifest"))
                    continue
                os.replace(tmp, target)
                mt = int(entry["mtime_ns"])
                os.utime(target, ns=(mt, mt))
                res.ok += 1
            # drain the rest so the shard's hash covers every byte
        while hr.read(CHUNK):
            pass
    if expect_sha is not None and hr.sha.hexdigest() != expect_sha:
        res.shard_ok = False
    return res


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def verify_tree(dest: Path, manifest: Manifest, workers: int = 8) -> list[tuple[str, str]]:
    """Every manifest file under ``dest`` with the right size and SHA-256; returns problems."""

    def one(path: str) -> tuple[str, str] | None:
        p = dest / path
        e = manifest.files[path]
        if not p.is_file():
            return (path, "missing")
        if p.stat().st_size != int(e["size"]):
            return (path, "size differs")
        if sha256_file(p) != e["sha256"]:
            return (path, "SHA-256 differs")
        return None

    with cf.ThreadPoolExecutor(workers) as pool:
        out = [r for r in pool.map(one, sorted(manifest.files)) if r is not None]
    return out


def count_files(root: Path) -> int:
    return sum(len(f) for _, _, f in os.walk(root))


# ---------------------------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------------------------

TOKEN_HELP = """No Hugging Face token on this machine. To add one (free):
  1. Open https://huggingface.co/settings/tokens while signed in as ASmith91.
  2. Create token > Fine-grained (or Write). Give it write access to your own repos
     (Repositories: Read and Write). Copy it.
  3. In this repo run:  .venv\\Scripts\\hf.exe auth login
     and paste the token (answer n to adding it as a git credential).
     It is saved under %USERPROFILE%\\.cache\\huggingface\\token, outside the repo.
Then run this script again."""


def hf_api() -> Any:
    from huggingface_hub import HfApi, get_token

    if not get_token():
        raise SystemExit(TOKEN_HELP)
    return HfApi()


def ensure_private_repo(api: Any, repo_id: str) -> None:
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)
    info = api.repo_info(repo_id, repo_type=REPO_TYPE)
    if info.private is not True:
        raise SystemExit(f"{repo_id} is not private; nothing uploaded")


def remote_manifest(api: Any, repo_id: str, cache: Path) -> tuple[Manifest, str]:
    """The repo's manifest and its SHA-256 ("" if the repo or manifest does not exist yet)."""
    from huggingface_hub import hf_hub_download

    if not api.repo_exists(repo_id, repo_type=REPO_TYPE):
        return Manifest(), ""
    if not api.file_exists(repo_id, MANIFEST, repo_type=REPO_TYPE):
        return Manifest(), ""
    cache.mkdir(parents=True, exist_ok=True)
    p = Path(
        hf_hub_download(
            repo_id, MANIFEST, repo_type=REPO_TYPE, local_dir=cache, force_download=True
        )
    )
    body = p.read_bytes()
    return Manifest.from_json(body.decode("utf-8")), hashlib.sha256(body).hexdigest()


def download(repo_id: str, filename: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename, repo_type=REPO_TYPE, local_dir=cache))


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1000:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1000
    return f"{n:.2f} TB"


def iter_lines(rows: Iterable[tuple[str, ...]]) -> Iterator[str]:
    rows = list(rows)
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        yield "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()


def rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
