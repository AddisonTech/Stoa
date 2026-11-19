"""Run the trial 17 data pipeline on a shared machine without starving it of memory.

Keeps free physical memory above FLOOR_GB: below it the parse/scores job is stopped first
(each filer's file is written whole, so a stopped parse loses nothing), then the pull. Work
restarts only when free memory is back above RESUME_GB. Parse workers scale with free
memory, at most MAX_WORKERS.

The pull runs until every ever-liquid filer (data/filings/liquid_symbols.json) has been
through it; then it is stopped, a last parse and scores run, the ticker map is built, and
a 'liquid_ready' line is logged. Logs to data/filings/logs/supervise.jsonl.
"""

import ctypes
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("data/filings")
LOGS = ROOT / "logs"
LOG = LOGS / "supervise.jsonl"
PY = sys.executable
FLOOR_GB = 6.2  # act below this (owner's line is 6 GB)
RESUME_GB = 7.5
MAX_WORKERS = 4
CYCLE_S = 600
TICK_S = 10


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def free_gb() -> float:
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30


def note(**kw: object) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        row = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "free_gb": round(free_gb(), 2),
        }
        fh.write(json.dumps({**row, **kw}) + "\n")


def kill(p: subprocess.Popen) -> None:
    subprocess.call(
        ["taskkill", "/T", "/F", "/PID", str(p.pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    p.wait()


def start(*args: str, name: str) -> subprocess.Popen:
    out = (LOGS / f"{name}.out").open("a", encoding="utf-8")
    err = (LOGS / f"{name}.err").open("a", encoding="utf-8")
    return subprocess.Popen(
        [PY, "-m", "stoa.filings", *args],
        stdout=out,
        stderr=err,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def pull_last() -> str:
    lines = (LOGS / "pull.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return lines[-1] if lines else ""


def needed() -> set[int]:
    from stoa.filings import pull

    liquid = json.loads((ROOT / "liquid_symbols.json").read_text(encoding="utf-8"))
    book = pull.filers()
    return {c for c, syms in book.items() if any(s in liquid for s in syms)}


def pulled() -> set[int]:
    seen: set[int] = set()
    for line in (LOGS / "pull.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if "cik" in rec and "error" not in rec:
            seen.add(int(rec["cik"]))
    return seen


def workers_for(free: float) -> int:
    if free >= 11:
        return MAX_WORKERS
    if free >= 9.5:
        return 3
    if free >= 8.5:
        return 2
    return 1


def main() -> None:
    need = needed()
    note(event="start", liquid_filers=len(need), pulled=len(need & pulled()))
    pull: subprocess.Popen | None = None
    pull_state = "run"  # run | waiting_memory | done | stopped_by_edgar
    job: subprocess.Popen | None = None
    job_name = ""
    queue: list[str] = []
    next_cycle = 0.0
    covered = False
    crashes = 0
    last_report = 0.0
    while True:
        free = free_gb()
        # 1. memory guard: stop parse/scores first, then the pull
        if free < FLOOR_GB and job is not None:
            kill(job)
            note(event="stopped_for_memory", job=job_name)
            job, queue = None, []
            next_cycle = time.time() + 120
        elif free < FLOOR_GB and pull is not None:
            kill(pull)
            pull = None
            pull_state = "waiting_memory"
            note(event="pull_paused_for_memory")
        # 2. the pull
        if pull is not None and pull.poll() is not None:
            rc = pull.returncode
            pull = None
            last = pull_last()
            if '"finished": true' in last:
                pull_state = "done"
            elif '"stopped"' in last:
                pull_state = "stopped_by_edgar"
            else:
                crashes += 1
                pull_state = "run" if crashes <= 3 else "crashed"
            note(event="pull_exit", rc=rc, state=pull_state)
        if not covered and need <= pulled():
            covered = True
            if pull is not None:
                kill(pull)
                pull = None
            pull_state = "done"
            note(event="liquid_covered", pulled=len(need))
            queue, next_cycle = [], 0.0
        if pull is None and pull_state in ("run", "waiting_memory") and free >= RESUME_GB:
            pull = start("pull", name="pull")
            pull_state = "run"
            note(event="pull_started", pid=pull.pid)
        # 3. parse, scores, then (once covered) tickers
        if job is not None and job.poll() is not None:
            note(event="job_done", job=job_name, rc=job.returncode)
            if job.returncode != 0:
                queue = []
            elif job_name == "tickers":
                note(event="liquid_ready")
                return
            job = None
        if job is None and not queue and time.time() >= next_cycle and free >= RESUME_GB:
            queue = ["parse", "scores"] + (["tickers"] if covered else [])
            next_cycle = time.time() + CYCLE_S
        if job is None and queue and free >= RESUME_GB:
            job_name = queue.pop(0)
            args = [job_name] if job_name == "tickers" else [job_name, str(workers_for(free))]
            job = start(*args, name=job_name)
            note(event="job_started", job=job_name, args=args[1:])
        if time.time() - last_report > 900:
            last_report = time.time()
            note(
                event="status",
                pull=pull_state,
                pulled=len(need & pulled()),
                of=len(need),
                job=job_name if job is not None else "",
            )
        time.sleep(TICK_S)


if __name__ == "__main__":
    main()
