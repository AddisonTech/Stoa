"""The shadow scoreboard: every stock the big account's news brain scores each morning,
locked before the open and graded after it, buys or not.

Each trading morning the 9:00 am picks job writes the full list it scored (symbol, p,
whether p cleared the 0.55 buy line, the model version and the dollar volume the cost
model needs) to the private feed dataset, under ``scoreboard/locked/<session>/``, with
the UTC time it was written and the file's sha256. The hash also goes into
``scoreboard/locks.jsonl`` in the account's state folder, which the workflow commits to
the stoa-state branch. A locked file is never replaced, and none is written at or after
9:30 am New York time.

The grader refuses any file written or uploaded at or after 9:30 am of its session, or
whose bytes do not match the logged hash. It grades each locked session once its exit
price exists: the trades' own rule, in at the 9:45 am price of the session and out at
the 9:45 am price of the next (the open of the first minute bar from 9:45 to 10:15, as
the trials took it), against SPY over the same span. A hit is the model's own label:
the stock beat SPY after a round trip of slippage and the fee. So the 5:15 pm summary
grades the morning before, and the first locked morning is graded a day later.

Backfilled days (``scoreboard/backfill/``) are scored after the fact, so they carry no
lock; they are graded the same way and kept in their own totals, never mixed with the
locked days.

Shadow models (``big_paper.shadow_models``) are scored beside the picks' model and never
trade: right after the picks, the same run scores the same rows with each of them and locks
the list under ``scoreboard/shadow/<version>/locked/<session>/``, with the same deadline, hash
and log. They are graded the same way, and the board keeps one column per model version,
each with its own totals.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stoa import clock, wide

LINE = 0.55  # the buy line
TOP = 10
SPY = "SPY"
FEE = 0.3e-4  # trial 9's fee per round trip, beside the slippage model (panel.FEE)
LOCK_AT = "09:30"  # nothing written at or after the open counts
PRICE_AT, PRICE_UNTIL = "09:45", "10:15"
PREFIX = "scoreboard"
KINDS = ("locked", "backfill")
SHADOW = "shadow"  # scoreboard/shadow/<version>/: a model scored beside the picks', never traded
# The calibration table: (label, low, high), p in [low, high).
BUCKETS = (
    ("under 0.50", 0.0, 0.50),
    ("0.50-0.55", 0.50, 0.55),
    ("0.55-0.60", 0.55, 0.60),
    ("0.60-0.65", 0.60, 0.65),
    ("0.65+", 0.65, 1.01),
)
COLUMNS = ["symbol", "p", "above", "model_version", "adv"]

Prices = Callable[[date, set[str]], Mapping[str, float]]


class LockRefused(RuntimeError):
    """A prediction file that may not be locked: too late, or already locked."""


class GradeError(RuntimeError):
    """A session that cannot be graded yet, or at all."""


# ---- the file -----------------------------------------------------------------------------


def lock_deadline(session: date) -> datetime:
    return clock.at_ny(session, LOCK_AT)


def predictions(rows: pd.DataFrame, version: str, extra: Iterable[str] = ()) -> pd.DataFrame:
    """Every scored stock, best first: symbol, p, above the line, model version, adv."""
    out = pd.DataFrame(
        {
            "symbol": rows["symbol"].astype(str).to_numpy(),
            "p": rows["p"].astype(float).to_numpy(),
            "model_version": version,
            "adv": rows["adv"].astype(float).to_numpy(),
        }
    )
    out["above"] = out["p"] >= LINE
    for c in extra:
        out[c] = rows[c].to_numpy()
    out = out.sort_values(["p", "symbol"], ascending=[False, True], kind="stable")
    return out[[*COLUMNS, *extra]].reset_index(drop=True)


def to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def lane(kind: str, shadow: str | None = None) -> str:
    """The folder of one kind of day: the picks' model's, or a shadow model's."""
    return f"{PREFIX}/{kind}" if shadow is None else f"{PREFIX}/{SHADOW}/{shadow}/{kind}"


def folder(kind: str, session: date, shadow: str | None = None) -> str:
    return f"{lane(kind, shadow)}/{session.isoformat()}"


def write(
    root: Path,
    kind: str,
    session: date,
    df: pd.DataFrame,
    now: datetime,
    shadow: str | None = None,
) -> dict[str, Any]:
    """The file and its meta.json under ``root/scoreboard/<kind>/<session>/`` (or a shadow
    model's own folder)."""
    out = root / folder(kind, session, shadow)
    out.mkdir(parents=True, exist_ok=True)
    blob = to_bytes(df)
    (out / "predictions.parquet").write_bytes(blob)
    versions = sorted(set(df["model_version"])) if len(df) else []
    meta = {
        "session": session.isoformat(),
        "kind": kind,
        "model_version": versions[0] if len(versions) == 1 else versions,
        "written_at": now.astimezone(UTC).isoformat(),
        "rows": len(df),
        "above": int(df["above"].sum()) if len(df) else 0,
        "sha256": sha256(blob),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta


def problems(
    meta: Mapping[str, Any], blob: bytes, logged: str | None, uploaded_at: datetime | None
) -> list[str]:
    """Why a prediction file may not be graded; empty when it may. A backfill is checked
    against its own hash only."""
    out = []
    if sha256(blob) != meta.get("sha256"):
        out.append("the file does not match the hash written beside it")
    if meta.get("kind") != "locked":
        return out
    session = date.fromisoformat(str(meta["session"]))
    deadline = lock_deadline(session)
    if logged is None:
        out.append("no hash for it on the state branch")
    elif logged != meta.get("sha256"):
        out.append("its hash differs from the one logged on the state branch")
    written = datetime.fromisoformat(str(meta["written_at"]))
    if written >= deadline:
        out.append(f"written at {written.isoformat()}, not before 9:30 am")
    if uploaded_at is None:
        out.append("its upload time could not be read")
    elif uploaded_at >= deadline:
        out.append(f"uploaded at {uploaded_at.isoformat()}, not before 9:30 am")
    return out


# ---- grading ------------------------------------------------------------------------------


def bucket_of(p: float) -> str:
    for label, lo, hi in BUCKETS:
        if lo <= p < hi:
            return label
    return BUCKETS[-1][0]


def buckets(p: Iterable[float], hit: Iterable[bool]) -> list[dict[str, Any]]:
    """Per bucket: count, sum of p and hits, so days add up."""
    rows: dict[str, dict[str, Any]] = {
        label: {"bucket": label, "n": 0, "p_sum": 0.0, "hits": 0} for label, _, _ in BUCKETS
    }
    for pi, hi in zip(p, hit, strict=True):
        b = rows[bucket_of(float(pi))]
        b["n"] += 1
        b["p_sum"] += float(pi)
        b["hits"] += int(bool(hi))
    return list(rows.values())


def returns(
    entry: Mapping[str, float], exit_: Mapping[str, float], symbols: Iterable[str]
) -> pd.Series:
    out = {}
    for s in symbols:
        a, b = entry.get(s), exit_.get(s)
        ok = a is not None and b is not None and math.isfinite(a) and math.isfinite(b) and a > 0
        out[s] = b / a - 1 if ok else math.nan  # type: ignore[operator]
    return pd.Series(out, dtype=float)


def grade(
    preds: pd.DataFrame, entry: Mapping[str, float], exit_: Mapping[str, float]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """One session: each stock's 9:45 to 9:45 return against SPY's, its hit after costs,
    and the day's sums."""
    spy = returns(entry, exit_, [SPY]).iloc[0]
    if not math.isfinite(spy):
        raise GradeError("no SPY price at 9:45 am on both days")
    g = preds.copy()
    g["r"] = returns(entry, exit_, g["symbol"]).reindex(g["symbol"]).to_numpy()
    g["vs_spy"] = g["r"] - spy
    slip = [wide.slippage_bps(v if math.isfinite(v) else None) / 1e4 for v in g["adv"]]
    g["net"] = g["vs_spy"] - 2 * pd.Series(slip, index=g.index) - FEE
    g["hit"] = g["net"] > 0
    ok = g[g["r"].notna()]
    above = ok[ok["above"]]
    top = ok.sort_values(["p", "symbol"], ascending=[False, True], kind="stable").head(TOP)
    day = {
        "scored": len(g),
        "graded": len(ok),
        "above": int(g["above"].sum()),
        "above_graded": len(above),
        "above_hits": int(above["hit"].sum()),
        "above_vs_spy_sum": float(above["vs_spy"].sum()),
        "top_n": len(top),
        "top_vs_spy_sum": float(top["vs_spy"].sum()),
        "hits": int(ok["hit"].sum()),
        "spy": float(spy),
        "buckets": buckets(ok["p"], ok["hit"]),
    }
    return day, g


def summarize(day: Mapping[str, Any]) -> dict[str, Any]:
    """Rates and averages from one day's sums, or from the running totals."""

    def ratio(a: float, b: float) -> float | None:
        return a / b if b else None

    return {
        "hit_rate": ratio(day["above_hits"], day["above_graded"]),
        "all_hit_rate": ratio(day["hits"], day["graded"]),
        "above_avg_vs_spy": ratio(day["above_vs_spy_sum"], day["above_graded"]),
        "top_avg_vs_spy": ratio(day["top_vs_spy_sum"], day["top_n"]),
        "calibration": [
            {
                "bucket": b["bucket"],
                "n": b["n"],
                "mean_p": ratio(b["p_sum"], b["n"]),
                "hit_rate": ratio(b["hits"], b["n"]),
            }
            for b in day["buckets"]
        ],
    }


SUMS = (
    "scored",
    "graded",
    "above",
    "above_graded",
    "above_hits",
    "above_vs_spy_sum",
    "top_n",
    "top_vs_spy_sum",
    "hits",
)


def totals(days: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The running total since day 1: the days' sums added up, then the rates."""
    out: dict[str, Any] = dict.fromkeys(SUMS, 0)
    out["days"] = len(days)
    out["first"] = min((d["session"] for d in days), default=None)
    out["last"] = max((d["session"] for d in days), default=None)
    rows: dict[str, dict[str, Any]] = {
        label: {"bucket": label, "n": 0, "p_sum": 0.0, "hits": 0} for label, _, _ in BUCKETS
    }
    for d in days:
        for k in SUMS:
            out[k] += d[k]
        for b in d["buckets"]:
            r = rows[b["bucket"]]
            r["n"] += b["n"]
            r["p_sum"] += b["p_sum"]
            r["hits"] += b["hits"]
    out["buckets"] = list(rows.values())
    return {**out, **summarize(out)}


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def shadow_line(
    day: Mapping[str, Any] | None, waiting: str | None = None, shadow: str | None = None
) -> str:
    """The 5:15 pm summary's line for the latest locked day graded; a shadow model's line
    names it and adds its top 10 against SPY."""
    head = "Shadow:" if shadow is None else f"Shadow {shadow}:"
    if day is None:
        return f"{head} " + (waiting or "no locked day graded yet.")
    got = summarize(day)
    top = "" if shadow is None else f", top 10 vs SPY {pct(got['top_avg_vs_spy'])}"
    return (
        f"{head} {day['scored']} scored, {day['above']} above line, "
        f"above-line avg vs SPY {pct(got['above_avg_vs_spy'])}{top} ({day['session']})"
    )


# ---- 9:45 am prices -----------------------------------------------------------------------


def alpaca_945(client: Any) -> Prices:
    """The open of each symbol's first minute bar from 9:45 to 10:15 am, adjusted, from the
    SIP feed: the price the trials and the trades take."""
    from stoa.net import request_with_retry

    def at(day: date, symbols: set[str]) -> dict[str, float]:
        start, end = clock.at_ny(day, PRICE_AT), clock.at_ny(day, PRICE_UNTIL)
        got: dict[str, float] = {}
        names = sorted(symbols)
        for i in range(0, len(names), 100):
            params: dict[str, Any] = {
                "symbols": ",".join(names[i : i + 100]),
                "timeframe": "1Min",
                "start": start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit": 10000,
                "adjustment": "all",
                "feed": "sip",
                "sort": "asc",
            }
            while True:
                resp = request_with_retry(
                    client, "GET", "/v2/stocks/bars", params=params, max_retries=4
                )
                if resp.status_code != 200:
                    raise GradeError(f"9:45 am prices: HTTP {resp.status_code}")
                body = resp.json()
                for sym, rows in (body.get("bars") or {}).items():
                    if rows and sym not in got:
                        got[sym] = float(rows[0]["o"])
                token = body.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
        return got

    return at


# ---- the dataset and the state branch -----------------------------------------------------


def locks_path(state: Path) -> Path:
    return state / PREFIX / "locks.jsonl"


def logged_hashes(state: Path) -> dict[str, str]:
    """Each locked file's hash, by its folder in the dataset."""
    path = locks_path(state)
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            where = rec.get("path") or folder("locked", date.fromisoformat(rec["session"]))
            out.setdefault(where, rec["sha256"])  # the first lock of a folder stands
    return out


def locked_sessions(state: Path, shadow: str | None = None) -> list[date]:
    """The sessions locked for the picks' model, or for one shadow model."""
    root = lane("locked", shadow) + "/"
    return sorted(
        date.fromisoformat(p.removeprefix(root)) for p in logged_hashes(state) if p.startswith(root)
    )


def lock(
    api: Any,
    repo: str,
    state: Path,
    work: Path,
    session: date,
    df: pd.DataFrame,
    now: datetime,
    shadow: str | None = None,
) -> dict[str, Any]:
    """Write the morning's file to the dataset and log its hash on the state branch.
    Refuses at or after 9:30 am, and refuses to replace a session already locked."""
    if now >= lock_deadline(session):
        raise LockRefused(f"it is {now.isoformat()}, at or after 9:30 am of {session}")
    path = folder("locked", session, shadow)
    if path in logged_hashes(state) or api.file_exists(
        repo, f"{path}/meta.json", repo_type="dataset"
    ):
        raise LockRefused(f"{path} is already locked")
    meta = write(work, "locked", session, df, now, shadow)
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=work,
        allow_patterns=[f"{path}/**"],
        commit_message=f"scoreboard lock {session}",
    )
    log = locks_path(state)
    log.parent.mkdir(parents=True, exist_ok=True)
    rec = {**meta, "path": path}
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return meta


def children(api: Any, repo: str, path: str) -> list[str]:
    try:
        tree = api.list_repo_tree(repo, path_in_repo=path, repo_type="dataset")
        return [str(getattr(t, "path", "")).rsplit("/", 1)[-1] for t in tree]
    except Exception as exc:  # no such folder yet is fine; anything else is a real failure
        if isinstance(exc, FileNotFoundError) or "NotFound" in type(exc).__name__:
            return []
        raise


def shadows_in(api: Any, repo: str) -> list[str]:
    return sorted(children(api, repo, f"{PREFIX}/{SHADOW}"))


def sessions_in(api: Any, repo: str, kind: str, shadow: str | None = None) -> list[date]:
    out = []
    for n in children(api, repo, lane(kind, shadow)):
        try:
            out.append(date.fromisoformat(n))
        except ValueError:
            continue
    return sorted(out)


def uploaded_at(api: Any, repo: str, path: str) -> datetime | None:
    """When the dataset's server recorded the file's last commit."""
    try:
        info = api.get_paths_info(repo, [path], expand=True, repo_type="dataset")
        when = info[0].last_commit.date
    except Exception:
        return None
    return when if isinstance(when, datetime) else None


def days_path(state: Path) -> Path:
    return state / PREFIX / "days"


def board_path(state: Path) -> Path:
    return state / PREFIX / "scoreboard.json"


def graded_days(state: Path) -> list[dict[str, Any]]:
    d = days_path(state)
    if not d.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))]


def path_of(day: Mapping[str, Any]) -> str:
    """A graded day's folder in the dataset."""
    return str(day.get("path") or folder(day["kind"], date.fromisoformat(day["session"])))


def day_file(state: Path, path: str) -> Path:
    return days_path(state) / (path.removeprefix(f"{PREFIX}/").replace("/", "-") + ".json")


def ready(session: date, now: datetime) -> date | None:
    """The exit session, once its 9:45 am price window has closed; else None."""
    exit_ = clock.next_session(session)
    return exit_ if now >= clock.at_ny(exit_, PRICE_UNTIL) else None


def grade_pending(
    api: Any,
    repo: str,
    state: Path,
    work: Path,
    prices: Prices,
    now: datetime,
    lineup: Sequence[tuple[str, str]] = (),
) -> list[dict[str, Any]]:
    """Grade every locked or backfilled session not graded yet whose exit price exists, the
    picks' model's and each shadow model's, save each day on the state branch, and rewrite
    the scoreboard (``lineup``: its columns, see write_board). Returns the days graded now."""
    done = {path_of(d) for d in graded_days(state)}
    logged = logged_hashes(state)
    new = []
    lanes: list[tuple[str, str | None]] = [(k, None) for k in KINDS]
    lanes += [("locked", v) for v in shadows_in(api, repo)]
    for kind, shadow in lanes:
        for session in sessions_in(api, repo, kind, shadow):
            path = folder(kind, session, shadow)
            if path in done:
                continue
            exit_ = ready(session, now)
            if exit_ is None:
                continue
            blob = Path(
                api.hf_hub_download(
                    repo, f"{path}/predictions.parquet", repo_type="dataset", local_dir=work
                )
            ).read_bytes()
            meta = json.loads(
                Path(
                    api.hf_hub_download(
                        repo, f"{path}/meta.json", repo_type="dataset", local_dir=work
                    )
                ).read_text(encoding="utf-8")
            )
            rec: dict[str, Any] = {
                "kind": kind,
                "session": session.isoformat(),
                "path": path,
                "shadow": shadow is not None,
                "exit_session": exit_.isoformat(),
                "model_version": meta.get("model_version"),
                "graded_at": now.astimezone(UTC).isoformat(),
            }
            up = uploaded_at(api, repo, f"{path}/predictions.parquet") if kind == "locked" else None
            bad = problems(meta, blob, logged.get(path), up)
            if shadow is not None and meta.get("model_version") != shadow:
                bad.append(f"its model is {meta.get('model_version')}, not {shadow}")
            if bad:
                rec["refused"] = bad
            else:
                preds = pd.read_parquet(io.BytesIO(blob))
                want = set(preds["symbol"]) | {SPY}
                try:
                    day, _ = grade(preds, prices(session, want), prices(exit_, want))
                except GradeError as exc:
                    rec["refused"] = [str(exc)]
                else:
                    rec.update(day)
            out = day_file(state, path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
            new.append(rec)
    write_board(state, lineup)
    return new


def version_of(day: Mapping[str, Any]) -> str:
    v = day.get("model_version")
    return ", ".join(map(str, v)) if isinstance(v, list) else str(v)


def write_board(state: Path, lineup: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
    """The public scoreboard: one column per model version (``lineup``'s (version, role)
    first, role "picks" or "shadow", then any other version graded), each with running totals
    for its locked days, separate ones for its backfill, and each day's numbers. Aggregates
    only; the per-stock files stay in the dataset."""
    days = graded_days(state)
    ok = [d for d in days if "refused" not in d]
    order = dict(lineup)
    for d in ok:
        order.setdefault(version_of(d), "shadow" if d.get("shadow") else "picks")

    def rows(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "session": d["session"],
                "scored": d["scored"],
                "above": d["above"],
                "model_version": d.get("model_version"),
                **{k: v for k, v in summarize(d).items() if k != "calibration"},
            }
            for d in sorted(group, key=lambda d: d["session"], reverse=True)
        ]

    models = []
    for version, role in order.items():
        mine = [d for d in ok if version_of(d) == version]
        locked: dict[str, dict[str, Any]] = {}
        # A version locked both as the picks' model and as a shadow on one morning counts once.
        for d in sorted(mine, key=lambda d: bool(d.get("shadow"))):
            if d["kind"] == "locked":
                locked.setdefault(d["session"], d)
        back = [d for d in mine if d["kind"] == "backfill"]
        models.append(
            {
                "version": version,
                "role": role,
                "locked": totals(list(locked.values())),
                "locked_days": rows(list(locked.values())),
                "backfill": totals(back),
                "backfill_days": rows(back),
            }
        )
    board: dict[str, Any] = {"line": LINE, "buckets": [b[0] for b in BUCKETS], "models": models}
    board["refused"] = [
        {
            "kind": d["kind"],
            "session": d["session"],
            "model_version": d.get("model_version"),
            "path": path_of(d),
            "why": d["refused"],
        }
        for d in days
        if "refused" in d
    ]
    path = board_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(board, indent=1, default=str) + "\n", encoding="utf-8")
    return board


def latest_locked(state: Path, shadow: str | None = None) -> dict[str, Any] | None:
    """The latest locked morning graded, the picks' model's or one shadow model's."""
    root = lane("locked", shadow) + "/"
    ok = [
        d
        for d in graded_days(state)
        if path_of(d).startswith(root) and d["kind"] == "locked" and "refused" not in d
    ]
    return max(ok, key=lambda d: d["session"]) if ok else None


def read_board(state: Path) -> dict[str, Any] | None:
    path = board_path(state)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
