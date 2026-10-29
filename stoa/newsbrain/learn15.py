"""Trial 15's daily learning (docs/DECISIONS.md, trial 15, ``daily_learning``).

Every night the live brain's design is refit on every row labelled through that day, recent
rows weighted more (``brain15.Brain``, half life 2 years). The refit is a candidate: the next
morning it ranks the same rows as the live brain, its top 5 are written with their sha256 to
``trial15/shadow/<session>.json`` before 9:30 am, and the 5:15 pm summary grades them exactly
as it grades the picks. It takes over only after 20 graded shadow sessions when, over the
last 20, its average edge is higher than the live picks' and its hit rate at least as high.

Where the rows come from: the decade's labelled rows (``learn/base.parquet`` in the feed's
private dataset, built once on the Dell from trial 12's holdout panel and the Dell's
rebuilds, as the retrain-all was), plus each morning's scored rows, which the picks job
keeps under ``learn/rows/<session>.parquet``. A morning's rows get their 1 session label
(open to open against SPY after costs) once the next session's open exists.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa import clock, wide
from stoa.newsbrain import ROOT, brain15, trial15
from stoa.newsbrain import panel as pn

BASE = "learn/base.parquet"
ROWS = "learn/rows"
KEEP = ["symbol", "sidx", "session", *pn.FEATURES, "adv", "y_1"]
LOCAL_BASE = ROOT / "trial15" / "learn_base.parquet"

Opens = Callable[[list[str], date, date], pd.DataFrame]  # date x symbol, adjusted opens


def build_base(last: date, out: Path = LOCAL_BASE) -> dict[str, Any]:
    """The decade's labelled rows through the label ending at the open of ``last``, on the
    Dell (the retrain-all's rows); uploaded once with scripts/push-learn-base.ps1."""
    from stoa.newsbrain import retrain

    rows, sources = retrain.training_rows(last)
    rows = rows[[c for c in KEEP if c in rows.columns]]
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out, index=False)
    return {"rows": len(rows), "last_session": str(rows["session"].max()), "sources": sources}


def label(rows: pd.DataFrame, opens: pd.DataFrame, entry: date, exit_: date) -> pd.DataFrame:
    """Trial 9's 1 session label for one morning's rows, as ``retrain.label`` gives it."""
    o_in, o_out = opens.loc[pd.Timestamp(entry)], opens.loc[pd.Timestamp(exit_)]
    r = (o_out / o_in - 1).reindex(rows["symbol"]).to_numpy(dtype=float)
    spy = [np.asarray(o.reindex(["SPY"]), dtype=float)[0] for o in (o_in, o_out)]
    b = float(spy[1] / spy[0] - 1)
    slip = np.array([wide.slippage_bps(v if np.isfinite(v) else None) / 1e4 for v in rows["adv"]])
    net = r - b - 2 * slip - pn.FEE
    y = pd.Series((net > 0).astype("float32"), index=rows.index).where(np.isfinite(net))
    return rows.assign(y_1=y)


def training_set(
    base: pd.DataFrame, mornings: Mapping[date, pd.DataFrame], opens: Opens
) -> pd.DataFrame:
    """The base rows plus every kept morning whose next open is known, labelled."""
    parts = [base[base["y_1"].notna()]]
    done = set(pd.to_datetime(base["session"]).dt.date)
    todo = sorted(d for d in mornings if d not in done)
    if todo:
        symbols = sorted({s for d in todo for s in mornings[d]["symbol"]} | {"SPY"})
        got = opens(symbols, todo[0], clock.next_session(todo[-1]))
        for d in todo:
            nxt = clock.next_session(d)
            if pd.Timestamp(nxt) not in got.index or pd.Timestamp(d) not in got.index:
                continue  # the next open is not in yet
            lab = label(mornings[d].assign(session=d), got, d, nxt)
            parts.append(lab[lab["y_1"].notna()][[c for c in KEEP if c in lab.columns]])
    return pd.concat(parts, ignore_index=True)


def fit(train: pd.DataFrame) -> brain15.Brain:
    """The live brain's design (trial 9's 13 inputs) refit with recency weights."""
    return brain15.Brain(tuple(pn.FEATURES)).fit(train)


def save(brain: brain15.Brain, train: pd.DataFrame, models: Path) -> dict[str, Any]:
    last = pd.to_datetime(train["session"]).max().date()
    version = f"t15-learn-{last:%Y%m%d}"
    blob = pickle.dumps(brain)
    sha = hashlib.sha256(blob).hexdigest()
    out = models / version
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.pkl").write_bytes(blob)
    meta = {
        "version": version,
        "sha256": sha,
        "rows": len(train),
        "trained_through": str(last),
        "features": list(pn.FEATURES),
        "half_life_years": trial15.HALF_LIFE_YEARS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta


# ---- the shadow and the takeover ------------------------------------------------------------


def lock_shadow(
    folder: Path, session: date, version: str, picks: Sequence[str], now: Any
) -> dict[str, Any]:
    """The candidate's top 5 for ``session``, written once with its sha256 before 9:30 am."""
    if clock.to_ny(now) >= clock.at_ny(session, "09:30"):
        raise RuntimeError("the shadow's picks are locked only before 9:30 am")
    out = folder / "shadow" / f"{session.isoformat()}.json"
    if out.exists():
        return dict(json.loads(out.read_text(encoding="utf-8")))
    body = {"session": session.isoformat(), "version": version, "picks": list(picks)}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    body["locked_at"] = now.isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    return body


def verify(body: Mapping[str, Any]) -> bool:
    core = {k: body[k] for k in ("session", "version", "picks")}
    return bool(
        hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest() == body["sha256"]
    )


def takeover(
    live_days: Sequence[Mapping[str, Any]], shadow_days: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Whether the candidate takes over: at least 20 graded shadow sessions and, over the
    last 20 sessions both were graded on, a higher average edge and a hit rate at least as
    high."""
    live = {d["session"]: d for d in live_days}
    both = [d for d in shadow_days if d["session"] in live][-trial15.WINDOW :]
    out: dict[str, Any] = {"shadow_sessions": len(shadow_days), "compared": len(both)}
    if len(shadow_days) < trial15.MIN_SHADOW or len(both) < trial15.WINDOW:
        out["takes_over"] = False
        return out
    s = trial15.tally(both)
    lv = trial15.tally([live[d["session"]] for d in both])
    out.update(shadow=s, live=lv)
    out["takes_over"] = bool(
        s["avg_edge"] is not None
        and lv["avg_edge"] is not None
        and s["avg_edge"] > lv["avg_edge"]
        and s["hit_rate"] >= lv["hit_rate"]
    )
    return out
