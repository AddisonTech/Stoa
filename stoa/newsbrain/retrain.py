"""Trial 12 retrain-all: trial 9's model refit on every labelled session, 2016 to the last
completed trading day, the test window and the spent holdout included.

The owner's decision of 2026-10-28 (docs/DECISIONS.md): same 13 features, the 1 session
label (open of t to the open of t+1, against SPY, after costs), trial 9's classifier and
its fit and calibration split, the same p >= 0.55 buy line and SPY core. Only the window
changes. Nothing is left out of sample, so the shadow scoreboard (stoa/scoreboard.py) is
its test.

The rows are the Dell's: trial 12's holdout panel (2016-02-02 to 2026-10-23, built on
Windows like the feed and picks runners, so log and log1p round as they do live) and the
Dell's rebuilds of the sessions after it (data/newsbrain/feedcheck/ref), which matched the
feed exactly. The Dell's daily bars end on 2026-10-23, so the labels of the sessions after
2026-10-22 use Yahoo's opens, adjusted for splits and dividends like the Dell's Alpaca bars
(the two agree to about 0.001% where both exist).

The model is saved under data/newsbrain/models/<version>/ with a meta.json: the train
window, row count, features, horizon, git sha and the pickle's sha256. The model it
replaces stays where it is (the feed state's static/model_h1.pkl), for rollback.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa import clock, wide
from stoa.experiments import code_sha
from stoa.newsbrain import ROOT, panel
from stoa.newsbrain import trial as t9

H = 1
SPY = "SPY"
NAME = "trial12:retrain-all"
MODELS = ROOT / "models"
HOLDOUT_PANEL = ROOT / "trial12_holdout" / "panel.parquet"
REF = ROOT / "feedcheck" / "ref"

Opens = Callable[[list[str], date, date], pd.DataFrame]  # date x symbol, adjusted opens


def yahoo_opens(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    """Daily opens from Yahoo, adjusted for splits and dividends, ``start`` to ``end``, in
    batches of 200 with a pause between, and one more try for any symbol missing."""
    import time

    import yfinance as yf

    yahoo = {s: s.replace(".", "-") for s in symbols}  # BRK.B is BRK-B there

    def fetch(names: list[str]) -> pd.DataFrame:
        parts = []
        for i in range(0, len(names), 200):
            got = yf.download(
                names[i : i + 200],
                start=start.isoformat(),
                end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=True,
                progress=False,
                threads=False,
            )["Open"]
            parts.append(got.dropna(axis=1, how="all"))
            time.sleep(3)
        return pd.concat(parts, axis=1) if parts else pd.DataFrame()

    got = fetch(list(yahoo.values()))
    missing = [y for y in yahoo.values() if y not in got.columns]
    if missing:
        time.sleep(30)
        got = pd.concat([got, fetch(missing)], axis=1)
    got.index = pd.DatetimeIndex(got.index).tz_localize(None).normalize()
    out: pd.DataFrame = got.rename(columns={v: k for k, v in yahoo.items()})
    return out


def label(rows: pd.DataFrame, entry: date, exit_: date, opens: pd.DataFrame) -> pd.DataFrame:
    """Trial 9's 1 session label for ``entry``'s rows: the open of ``entry`` to the open of
    ``exit_``, minus SPY's, minus a round trip of slippage and the fee, above zero."""
    o_in, o_out = opens.loc[pd.Timestamp(entry)], opens.loc[pd.Timestamp(exit_)]
    r = (o_out / o_in - 1).reindex(rows["symbol"]).to_numpy(dtype=float)
    b = (
        float(o_out.to_numpy(dtype=float)[list(o_out.index).index(SPY)])
        / float(o_in.to_numpy(dtype=float)[list(o_in.index).index(SPY)])
        - 1
    )
    slip = np.array([wide.slippage_bps(v if np.isfinite(v) else None) / 1e4 for v in rows["adv"]])
    net = r - b - 2 * slip - panel.FEE
    out = rows.copy()
    out["slip"] = slip
    out[f"excess_{H}"] = net
    out[f"y_{H}"] = pd.Series((net > 0).astype("float32"), index=out.index).where(np.isfinite(net))
    return out


def training_rows(
    last: date, opens: Opens = yahoo_opens, panel_path: Path = HOLDOUT_PANEL, ref: Path = REF
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every row whose 1 session label ends by the open of ``last``, and where each part
    came from."""
    keep = ["symbol", "sidx", "session", *panel.FEATURES, "adv", f"y_{H}"]
    base = pd.read_parquet(panel_path, columns=keep)
    panel_last = base["session"].max()
    labelled = base[base[f"y_{H}"].notna()]
    days = clock.sessions_between(panel.PANEL_START, last)
    tail = [d for d in days if d >= panel_last and d < last]  # sessions still to label
    parts, sources = [labelled], {}
    todo: list[tuple[date, pd.DataFrame, str]] = []
    for d in tail:
        if d == panel_last:
            rows = base[base["session"] == d].drop(columns=[f"y_{H}"])
            src = "trial12 holdout panel"
        else:
            path = ref / f"dry-{d:%Y%m%d}T1300Z-picks.parquet"  # matched the feed exactly
            if not path.exists():
                path = ref / f"{d.isoformat()}.parquet"
            rows = pd.read_parquet(path)
            rows = rows.assign(session=d)[[c for c in keep if c != f"y_{H}"]]
            src = f"Dell rebuild {path.name}"
        todo.append((d, rows, src))
    if todo:
        symbols = sorted({s for _, r, _ in todo for s in r["symbol"]} | {SPY})
        cache = ROOT / "retrain" / f"opens-{tail[0]}-{last}.parquet"  # fetched once, kept
        if cache.exists():
            got = pd.read_parquet(cache)
        else:
            got = opens(symbols, tail[0], last)
            cache.parent.mkdir(parents=True, exist_ok=True)
            got.to_parquet(cache)
        for d, rows, src in todo:
            nxt = days[days.index(d) + 1]
            done = label(rows, d, nxt, got)[keep]
            parts.append(done[done[f"y_{H}"].notna()])
            sources[d.isoformat()] = (
                f"{src}; {int(done[f'y_{H}'].notna().sum())} of {len(done)} rows labelled "
                f"from Yahoo opens {d} to {nxt}"
            )
    rows = pd.concat(parts, ignore_index=True)
    return rows, {
        "panel": str(panel_path),
        "panel_labels_to": str(labelled["session"].max()),
        "after_panel": sources,
    }


def fit(rows: pd.DataFrame, last_sidx: int, days: list[date]) -> t9.Frozen:
    """Trial 9's fit and calibration split (``trial.fit_calibrated``) on every row."""
    train = t9.labelled(rows, H, last_sidx)
    clf, cal = t9.fit_calibrated(train, H)
    return t9.Frozen(H, clf, cal, len(train), str(days[int(train["sidx"].max()) + H]))


def version_for(last_train: date) -> str:
    return f"t12-all-{last_train:%Y%m%d}"


def save(model: t9.Frozen, meta: dict[str, Any], out: Path) -> dict[str, Any]:
    """The pickle and its meta.json; the meta carries the pickle's sha256."""
    out.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(model)
    (out / f"model_h{H}.pkl").write_bytes(blob)
    meta = {**meta, "sha256": hashlib.sha256(blob).hexdigest()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta


def load(version: str, models: Path = MODELS) -> t9.Frozen:
    """A saved version, refused if the pickle does not match its meta.json."""
    meta = json.loads((models / version / "meta.json").read_text(encoding="utf-8"))
    blob = (models / version / f"model_h{H}.pkl").read_bytes()
    if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
        raise ValueError(f"{version}: the model does not match its meta.json")
    model: t9.Frozen = pickle.loads(blob)
    return model


def session_rows(d: date, panel_path: Path = HOLDOUT_PANEL, ref: Path = REF) -> pd.DataFrame:
    """One session's rows as the Dell built them: the holdout panel through its last
    session, then the Dell's rebuilds that matched the feed exactly."""
    cols = ["symbol", "sidx", *panel.FEATURES, "adv"]
    for path in (ref / f"dry-{d:%Y%m%d}T1300Z-picks.parquet", ref / f"{d.isoformat()}.parquet"):
        if path.exists():
            rows = pd.read_parquet(path)
            if len(rows):
                return rows[cols].reset_index(drop=True)
    got = pd.read_parquet(panel_path, columns=[*cols, "session"], filters=[("session", "==", d)])
    if got.empty:
        raise ValueError(f"no Dell rows for {d}")
    return got[cols].reset_index(drop=True)


BACKFILL = ROOT / "scoreboard-out"


def backfill(
    version: str, sessions: list[date], out: Path = BACKFILL, models: Path = MODELS
) -> list[dict[str, Any]]:
    """Score past sessions' Dell rows with ``version`` for the shadow scoreboard, marked
    backfill. The rows' 13 features ride along, so the runner can score them again and
    require the same p."""
    from datetime import UTC, datetime

    from stoa import shadowboard
    from stoa.newsbrain import spycore as sc

    model = load(version, models)
    t9.register_kind(sc.GRADED)
    metas = []
    for d in sessions:
        rows = session_rows(d).sort_values(["sidx", "symbol"]).reset_index(drop=True)
        p = model.predict(rows)
        rows = rows.assign(p=p)
        sigs = sc.signals(rows, p, sc.GRADED)
        df = shadowboard.predictions(rows, version, extra=panel.FEATURES)
        meta = shadowboard.write(out, "backfill", d, df, datetime.now(UTC))
        meta["picks"] = [(x.symbol, round(x.score, 4)) for x in sigs]
        metas.append(meta)
    return metas


def push(version: str, repo: str, token: str, models: Path = MODELS, out: Path = BACKFILL) -> None:
    """Upload the version and its backfill to the private dataset: models/<version>/ and
    scoreboard/backfill/. Nothing else there is touched."""
    from huggingface_hub import HfApi

    from stoa.newsbrain.feed import private_or_refuse

    load(version, models)  # refuses a pickle that does not match its meta.json
    api = HfApi(token=token)
    private_or_refuse(api, repo)
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=models / version,
        path_in_repo=f"models/{version}",
        commit_message=f"model {version}",
    )
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=out,
        allow_patterns=["scoreboard/backfill/**"],
        commit_message=f"scoreboard backfill {version}",
    )


def run(last: date, opens: Opens = yahoo_opens, models: Path = MODELS) -> dict[str, Any]:
    """Retrain on every session whose label ends by the open of ``last`` and save it."""
    import sklearn

    rows, sources = training_rows(last, opens)
    days = clock.sessions_between(panel.PANEL_START, last)
    model = fit(rows, days.index(last), days)
    train = t9.labelled(rows, H, days.index(last))
    first, final = train["session"].min(), train["session"].max()
    version = version_for(final)
    meta = {
        "version": version,
        "name": NAME,
        "horizon": H,
        "buy_line": 0.55,
        "core": "SPY",
        "train_window": [str(first), str(final)],
        "last_label_session": model.last_label_session,
        "rows": model.rows,
        "sessions": int(train["sidx"].nunique()),
        "positive_share": round(float(train[f"y_{H}"].mean()), 6),
        "features": list(panel.FEATURES),
        "classifier": t9.HGB,
        "calibration": f"isotonic on the last {t9.CAL_SHARE:.0%} of sessions",
        "git_sha": code_sha(),
        "built_on": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "sources": sources,
        "replaces": "trial 9's frozen model_h1.pkl (feed state static/), kept for rollback",
    }
    return save(model, meta, models / version)
