"""Trial 15's daily learning: the refit's labels, its shadow picks locked before 9:30 am, and
the takeover rule (20 graded sessions, better edge, hit rate at least as high)."""

from __future__ import annotations

from datetime import UTC, date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stoa import clock
from stoa.newsbrain import learn15
from stoa.newsbrain import panel as pn

D1, D2 = date(2026, 10, 29), date(2026, 10, 30)


def test_a_morning_is_labelled_once_its_next_open_is_in() -> None:
    feats = dict.fromkeys(pn.FEATURES, 0.0)
    base = pd.DataFrame(
        [{"symbol": "X", "sidx": 1, "session": date(2026, 10, 28), **feats, "adv": 1e9, "y_1": 1.0}]
    )
    morning = pd.DataFrame(
        [
            {"symbol": "A", "sidx": 2, **feats, "adv": 5e9},
            {"symbol": "B", "sidx": 2, **feats, "adv": 5e9},
        ]
    )

    def opens(symbols: list[str], start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame(
            {"SPY": [100.0, 101.0], "A": [10.0, 10.5], "B": [10.0, 9.9]},
            index=pd.DatetimeIndex([pd.Timestamp(D1), pd.Timestamp(D2)]),
        )

    got = learn15.training_set(base, {D1: morning}, opens)
    assert len(got) == 3
    lab = got[got["session"] == D1].set_index("symbol")["y_1"]
    assert lab["A"] == 1.0 and lab["B"] == 0.0  # A beat SPY by 4%, B lost 2%

    def not_yet(symbols: list[str], start: date, end: date) -> pd.DataFrame:
        return opens(symbols, start, end).iloc[:1]

    assert len(learn15.training_set(base, {D1: morning}, not_yet)) == 1


def test_the_shadow_is_locked_before_930_and_its_hash_checks(tmp_path: Path) -> None:
    early = clock.at_ny(D2, "09:05").astimezone(UTC)
    body = learn15.lock_shadow(tmp_path, D2, "t15-learn-20261029", ["A", "B"], early)
    assert learn15.verify(body)
    assert learn15.lock_shadow(tmp_path, D2, "other", ["C"], early)["picks"] == ["A", "B"]
    with pytest.raises(RuntimeError):
        learn15.lock_shadow(
            tmp_path,
            date(2026, 11, 2),
            "v",
            ["A"],
            clock.at_ny(date(2026, 11, 2), "09:31").astimezone(UTC),
        )
    assert not learn15.verify({**body, "picks": ["Z"]})


def day(session: int, nets: list[float]) -> dict[str, object]:
    return {
        "session": f"s{session:03d}",
        "picks": [{"symbol": f"S{i}", "net": n, "hit": n > 0} for i, n in enumerate(nets)],
    }


def test_the_takeover_needs_20_sessions_better_edge_and_no_worse_hit_rate() -> None:
    live = [day(i, [0.001, -0.002, 0.0005, -0.001, 0.002]) for i in range(30)]
    better = [day(i, [0.003, -0.001, 0.001, 0.002, -0.0005]) for i in range(30)]
    assert learn15.takeover(live, better)["takes_over"]
    assert not learn15.takeover(live, better[:19])["takes_over"]  # too few sessions
    worse = [day(i, [0.0, -0.002, 0.0, -0.001, 0.0]) for i in range(30)]
    assert not learn15.takeover(live, worse)["takes_over"]


def test_the_refit_weighs_recent_rows_more() -> None:
    rng = np.random.default_rng(1)
    n = 3000
    rows = pd.DataFrame({f: rng.normal(size=n) for f in pn.FEATURES})
    rows["sidx"] = np.arange(n) // 10
    rows["session"] = [date(2020, 1, 1)] * n
    rows["y_1"] = (rows["sent"] > 0).astype(float)
    brain = learn15.fit(rows)
    assert brain.rows == n and brain.last_sidx == rows["sidx"].max()
    hi = brain.predict(rows.assign(sent=2.0))
    lo = brain.predict(rows.assign(sent=-2.0))
    assert (hi > lo).all()
