"""Trial 12 retrain-all (stoa/newsbrain/retrain.py): the label after the Dell's bars end,
and the saved model's hash check that rollback and the picks rely on."""

from __future__ import annotations

import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stoa import wide
from stoa.newsbrain import panel, retrain


def test_the_label_is_trial_9s_open_to_open_against_spy_after_costs() -> None:
    d0, d1 = date(2026, 10, 26), date(2026, 10, 27)
    opens = pd.DataFrame(
        {"SPY": [500.0, 505.0], "AAA": [10.0, 10.3], "BBB": [20.0, 20.0], "CCC": [5.0, np.nan]},
        index=pd.DatetimeIndex([pd.Timestamp(d0), pd.Timestamp(d1)]),
    )
    rows = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "adv": [1e8, 1e8, 1e8]})
    got = retrain.label(rows, d0, d1, opens).set_index("symbol")
    cost = 2 * wide.slippage_bps(1e8) / 1e4 + panel.FEE
    assert got.loc["AAA", "excess_1"] == pytest.approx(0.03 - 0.01 - cost)
    assert got.loc["AAA", "y_1"] == 1.0 and got.loc["BBB", "y_1"] == 0.0
    assert np.isnan(got.loc["CCC", "y_1"])  # no exit open: no label, as in the panel


def test_a_saved_model_loads_only_when_it_matches_its_hash(tmp_path: Path) -> None:
    from stoa.newsbrain import trial as t9

    model = t9.Frozen(1, clf=None, cal=None, rows=3, last_label_session="2026-10-27")  # type: ignore[arg-type]
    meta = retrain.save(model, {"version": "v1", "train_window": ["a", "b"]}, tmp_path / "v1")
    assert json.loads((tmp_path / "v1" / "meta.json").read_text(encoding="utf-8")) == meta
    assert retrain.load("v1", tmp_path).rows == 3
    (tmp_path / "v1" / "model_h1.pkl").write_bytes(pickle.dumps(model) + b" ")
    with pytest.raises(ValueError, match="does not match"):
        retrain.load("v1", tmp_path)
    assert retrain.version_for(date(2026, 10, 27)) == "t12-all-20261027"
