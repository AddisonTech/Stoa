from __future__ import annotations

import math

import pandas as pd
import pytest

from stoa.config import QuantConfig
from stoa.models import Candidate
from stoa.roles.quant import apply_gates, gate_one

CFG = QuantConfig()
GOOD = {
    "close": 110.0,
    "sma_50": 100.0,
    "ret_5": 0.02,
    "ret_60": 0.10,
    "vol_20": 0.25,
    "rsi_14": 60.0,
    "dollar_vol_20": 2e8,
    "atr_14": 2.0,
}


def cand(sym: str = "AAPL") -> Candidate:
    return Candidate(symbol=sym, conviction=3, thesis="t", score=1.0, source="rule")


def row(**changes: float) -> pd.Series:
    return pd.Series({**GOOD, **changes})


def test_good_idea_passes() -> None:
    assert gate_one(cand(), row(), set(), CFG) is None


@pytest.mark.parametrize(
    ("changes", "gate"),
    [
        ({"close": 99.0}, "uptrend"),
        ({"ret_60": -0.01}, "momentum"),
        ({"vol_20": 0.9}, "volatility"),
        ({"ret_5": 0.2}, "spike"),
        ({"rsi_14": 90.0}, "overbought"),
        ({"dollar_vol_20": 1e6}, "liquidity"),
        ({"sma_50": math.nan}, "insufficient_history"),
    ],
)
def test_each_gate_kills_with_its_name(changes: dict[str, float], gate: str) -> None:
    kill = gate_one(cand(), row(**changes), set(), CFG)
    assert kill is not None and kill.gate == gate and kill.symbol == "AAPL"


def test_already_held_and_no_data() -> None:
    assert gate_one(cand(), row(), {"AAPL"}, CFG).gate == "already_held"  # type: ignore[union-attr]
    assert gate_one(cand(), None, set(), CFG).gate == "no_data"  # type: ignore[union-attr]


def test_apply_gates_splits() -> None:
    snap = pd.DataFrame([GOOD, {**GOOD, "close": 50.0}], index=["AAPL", "KO"])
    passed, kills = apply_gates([cand("AAPL"), cand("KO"), cand("ZZZ")], snap, set(), CFG)
    assert [c.symbol for c in passed] == ["AAPL"]
    assert {(k.symbol, k.gate) for k in kills} == {("KO", "uptrend"), ("ZZZ", "no_data")}
