from __future__ import annotations

import math
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from stoa import metrics
from stoa.models import ClosedTrade


def curve(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def trade(pnl: str) -> ClosedTrade:
    return ClosedTrade(
        symbol="X",
        entry_session=date(2024, 1, 2),
        exit_session=date(2024, 1, 5),
        qty=Decimal(1),
        entry_price=Decimal(10),
        exit_price=Decimal(10),
        fees=Decimal(0),
        pnl=Decimal(pnl),
        reason="t",
    )


def test_total_return_and_cagr() -> None:
    e = curve([100.0] + [0.0] * 252)
    e[:] = 100 * (1.10 ** (np.arange(253) / 252))
    assert metrics.total_return(e) == pytest.approx(0.10)
    assert metrics.cagr(e) == pytest.approx(0.10)


def test_max_drawdown_and_duration() -> None:
    e = curve([100, 120, 90, 100, 130, 125])
    mdd, dur = metrics.max_drawdown(e)
    assert mdd == pytest.approx(0.25)
    assert dur == 2


def test_sharpe_sortino_signs() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    assert metrics.sharpe(r) > 0
    assert metrics.sortino(r) > metrics.sharpe(r)
    assert metrics.sharpe(pd.Series([0.01, 0.01, 0.01])) == 0.0
    expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert metrics.sharpe(r) == pytest.approx(expected)


def test_trade_stats() -> None:
    pf, wr, aw, al, ex = metrics.trade_stats([trade("10"), trade("-5"), trade("20"), trade("-5")])
    assert pf == pytest.approx(3.0)
    assert wr == pytest.approx(0.5)
    assert aw == pytest.approx(15.0)
    assert al == pytest.approx(-5.0)
    assert ex == pytest.approx(5.0)
    assert metrics.trade_stats([trade("5")])[0] == math.inf
    assert metrics.trade_stats([]) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_monthly_returns_table() -> None:
    idx = pd.to_datetime(["2023-12-29", "2024-01-31", "2024-02-29", "2024-03-28"])
    e = pd.Series([100.0, 110.0, 99.0, 108.9], index=idx)
    table = metrics.monthly_returns(e)
    assert table.loc[2024, "01"] == pytest.approx(0.10)
    assert table.loc[2024, "02"] == pytest.approx(-0.10)
    assert table.loc[2024, "year"] == pytest.approx(0.089)


def test_bootstrap_is_seeded_and_bounded() -> None:
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 400))
    a = metrics.bootstrap_max_drawdown(r, 200, 20, seed=3)
    b = metrics.bootstrap_max_drawdown(r, 200, 20, seed=3)
    assert a == b
    assert 0 <= a["p05"] <= a["p50"] <= a["p95"] <= 1


def test_compute_runs_end_to_end() -> None:
    e = curve([100, 101, 99, 103, 104])
    exp = pd.Series([0, 0.5, 0.5, 0.5, 0.0], index=e.index)
    m = metrics.compute(e, [trade("4")], [], exp)
    assert m.trade_count == 1
    assert m.exposure == pytest.approx(0.375)
    assert m.final_equity == 104
    assert set(m.to_dict()) >= {"sharpe", "calmar", "turnover"}
