from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace as Ns

import numpy as np
import pandas as pd
import pytest

from stoa import clock
from stoa.models import Side
from stoa.newsbrain import harness as hz
from stoa.newsbrain import sealed
from stoa.newsbrain import trial as t9

D = date(2024, 3, 12)


class FakeStore:
    def __init__(self) -> None:
        self.calls = 0

    def day(self, symbol: str, day: date) -> pd.DataFrame:
        self.calls += 1
        t = [pd.Timestamp(clock.at_ny(day, m)) for m in ("09:44", "09:46", "10:20")]
        return pd.DataFrame({"t": t, "o": [9.0, 10.0, 11.0], "c": [9.0, 10.0, 11.0]})


def test_prices_take_the_cache_first_then_the_first_bar_from_945(tmp_path: Path) -> None:
    pd.DataFrame({"symbol": ["AAA"], "price": [20.5]}).to_parquet(tmp_path / f"{D}.parquet")
    store = FakeStore()
    p = hz.Prices(tmp_path, store)  # type: ignore[arg-type]
    got = p(D, {"AAA"}, {"BBB"})
    assert got == {"AAA": Decimal("20.5"), "BBB": Decimal("10.0")}
    p(D, {"BBB"}, set())
    assert store.calls == 1 and p.sources == {"cache": 1, "minute bars": 1, "none": 0}
    with pytest.raises(sealed.HoldoutBreach):  # the holdout wall
        p.one(date(2026, 8, 3), "AAA")


def rows() -> pd.DataFrame:
    days = clock.sessions_between(date(2024, 3, 1), date(2024, 3, 8))
    out = []
    for i, d in enumerate(days):
        for k in range(12):
            out.append(
                {
                    "symbol": f"S{k:02d}",
                    "sidx": i,
                    "session": d,
                    "excess_5": (k - 5) / 100,
                    "slip": 0.001,
                    "adv": 1e8,
                }
            )
    return pd.DataFrame(out)


def test_cheater_takes_each_sessions_best_realised_rows_and_random_takes_two() -> None:
    hz.register()
    test = rows()
    sigs = hz.cheater_signals(test)
    per = pd.Series([s.session for s in sigs]).value_counts()
    assert per.max() == 6  # only the 6 rows with a positive excess return
    assert {s.symbol for s in sigs} == {f"S{k:02d}" for k in range(6, 12)}
    assert all(s.kind == t9.kind_for(5) for s in sigs)
    a, b = hz.random_signals(test, 1), hz.random_signals(test, 1)
    assert [s.symbol for s in a] == [s.symbol for s in b]  # seeded
    assert len(a) == 2 * test["sidx"].nunique()
    assert [s.symbol for s in a] != [s.symbol for s in hz.random_signals(test, 2)]
    spy = hz.spy_signals([date(2024, 3, 4), date(2024, 3, 5)])
    assert all(s.symbol == "SPY" and s.kind == hz.HOLD_KIND for s in spy)


def test_charged_costs_are_measured_against_the_price_handed_to_the_engine() -> None:
    prices = Ns(memo={(D, "AAA"): 100.0})
    fills = [
        Ns(
            session=D,
            symbol="AAA",
            side=Side.BUY,
            price=Decimal("100.10"),
            fees=Decimal("0"),
            gross=Decimal("100.10"),
        ),
        Ns(
            session=D,
            symbol="AAA",
            side=Side.SELL,
            price=Decimal("99.90"),
            fees=Decimal("0.01"),
            gross=Decimal("99.90"),
        ),
    ]
    c = hz.charged(Ns(fills=fills), prices)  # type: ignore[arg-type]
    assert c.buy_slip_bps == pytest.approx(10.0) and c.sell_slip_bps == pytest.approx(10.0)
    assert c.fee_bps == pytest.approx(1.001, rel=1e-3) and c.round_trip_bps == pytest.approx(
        21.0, rel=1e-3
    )


def test_spy_expectation_is_exposure_times_spy() -> None:
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=4))
    close = pd.Series([100.0, 110.0, 99.0, 99.0], index=idx)
    res = Ns(equity=pd.Series(1.0, index=idx), exposure=pd.Series([0.0, 0.2, 0.2, 0.2], index=idx))
    # day 2 at 0 exposure (yesterday's), day 3 at 20% of -10%, day 4 flat
    assert hz.spy_expected(res, close) == pytest.approx(0.98 - 1)  # type: ignore[arg-type]
    assert np.isfinite(hz.SPY_SHARPE_GAP)
