from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stoa.data.base import Bars, DataError, require_fresh, validate_bars
from stoa.data.cache import BarCache, load_cached_bars
from stoa.data.csv import CSVProvider
from stoa.data.synthetic import SyntheticProvider, synthetic_bars


def test_synthetic_is_seeded_and_on_calendar() -> None:
    a = synthetic_bars(["SPY", "AAPL"], date(2026, 11, 20), date(2026, 12, 31), seed=3)
    b = synthetic_bars(["SPY", "AAPL"], date(2026, 11, 20), date(2026, 12, 31), seed=3)
    pd.testing.assert_frame_equal(a["AAPL"], b["AAPL"])
    days = {d.date() for d in a["SPY"].index}
    assert date(2026, 11, 26) not in days and date(2026, 12, 25) not in days


def test_validate_rejects_bad_bars() -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    good = pd.DataFrame(
        {"open": [1, 2], "high": [2, 3], "low": [0.5, 1], "close": [1.5, 2], "volume": [1, 1]},
        index=idx,
    )
    validate_bars("X", good)
    with pytest.raises(DataError):
        validate_bars("X", good.assign(close=[1.5, -1]))
    with pytest.raises(DataError):
        validate_bars("X", good.assign(high=[1.0, 3]))
    with pytest.raises(DataError):
        validate_bars("X", good.drop(columns=["volume"]))
    with pytest.raises(DataError):
        validate_bars("X", pd.concat([good, good]))
    with pytest.raises(DataError):
        validate_bars("X", good.assign(open=[np.nan, 2]))


def test_require_fresh(bars: Bars) -> None:
    last = bars["SPY"].index[-1].date()
    require_fresh(bars, ["SPY", "AAPL"], last)
    stale = {**bars, "AAPL": bars["AAPL"].iloc[:-1]}
    with pytest.raises(DataError, match="AAPL"):
        require_fresh(stale, ["SPY", "AAPL"], last)
    with pytest.raises(DataError, match="missing"):
        require_fresh(bars, ["ZZZ"], last)


def test_cache_round_trip(tmp_path: Path, bars: Bars) -> None:
    cache = BarCache(tmp_path, "default")
    cache.write({"SPY": bars["SPY"]})
    got = load_cached_bars(tmp_path, ["SPY"], date(2017, 1, 1), date(2017, 12, 31))
    assert got["SPY"].index[0] >= pd.Timestamp("2017-01-01")
    assert got["SPY"].index[-1] <= pd.Timestamp("2017-12-31")
    with pytest.raises(DataError, match="not in cache"):
        cache.read(["NOPE"], date(2017, 1, 1), date(2017, 2, 1))


def test_csv_provider(tmp_path: Path, bars: Bars) -> None:
    bars["SPY"].reset_index().to_csv(tmp_path / "SPY.csv", index=False)
    got = CSVProvider(tmp_path).fetch(["SPY"], date(2016, 1, 1), date(2030, 1, 1))
    pd.testing.assert_frame_equal(got["SPY"], bars["SPY"], check_freq=False)
    with pytest.raises(DataError):
        CSVProvider(tmp_path).fetch(["QQQ"], date(2016, 1, 1), date(2030, 1, 1))


def test_synthetic_provider_slices() -> None:
    got = SyntheticProvider(seed=2).fetch(["SPY"], date(2020, 1, 1), date(2020, 3, 1))
    assert got["SPY"].index[0] >= pd.Timestamp("2020-01-01")
