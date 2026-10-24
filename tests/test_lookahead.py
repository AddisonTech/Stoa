"""No lookahead: anything decided at t must not change when bars after t change."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from stoa.config import Settings
from stoa.data.base import Bars, validate_bars
from stoa.exits import evaluate_exits
from stoa.features import build_features, regime_is_bull, slice_features, snapshot
from stoa.models import Holding, PositionMeta, dec
from stoa.planner import DecisionInput
from stoa.roles.quant import apply_gates
from stoa.roles.scout import RuleScout, ScoutInput
from stoa.strategy import AgentStrategy
from stoa.universe import Universe


def perturb_after(bars: Bars, t: pd.Timestamp, seed: int) -> Bars:
    rng = np.random.default_rng(seed)
    out: Bars = {}
    for sym, original in bars.items():
        df = original.copy()
        after = df.index > t
        n = int(after.sum())
        shock = np.exp(rng.normal(0, 0.2, n))
        for col in ("open", "high", "low", "close"):
            df.loc[after, col] = df.loc[after, col].to_numpy() * shock
        df.loc[after, "high"] = df.loc[after, ["open", "high", "low", "close"]].max(axis=1) * 1.01
        df.loc[after, "low"] = df.loc[after, ["open", "high", "low", "close"]].min(axis=1) * 0.99
        df.loc[after, "volume"] = df.loc[after, "volume"].to_numpy() * rng.uniform(0.1, 10, n)
        out[sym] = validate_bars(sym, df)
    return out


def holdings_at(bars: Bars, t: pd.Timestamp) -> dict[str, Holding]:
    out = {}
    for sym in ("AAPL", "KO"):
        df = bars[sym].loc[:t]
        entry_ts = df.index[-12]
        meta = PositionMeta(
            entry_session=entry_ts.date(),
            entry_price=dec(float(df["close"].iloc[-12])),
            entry_atr=Decimal("1.5"),
        )
        out[sym] = Holding(
            symbol=sym, qty=Decimal("0.25"), avg_entry_price=meta.entry_price, meta=meta
        )
    return out


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_no_lookahead_any_stage(small_settings: Settings, bars: Bars, seed: int) -> None:
    rng = np.random.default_rng(seed)
    idx = bars["SPY"].index
    t = idx[int(rng.integers(300, len(idx) - 20))]
    day: date = t.date()
    other = perturb_after(bars, t, seed + 100)
    assert (
        not other["SPY"]
        .loc[t + pd.Timedelta(days=1) :]
        .equals(bars["SPY"].loc[t + pd.Timedelta(days=1) :])
    )

    fa = build_features(bars, "SPY")
    fb = build_features(other, "SPY")
    pd.testing.assert_frame_equal(snapshot(fa, day), snapshot(fb, day))
    assert regime_is_bull(fa, "SPY", day) == regime_is_bull(fb, "SPY", day)
    for sym in bars:
        pd.testing.assert_frame_equal(fa[sym].loc[:t], fb[sym].loc[:t])

    held = holdings_at(bars, t)
    ex_a = evaluate_exits(held, slice_features(fa, day), day, small_settings.exits)
    ex_b = evaluate_exits(held, slice_features(fb, day), day, small_settings.exits)
    assert ex_a == ex_b

    scout = RuleScout(small_settings.scout)

    def scout_in(f: dict[str, pd.DataFrame]) -> ScoutInput:
        return ScoutInput(day, snapshot(f, day), "SPY", regime_is_bull(f, "SPY", day), {}, ())

    ca, _ = scout.propose(scout_in(fa))
    cb, _ = scout.propose(scout_in(fb))
    assert ca == cb
    assert apply_gates(ca, snapshot(fa, day), set(), small_settings.quant) == apply_gates(
        cb, snapshot(fb, day), set(), small_settings.quant
    )

    universe = Universe.assume_all_fractionable(small_settings.watchlist, "SPY")
    strat = AgentStrategy(small_settings, scout)

    def decide(f: dict[str, pd.DataFrame]):  # type: ignore[no-untyped-def]
        return strat.decide(
            DecisionInput(
                session=day,
                execute_session=day,
                features=slice_features(f, day),
                holdings=held,
                equity=Decimal(100),
                cash=Decimal(60),
                prev_equity=Decimal(99),
                high_water_mark=Decimal(105),
                kill_switch_active=False,
                universe=universe,
            )
        )

    da, db = decide(fa), decide(fb)
    assert da.orders == db.orders
    assert da.rejections == db.rejections
    assert da.kills == db.kills
