from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd

from stoa.clock import sessions_between
from stoa.config import ExitConfig
from stoa.exits import evaluate_exits, evaluate_position, initial_stop, trailing_stop
from stoa.models import Holding, PositionMeta

CFG = ExitConfig()
DAYS = sessions_between(date(2024, 1, 2), date(2024, 4, 30))


def history(
    closes: list[float], highs: list[float] | None = None, sma50: list[float] | None = None
) -> pd.DataFrame:
    n = len(closes)
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in DAYS[:n]])
    return pd.DataFrame(
        {
            "close": closes,
            "high": highs or closes,
            "sma_50": sma50 or [c * 0.9 for c in closes],
        },
        index=idx,
    )


def holding(entry: float = 100.0, atr: float = 2.0, entry_idx: int = 0) -> Holding:
    meta = PositionMeta(
        entry_session=DAYS[entry_idx], entry_price=Decimal(str(entry)), entry_atr=Decimal(str(atr))
    )
    return Holding(symbol="AAPL", qty=Decimal(1), avg_entry_price=Decimal(str(entry)), meta=meta)


def at(df: pd.DataFrame) -> date:
    return df.index[-1].date()


def test_stop_loss_on_close() -> None:
    assert initial_stop(100, 2, CFG) == 95.0
    df = history([100, 98, 95.0])
    sig = evaluate_position(holding(), df, at(df), CFG)
    assert sig is not None and sig.reason == "stop_loss"
    df2 = history([100, 98, 95.01])
    assert evaluate_position(holding(), df2, at(df2), CFG) is None


def test_trailing_stop_rises_with_highs() -> None:
    assert trailing_stop(110, 100, 2, CFG) == 104.0
    assert trailing_stop(101, 100, 2, CFG) == 95.0
    df = history([100, 105, 110, 104.0], highs=[100, 106, 110, 105])
    sig = evaluate_position(holding(), df, at(df), CFG)
    assert sig is not None and sig.reason == "trailing_stop"
    off = ExitConfig(trailing_enabled=False)
    assert evaluate_position(holding(), df, at(df), off) is None


def test_take_profit_optional() -> None:
    df = history([100, 106.0])
    assert evaluate_position(holding(), df, at(df), CFG) is None
    tp = ExitConfig(take_profit_atr_multiple=3.0)
    sig = evaluate_position(holding(), df, at(df), tp)
    assert sig is not None and sig.reason == "take_profit"


def test_time_exit_after_max_hold_sessions() -> None:
    closes = list(np.linspace(100, 101, 31))
    df = history(closes, highs=closes)
    sig = evaluate_position(holding(), df, at(df), CFG)
    assert sig is not None and sig.reason == "time_exit"
    df29 = history(closes[:30], highs=closes[:30])
    assert evaluate_position(holding(), df29, at(df29), CFG) is None


def test_trend_break_needs_consecutive_closes() -> None:
    closes = [100, 101, 99, 99.5]
    sma = [98, 98, 100, 98]  # only one close below
    df = history(closes, sma50=sma)
    assert evaluate_position(holding(), df, at(df), CFG) is None
    sma2 = [98, 98, 100, 100]
    df2 = history(closes, sma50=sma2)
    sig = evaluate_position(holding(), df2, at(df2), CFG)
    assert sig is not None and sig.reason == "trend_break"


def test_stale_history_produces_no_signal() -> None:
    df = history([100, 90.0])
    assert evaluate_position(holding(), df, DAYS[5], CFG) is None


def test_evaluate_exits_only_held_with_data() -> None:
    df = history([100, 90.0])
    sigs = evaluate_exits({"AAPL": holding()}, {"AAPL": df}, at(df), CFG)
    assert [s.symbol for s in sigs] == ["AAPL"]
    assert evaluate_exits({"AAPL": holding()}, {}, at(df), CFG) == []
