"""The two baselines every run is graded against.

Both run through the same planner, simulated broker, costs and calendar as the
agents. They are not risk managed: a baseline that kept a cash buffer or a 20%
position cap would no longer be the thing it claims to measure.
"""

from __future__ import annotations

import math

import pandas as pd

from stoa.features import cell
from stoa.models import PlannedOrder, Side, cents_down, dec
from stoa.planner import Decision, DecisionInput


def _close(inp: DecisionInput, symbol: str) -> float:
    return cell(inp.features[symbol], pd.Timestamp(inp.session), "close")


def _buy_all(inp: DecisionInput, symbol: str, reason: str) -> PlannedOrder | None:
    notional = cents_down(inp.cash)
    if notional < dec("1.00"):
        return None
    return PlannedOrder(
        symbol=symbol,
        side=Side.BUY,
        notional=notional,
        ref_price=dec(_close(inp, symbol)),
        reason=reason,
    )


def _sell_all(inp: DecisionInput, symbol: str, reason: str) -> PlannedOrder:
    h = inp.holdings[symbol]
    return PlannedOrder(
        symbol=symbol, side=Side.SELL, qty=h.qty, ref_price=dec(_close(inp, symbol)), reason=reason
    )


class BuyAndHold:
    """Buy the benchmark with all cash at the first open and never sell."""

    risk_managed = False

    def __init__(self, symbol: str = "SPY", name: str = "spy") -> None:
        self.symbol = symbol
        self.name = name

    def decide(self, inp: DecisionInput) -> Decision:
        if self.symbol in inp.holdings:
            return Decision(orders=())
        order = _buy_all(inp, self.symbol, "buy_and_hold")
        return Decision(orders=(order,) if order else ())


def sma_signal(features: pd.DataFrame, ts: pd.Timestamp) -> bool | None:
    """True when the 50 day average is above the 200 day average, None while warming up."""
    fast = cell(features, ts, "sma_50")
    slow = cell(features, ts, "sma_200")
    if math.isnan(fast) or math.isnan(slow):
        return None
    return fast > slow


class SMACrossover:
    """Fully in the benchmark when its 50 day average is above its 200 day average, else cash."""

    risk_managed = False

    def __init__(self, symbol: str = "SPY") -> None:
        self.symbol = symbol
        self.name = "sma"

    def decide(self, inp: DecisionInput) -> Decision:
        signal = sma_signal(inp.features[self.symbol], pd.Timestamp(inp.session))
        held = self.symbol in inp.holdings
        if signal is True and not held:
            order = _buy_all(inp, self.symbol, "sma_bull")
            return Decision(orders=(order,) if order else ())
        if signal is False and held:
            return Decision(orders=(_sell_all(inp, self.symbol, "sma_bear"),))
        return Decision(orders=())
