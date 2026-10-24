"""Deterministic gates on every idea. Every kill names the gate that fired."""

from __future__ import annotations

import math
from collections.abc import Collection

import pandas as pd

from stoa.config import QuantConfig
from stoa.models import Candidate, QuantKill

REQUIRED = ("close", "sma_50", "ret_5", "ret_60", "vol_20", "rsi_14", "dollar_vol_20", "atr_14")


def _num(row: pd.Series, key: str) -> float:
    value = row.get(key)
    return float(value) if value is not None else math.nan


def gate_one(
    cand: Candidate, row: pd.Series | None, held: Collection[str], cfg: QuantConfig
) -> QuantKill | None:
    sym = cand.symbol

    def kill(gate: str, detail: str) -> QuantKill:
        return QuantKill(symbol=sym, gate=gate, detail=detail)

    if sym in held:
        return kill("already_held", "position already open")
    if row is None:
        return kill("no_data", "no bar for this session")
    values = {k: _num(row, k) for k in REQUIRED}
    missing = [k for k, v in values.items() if math.isnan(v)]
    if missing:
        return kill("insufficient_history", f"missing {', '.join(missing)}")
    if values["close"] <= values["sma_50"]:
        return kill("uptrend", f"close {values['close']:.2f} <= SMA50 {values['sma_50']:.2f}")
    if values["ret_60"] <= cfg.min_return_60:
        return kill("momentum", f"60 day return {values['ret_60']:.3f} <= {cfg.min_return_60}")
    if values["vol_20"] > cfg.max_volatility_20:
        return kill("volatility", f"20 day vol {values['vol_20']:.3f} > {cfg.max_volatility_20}")
    if values["ret_5"] > cfg.max_return_5:
        return kill("spike", f"5 day return {values['ret_5']:.3f} > {cfg.max_return_5}")
    if values["rsi_14"] > cfg.max_rsi_14:
        return kill("overbought", f"RSI14 {values['rsi_14']:.1f} > {cfg.max_rsi_14}")
    if values["dollar_vol_20"] < cfg.min_dollar_volume_20:
        return kill(
            "liquidity",
            f"20 day dollar volume {values['dollar_vol_20']:,.0f} "
            f"< {cfg.min_dollar_volume_20:,.0f}",
        )
    if values["atr_14"] <= 0:
        return kill("insufficient_history", "ATR is zero")
    return None


def apply_gates(
    candidates: list[Candidate], snapshot: pd.DataFrame, held: Collection[str], cfg: QuantConfig
) -> tuple[list[Candidate], list[QuantKill]]:
    passed: list[Candidate] = []
    kills: list[QuantKill] = []
    for cand in candidates:
        row = snapshot.loc[cand.symbol] if cand.symbol in snapshot.index else None
        assert row is None or isinstance(row, pd.Series)
        k = gate_one(cand, row, held, cfg)
        if k is None:
            passed.append(cand)
        else:
            kills.append(k)
    return passed, kills
