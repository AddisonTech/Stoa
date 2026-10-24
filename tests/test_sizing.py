"""Sizing by confidence and volatility, and every new cap in the risk engine."""

from __future__ import annotations

import math
from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from stoa import risk, sizing
from stoa.config import RiskLimits
from stoa.models import Candidate
from stoa.universe import AssetInfo, Universe

D = Decimal
LIMITS = RiskLimits()
WATCH = ("SPY", "AAPL", "MSFT", "NVDA", "AMD", "KO", "XOM", "JPM")
UNIVERSE = Universe(
    watchlist=WATCH, benchmark="SPY", assets={s: AssetInfo(s, True, True) for s in WATCH}
)
SECTORS = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "NVDA": "Information Technology",
    "AMD": "Information Technology",
    "KO": "Consumer Staples",
    "XOM": "Energy",
    "JPM": "Financials",
    "SPY": "ETF",
}


# ---- edge -----------------------------------------------------------------------


def test_linear_edge() -> None:
    assert sizing.linear_edge(0.6) == pytest.approx(0.1)
    assert sizing.linear_edge(0.5) == 0.0
    assert sizing.linear_edge(0.4) == 0.0


def test_kelly_edge_matches_formula_and_is_never_negative() -> None:
    # Even payoff: full Kelly is 2p - 1.
    assert sizing.kelly_edge(0.6, 1.0, 1.0) == pytest.approx(0.2)
    assert sizing.kelly_edge(0.6, 1.0, 0.25) == pytest.approx(0.05)
    # Payoff 2:1 at p = 0.4: f = 0.4 - 0.6 / 2 = 0.1.
    assert sizing.kelly_edge(0.4, 2.0, 1.0) == pytest.approx(0.1)
    assert sizing.kelly_edge(0.3, 1.0, 1.0) == 0.0
    assert sizing.kelly_edge(0.9, 0.0, 1.0) == 0.0
    with pytest.raises(ValueError):
        sizing.kelly_edge(1.5, 1.0, 1.0)


def test_bigger_when_sure_smaller_when_volatile() -> None:
    items = [
        sizing.SizingInput("A", 0.60, 0.20),
        sizing.SizingInput("B", 0.70, 0.20),
        sizing.SizingInput("C", 0.70, 0.40),
    ]
    w = sizing.raw_weights(items, edge="linear", payoff_ratio=1, kelly_fraction=1, vol_floor=0.1)
    assert w["B"] > w["A"]
    assert w["B"] == pytest.approx(2 * w["C"])


def test_vol_floor_and_min_edge() -> None:
    items = [sizing.SizingInput("Q", 0.6, 0.01), sizing.SizingInput("R", 0.52, 0.2)]
    w = sizing.raw_weights(
        items, edge="linear", payoff_ratio=1, kelly_fraction=1, vol_floor=0.10, min_edge=0.05
    )
    assert w["Q"] == pytest.approx(0.1 / 0.10)
    assert w["R"] == 0.0


# ---- volatility targeting -----------------------------------------------------------


def diag(vols: dict[str, float]) -> dict[str, dict[str, float]]:
    return {a: {b: (vols[a] ** 2 if a == b else 0.0) for b in vols} for a in vols}


def test_portfolio_vol_uncorrelated() -> None:
    cov = diag({"A": 0.2, "B": 0.2})
    assert sizing.portfolio_vol({"A": 0.5, "B": 0.5}, cov) == pytest.approx(math.sqrt(0.02))


def test_scale_to_target_hits_target_with_existing_book() -> None:
    cov = diag({"A": 0.3, "B": 0.2, "C": 0.25})
    existing = {"A": 0.2}
    new = {"B": 1.0, "C": 0.5}
    k = sizing.scale_to_target(new, existing, cov, 0.12)
    combined = {"A": 0.2, "B": k, "C": 0.5 * k}
    assert sizing.portfolio_vol(combined, cov) == pytest.approx(0.12, rel=1e-6)


def test_scale_is_zero_when_existing_book_is_already_at_target() -> None:
    cov = diag({"A": 0.5, "B": 0.2})
    assert sizing.scale_to_target({"B": 1.0}, {"A": 0.5}, cov, 0.15) == 0.0


def test_size_positions_applies_event_multiplier() -> None:
    cov = diag({"A": 0.2})
    full = sizing.size_positions([sizing.SizingInput("A", 0.6, 0.2)], {}, cov, vol_target=0.1)
    half = sizing.size_positions(
        [sizing.SizingInput("A", 0.6, 0.2)], {}, cov, vol_target=0.1, multiplier=0.5
    )
    assert full["A"] == pytest.approx(0.5)  # 0.1 target / 0.2 vol
    assert half["A"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        sizing.size_positions([], {}, cov, multiplier=0)


def test_covariance_and_correlation() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.01, 200)
    b = a * 0.9 + rng.normal(0, 0.002, 200)
    cov = sizing.covariance({"A": list(a), "B": list(b)})
    assert cov["A"]["A"] == pytest.approx(np.var(a, ddof=1) * 252)
    c = sizing.correlation(list(a), list(b))
    assert c is not None and c > 0.9
    assert sizing.correlation([1, 1, 1], [1, 2, 3]) is None
    assert sizing.correlation([1, 2], [1, 2]) is None


# ---- risk engine caps ---------------------------------------------------------------


def idea(sym: str, weight: str | None = None, max_weight: str | None = None) -> risk.EntryIdea:
    c = Candidate(symbol=sym, conviction=4, thesis="t", score=1.0, source="model")
    return risk.EntryIdea(
        candidate=c,
        ref_price=D(100),
        atr=D(2),
        stop_price=D(95),
        target_weight=D(weight) if weight else None,
        max_weight=D(max_weight) if max_weight else None,
    )


def ctx(
    positions: dict[str, risk.HeldPosition] | None = None,
    cash: str | None = None,
    **kw: object,
) -> risk.RiskContext:
    positions = positions or {}
    invested = sum((p.value for p in positions.values()), D(0))
    base: dict[str, object] = {
        "session": date(2024, 6, 3),
        "equity": D(100),
        "cash": D(cash) if cash else D(100) - invested,
        "positions": positions,
        "prev_equity": None,
        "high_water_mark": D(100),
        "kill_switch_active": False,
        "regime_bull": True,
        "universe": UNIVERSE,
        "limits": LIMITS,
        "sectors": SECTORS,
    }
    base.update(kw)
    return risk.RiskContext(**base)  # type: ignore[arg-type]


def held(sym: str, value: str) -> risk.HeldPosition:
    return risk.HeldPosition(sym, D(value) / D(100), D(100))


def notional(res: risk.RiskResult) -> dict[str, Decimal]:
    return {o.symbol: o.notional for o in res.orders if o.notional is not None}


def reasons(res: risk.RiskResult) -> dict[str, str]:
    return {r.symbol: r.reason for r in res.rejections}


def test_target_weight_is_used_and_clipped_at_position_cap() -> None:
    res = risk.evaluate(ctx(), [], [idea("AAPL", "0.07"), idea("KO", "0.5")])
    assert notional(res) == {"AAPL": D("7.00"), "KO": D("20.00")}


def test_sector_cap_pass_and_fail() -> None:
    # Two tech positions at 20% each: the sector is at 40%, a third is refused.
    pos = {"AAPL": held("AAPL", "20"), "MSFT": held("MSFT", "20")}
    res = risk.evaluate(ctx(pos), [], [idea("NVDA"), idea("KO")])
    assert reasons(res) == {"NVDA": "sector_cap"}
    assert notional(res) == {"KO": D("20.00")}
    # One tech position at 25%: a new tech buy is cut to the 15% left in the sector.
    pos2 = {"AAPL": held("AAPL", "25")}
    res2 = risk.evaluate(ctx(pos2), [], [idea("NVDA")])
    assert notional(res2) == {"NVDA": D("15.00")}


def test_sector_cap_counts_buys_in_the_same_plan() -> None:
    res = risk.evaluate(ctx(), [], [idea("AAPL"), idea("MSFT"), idea("NVDA")])
    assert notional(res) == {"AAPL": D("20.00"), "MSFT": D("20.00")}
    assert reasons(res) == {"NVDA": "sector_cap"}


def test_selling_frees_sector_room() -> None:
    from stoa.models import ExitSignal

    pos = {"AAPL": held("AAPL", "20"), "MSFT": held("MSFT", "20")}
    exits = [ExitSignal(symbol="AAPL", reason="stop_loss", detail="")]
    res = risk.evaluate(ctx(pos, cash="60"), exits, [idea("NVDA")])
    assert notional(res) == {"NVDA": D("20.00")}


def test_correlation_check_pass_and_fail() -> None:
    rng = np.random.default_rng(5)
    base = rng.normal(0, 0.01, 60)
    returns = {
        "AAPL": list(base),
        "MSFT": list(base * 1.1 + rng.normal(0, 0.001, 60)),  # near copy
        "KO": list(rng.normal(0, 0.01, 60)),  # independent
    }
    pos = {"AAPL": held("AAPL", "20")}
    res = risk.evaluate(ctx(pos, returns=returns), [], [idea("MSFT"), idea("KO")])
    assert reasons(res)["MSFT"] == "correlation" and "AAPL" in res.rejections[0].detail
    assert "KO" in notional(res)


def test_correlation_checks_buys_planned_together() -> None:
    rng = np.random.default_rng(6)
    base = rng.normal(0, 0.01, 60)
    returns = {"NVDA": list(base), "AMD": list(base + rng.normal(0, 0.0005, 60))}
    res = risk.evaluate(ctx(returns=returns), [], [idea("NVDA"), idea("AMD")])
    assert reasons(res) == {"AMD": "correlation"}


def test_correlation_without_history_is_refused_when_history_is_supplied() -> None:
    pos = {"AAPL": held("AAPL", "20")}
    res = risk.evaluate(ctx(pos, returns={"AAPL": [0.01] * 60}), [], [idea("KO")])
    assert reasons(res) == {"KO": "correlation"}


def test_entry_blocks_are_rejections_with_their_reason() -> None:
    blocks = {"AAPL": ("earnings", "earnings on 2024-06-06 inside the 10 session window")}
    res = risk.evaluate(ctx(entry_blocks=blocks), [], [idea("AAPL"), idea("KO")])
    assert reasons(res) == {"AAPL": "earnings"} and "KO" in notional(res)


def test_event_day_multiplier_halves_new_positions_only() -> None:
    from stoa.models import ExitSignal

    pos = {"XOM": held("XOM", "20")}
    exits = [ExitSignal(symbol="XOM", reason="time_exit", detail="")]
    res = risk.evaluate(
        ctx(pos, size_multiplier=D("0.5"), size_note="FOMC 2024-06-12"),
        exits,
        [idea("AAPL"), idea("KO", "0.08")],
    )
    assert notional(res) == {"AAPL": D("10.00"), "KO": D("4.00")}
    sells = [o for o in res.orders if o.side.value == "sell"]
    assert sells and sells[0].qty == D("0.2")
    buy = next(o for o in res.orders if o.symbol == "AAPL")
    assert "FOMC" in buy.thesis


def test_zero_weight_is_rejected() -> None:
    res = risk.evaluate(ctx(), [], [idea("AAPL", "0")])
    assert reasons(res) == {"AAPL": "zero_size"}


def test_halt_blocks_entries_but_not_exits() -> None:
    from stoa.models import ExitSignal

    pos = {"KO": held("KO", "20")}
    exits = [ExitSignal(symbol="KO", reason="stop_loss", detail="")]
    res = risk.evaluate(ctx(pos, halt_reason="drift fallback"), exits, [idea("AAPL")])
    assert reasons(res) == {"AAPL": "halted"}
    assert [o.symbol for o in res.orders] == ["KO"]


def test_benchmark_cap_only_in_fallback_and_only_for_benchmark() -> None:
    # Not in fallback: max_weight is ignored, the 20% cap holds.
    res = risk.evaluate(ctx(), [], [idea("SPY", "0.98", max_weight="0.98")])
    assert notional(res) == {"SPY": D("20.00")}
    # In fallback: the benchmark may go to 98%, nothing else may be bought.
    fb = ctx(fallback_active=True, halt_reason="SPY-hold fallback")
    res2 = risk.evaluate(
        fb, [], [idea("SPY", "0.98", max_weight="0.98"), idea("AAPL", max_weight="0.98")]
    )
    assert notional(res2) == {"SPY": D("98.00")}
    assert reasons(res2) == {"AAPL": "halted"}
    # Even in fallback the cash buffer is kept: 98% is the most.
    res3 = risk.evaluate(fb, [], [idea("SPY", "1", max_weight="1")])
    assert notional(res3) == {"SPY": D("98.00")}


def test_fallback_ignores_risk_off_but_not_the_kill_switch() -> None:
    fb = ctx(fallback_active=True, halt_reason="fallback", regime_bull=False)
    assert notional(risk.evaluate(fb, [], [idea("SPY", "0.98", "0.98")])) == {"SPY": D("98.00")}
    ks = ctx(fallback_active=True, halt_reason="fallback", kill_switch_active=True)
    assert notional(risk.evaluate(ks, [], [idea("SPY", "0.98", "0.98")])) == {}


def test_hard_caps_hold_for_any_sizing_output() -> None:
    """Whatever sizing asks for, the caps hold: 20% each, 5 positions, 98% gross, 40% sector."""
    rng = np.random.default_rng(11)
    syms = ["AAPL", "MSFT", "NVDA", "AMD", "KO", "XOM", "JPM"]
    for _ in range(200):
        weights = rng.uniform(0, 2, len(syms))
        ideas = [idea(s, f"{w:.4f}") for s, w in zip(syms, weights, strict=True)]
        res = risk.evaluate(ctx(), [], ideas)
        n = notional(res)
        assert len(n) <= LIMITS.max_positions
        assert all(v <= D(20) for v in n.values())
        assert sum(n.values(), D(0)) <= D(98)
        tech = sum((v for s, v in n.items() if SECTORS[s] == "Information Technology"), D(0))
        assert tech <= D(40)


# ---- execution-time checks -----------------------------------------------------------


def exec_ctx(**kw: object) -> risk.ExecutionContext:
    base: dict[str, object] = {
        "equity": D(100),
        "cash": D(100),
        "invested": D(0),
        "prev_equity": None,
        "high_water_mark": D(100),
        "kill_switch_active": False,
        "held": frozenset(),
        "open_positions": 0,
        "orders_today": 0,
        "traded_today": {},
        "universe": UNIVERSE,
        "limits": LIMITS,
        "sectors": SECTORS,
    }
    base.update(kw)
    return risk.ExecutionContext(**base)  # type: ignore[arg-type]


def planned(sym: str, amount: str, max_weight: str | None = None) -> object:
    from stoa.models import PlannedOrder, Side

    return PlannedOrder(
        symbol=sym,
        side=Side.BUY,
        notional=D(amount),
        ref_price=D(100),
        reason="entry",
        max_weight=D(max_weight) if max_weight else None,
    )


def test_execution_sector_cap_rechecked_with_live_positions() -> None:
    ctx_ = exec_ctx(
        cash=D(60),
        invested=D(40),
        held=frozenset({"AAPL", "MSFT"}),
        open_positions=2,
        position_values={"AAPL": D(20), "MSFT": D(20)},
    )
    ok, rej = risk.check_buys(
        [planned("NVDA", "20"), planned("KO", "20")],  # type: ignore[list-item]
        {"NVDA": D(100), "KO": D(100)},
        ctx_,
    )
    assert [o.symbol for o in ok] == ["KO"] and rej[0].reason == "sector_cap"


def test_execution_halt_and_fallback_cap() -> None:
    ok, rej = risk.check_buys(
        [planned("AAPL", "20")],  # type: ignore[list-item]
        {"AAPL": D(100)},
        exec_ctx(halt_reason="drift"),
    )
    assert ok == [] and rej[0].reason == "halted"
    ok2, _ = risk.check_buys(
        [planned("SPY", "98", "0.98")],  # type: ignore[list-item]
        {"SPY": D(100)},
        exec_ctx(halt_reason="fallback", fallback_active=True),
    )
    assert ok2[0].notional == D("98.00")
    ok3, _ = risk.check_buys(
        [planned("SPY", "98", "0.98")],  # type: ignore[list-item]
        {"SPY": D(100)},
        exec_ctx(),
    )
    assert ok3[0].notional == D("20.00")
