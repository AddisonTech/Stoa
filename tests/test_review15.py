"""Trial 15's nightly self-review: a reason for each losing pick, a running lessons file, and
rules that switch on only when their reason recurs and they pass the decade check."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stoa.newsbrain import review15 as rv


def pick(symbol: str, **saw: float) -> dict[str, object]:
    return {"symbol": symbol, "rank": 1, "conviction": 0.6, "saw": {"n": 2.0, **saw}}


def graded(symbol: str, r: float, spy: float, net: float) -> dict[str, object]:
    return {"symbol": symbol, "r": r, "spy": spy, "net": net, "hit": net > 0}


def test_each_loss_gets_the_first_reason_that_fits() -> None:
    g = graded("A", 0.012, 0.010, -0.001)
    assert rv.reason_for(pick("A"), g) == "costs"  # beat SPY before costs
    g = graded("A", -0.03, 0.0, -0.031)
    assert rv.reason_for(pick("A"), g, later_news=-0.7) == "bad_news"
    assert rv.reason_for(pick("A", ret_1=0.09), g) == "chased_jump"
    assert rv.reason_for(pick("A", n=0.0), g) == "no_news"
    assert rv.reason_for(pick("A", vol_20=0.05), g) == "noise"
    assert rv.reason_for(pick("A", vol_20=0.01), g) == "unexplained"


def test_the_review_and_the_lessons_count() -> None:
    picks = [pick("A", ret_1=0.08), pick("B")]
    day = [graded("A", -0.04, 0.0, -0.041), graded("B", 0.02, 0.0, 0.019)]
    reviews = rv.review_day(picks, day)
    assert [r["symbol"] for r in reviews] == ["A", "B"]
    assert reviews[0]["reason"] == "chased_jump" and "reason" not in reviews[1]
    lessons: dict[str, object] = {}
    for i in range(5):
        lessons = rv.update_lessons(lessons, f"2026-11-0{i + 2}", reviews)
    c = lessons["reasons"]["chased_jump"]  # type: ignore[index]
    assert c["count"] == 5 and c["rule"] == "skip_jumps" and c["rule_due"]
    assert len(c["examples"]) == 5
    # Due by its count, but used only when the decade check passed.
    assert rv.active_rules(lessons, {"skip_jumps": {"passed": False}}) == []
    assert rv.active_rules(lessons, {"skip_jumps": {"passed": True}}) == ["skip_jumps"]


def test_the_decade_check_needs_overall_and_7_of_10_years() -> None:
    rng = np.random.default_rng(0)
    n_days, per_day = 10 * 20, 8
    sidx = np.repeat(np.arange(n_days), per_day)
    jump = rng.random(n_days * per_day) < 0.3
    rows = pd.DataFrame(
        {
            "sidx": sidx,
            "symbol": [f"S{i % per_day}" for i in range(n_days * per_day)],
            "ret_1": np.where(jump, 0.1, 0.0),
            "excess_1": np.where(jump, -0.02, 0.01) + rng.normal(0, 0.001, n_days * per_day),
            "y_1": 0.0,
        }
    )
    conv = np.where(jump, 0.9, 0.5) + rng.random(n_days * per_day) / 100
    year_of = {i: 2016 + i // 20 for i in range(n_days)}
    got = rv.decade_check(rows, conv, year_of, "skip_jumps")
    assert got["passed"] and got["years"] == 10 and got["years_better"] == 10
    assert got["edge_with"] > got["edge_without"]
    kept = rv.apply_rules(rows, ["skip_jumps"])
    assert (kept["ret_1"] <= 0.05).all()
