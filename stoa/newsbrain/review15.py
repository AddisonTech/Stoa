"""Trial 15's nightly self-review and its lessons (docs/DECISIONS.md, trial 15).

After the 5:15 pm summary grades the picks of the session before, each pick gets a review:
what it saw (its inputs and conviction), what happened against SPY after costs, and when it
was wrong, the most likely reason, from a fixed list by fixed rules (``REASONS``, checked in
order, the first that fits). The lessons file counts the reasons.

A reason seen at least 5 times may turn into a rule from a fixed menu (``RULES``). A rule is
used only if over 2016 to the last full year it raises the top 5's average edge overall and
in at least 7 of the 10 years (``decade_check``), so one bad day never changes anything. The
check runs once per rule on the Dell, where the decade's rows are, and is logged in the
registry and in ``rules.json``; a rule switches on only when both its count and its check
say so.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa.newsbrain import trial15

JUMP = 0.05  # a pick up more than 5% the day before was chasing a jump
COST_CAP = 0.005  # a round trip dearer than 0.5% of the money
VOLATILE_SHARE = 0.10  # the most volatile tenth of the day's rows


@dataclass(frozen=True)
class Reason:
    key: str
    words: str  # plain words for the summary and the lessons file
    rule: str | None  # the menu rule it can turn into, if any


REASONS = (
    Reason("costs", "it beat SPY, but not by enough to pay its trading costs", "cost_cap"),
    Reason("bad_news", "bad news about it came out after the buy", None),
    Reason(
        "chased_jump", "it had jumped more than 5% the day before and gave some back", "skip_jumps"
    ),
    Reason("no_news", "there was no news behind it that day", "need_news"),
    Reason("noise", "a normal day's wobble for a stock this jumpy", "skip_volatile"),
    Reason("unexplained", "a loss bigger than its usual wobble, with no clear cause", None),
)
BY_KEY = {r.key: r for r in REASONS}


def reason_for(
    pick: Mapping[str, Any], graded: Mapping[str, Any], later_news: float | None = None
) -> str:
    """The most likely reason a losing pick lost. ``pick["saw"]`` holds its inputs;
    ``graded`` its r, spy and net; ``later_news`` the reader's direction for news about it
    between the buy and the sale (None when not known)."""
    saw = pick.get("saw", {})
    r, spy = graded.get("r"), graded.get("spy")
    if r is not None and spy is not None and r - spy > 0:
        return "costs"
    if later_news is not None and later_news < -0.2:
        return "bad_news"
    if float(saw.get("ret_1", 0.0)) > JUMP:
        return "chased_jump"
    if float(saw.get("n", 0.0)) <= 0:
        return "no_news"
    vol = float(saw.get("vol_20", math.nan))
    net = graded.get("net")
    if net is not None and math.isfinite(vol) and abs(float(net)) <= vol:
        return "noise"
    return "unexplained"


def review_day(
    picks: Sequence[Mapping[str, Any]],
    graded: Sequence[Mapping[str, Any]],
    later_news: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """One review per graded pick: what it saw, what happened, and why when it lost."""
    by_symbol = {str(g["symbol"]): g for g in graded}
    out = []
    for p in picks:
        g = by_symbol.get(str(p["symbol"]))
        if g is None or g.get("hit") is None:
            continue
        rec: dict[str, Any] = {
            "symbol": p["symbol"],
            "rank": p.get("rank"),
            "conviction": p.get("conviction"),
            "saw": p.get("saw", {}),
            "return": g["r"],
            "spy": g["spy"],
            "net": g["net"],
            "hit": g["hit"],
        }
        if not g["hit"]:
            key = reason_for(p, g, (later_news or {}).get(str(p["symbol"])))
            rec["reason"] = key
            rec["why"] = BY_KEY[key].words
        out.append(rec)
    return out


def update_lessons(
    lessons: dict[str, Any], session: str, reviews: list[dict[str, Any]]
) -> dict[str, Any]:
    """Count each reason and keep its last 5 examples; say when a reason's rule is due."""
    counts: dict[str, Any] = lessons.setdefault("reasons", {})
    for rv in reviews:
        key = rv.get("reason")
        if key is None:
            continue
        c = counts.setdefault(key, {"count": 0, "words": BY_KEY[key].words, "examples": []})
        c["count"] += 1
        c["examples"] = [*c["examples"], f"{session} {rv['symbol']} {rv['net']:+.2%}"][-5:]
        rule = BY_KEY[key].rule
        c["rule"] = rule
        c["rule_due"] = rule is not None and c["count"] >= trial15.LESSON_MIN_COUNT
    lessons["reviewed_days"] = sorted({*lessons.get("reviewed_days", []), session})
    return lessons


# ---- the rule menu and the decade check ---------------------------------------------------------

RuleFn = Callable[[pd.DataFrame], pd.Series]  # rows to a keep mask


def _skip_jumps(rows: pd.DataFrame) -> pd.Series:
    return ~(rows["ret_1"].astype(float) > JUMP)


def _need_news(rows: pd.DataFrame) -> pd.Series:
    return rows["n"].fillna(0).astype(float) > 0


def _cost_cap(rows: pd.DataFrame) -> pd.Series:
    from stoa import wide

    cost = rows["adv"].map(lambda v: 2 * wide.slippage_bps(v if pd.notna(v) else None) / 1e4)
    return cost <= COST_CAP


def _skip_volatile(rows: pd.DataFrame) -> pd.Series:
    cut = rows.groupby("sidx")["vol_20"].transform(lambda v: v.quantile(1 - VOLATILE_SHARE))
    return ~(rows["vol_20"] > cut)


RULES: dict[str, tuple[str, RuleFn]] = {
    "skip_jumps": ("skip a stock up more than 5% the day before", _skip_jumps),
    "need_news": ("skip a stock with no article that day", _need_news),
    "cost_cap": ("skip a stock whose round trip costs more than 0.5%", _cost_cap),
    "skip_volatile": ("skip the most volatile tenth of the day's stocks", _skip_volatile),
}


def decade_check(
    rows: pd.DataFrame, conviction: np.ndarray, year_of: Mapping[int, int], rule: str
) -> dict[str, Any]:
    """The top 5 a day with and without ``rule`` (``year_of``: session index to year): it
    passes when the average edge is higher over the decade and in at least 7 of its years."""
    from stoa.newsbrain import brain15

    keep = RULES[rule][1](rows).to_numpy()
    base = brain15.top5(rows, conviction)
    with_rule = brain15.top5(rows[keep], conviction[keep])
    years: dict[int, dict[str, float]] = {}
    for label, picks in (("without", base), ("with", with_rule)):
        for y, g in picks.groupby(picks["sidx"].map(year_of)):
            years.setdefault(int(y), {})[label] = float(g["excess_1"].mean())
    better = sum(1 for v in years.values() if v.get("with", -1.0) > v.get("without", 1.0))
    overall = (float(with_rule["excess_1"].mean()), float(base["excess_1"].mean()))
    passed = overall[0] > overall[1] and better >= trial15.LESSON_MIN_YEARS
    return {
        "rule": rule,
        "words": RULES[rule][0],
        "edge_with": overall[0],
        "edge_without": overall[1],
        "years_better": better,
        "years": len(years),
        "by_year": years,
        "passed": bool(passed),
    }


def active_rules(lessons: Mapping[str, Any], checks: Mapping[str, Any]) -> list[str]:
    """Rules in use: due by their reason's count and passed on the decade."""
    due = {c.get("rule") for c in lessons.get("reasons", {}).values() if c.get("rule_due")}
    return sorted(r for r in due if r and checks.get(r, {}).get("passed"))


def apply_rules(rows: pd.DataFrame, rules: Sequence[str]) -> pd.DataFrame:
    """The rows the active rules keep."""
    keep = pd.Series(True, index=rows.index)
    for r in rules:
        keep &= RULES[r][1](rows)
    return rows[keep]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    out: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return out
