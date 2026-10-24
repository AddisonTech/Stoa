"""Idea generation: a rule-based scout and an LLM scout that falls back to it."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd

from stoa.config import LLMConfig, ScoutConfig
from stoa.models import Candidate, ClosedTrade, Holding
from stoa.planner import ScoutRun
from stoa.roles.llm import LLMClient, LLMFailure
from stoa.roles.prompts import SYSTEM_PROMPT, ScoutResponse, build_user_prompt

log = logging.getLogger(__name__)

MAX_THESIS_CHARS = 280


@dataclass(frozen=True)
class ScoutInput:
    session: date
    snapshot: pd.DataFrame
    benchmark: str
    regime_bull: bool
    holdings: Mapping[str, Holding]
    recent_trades: Sequence[ClosedTrade]


class Scout(Protocol):
    def propose(self, inp: ScoutInput) -> tuple[list[Candidate], ScoutRun]: ...


def momentum_score(row: pd.Series, cfg: ScoutConfig) -> float:
    return cfg.weight_return_60 * float(row["ret_60"]) + cfg.weight_return_120 * float(
        row["ret_120"]
    )


def passes_trend_filter(row: pd.Series) -> bool:
    vals = [
        float(row.get(k, math.nan)) for k in ("close", "sma_50", "sma_200", "ret_60", "ret_120")
    ]
    if any(math.isnan(v) for v in vals):
        return False
    close, sma50, sma200, _, _ = vals
    return close > sma50 > sma200


class RuleScout:
    """Momentum plus trend. Deterministic, free, and the stand-in for tests."""

    def __init__(self, cfg: ScoutConfig) -> None:
        self.cfg = cfg

    def propose(self, inp: ScoutInput) -> tuple[list[Candidate], ScoutRun]:
        run = ScoutRun(source="rule")
        if not inp.regime_bull:
            return [], run
        scored = []
        for sym in inp.snapshot.index:
            if sym in inp.holdings:
                continue
            row = inp.snapshot.loc[sym]
            if passes_trend_filter(row):
                scored.append((momentum_score(row, self.cfg), sym, row))
        scored.sort(key=lambda t: (-t[0], t[1]))
        out = []
        for rank, (score, sym, row) in enumerate(scored[: self.cfg.max_candidates]):
            thesis = (
                f"Momentum score {score:.3f}: 60 day {float(row['ret_60']):+.1%}, "
                f"120 day {float(row['ret_120']):+.1%}, above its 50 and 200 day averages."
            )
            out.append(
                Candidate(
                    symbol=sym,
                    conviction=max(1, 5 - rank),
                    thesis=thesis,
                    score=score,
                    source="rule",
                )
            )
        return out, run


def validate_picks(
    response: ScoutResponse, to_symbol: Mapping[str, str], max_candidates: int
) -> tuple[list[Candidate], list[str]]:
    """De-anonymize and range-check model output. Returns candidates and problems found."""
    out: list[Candidate] = []
    problems: list[str] = []
    seen: set[str] = set()
    for pick in response.candidates:
        sym = to_symbol.get(pick.asset_id.strip())
        if sym is None:
            problems.append(f"unknown id {pick.asset_id!r}")
            continue
        if sym in seen:
            problems.append(f"duplicate id {pick.asset_id!r}")
            continue
        if not 1 <= pick.conviction <= 5:
            problems.append(f"conviction {pick.conviction} out of range for {pick.asset_id!r}")
            continue
        seen.add(sym)
        thesis = " ".join(pick.thesis.split())[:MAX_THESIS_CHARS]
        out.append(
            Candidate(
                symbol=sym,
                conviction=pick.conviction,
                thesis=thesis,
                score=float(pick.conviction),
                source="llm",
            )
        )
        if len(out) >= max_candidates:
            break
    return out, problems


class LLMScout:
    """Asks Claude for ideas. On any failure it returns the rule scout's ideas instead."""

    def __init__(self, scout_cfg: ScoutConfig, llm_cfg: LLMConfig, client: LLMClient) -> None:
        self.cfg = scout_cfg
        self.llm_cfg = llm_cfg
        self.client = client
        self.fallback = RuleScout(scout_cfg)

    def propose(self, inp: ScoutInput) -> tuple[list[Candidate], ScoutRun]:
        if not inp.regime_bull:
            return [], ScoutRun(source="llm", model=self.llm_cfg.model)
        user, amap = build_user_prompt(
            session=inp.session,
            snap=inp.snapshot,
            benchmark=inp.benchmark,
            regime_bull=inp.regime_bull,
            holdings=inp.holdings,
            recent_trades=inp.recent_trades,
            max_candidates=self.cfg.max_candidates,
            anonymize=self.llm_cfg.anonymize,
            decimals=self.llm_cfg.feature_decimals,
        )
        try:
            result = self.client.parse(SYSTEM_PROMPT, user, ScoutResponse)
        except LLMFailure as exc:
            log.warning("LLM scout failed (%s); using rule scout", exc.reason)
            cands, _ = self.fallback.propose(inp)
            return cands, ScoutRun(
                source="rule", model=self.llm_cfg.model, fallback_reason=exc.reason
            )
        assert isinstance(result.parsed, ScoutResponse)
        cands, problems = validate_picks(result.parsed, amap.to_symbol, self.cfg.max_candidates)
        for p in problems:
            log.warning("LLM scout output rejected: %s", p)
        cands = [c for c in cands if c.symbol not in inp.holdings]
        return cands, ScoutRun(
            source="llm",
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            charged_usd=result.charged_usd,
            cache_hit=result.cache_hit,
            fallback_reason=None,
        )
