from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from stoa.config import Settings
from stoa.data.base import Bars
from stoa.features import build_features, regime_is_bull, snapshot
from stoa.models import Holding, PositionMeta
from stoa.roles.llm import Budget, LLMClient, ReplayCache
from stoa.roles.prompts import ScoutPick, ScoutResponse, build_anon_map, build_user_prompt
from stoa.roles.scout import LLMScout, RuleScout, ScoutInput, passes_trend_filter, validate_picks
from tests.fakes import FakeLLMClient, FakeResponse


def bull_day(bars: Bars) -> date:
    f = build_features(bars, "SPY")
    for ts in reversed(f["SPY"].index):
        if regime_is_bull(f, "SPY", ts.date()):
            return ts.date()
    raise AssertionError("no bull day")


def bear_day(bars: Bars) -> date:
    f = build_features(bars, "SPY")
    for ts in f["SPY"].index[260:]:
        if not regime_is_bull(f, "SPY", ts.date()):
            return ts.date()
    raise AssertionError("no bear day")


def scout_input(bars: Bars, day: date, holdings: dict[str, Holding] | None = None) -> ScoutInput:
    f = build_features(bars, "SPY")
    return ScoutInput(
        session=day,
        snapshot=snapshot(f, day),
        benchmark="SPY",
        regime_bull=regime_is_bull(f, "SPY", day),
        holdings=holdings or {},
        recent_trades=(),
    )


def holding(sym: str, day: date) -> Holding:
    meta = PositionMeta(entry_session=day, entry_price=Decimal(100), entry_atr=Decimal(2))
    return Holding(symbol=sym, qty=Decimal("0.5"), avg_entry_price=Decimal(100), meta=meta)


def test_rule_scout_ranks_by_momentum(small_settings: Settings, bars: Bars) -> None:
    inp = scout_input(bars, bull_day(bars))
    cands, run = RuleScout(small_settings.scout).propose(inp)
    assert run.source == "rule"
    assert cands, "expected candidates on a bull day"
    scores = [c.score for c in cands]
    assert scores == sorted(scores, reverse=True)
    for c in cands:
        assert passes_trend_filter(inp.snapshot.loc[c.symbol])
        assert 1 <= c.conviction <= 5
    assert len(cands) <= small_settings.scout.max_candidates


def test_rule_scout_risk_off_returns_nothing(small_settings: Settings, bars: Bars) -> None:
    inp = scout_input(bars, bear_day(bars))
    assert not inp.regime_bull
    cands, _ = RuleScout(small_settings.scout).propose(inp)
    assert cands == []


def test_rule_scout_skips_holdings(small_settings: Settings, bars: Bars) -> None:
    day = bull_day(bars)
    first, _ = RuleScout(small_settings.scout).propose(scout_input(bars, day))
    top = first[0].symbol
    again, _ = RuleScout(small_settings.scout).propose(
        scout_input(bars, day, {top: holding(top, day)})
    )
    assert top not in {c.symbol for c in again}


# ---- LLM scout -----------------------------------------------------------------


def make_llm(
    settings: Settings, responder: object, tmp: Path | None, spent: float = 0.0
) -> tuple[LLMScout, FakeLLMClient]:
    fake = FakeLLMClient(responder)
    ledger = {}
    budget = Budget(settings.llm.monthly_cost_cap_usd, ledger)
    if spent:
        budget.record(spent)
    client = LLMClient(settings.llm, fake, ReplayCache(tmp), budget)
    return LLMScout(settings.scout, settings.llm, client), fake


def picks_for(inp: ScoutInput, symbols: list[str], conviction: int = 4) -> ScoutResponse:
    amap = build_anon_map([s for s in inp.snapshot.index if s != "SPY"], inp.session)
    return ScoutResponse(
        candidates=[
            ScoutPick(asset_id=amap.to_alias[s], conviction=conviction, thesis=f"likes {s}")
            for s in symbols
        ]
    )


def test_llm_scout_valid_parse_deanonymizes(
    small_settings: Settings, bars: Bars, tmp_path: Path
) -> None:
    inp = scout_input(bars, bull_day(bars))
    scout, fake = make_llm(small_settings, None, tmp_path)
    fake.messages.responder = FakeResponse(picks_for(inp, ["NVDA", "KO"]))
    cands, run = scout.propose(inp)
    assert [c.symbol for c in cands] == ["NVDA", "KO"]
    assert all(c.source == "llm" for c in cands)
    assert run.source == "llm" and run.fallback_reason is None
    assert run.input_tokens == 1200 and run.output_tokens == 300
    assert run.cost_usd == pytest.approx((1200 * 0.10 + 300 * 0.50) / 1e6)
    call = fake.messages.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_format"] is ScoutResponse
    prompt = call["messages"][0]["content"]
    for sym in inp.snapshot.index:
        assert f"{sym}," not in prompt and f" {sym}:" not in prompt
    assert str(inp.session) not in prompt


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_llm_scout_bad_stop_reason_falls_back(
    small_settings: Settings, bars: Bars, tmp_path: Path, stop: str
) -> None:
    inp = scout_input(bars, bull_day(bars))
    scout, fake = make_llm(small_settings, None, tmp_path)
    fake.messages.responder = FakeResponse(picks_for(inp, ["NVDA"]), stop_reason=stop)
    cands, run = scout.propose(inp)
    rule, _ = RuleScout(small_settings.scout).propose(inp)
    assert run.fallback_reason == stop
    assert run.source == "rule"
    assert [c.symbol for c in cands] == [c.symbol for c in rule]


def test_llm_scout_api_error_falls_back(
    small_settings: Settings, bars: Bars, tmp_path: Path
) -> None:
    inp = scout_input(bars, bull_day(bars))
    scout, _ = make_llm(small_settings, RuntimeError("boom"), tmp_path)
    _, run = scout.propose(inp)
    assert run.fallback_reason == "api_error"


def test_llm_scout_unknown_and_out_of_range_rejected(
    small_settings: Settings, bars: Bars, tmp_path: Path
) -> None:
    inp = scout_input(bars, bull_day(bars))
    good = picks_for(inp, ["AAPL"])
    resp = ScoutResponse(
        candidates=[
            ScoutPick(asset_id="Asset ZZ", conviction=5, thesis="made up"),
            ScoutPick(asset_id="SPY", conviction=5, thesis="ticker leak"),
            ScoutPick(asset_id=good.candidates[0].asset_id, conviction=9, thesis="too sure"),
            good.candidates[0],
        ]
    )
    scout, _ = make_llm(small_settings, FakeResponse(resp), tmp_path)
    cands, _ = scout.propose(inp)
    assert [c.symbol for c in cands] == ["AAPL"]


def test_llm_scout_cost_cap_falls_back_without_calling(
    small_settings: Settings, bars: Bars, tmp_path: Path
) -> None:
    inp = scout_input(bars, bull_day(bars))
    scout, fake = make_llm(
        small_settings, FakeResponse(picks_for(inp, ["NVDA"])), tmp_path, spent=5.0
    )
    _, run = scout.propose(inp)
    assert run.fallback_reason == "cost_cap"
    assert fake.messages.calls == []


def test_llm_replay_cache_avoids_second_call(
    small_settings: Settings, bars: Bars, tmp_path: Path
) -> None:
    inp = scout_input(bars, bull_day(bars))
    scout, fake = make_llm(small_settings, FakeResponse(picks_for(inp, ["NVDA"])), tmp_path)
    first, r1 = scout.propose(inp)
    second, r2 = scout.propose(inp)
    assert len(fake.messages.calls) == 1
    assert [c.symbol for c in first] == [c.symbol for c in second]
    assert not r1.cache_hit and r2.cache_hit
    assert r1.charged_usd > 0 and r2.charged_usd == 0.0
    assert r2.cost_usd == r1.cost_usd


def test_llm_no_client_falls_back(small_settings: Settings, bars: Bars, tmp_path: Path) -> None:
    inp = scout_input(bars, bull_day(bars))
    client = LLMClient(small_settings.llm, None, ReplayCache(tmp_path), Budget(5.0))
    _, run = LLMScout(small_settings.scout, small_settings.llm, client).propose(inp)
    assert run.fallback_reason == "no_client"


def test_llm_scout_risk_off_makes_no_call(
    small_settings: Settings, bars: Bars, tmp_path: Path
) -> None:
    inp = scout_input(bars, bear_day(bars))
    scout, fake = make_llm(small_settings, FakeResponse(ScoutResponse(candidates=[])), tmp_path)
    cands, _ = scout.propose(inp)
    assert cands == [] and fake.messages.calls == []


def test_anonymization_round_trip(bars: Bars) -> None:
    syms = ["AAPL", "MSFT", "NVDA", "KO"]
    amap = build_anon_map(syms, date(2024, 3, 1))
    assert sorted(amap.to_alias) == sorted(syms)
    assert all(a.startswith("Asset ") for a in amap.to_alias.values())
    assert {amap.to_symbol[amap.to_alias[s]] for s in syms} == set(syms)
    other = build_anon_map(syms, date(2024, 3, 4))
    assert amap == build_anon_map(syms, date(2024, 3, 1))
    assert sorted(other.to_alias.values()) == sorted(amap.to_alias.values())


def test_prompt_without_anonymization_names_tickers(small_settings: Settings, bars: Bars) -> None:
    inp = scout_input(bars, bull_day(bars))
    text, amap = build_user_prompt(
        session=inp.session,
        snap=inp.snapshot,
        benchmark="SPY",
        regime_bull=True,
        holdings={},
        recent_trades=(),
        max_candidates=3,
        anonymize=False,
        decimals=3,
    )
    assert "AAPL," in text and amap.to_symbol["AAPL"] == "AAPL"
    assert str(inp.session) in text


def test_validate_picks_dedupes_and_caps() -> None:
    resp = ScoutResponse(
        candidates=[ScoutPick(asset_id="Asset A", conviction=3, thesis="x " * 400)] * 2
        + [ScoutPick(asset_id="Asset B", conviction=1, thesis="y")]
    )
    amap = {"Asset A": "AAPL", "Asset B": "KO"}
    cands, problems = validate_picks(resp, amap, max_candidates=3)
    assert [c.symbol for c in cands] == ["AAPL", "KO"]
    assert len(cands[0].thesis) <= 280
    assert any("duplicate" in p for p in problems)
    capped, _ = validate_picks(resp, amap, max_candidates=1)
    assert [c.symbol for c in capped] == ["AAPL"]


def test_prompt_snapshot_rounding(bars: Bars) -> None:
    inp = scout_input(bars, bull_day(bars))
    text, _ = build_user_prompt(
        session=inp.session,
        snap=inp.snapshot,
        benchmark="SPY",
        regime_bull=True,
        holdings={},
        recent_trades=(),
        max_candidates=3,
        anonymize=True,
        decimals=2,
    )
    table = [ln for ln in text.splitlines() if ln.startswith("Asset ")]
    assert table
    for value in table[0].split(",")[1:]:
        assert value == "na" or len(value.split(".")[-1]) == 2
    assert isinstance(pd.Timestamp(inp.session), pd.Timestamp)
