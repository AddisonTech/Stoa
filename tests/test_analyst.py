"""The analyst layer: schema, prompt caching flag, replay cache, batch resume, cost cap."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from stoa import analyst as an
from stoa.config import AnalystConfig, BudgetConfig, ModelPrice

PRICES = {"claude-haiku-5-5": ModelPrice(input_per_mtok=0.10, output_per_mtok=0.50)}
CFG = AnalystConfig(batch_poll_seconds=1, batch_max_requests=2)
BUDGET = BudgetConfig()
NOW = datetime(2026, 9, 15, 12, tzinfo=UTC)


def item(i: int, day: str = "2026-09-01") -> an.Item:
    return an.Item(
        item_id=f"alpaca:{i}",
        symbol="ACME",
        known_at=datetime.fromisoformat(f"{day}T14:00:00+00:00"),
        kind="news",
        title=f"Acme headline {i}",
        body="Acme raised its full-year outlook after revenue beat estimates." * 3,
    )


GOOD = an.AnalystScore(
    sentiment=0.7,
    guidance="raised",
    surprise="beat",
    risk_flags=[],
    materiality=0.8,
    confidence=0.9,
)


@dataclass
class Usage:
    input_tokens: int = 300
    output_tokens: int = 60
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 700


@dataclass
class Parsed:
    parsed_output: Any
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)


@dataclass
class Block:
    text: str
    type: str = "text"


@dataclass
class Message:
    content: list[Block]
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)


@dataclass
class Result:
    type: str
    message: Message | None = None


@dataclass
class Line:
    custom_id: str
    result: Result


@dataclass
class BatchObj:
    id: str
    processing_status: str


class FakeBatches:
    def __init__(self) -> None:
        self.created: list[list[dict[str, Any]]] = []
        self.polls: dict[str, int] = {}
        self.fail_ids: set[str] = set()

    def create(self, requests: list[dict[str, Any]]) -> BatchObj:
        bid = f"msgbatch_{len(self.created)}"
        self.created.append(list(requests))
        self.polls[bid] = 0
        return BatchObj(bid, "in_progress")

    def retrieve(self, batch_id: str) -> BatchObj:
        self.polls[batch_id] += 1
        return BatchObj(batch_id, "ended" if self.polls[batch_id] >= 2 else "in_progress")

    def results(self, batch_id: str) -> list[Line]:
        reqs = self.created[int(batch_id.split("_")[1])]
        out = []
        for r in reqs:
            cid = r["custom_id"]
            if cid in self.fail_ids:
                out.append(Line(cid, Result("errored")))
            else:
                out.append(Line(cid, Result("succeeded", Message([Block(GOOD.model_dump_json())]))))
        return out


class FakeMessages:
    def __init__(self, response: Any = None, exc: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response if response is not None else Parsed(GOOD)
        self.exc = exc
        self.batches = FakeBatches()

    def parse(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return self.response


class FakeClient:
    def __init__(self, **kw: Any) -> None:
        self.messages = FakeMessages(**kw)


def make(
    tmp: Path, client: Any = None, cap: float = 25.0, spent: Decimal = Decimal(0)
) -> tuple[an.Analyst, an.Ledger]:
    ledger = an.Ledger(cap, {"2026-09": spent}, clock=lambda: NOW)
    a = an.Analyst(
        CFG, BUDGET, PRICES, client, an.ScoreCache(tmp / "cache"), ledger, sleep=lambda _s: None
    )
    return a, ledger


def test_schema_parses_and_rejects_bad_values() -> None:
    s = an.AnalystScore.model_validate_json(GOOD.model_dump_json())
    assert s.guidance == "raised"
    with pytest.raises(ValueError):
        an.AnalystScore.model_validate({**GOOD.model_dump(), "guidance": "up"})
    with pytest.raises(ValueError):
        an.AnalystScore.model_validate({**GOOD.model_dump(), "price_target": 100})
    row = an.to_row(
        item(1), an.AnalystScore(**{**GOOD.model_dump(), "sentiment": 3.0}), "m", date(2026, 7, 1)
    )
    assert row["sentiment"] == 1.0 and row["guidance"] == 1.0 and row["surprise"] == 1.0
    assert row["contaminated"] is False
    old = an.to_row(item(1, "2025-03-03"), GOOD, "m", date(2026, 7, 1))
    assert old["contaminated"] is True


def test_prompt_never_mentions_prices_and_never_asks_to_predict() -> None:
    text = item(1).text(CFG.text_chars).lower()
    assert "price" not in text and "return" not in text
    assert "not a forecaster" in an.SYSTEM_PROMPT


def test_prompt_caching_flag_on_system_prompt(tmp_path: Path) -> None:
    client = FakeClient()
    a, _ = make(tmp_path, client)
    a.score_now([item(1)])
    call = client.messages.calls[0]
    assert call["output_format"] is an.AnalystScore
    sys_block = call["system"][0]
    assert sys_block["cache_control"]["type"] == "ephemeral"
    assert sys_block["text"] == an.SYSTEM_PROMPT
    breq = a.batch_request(item(2))["params"]
    assert breq["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert breq["output_config"]["format"]["type"] == "json_schema"


def test_replay_cache_never_pays_twice(tmp_path: Path) -> None:
    client = FakeClient()
    a, ledger = make(tmp_path, client)
    r1 = a.score_now([item(1), item(2)])
    assert r1.scored == 2 and r1.spent_usd > 0
    spent = ledger.spent()
    r2 = a.score_now([item(1), item(2)])
    assert r2.cached == 2 and r2.scored == 0 and ledger.spent() == spent
    assert len(client.messages.calls) == 2
    rows = a.rows([item(1), item(2)])
    assert list(rows.columns) == list(an.SCORE_COLUMNS) and len(rows) == 2


def test_cost_pricing_with_cache_reads_and_batch_discount() -> None:
    u = an.Usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    p = PRICES["claude-haiku-5-5"]
    # 0.10 + 1.25 * 0.10 + 0.1 * 0.10 + 0.50 = 0.735
    assert an.price(u, p, BUDGET, batch=False) == pytest.approx(0.735)
    assert an.price(u, p, BUDGET, batch=True) == pytest.approx(0.3675)
    # 1 hour cache writes cost twice the input price.
    assert an.price(u, p, BUDGET, batch=False, ttl="1h") == pytest.approx(0.81)


def test_cost_cap_refuses_backfill_unless_allowed(tmp_path: Path) -> None:
    client = FakeClient()
    a, _ = make(tmp_path, client, cap=0.0001)
    items = [item(i) for i in range(50)]
    with pytest.raises(an.BudgetExceeded, match="allow-spend"):
        a.run_batches(items, an.BatchJournal(tmp_path / "j.json"), allow_spend=False)
    assert client.messages.batches.created == []
    # --allow-spend overrides the estimate, but the monthly cap is still a hard stop:
    # the first batch runs, and nothing more is submitted once the cap is reached.
    rep = a.run_batches(items, an.BatchJournal(tmp_path / "j.json"), allow_spend=True)
    assert 0 < rep.scored < 50 and "cap" in rep.failures[-1]
    roomy, _ = make(tmp_path / "roomy", FakeClient(), cap=25.0)
    assert (
        roomy.run_batches(items, an.BatchJournal(tmp_path / "j2.json"), allow_spend=False).scored
        == 50
    )


def test_hard_stop_when_cap_is_reached_in_daily_scoring(tmp_path: Path) -> None:
    client = FakeClient()
    a, _ = make(tmp_path, client, cap=1.0, spent=Decimal("1.0"))
    rep = a.score_now([item(1)])
    assert rep.failed == 1 and "cap" in rep.failures[0] and client.messages.calls == []


def test_batch_flow_polls_stores_and_resumes(tmp_path: Path) -> None:
    client = FakeClient()
    a, ledger = make(tmp_path, client)
    items = [item(i) for i in range(5)]
    journal = an.BatchJournal(tmp_path / "batches.json")
    # Simulate an interruption: one batch was created and recorded, then the process died.
    first = client.messages.batches.create([a.batch_request(i) for i in items[:2]])
    journal.opened(first.id, [a.key(i)[:64] for i in items[:2]])
    rep = a.run_batches(items, journal, allow_spend=False)
    created = client.messages.batches.created
    assert rep.scored == 5 and rep.failed == 0
    # The resumed run collected the open batch and only submitted the other three items.
    assert len(created) == 1 + 2  # first batch, then 3 items in chunks of 2
    assert sum(len(c) for c in created[1:]) == 3
    assert journal.open_batches() == []
    assert ledger.spent() > 0 and a.pending(items) == []
    again = a.run_batches(items, journal, allow_spend=False)
    assert again.cached == 5 and len(client.messages.batches.created) == 3


def test_batch_errors_are_reported_not_cached(tmp_path: Path) -> None:
    client = FakeClient()
    a, _ = make(tmp_path, client)
    items = [item(1), item(2)]
    client.messages.batches.fail_ids = {a.key(items[0])[:64]}
    rep = a.run_batches(items, an.BatchJournal(tmp_path / "j.json"), allow_spend=False)
    assert rep.scored == 1 and rep.failed == 1 and a.pending(items) == [items[0]]


def test_refusal_and_errors_are_failures(tmp_path: Path) -> None:
    a, _ = make(tmp_path, FakeClient(response=Parsed(None, stop_reason="refusal")))
    assert a.score_now([item(1)]).failed == 1
    b, _ = make(tmp_path / "b", FakeClient(exc=RuntimeError("boom")))
    assert b.score_now([item(1)]).failed == 1
    c, _ = make(tmp_path / "c", None)
    assert "ANTHROPIC_API_KEY" in c.score_now([item(1)]).failures[0]


def test_estimate_cost_is_in_the_expected_range() -> None:
    items = [item(i) for i in range(1000)]
    est, n = an.estimate_cost(items, CFG, PRICES["claude-haiku-5-5"], BUDGET, batch=True)
    assert n == 1000 and 0.005 < est < 0.10


def test_items_from_news_caps_per_symbol_per_day() -> None:
    df = pd.DataFrame(
        {
            "id": [f"a{i}" for i in range(5)],
            "symbol": ["ACME"] * 5,
            "known_at": [datetime(2026, 9, 1, 14, i, tzinfo=UTC) for i in range(5)],
            "headline": ["h"] * 5,
            "summary": ["s"] * 5,
            "content": ["x" * (i + 1) for i in range(5)],
            "kind": ["news"] * 5,
        }
    )
    out = an.items_from_news(df, 500, 3)
    assert len(out) == 3 and {i.item_id for i in out} == {"a2", "a3", "a4"}


def test_merge_scores_dedupes(tmp_path: Path) -> None:
    rows = pd.DataFrame([an.to_row(item(1), GOOD, "m", date(2026, 7, 1))])
    an.merge_scores(tmp_path / "s.parquet", rows)
    out = an.merge_scores(tmp_path / "s.parquet", rows)
    assert len(out) == 1
    assert json.loads(GOOD.model_dump_json())["materiality"] == 0.8
