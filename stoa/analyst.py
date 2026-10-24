"""The analyst: Claude reads news and filings into structured scores. It never predicts.

What it sees: the text of one item (headline, summary, a slice of the body)
and the company's ticker. What it never sees: prices, returns, charts,
positions, or any request to forecast. Each score carries ``known_at`` equal to
the item's own timestamp, so it enters the point-in-time store like any other
record and is invisible to any decision made before the item was public.

Cost control:
- Prompt caching: the long, fixed system prompt carries ``cache_control``, so
  repeated calls pay the cache-read rate for it.
- Replay cache: every result is stored by a hash of (model, system prompt,
  item text, schema). The same item is never paid for twice.
- Batch API: historical backfill and rescoring go through the Message Batches
  API at half price. Batch ids are saved before polling, so an interrupted
  backfill resumes by collecting the same batch instead of paying again.
- A monthly dollar cap (``budget.monthly_cap_usd``, $25 by default) shared with
  every other Claude call. A backfill estimates its cost first and refuses if
  it would go over the remaining budget, unless ``--allow-spend`` is given.

Contamination: Claude has a knowledge cutoff (``analyst.contamination_cutoff``).
For items before it, the model may remember what happened next. Those scores
are flagged ``contaminated`` and the ablation reports them separately.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from stoa.config import AnalystConfig, BudgetConfig, ModelPrice
from stoa.models import dec

log = logging.getLogger(__name__)

SCORE_COLUMNS = (
    "item_id",
    "symbol",
    "known_at",
    "kind",
    "sentiment",
    "guidance",
    "surprise",
    "risk",
    "materiality",
    "confidence",
    "contaminated",
    "model",
)
CHARS_PER_TOKEN = 4.0

SYSTEM_PROMPT = """You are a careful equity research analyst. You read one piece of company
news or one regulatory filing and turn it into a few structured scores.

You are not a forecaster. Do not predict prices, returns or what the stock will do.
Do not use anything you may remember about later events. Judge only what the text
in front of you says, as a reader on the day it was published would.

Return exactly these fields:
- sentiment: the tone toward the company, from -1 (clearly negative) to 1 (clearly
  positive). 0 when neutral or mixed.
- guidance: the direction of any forward guidance the company itself gave:
  "raised", "maintained", "lowered", or "none" when the text has no company guidance.
- surprise: how reported results compared with expectations the text mentions:
  "beat", "inline", "miss", or "none" when the text reports no results against
  expectations.
- risk_flags: zero or more of these labels, only when the text clearly supports them:
  "litigation", "regulatory", "accounting", "guidance_cut", "management_change",
  "liquidity", "dilution", "cybersecurity", "product_issue", "macro".
- materiality: how much this item could matter to a long-term holder, from 0 (trivial,
  routine or promotional) to 1 (major: results, guidance, deals, legal or leadership
  changes).
- confidence: how sure you are of these scores given how clear the text is, 0 to 1.

Rules:
- Promotional, listicle, price-target-only or "stocks to watch" items get materiality
  at most 0.2 and sentiment near 0.
- If the item is mostly about another company, keep materiality low for this one.
- Never mention prices, returns or predictions in any field.

Scoring rubric, to keep scores consistent from item to item:

sentiment
-  0.8 to  1.0: unambiguous good news for the company itself (record results with raised
   guidance, a large contract win, regulatory approval of a key product).
-  0.3 to  0.7: mostly positive with some caveats.
- -0.2 to  0.2: neutral, routine, mixed, or not really about this company.
- -0.7 to -0.3: mostly negative (a guidance trim, a delay, a downgrade with reasons).
- -1.0 to -0.8: unambiguous bad news (fraud allegations, a failed trial, a large recall).

materiality
- 0.0 to 0.2: routine or promotional (conference appearances, minor partnerships,
  analyst price targets alone, market wrap-ups that mention the ticker).
- 0.3 to 0.5: worth knowing (a product launch, a mid-size contract, an executive hire).
- 0.6 to 0.8: important (quarterly results, guidance changes, a large deal, a lawsuit).
- 0.9 to 1.0: transformative (a merger agreement, bankruptcy, a CEO departure under
  pressure, a regulator blocking the core business).

confidence
- High (0.8 to 1.0) when the text states facts plainly. Medium (0.4 to 0.7) when the
  text is a summary, a headline only, or hedged. Low (under 0.4) when the text is
  vague, truncated, or mostly about something else.

guidance and surprise
- Only the company's own forward guidance counts for guidance; analyst estimates do not.
- surprise needs both a reported figure and a stated expectation in the text. Words like
  "topped estimates" or "fell short of forecasts" are enough. Otherwise use "none".

Examples (described, not quoted):
- A headline that a company "beat on revenue and earnings and raised its full-year
  outlook": sentiment about 0.8, guidance "raised", surprise "beat", materiality about 0.8.
- An item listing ten stocks "to watch this week" that includes the company:
  sentiment 0, guidance "none", surprise "none", materiality about 0.1, confidence 0.8.
- A report that a regulator opened an investigation into the company's accounting:
  sentiment about -0.7, risk_flags ["regulatory", "accounting"], materiality about 0.8."""

RISK_FLAGS = (
    "litigation",
    "regulatory",
    "accounting",
    "guidance_cut",
    "management_change",
    "liquidity",
    "dilution",
    "cybersecurity",
    "product_issue",
    "macro",
)


class AnalystScore(BaseModel):
    """The strict schema Claude answers in. Ranges are checked after parsing."""

    model_config = ConfigDict(extra="forbid")

    sentiment: float
    guidance: Literal["raised", "maintained", "lowered", "none"]
    surprise: Literal["beat", "inline", "miss", "none"]
    risk_flags: list[
        Literal[
            "litigation",
            "regulatory",
            "accounting",
            "guidance_cut",
            "management_change",
            "liquidity",
            "dilution",
            "cybersecurity",
            "product_issue",
            "macro",
        ]
    ]
    materiality: float
    confidence: float


GUIDANCE_VALUE = {"raised": 1.0, "maintained": 0.0, "lowered": -1.0, "none": 0.0}
SURPRISE_VALUE = {"beat": 1.0, "inline": 0.0, "miss": -1.0, "none": 0.0}


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if not math.isfinite(x) else max(lo, min(hi, x))


def to_row(item: Item, score: AnalystScore, model: str, cutoff: date) -> dict[str, Any]:
    known = pd.Timestamp(item.known_at).tz_convert(UTC)
    return {
        "item_id": item.item_id,
        "symbol": item.symbol,
        "known_at": known,
        "kind": item.kind,
        "sentiment": clamp(score.sentiment, -1.0, 1.0),
        "guidance": GUIDANCE_VALUE[score.guidance],
        "surprise": SURPRISE_VALUE[score.surprise],
        "risk": float(len(score.risk_flags)),
        "materiality": clamp(score.materiality, 0.0, 1.0),
        "confidence": clamp(score.confidence, 0.0, 1.0),
        "contaminated": known.date() < cutoff,
        "model": model,
    }


@dataclass(frozen=True)
class Item:
    item_id: str
    symbol: str
    known_at: datetime
    kind: str  # news | filing | transcript
    title: str
    body: str

    def text(self, max_chars: int) -> str:
        body = " ".join(self.body.split())[:max_chars]
        return f"Company ticker: {self.symbol}\nType: {self.kind}\nTitle: {self.title}\n\n{body}"


def items_from_news(df: pd.DataFrame, max_chars: int, per_symbol_day: int) -> list[Item]:
    """News rows to analyst items, at most ``per_symbol_day`` per symbol per day (longest first)."""
    if df.empty:
        return []
    d = df.copy()
    d["known_at"] = pd.to_datetime(d["known_at"], utc=True)
    d["_day"] = d["known_at"].dt.tz_convert("America/New_York").dt.date
    d["_len"] = d["content"].fillna("").str.len() + d["summary"].fillna("").str.len()
    d = d.sort_values(["symbol", "_day", "_len"], ascending=[True, True, False])
    d = d.groupby(["symbol", "_day"], sort=False).head(per_symbol_day)
    out = []
    for r in d.itertuples(index=False):
        body = str(r.content or "") or str(r.summary or "")
        out.append(
            Item(
                item_id=str(r.id),
                symbol=str(r.symbol),
                known_at=pd.Timestamp(cast(Any, r.known_at)).to_pydatetime(),
                kind=str(getattr(r, "kind", "news") or "news"),
                title=str(r.headline or ""),
                body=body[: max_chars * 2],
            )
        )
    return sorted(out, key=lambda i: (i.known_at, i.item_id))


# ---- cost ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def usage_of(response: Any) -> Usage:
    u = getattr(response, "usage", None)

    def g(name: str) -> int:
        return int(getattr(u, name, 0) or 0)

    return Usage(
        g("input_tokens"),
        g("output_tokens"),
        g("cache_creation_input_tokens"),
        g("cache_read_input_tokens"),
    )


def price(u: Usage, p: ModelPrice, budget: BudgetConfig, *, batch: bool, ttl: str = "5m") -> float:
    """Dollars for one call. ``input_tokens`` excludes cached tokens, per the API."""
    write_mult = (
        budget.cache_write_multiplier_1h if ttl == "1h" else budget.cache_write_multiplier_5m
    )
    inp = (
        u.input_tokens
        + u.cache_creation_input_tokens * write_mult
        + u.cache_read_input_tokens * budget.cache_read_multiplier
    ) * p.input_per_mtok
    out = u.output_tokens * p.output_per_mtok
    cost = (inp + out) / 1_000_000
    return cost * (budget.batch_discount if batch else 1.0)


def estimate_cost(
    items: Iterable[Item],
    cfg: AnalystConfig,
    p: ModelPrice,
    budget: BudgetConfig,
    *,
    batch: bool,
    cached_fraction: float = 0.9,
) -> tuple[float, int]:
    """Upper-bound-ish dollar estimate before running, and the item count.

    Assumes the system prompt is a cache read on ``cached_fraction`` of calls
    and a cache write on the rest, item text at 4 characters per token, and
    ``est_output_tokens`` out per item.
    """
    n = 0
    item_tokens = 0.0
    for it in items:
        n += 1
        item_tokens += len(it.text(cfg.text_chars)) / CHARS_PER_TOKEN + 20
    sys_t = cfg.est_system_tokens
    ttl = cfg.batch_cache_ttl if batch else cfg.cache_ttl
    u = Usage(
        input_tokens=int(item_tokens),
        output_tokens=n * cfg.est_output_tokens,
        cache_creation_input_tokens=int(n * (1 - cached_fraction) * sys_t),
        cache_read_input_tokens=int(n * cached_fraction * sys_t),
    )
    return price(u, p, budget, batch=batch, ttl=ttl), n


class Ledger:
    """Monthly Claude spend in USD (UTC calendar month of the real call), shared by every caller."""

    def __init__(
        self,
        cap_usd: float,
        months: MutableMapping[str, Decimal],
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.cap_usd = cap_usd
        self.months = months
        self.clock = clock

    def month(self) -> str:
        return self.clock().strftime("%Y-%m")

    def spent(self) -> float:
        return float(self.months.get(self.month(), Decimal(0)))

    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self.spent())

    def exhausted(self) -> bool:
        return self.spent() >= self.cap_usd

    def record(self, usd: float) -> None:
        k = self.month()
        self.months[k] = self.months.get(k, Decimal(0)) + dec(round(usd, 8))


class BudgetExceeded(RuntimeError):
    pass


# ---- replay cache ---------------------------------------------------------------------------


def cache_key(model: str, system: str, text: str) -> str:
    schema = json.dumps(AnalystScore.model_json_schema(), sort_keys=True)
    blob = json.dumps([model, system, text, schema], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class ScoreCache:
    def __init__(self, root: Path | None) -> None:
        self.root = root

    def _p(self, key: str) -> Path:
        assert self.root is not None
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> AnalystScore | None:
        if self.root is None or not self._p(key).exists():
            return None
        try:
            data = json.loads(self._p(key).read_text(encoding="utf-8"))
            return AnalystScore.model_validate(data["score"])
        except (ValueError, KeyError, ValidationError):
            return None

    def put(self, key: str, score: AnalystScore, cost: float) -> None:
        if self.root is None:
            return
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"score": score.model_dump(mode="json"), "cost_usd": cost}),
            encoding="utf-8",
        )
        tmp.replace(p)


# ---- the client --------------------------------------------------------------------------------


class MessagesAPI(Protocol):
    def parse(self, **kwargs: Any) -> Any: ...

    @property
    def batches(self) -> Any: ...


class Client(Protocol):
    @property
    def messages(self) -> MessagesAPI: ...


def system_blocks(ttl: str) -> list[dict[str, Any]]:
    """The system prompt as one text block marked for prompt caching."""
    cc: dict[str, Any] = {"type": "ephemeral"}
    if ttl == "1h":
        cc["ttl"] = "1h"
    return [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": cc}]


def output_config() -> dict[str, Any]:
    """Structured output config for batch requests, built the same way ``parse`` builds it."""
    from anthropic.lib._parse._transform import transform_schema

    return {"format": {"type": "json_schema", "schema": transform_schema(AnalystScore)}}


@dataclass
class RunReport:
    scored: int = 0
    cached: int = 0
    failed: int = 0
    spent_usd: float = 0.0
    failures: list[str] = field(default_factory=list)


class Analyst:
    def __init__(
        self,
        cfg: AnalystConfig,
        budget_cfg: BudgetConfig,
        prices: dict[str, ModelPrice],
        client: Client | None,
        cache: ScoreCache,
        ledger: Ledger,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if cfg.model not in prices:
            raise ValueError(f"no price configured for analyst model {cfg.model!r}")
        self.cfg = cfg
        self.budget_cfg = budget_cfg
        self.price = prices[cfg.model]
        self.client = client
        self.cache = cache
        self.ledger = ledger
        self.sleep = sleep

    def key(self, item: Item) -> str:
        return cache_key(self.cfg.model, SYSTEM_PROMPT, item.text(self.cfg.text_chars))

    def pending(self, items: Iterable[Item]) -> list[Item]:
        return [i for i in items if self.cache.get(self.key(i)) is None]

    def rows(self, items: Iterable[Item]) -> pd.DataFrame:
        """Scores for every cached item, as point-in-time rows."""
        out = []
        for it in items:
            s = self.cache.get(self.key(it))
            if s is not None:
                out.append(to_row(it, s, self.cfg.model, self.cfg.contamination_cutoff))
        return pd.DataFrame(out, columns=list(SCORE_COLUMNS))

    def check_budget(self, items: list[Item], *, batch: bool, allow_spend: bool) -> float:
        est, _ = estimate_cost(items, self.cfg, self.price, self.budget_cfg, batch=batch)
        if est > self.ledger.remaining() and not allow_spend:
            left = self.ledger.remaining()
            raise BudgetExceeded(
                f"estimated ${est:.2f} for {len(items)} items is over the ${left:.2f} left "
                f"this month (cap ${self.ledger.cap_usd:.2f}); pass --allow-spend to run anyway"
            )
        return est

    # -- synchronous, for the daily run ------------------------------------------------------

    def score_now(self, items: Iterable[Item]) -> RunReport:
        rep = RunReport()
        for it in items:
            key = self.key(it)
            if self.cache.get(key) is not None:
                rep.cached += 1
                continue
            if self.client is None:
                rep.failed += 1
                rep.failures.append(f"{it.item_id}: no ANTHROPIC_API_KEY")
                continue
            if self.ledger.exhausted():
                rep.failed += 1
                rep.failures.append(f"{it.item_id}: monthly cap reached")
                continue
            try:
                resp = self.client.messages.parse(
                    model=self.cfg.model,
                    max_tokens=self.cfg.max_tokens,
                    system=system_blocks(self.cfg.cache_ttl),
                    messages=[{"role": "user", "content": it.text(self.cfg.text_chars)}],
                    output_format=AnalystScore,
                )
            except Exception as exc:
                rep.failed += 1
                rep.failures.append(f"{it.item_id}: {type(exc).__name__}")
                continue
            cost = price(
                usage_of(resp), self.price, self.budget_cfg, batch=False, ttl=self.cfg.cache_ttl
            )
            self.ledger.record(cost)
            rep.spent_usd += cost
            parsed = getattr(resp, "parsed_output", None)
            if getattr(resp, "stop_reason", None) in ("refusal", "max_tokens") or not isinstance(
                parsed, AnalystScore
            ):
                rep.failed += 1
                rep.failures.append(f"{it.item_id}: no usable output")
                continue
            self.cache.put(key, parsed, cost)
            rep.scored += 1
        return rep

    # -- batch, for backfill and rescoring ------------------------------------------------------

    def batch_request(self, it: Item) -> dict[str, Any]:
        return {
            "custom_id": self.key(it)[:64],
            "params": {
                "model": self.cfg.model,
                "max_tokens": self.cfg.max_tokens,
                "system": system_blocks(self.cfg.batch_cache_ttl),
                "messages": [{"role": "user", "content": it.text(self.cfg.text_chars)}],
                "output_config": output_config(),
            },
        }

    def run_batches(
        self,
        items: list[Item],
        journal: BatchJournal,
        *,
        allow_spend: bool,
        progress: Callable[[str], None] | None = None,
    ) -> RunReport:
        """Submit pending items in batches, poll to completion, store results. Resumable."""
        rep = RunReport()
        if self.client is None:
            raise RuntimeError("the analyst backfill needs ANTHROPIC_API_KEY")
        by_key = {self.key(i)[:64]: i for i in items}
        # 1. Finish anything an earlier run left in flight.
        for batch_id in journal.open_batches():
            self._collect(batch_id, by_key, journal, rep, progress)
        # 2. Submit what is still not cached.
        todo = self.pending(items)
        rep.cached = len(items) - len(todo)
        if todo:
            self.check_budget(todo, batch=True, allow_spend=allow_spend)
        for start in range(0, len(todo), self.cfg.batch_max_requests):
            if self.ledger.exhausted():
                rep.failures.append("monthly cap reached; remaining items not submitted")
                rep.failed += len(todo) - start
                break
            chunk = todo[start : start + self.cfg.batch_max_requests]
            batch = self.client.messages.batches.create(
                requests=[self.batch_request(i) for i in chunk]
            )
            journal.opened(str(batch.id), [self.key(i)[:64] for i in chunk])
            if progress:
                progress(f"submitted batch {batch.id} with {len(chunk)} requests")
            self._collect(str(batch.id), by_key, journal, rep, progress)
        return rep

    def _collect(
        self,
        batch_id: str,
        by_key: dict[str, Item],
        journal: BatchJournal,
        rep: RunReport,
        progress: Callable[[str], None] | None,
    ) -> None:
        assert self.client is not None
        while True:
            b = self.client.messages.batches.retrieve(batch_id)
            if str(b.processing_status) == "ended":
                break
            if progress:
                progress(f"batch {batch_id} {b.processing_status}")
            self.sleep(self.cfg.batch_poll_seconds)
        for line in self.client.messages.batches.results(batch_id):
            cid = str(line.custom_id)
            it = by_key.get(cid)
            result = line.result
            kind = str(getattr(result, "type", ""))
            if kind != "succeeded":
                rep.failed += 1
                rep.failures.append(f"{cid}: {kind}")
                continue
            message = result.message
            cost = price(
                usage_of(message),
                self.price,
                self.budget_cfg,
                batch=True,
                ttl=self.cfg.batch_cache_ttl,
            )
            self.ledger.record(cost)
            rep.spent_usd += cost
            score = parse_message(message)
            if score is None or it is None:
                rep.failed += 1
                rep.failures.append(f"{cid}: unparsable")
                continue
            self.cache.put(self.key(it), score, cost)
            rep.scored += 1
        journal.closed(batch_id)


def parse_message(message: Any) -> AnalystScore | None:
    """The structured JSON from a batch result message's text block."""
    if getattr(message, "stop_reason", None) in ("refusal", "max_tokens"):
        return None
    for block in getattr(message, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            return AnalystScore.model_validate_json(text)
        except ValidationError:
            return None
    return None


class BatchJournal:
    """Batch ids written before polling, so an interrupted backfill picks up where it stopped."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"open": {}, "closed": []}
        data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def opened(self, batch_id: str, keys: list[str]) -> None:
        d = self._load()
        d["open"][batch_id] = {"keys": keys, "at": datetime.now(UTC).isoformat()}
        self._save(d)

    def closed(self, batch_id: str) -> None:
        d = self._load()
        d["open"].pop(batch_id, None)
        d["closed"] = [*d["closed"], batch_id][-200:]
        self._save(d)

    def open_batches(self) -> list[str]:
        return list(self._load()["open"])


def make_client(api_key: str, timeout: float = 60.0, max_retries: int = 2) -> Client:
    import anthropic

    return anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)  # type: ignore[return-value]


def merge_scores(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=list(SCORE_COLUMNS))
    both = pd.concat([old, new], ignore_index=True) if not old.empty else new
    both = both.drop_duplicates(["item_id", "symbol", "model"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    both.reset_index(drop=True).to_parquet(tmp)
    tmp.replace(path)
    return both
