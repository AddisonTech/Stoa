"""Claude client wrapper: schema output, replay cache, cost pricing and a monthly cap.

Every response is cached by sha256(model, system prompt, user prompt, schema).
Backtests replay from the cache, so they are reproducible and pay only once.
A refusal, a truncated answer or a schema failure is a failure: the caller
falls back to the rule-based scout and an alert fires.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ValidationError

from stoa.config import LLMConfig, ModelPrice
from stoa.models import dec

log = logging.getLogger(__name__)


class LLMFailure(RuntimeError):
    """The model did not produce a usable answer. ``reason`` is a short code."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class MessagesAPI(Protocol):
    def parse(self, **kwargs: Any) -> Any: ...


class MessagesClient(Protocol):
    """The slice of the SDK client Stoa uses, so tests can pass a fake."""

    @property
    def messages(self) -> MessagesAPI: ...


@dataclass(frozen=True)
class LLMResult:
    parsed: BaseModel
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float  # what the call cost (or cost when first made, for a cache hit)
    charged_usd: float  # what this run actually paid: zero on a cache hit
    cache_hit: bool


def price_usage(
    prices: dict[str, ModelPrice], model: str, input_tokens: int, output_tokens: int
) -> float:
    if model not in prices:
        raise LLMFailure("unpriced_model", model)
    p = prices[model]
    return (input_tokens * p.input_per_mtok + output_tokens * p.output_per_mtok) / 1_000_000


def cache_key(model: str, system: str, user: str, schema: dict[str, Any]) -> str:
    blob = json.dumps(
        {"model": model, "system": system, "user": user, "schema": schema},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class ReplayCache:
    def __init__(self, root: Path | None) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        assert self.root is not None
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if self.root is None:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        return data

    def put(self, key: str, record: dict[str, Any]) -> None:
        if self.root is None:
            return
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, sort_keys=True, indent=1), encoding="utf-8")
        tmp.replace(p)


class Budget:
    """Monthly API spend in USD, keyed by calendar month (UTC) of the real call."""

    def __init__(
        self,
        cap_usd: float,
        ledger: MutableMapping[str, Decimal] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.cap_usd = cap_usd
        self.ledger: MutableMapping[str, Decimal] = ledger if ledger is not None else {}
        self.clock = clock

    def month(self) -> str:
        return self.clock().strftime("%Y-%m")

    def spent(self) -> float:
        return float(self.ledger.get(self.month(), Decimal(0)))

    def exhausted(self) -> bool:
        return self.spent() >= self.cap_usd

    def record(self, usd: float) -> None:
        key = self.month()
        self.ledger[key] = self.ledger.get(key, Decimal(0)) + dec(round(usd, 8))


class LLMClient:
    def __init__(
        self,
        cfg: LLMConfig,
        client: MessagesClient | None,
        cache: ReplayCache,
        budget: Budget,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.cache = cache
        self.budget = budget

    def parse(self, system: str, user: str, schema: type[BaseModel]) -> LLMResult:
        model = self.cfg.model
        key = cache_key(model, system, user, schema.model_json_schema())
        hit = self.cache.get(key)
        if hit is not None:
            try:
                cached = schema.model_validate(hit["parsed"])
            except ValidationError as exc:
                raise LLMFailure("cache_corrupt", str(exc)[:200]) from exc
            return LLMResult(
                parsed=cached,
                model=model,
                input_tokens=int(hit["input_tokens"]),
                output_tokens=int(hit["output_tokens"]),
                cost_usd=float(hit["cost_usd"]),
                charged_usd=0.0,
                cache_hit=True,
            )
        if self.client is None:
            raise LLMFailure("no_client", "ANTHROPIC_API_KEY is not set")
        if self.budget.exhausted():
            raise LLMFailure(
                "cost_cap", f"spent ${self.budget.spent():.2f} of ${self.cap:.2f} this month"
            )
        try:
            response = self.client.messages.parse(
                model=model,
                max_tokens=self.cfg.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except ValidationError as exc:
            raise LLMFailure("schema_invalid", str(exc)[:200]) from exc
        except Exception as exc:
            raise LLMFailure("api_error", f"{type(exc).__name__}: {str(exc)[:200]}") from exc

        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        cost = price_usage(self.cfg.prices, model, in_tok, out_tok)
        self.budget.record(cost)

        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            raise LLMFailure("refusal", "model refused")
        if stop == "max_tokens":
            raise LLMFailure("max_tokens", f"hit max_tokens={self.cfg.max_tokens}")
        output = getattr(response, "parsed_output", None)
        if not isinstance(output, schema):
            raise LLMFailure("schema_invalid", "no parsed output")
        parsed: BaseModel = output
        self.cache.put(
            key,
            {
                "model": model,
                "parsed": parsed.model_dump(mode="json"),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": cost,
                "created": datetime.now(UTC).isoformat(),
            },
        )
        return LLMResult(
            parsed=parsed,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            charged_usd=cost,
            cache_hit=False,
        )

    @property
    def cap(self) -> float:
        return self.budget.cap_usd


def make_llm_client(api_key: str, cfg: LLMConfig) -> MessagesClient:
    import anthropic

    client = anthropic.Anthropic(
        api_key=api_key, timeout=cfg.timeout_seconds, max_retries=cfg.max_retries
    )
    return cast(MessagesClient, client)
