"""Test doubles shared across test modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass
class FakeUsage:
    input_tokens: int = 1200
    output_tokens: int = 300


@dataclass
class FakeResponse:
    parsed_output: BaseModel | None
    stop_reason: str = "end_turn"
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeMessages:
    def __init__(self, responder: Any) -> None:
        self.responder = responder
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.responder, Exception):
            raise self.responder
        if callable(self.responder):
            return self.responder(**kwargs)
        return self.responder


class FakeLLMClient:
    def __init__(self, responder: Any) -> None:
        self.messages = FakeMessages(responder)
