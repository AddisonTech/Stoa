"""HTTP retry helper shared by the Alpaca trading and market data clients.

Retries only on 429 and 5xx responses, plus connection errors on GET. A 4xx other
than 429 is a real answer and is returned immediately. Order submission is safe
to retry on 429/5xx because every order carries a deterministic client order id.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx

log = logging.getLogger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def backoff_seconds(
    attempt: int, retry_after: str | None, base: float = 0.5, cap: float = 30.0
) -> float:
    if retry_after:
        try:
            return min(cap, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return float(min(cap, base * (2**attempt)))


def request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_retries: int,
    sleep: Callable[[float], None] = time.sleep,
    params: Mapping[str, Any] | None = None,
    json: Any = None,
) -> httpx.Response:
    attempt = 0
    while True:
        try:
            response = client.request(method, url, params=params, json=json)
        except httpx.TransportError as exc:
            if method.upper() != "GET" or attempt >= max_retries:
                raise
            wait = backoff_seconds(attempt, None)
            log.warning(
                "transport error on %s %s (%s), retry in %.1fs",
                method,
                _path(url),
                type(exc).__name__,
                wait,
            )
            sleep(wait)
            attempt += 1
            continue
        if response.status_code in RETRY_STATUS and attempt < max_retries:
            wait = backoff_seconds(attempt, response.headers.get("Retry-After"))
            log.warning(
                "HTTP %s on %s %s, retry in %.1fs", response.status_code, method, _path(url), wait
            )
            sleep(wait)
            attempt += 1
            continue
        return response


def _path(url: str) -> str:
    """Log only the path, never a query string that might carry identifiers."""
    return httpx.URL(url).path
