"""Dead man's switch: an outside service notices when Stoa goes quiet.

Stoa pings a healthchecks.io check (``STOA_HEALTHCHECK_URL``):
- ``<url>/start`` when DECIDE starts,
- ``<url>`` after DECIDE has sent the daily summary,
- ``<url>/fail`` when DECIDE fails or refuses.

healthchecks.io expects the success ping on a cron schedule (set it in New
York time with a grace period). If the ping does not arrive, it alerts the
phone through its ntfy integration, even when the GitHub job never started
at all. That is the case Stoa's own failure alerts cannot cover.

The ping URL is a secret: it is never logged and never put in an alert.
Ping failures are logged by kind only and never fail the job.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Literal, Protocol

import httpx

log = logging.getLogger(__name__)

Signal = Literal["start", "success", "fail"]


class Pinger(Protocol):
    def ping(self, signal: Signal, body: str = "") -> bool: ...


class NullPinger:
    def ping(self, signal: Signal, body: str = "") -> bool:
        return False


class HealthcheckPinger:
    def __init__(
        self,
        url: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
        retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not url.startswith("https://"):
            raise ValueError("healthcheck URL must be https")
        self._url = url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)
        self.retries = retries
        self.sleep = sleep

    def __repr__(self) -> str:
        return "HealthcheckPinger(url=<hidden>)"

    def _target(self, signal: Signal) -> str:
        return self._url if signal == "success" else f"{self._url}/{signal}"

    def ping(self, signal: Signal, body: str = "") -> bool:
        for attempt in range(self.retries + 1):
            try:
                resp = self.client.post(self._target(signal), content=body[:10000].encode())
                if resp.status_code < 300:
                    return True
                log.warning("healthcheck %s ping returned HTTP %s", signal, resp.status_code)
            except httpx.HTTPError as exc:
                log.warning("healthcheck %s ping failed: %s", signal, type(exc).__name__)
            if attempt < self.retries:
                self.sleep(1.0 + attempt)
        return False


def safe_ping(pinger: Pinger, signal: Signal, body: str = "") -> bool:
    try:
        return pinger.ping(signal, body)
    except Exception as exc:
        log.warning("healthcheck pinger raised %s", type(exc).__name__)
        return False
