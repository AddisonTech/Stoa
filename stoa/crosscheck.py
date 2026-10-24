"""Price cross-check: a buy is only sent when a second, independent source agrees.

Alpaca's latest trade (the execution reference) is compared with a quote from
another source. If they differ by more than ``crosscheck.tolerance_pct``, or
the second source cannot be reached (unless ``on_unavailable: proceed``), the
buy is skipped and an alert says why. Sells are never blocked: skipping a stop
because a second quote was slow would keep the account in the position it is
trying to leave (the same rule as price sanity).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from stoa.config import CrossCheckConfig
from stoa.data.external import PriceSource


@dataclass(frozen=True)
class CheckResult:
    symbol: str
    ok: bool
    reason: str  # agree | disagree | unavailable | no_primary | disabled
    primary: Decimal | None
    secondary: Decimal | None

    @property
    def detail(self) -> str:
        if self.reason == "disagree":
            assert self.primary is not None and self.secondary is not None
            gap = abs(self.primary - self.secondary) / self.secondary
            return (
                f"Alpaca {self.primary:.2f} vs second source {self.secondary:.2f} ({gap:.2%} apart)"
            )
        if self.reason == "unavailable":
            return "second price source unavailable"
        if self.reason == "no_primary":
            return "no Alpaca price"
        return self.reason


def compare(primary: Decimal | None, secondary: Decimal | None, tolerance: float) -> str:
    if primary is None or primary <= 0:
        return "no_primary"
    if secondary is None or secondary <= 0:
        return "unavailable"
    gap = abs(primary - secondary) / secondary
    return "agree" if gap <= Decimal(str(tolerance)) else "disagree"


def check_prices(
    symbols: list[str],
    primary: Mapping[str, Decimal],
    source: PriceSource | None,
    cfg: CrossCheckConfig,
) -> dict[str, CheckResult]:
    out: dict[str, CheckResult] = {}
    for sym in symbols:
        p = primary.get(sym)
        if not cfg.enabled:
            out[sym] = CheckResult(sym, True, "disabled", p, None)
            continue
        s = source.price(sym) if source is not None else None
        reason = compare(p, s, cfg.tolerance_pct)
        ok = reason == "agree" or (reason == "unavailable" and cfg.on_unavailable == "proceed")
        out[sym] = CheckResult(sym, ok, reason, p, s)
    return out
