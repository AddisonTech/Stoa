"""Trial 13's risk overlay: how much to put in each pick, and when to hold back.

It never picks a stock and never changes which stocks the model likes. It only sizes
the picks the strategy already made and pulls money back when the account is losing:

a. Volatility-scaled size: each new position risks the same share of the account,
   measured to the engine's existing ATR stop. Calm stocks get more money, jumpy
   stocks less.
b. A per-position cap on top of that.
c. A total exposure cap: picks plus the SPY core. Normally the no-borrowing rail
   (100%); while the account is in a drawdown, a lower cap, and money above it sits in
   cash, not SPY.
d. A drawdown brake: past the first line, new positions are cut in half; past the
   second, no new positions until the account is back within the first.
e. Broker-side ATR stops on every position, from the existing stop code (stops.py).

The numbers are fixed in ``OverlayConfig`` and registered before any run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from stoa.config import ExitConfig, RiskLimits, Settings
from stoa.risk import drawdown

HALT_NOTE = (
    "the drawdown brake: the account fell 15% from its high; "
    "no new positions until it is back within 10%"
)
BRAKE_NOTE = "the drawdown brake (10% or more off the high)"


@dataclass(frozen=True)
class OverlayConfig:
    risk_per_position: Decimal = Decimal("0.01")  # (a) share of equity lost at the ATR stop
    max_position_pct: Decimal = Decimal("0.05")  # (b)
    drawdown_exposure_cap: Decimal = Decimal("0.50")  # (c) picks plus SPY while in a drawdown
    brake_drawdown: Decimal = Decimal("0.10")  # (d) half size from here
    brake_size_multiplier: Decimal = Decimal("0.5")
    halt_drawdown: Decimal = Decimal("0.15")  # (d) no new positions from here ...
    resume_drawdown: Decimal = Decimal("0.10")  # ... until the drawdown is back to this
    broker_stops: bool = True  # (e)


@dataclass(frozen=True)
class OverlayState:
    """The overlay's settings for one decision, from the close it was made at."""

    session: date
    drawdown: Decimal
    size_multiplier: Decimal
    halted: bool
    exposure_cap: Decimal  # most of the account in picks plus SPY

    @property
    def in_drawdown(self) -> bool:
        return self.size_multiplier < 1 or self.halted


@dataclass
class Overlay:
    """The overlay for one account. ``halted`` carries over between sessions."""

    cfg: OverlayConfig = field(default_factory=OverlayConfig)
    halted: bool = False
    current: OverlayState | None = None
    history: list[OverlayState] = field(default_factory=list)

    def step(
        self, session: date, equity: Decimal, high_water_mark: Decimal, full_cap: Decimal
    ) -> OverlayState:
        """Move the brake on the close of ``session``; ``full_cap`` is the normal exposure cap."""
        c = self.cfg
        dd = drawdown(equity, high_water_mark)
        if dd >= c.halt_drawdown:
            self.halted = True
        elif self.halted and dd <= c.resume_drawdown:
            self.halted = False
        braking = dd >= c.brake_drawdown
        st = OverlayState(
            session=session,
            drawdown=dd,
            size_multiplier=c.brake_size_multiplier if braking else Decimal(1),
            halted=self.halted,
            exposure_cap=min(full_cap, c.drawdown_exposure_cap)
            if braking or self.halted
            else full_cap,
        )
        self.current = st
        if not self.history or self.history[-1].session != session:
            self.history.append(st)
        return st

    def weight(self, close: float, atr: float, exits: ExitConfig) -> Decimal | None:
        """(a): the share of equity whose loss at the ATR stop is ``risk_per_position``."""
        if not (close > 0 and atr > 0):
            return None
        stop_distance = exits.stop_atr_multiple * atr / close
        return self.cfg.risk_per_position / Decimal(str(float(stop_distance)))

    def limits(self, base: RiskLimits, st: OverlayState) -> RiskLimits:
        """The rails for one decision: (b) the size cap and (c) the exposure cap."""
        return base.model_copy(
            update={
                "max_position_pct": min(base.max_position_pct, self.cfg.max_position_pct),
                "max_gross_exposure_pct": min(base.max_gross_exposure_pct, st.exposure_cap),
            }
        )


def account_settings(settings: Settings, cfg: OverlayConfig, starting_equity: Decimal) -> Settings:
    """Settings for an account run with the overlay: its size cap and starting money."""
    return settings.model_copy(
        update={
            "risk": settings.risk.model_copy(
                update={
                    "max_position_pct": min(settings.risk.max_position_pct, cfg.max_position_pct)
                }
            ),
            "backtest": settings.backtest.model_copy(update={"starting_equity": starting_equity}),
        }
    )
