"""Watchlist and asset metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetInfo:
    symbol: str
    tradable: bool
    fractionable: bool

    @property
    def eligible(self) -> bool:
        return self.tradable and self.fractionable


@dataclass(frozen=True)
class Universe:
    watchlist: tuple[str, ...]
    benchmark: str
    assets: dict[str, AssetInfo]

    def is_listed(self, symbol: str) -> bool:
        return symbol in self.watchlist

    def is_fractionable(self, symbol: str) -> bool:
        info = self.assets.get(symbol)
        return info is not None and info.eligible

    @classmethod
    def assume_all_fractionable(cls, watchlist: tuple[str, ...], benchmark: str) -> Universe:
        """Backtest and signal mode universe. Live modes load real asset flags from the broker."""
        return cls(
            watchlist=watchlist,
            benchmark=benchmark,
            assets={s: AssetInfo(s, tradable=True, fractionable=True) for s in watchlist},
        )
