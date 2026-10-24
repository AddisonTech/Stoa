"""Watchlist and asset metadata."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    # Tradable but not fractionable symbols that may be bought in whole shares (trial 8's
    # account fit). Empty by default: every other strategy buys fractionable names only.
    whole_shares: frozenset[str] = field(default_factory=frozenset)

    def is_listed(self, symbol: str) -> bool:
        return symbol in self.watchlist

    def is_fractionable(self, symbol: str) -> bool:
        info = self.assets.get(symbol)
        return info is not None and info.eligible

    def needs_whole_shares(self, symbol: str) -> bool:
        info = self.assets.get(symbol)
        return (
            symbol in self.whole_shares
            and info is not None
            and info.tradable
            and not info.fractionable
        )

    def can_buy(self, symbol: str) -> bool:
        return self.is_fractionable(symbol) or self.needs_whole_shares(symbol)

    @classmethod
    def assume_all_fractionable(cls, watchlist: tuple[str, ...], benchmark: str) -> Universe:
        """Backtest and signal mode universe. Live modes load real asset flags from the broker."""
        return cls(
            watchlist=watchlist,
            benchmark=benchmark,
            assets={s: AssetInfo(s, tradable=True, fractionable=True) for s in watchlist},
        )
