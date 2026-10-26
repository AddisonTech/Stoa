from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from stoa import membership, risk
from stoa.universe import AssetInfo, Universe
from tests.test_risk import ctx, idea

PICKS = ("CRM", "MSFT", "KO", "AAPL", "LLY")  # the 2026-10-26 picks, in plan order


@pytest.mark.parametrize(
    ("sic", "sector"),
    [
        ("3571", "Information Technology"),  # computers
        ("7372", "Information Technology"),  # software
        ("3674", "Information Technology"),  # semiconductors
        ("2834", "Health Care"),  # drugs
        ("3841", "Health Care"),  # medical instruments
        ("2080", "Consumer Staples"),  # beverages
        ("1311", "Energy"),  # crude oil
        ("6022", "Financials"),  # state banks
        ("6798", "Real Estate"),  # REITs
        ("4911", "Utilities"),
        ("3711", "Consumer Discretionary"),  # motor vehicles
        ("4813", "Communication Services"),  # telephone
        ("3721", "Industrials"),  # aircraft
        ("2800", "Materials"),  # chemicals
        ("", membership.UNKNOWN_SECTOR),
        (None, membership.UNKNOWN_SECTOR),
        ("9999", membership.UNKNOWN_SECTOR),
    ],
)
def test_sic_codes_map_to_gics_sector_names(sic: str | None, sector: str) -> None:
    assert membership.sic_sector(sic) == sector


def test_every_mapped_name_is_a_gics_sector_from_the_sp500_file() -> None:
    gics = set(membership._read_sectors("sectors.csv").values())
    assert {name for _, _, name in membership.SIC_SECTORS} <= gics


def test_real_sectors_for_the_picks_and_the_wide_universe() -> None:
    assert [membership.sector_of(s) for s in PICKS] == [
        "Information Technology",
        "Information Technology",
        "Consumer Staples",
        "Information Technology",
        "Health Care",
    ]
    wide = membership._read_sectors("sectors_wide.csv")
    assert len(wide) > 7000
    assert membership.sector_of("SOFI") == "Financials"  # outside the S&P 500: from its SIC
    # GICS wins where both exist: Alphabet files under SIC 7370 (software).
    assert membership.sector_of("GOOGL") == "Communication Services"
    assert membership.sector_of("NOT-A-SYMBOL") == membership.UNKNOWN_SECTOR


def test_sector_cap_blocks_only_same_sector_picks() -> None:
    universe = Universe(
        watchlist=("SPY", *PICKS),
        benchmark="SPY",
        assets={s: AssetInfo(s, True, True) for s in ("SPY", *PICKS)},
    )
    base = ctx(equity="100")
    c = dataclasses.replace(
        base,
        universe=universe,
        sectors={s: membership.sector_of(s) for s in PICKS},
    )
    assert c.limits.max_sector_pct == Decimal("0.40")
    res = risk.evaluate(c, [], [idea(s) for s in PICKS])
    bought = {o.symbol: o for o in res.orders}
    blocked = {r.symbol: r.reason for r in res.rejections}
    # A third $20 tech buy would make tech 60% of $100: only that one is blocked.
    assert blocked == {"AAPL": "sector_cap"}
    assert set(bought) == {"CRM", "MSFT", "KO", "LLY"}
    assert all(o.sector != membership.UNKNOWN_SECTOR for o in bought.values())
    # The same five with five different sectors all pass.
    spread = dataclasses.replace(c, sectors={s: f"Sector {s}" for s in PICKS})
    assert not risk.evaluate(spread, [], [idea(s) for s in PICKS]).rejections
