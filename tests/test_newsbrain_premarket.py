from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from stoa import clock
from stoa.broker.sim import buy_fill_price
from stoa.experiments import Registry
from stoa.newsbrain import panel as pn
from stoa.newsbrain import premarket as pm
from tests.conftest import ROOT

DAYS = clock.sessions_between(date(2024, 1, 2), date(2024, 6, 28))
CUT = np.array([pd.Timestamp(clock.at_ny(d, "09:00")).value for d in DAYS], dtype=np.int64)


def ny(d: date, hhmm: str, seconds: int = 0) -> pd.Timestamp:
    return pd.Timestamp(clock.at_ny(d, hhmm)) + pd.Timedelta(seconds=seconds)


# ---- cost and fill ------------------------------------------------------------------------


def test_cost_model_is_wider_for_premarket_thin_minutes_and_small_caps() -> None:
    # Liquid large cap, busy minute: 2 x 5 bps floor.
    assert pm.cost_bps(1e9, 1e6, "large") == pytest.approx(10.0)
    # Same name in a $12,500 minute: volume factor 2.
    assert pm.cost_bps(1e9, 12_500, "large") == pytest.approx(20.0)
    # Thin small cap: capped at 300.
    assert pm.cost_bps(5e6, 1_000, "small") == pm.COST_CAP_BPS
    # Unknown dollar volume costs trial 9's maximum before the pre-market multiples.
    assert pm.cost_bps(None, 1e6, "mid") == pytest.approx(2 * 75 * 1.2)
    for b in ("small", "mid", "large"):
        assert pm.cost_bps(2e7, 5_000, b) > pm.cost_bps(2e7, 5e5, b)
    assert (
        pm.cost_bps(2e7, 5_000, "small")
        > pm.cost_bps(2e7, 5_000, "mid")
        > pm.cost_bps(2e7, 5_000, "large")
    )


def bars(d: date, rows: list[tuple[str, float, float, float, int]]) -> pd.DataFrame:
    """(hh:mm, close, volume, vw, trades) per minute."""
    return pd.DataFrame(
        {
            "t": [ny(d, m) for m, *_ in rows],
            "c": [r[1] for r in rows],
            "v": [r[2] for r in rows],
            "vw": [r[3] for r in rows],
            "n": [r[4] for r in rows],
        }
    )


D = date(2024, 3, 12)


def test_fill_uses_only_minutes_from_the_decision_and_honours_the_limit_and_volume() -> None:
    b = bars(
        D,
        [
            ("06:00", 10.00, 500, 10.00, 3),  # before the decision: the reference
            ("06:15", 10.50, 50, 10.50, 1),  # thin: $525 < $1,000
            ("06:16", 10.40, 500, 10.40, 4),  # vw x cost over the 10.10 limit
            ("06:17", 10.02, 10000, 10.02, 2),  # fills ($100k minute, cost 42 bps)
            ("09:26", 9.00, 9000, 9.00, 9),  # past 9:25, never used
        ],
    )
    f = pm.find_fill(b, ny(D, "06:15"), 9.5, 2e7, "mid", ny(D, "09:25"))
    assert f is not None
    assert f.reference == 10.00 and f.limit == pytest.approx(10.10)
    assert f.minute == ny(D, "06:17")
    c = pm.cost_bps(2e7, 10000 * 10.02, "mid")
    assert f.cost_bps == pytest.approx(c) and f.price == pytest.approx(10.02 * (1 + c / 1e4))
    assert f.price <= f.limit


def test_no_fill_means_no_trade_and_the_previous_close_is_the_fallback_reference() -> None:
    b = bars(D, [("07:00", 12.0, 1000, 12.0, 5), ("09:20", 12.5, 1000, 12.5, 5)])
    assert pm.find_fill(b, ny(D, "07:00"), 10.0, 2e7, "small", ny(D, "09:25")) is None
    f = pm.find_fill(b.iloc[0:0], ny(D, "07:00"), 10.0, 2e7, "small", ny(D, "09:25"))
    assert f is None
    assert pm.decision_minute(ny(D, "06:10", 20)) == ny(D, "06:11")
    assert pm.decision_minute(ny(D, "06:10")) == ny(D, "06:10")


# ---- features as known at the decision minute ----------------------------------------------


def news_frame(items: list[tuple[str, pd.Timestamp, float, str]]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "id": range(len(items)),
            "created_at": [i[1] for i in items],
            "known_at": [i[1] for i in items],
            "symbol": [i[0] for i in items],
            "headline": [i[3] for i in items],
            "net": np.array([i[2] for i in items], dtype="float32"),
        }
    )
    df["sidx"] = pn.to_session(df["known_at"], CUT)
    df = df.sort_values(["symbol", "created_at"]).reset_index(drop=True)
    return df.assign(first=pn.first_breaks(df))


def history() -> list[tuple[str, pd.Timestamp, float, str]]:
    rng = np.random.default_rng(3)
    out = []
    for k, d in enumerate(DAYS[:70]):
        if k % 2 == 0:
            out.append(
                ("ACME", ny(d, "14:00"), float(rng.uniform(-0.5, 0.5)), f"acme story {k} widget")
            )
    return out


def test_features_equal_trial_9s_when_every_article_is_known_by_the_decision() -> None:
    t = 75
    day = DAYS[t]
    items = [
        *history(),
        ("ACME", ny(DAYS[t - 1], "15:30"), 0.6, "acme wins large contract award"),
        ("ACME", ny(day, "06:40", 5), 0.8, "acme raises outlook sharply today"),
    ]
    news = news_frame(items)
    trig = pm.with_baselines(pm.triggers(news, DAYS, CUT), pm.news_baselines(news))
    row = trig[trig["sidx"] == t].iloc[0]
    assert row["minute"] == ny(day, "06:41")
    full = pn.daily_news(news.drop(columns=["first"]), len(DAYS))
    ref = full[(full["symbol"] == "ACME") & (full["sidx"] == t)].iloc[0]
    for c in ("n", "sent", "first_break", "surprise", "quiet_spike", "coverage"):
        assert row[c] == pytest.approx(ref[c], abs=1e-6), c


def test_articles_after_the_decision_are_left_out_and_9_to_925_counts_for_today() -> None:
    t = 75
    day = DAYS[t]
    items = [
        *history(),
        ("ACME", ny(day, "03:50"), 0.5, "acme overnight item one"),  # fresh, before the window
        (
            "ACME",
            ny(day, "09:10"),
            -0.9,
            "acme first premarket headline",
        ),  # binned to t+1 by trial 9
        ("ACME", ny(day, "09:24"), 0.9, "acme later headline ignored"),  # after the decision
        ("BETA", ny(day, "09:30"), 0.9, "beta at the open is not premarket"),
        ("GAMA", ny(DAYS[t + 2], "03:59"), 0.9, "gama before four"),
    ]
    news = news_frame(items)
    trig = pm.triggers(news, DAYS, CUT)
    assert set(trig["symbol"]) == {"ACME"}
    row = trig[trig["sidx"] == t].iloc[0]
    assert row["minute"] == ny(day, "09:10")
    assert row["n"] == 2  # the 3:50 item and the 9:10 one; not the 9:24 one
    assert news.loc[news["headline"].str.contains("first premarket"), "sidx"].iloc[0] == t + 1


# ---- signals and fills ------------------------------------------------------------------


def test_signals_come_earliest_first_filled_only_and_at_most_ten() -> None:
    day = DAYS[80]
    test = pd.DataFrame(
        {
            "symbol": [f"S{i:02d}" for i in range(14)],
            "sidx": 80,
            "session": day,
            "minute": [ny(day, "04:00") + pd.Timedelta(minutes=10 * (14 - i)) for i in range(14)],
            "sent": 0.5,
            "n": 1,
            "limit": 10.0,
            "fill_price": [10.0] * 13 + [np.nan],
        }
    )
    p = np.array([0.9] * 12 + [0.5, 0.95])
    sigs = pm.signals_for(test, p, 5)
    assert len(sigs) == 10
    assert [s.symbol for s in sigs] == [f"S{i:02d}" for i in range(11, 1, -1)]
    scores = [s.score for s in sigs]
    assert scores == sorted(scores, reverse=True)
    assert all(s.session == clock.previous_session(day) for s in sigs)


def test_premarket_buys_land_on_the_fill_after_the_engine_adds_its_slippage() -> None:
    def at_945(d: date, syms: set[str]) -> dict[str, Decimal]:
        return {s: Decimal("20") for s in syms}

    fills = pm.PremarketFills({(D, "AAA"): 10.123}, at_945, lambda s, d: 37.0)
    got = fills(D, {"AAA", "BBB"}, {"HELD"})
    assert float(buy_fill_price(got["AAA"], 37.0)) == pytest.approx(10.123, rel=1e-12)
    assert got["BBB"] == Decimal("20") and got["HELD"] == Decimal("20")


# ---- pre-registration -----------------------------------------------------------------------


def test_the_preregistration_is_written_once_and_is_not_a_trial(tmp_path) -> None:  # type: ignore[no-untyped-def]
    reg = Registry(tmp_path / "registry.jsonl")
    pm.preregister(reg)
    pm.preregister(reg)
    entries = reg.load()
    assert len(entries) == 1 and entries[0].kind == "preregistration"
    assert entries[0].params == pm.spec() and entries[0].config_hash == pm.spec_hash()
    assert reg.trials() == [] and reg.penalty_inputs("backtest")[0] == 1  # the floor, no trials


def test_decisions_md_states_the_registered_numbers() -> None:
    text = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    sec = text[text.index("## Trial 11") :]
    for needle in (
        "4:00",
        "9:25",
        "1%",
        "$1,000",
        "$50,000",
        "2 x",
        "1.5",
        "1.2",
        "300 bps",
        "2016-01-01 to 2022-12-31",
        "2023-01-01 to 2026-06-30",
        "earliest first",
        "9:45 am",
        pm.spec_hash(),
    ):
        assert needle in sec, needle
