from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stoa.newsbrain import news_timing as nt


def fake_events(n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    w = np.r_[np.full(4, 0.01), np.full(6, 0.08), np.full(8, 0.04), np.full(6, 0.02)]
    hours = rng.choice(24, n, p=w / w.sum())
    big = (hours >= 4) & (hours < 10)
    return pd.DataFrame(
        {
            "symbol": rng.choice(["A", "B", "C"], n),
            "at": pd.Timestamp("2020-01-02", tz="UTC"),
            "hour": hours,
            "weekend": rng.random(n) < 0.1,
            "bucket": rng.choice(list(nt.CAPS), n),
            "excess_1": np.where(big, 0.05, 0.005) * rng.choice([-1, 1], n),
            "excess_5": np.where(big, 0.08, 0.01) * rng.choice([-1, 1], n),
        }
    )


def test_hour_table_shares_add_up_per_size() -> None:
    tab = nt.by_hour(fake_events())
    for b in (*nt.CAPS, "all"):
        t = tab[tab["bucket"] == b]
        assert t["move_share_5"].sum() == pytest.approx(1.0)
        assert t["move_share_1"].sum() == pytest.approx(1.0)
        for day in ("weekday", "weekend"):
            assert t[t["day"] == day]["article_share"].sum() == pytest.approx(1.0)
    assert tab[tab["bucket"] == "all"]["articles"].sum() == 4000


def test_best_window_wraps_past_midnight_and_grows() -> None:
    share = np.zeros(24)
    share[[22, 23, 0, 1]] = 0.2
    share[12] = 0.2
    wins = nt.best_windows(share)
    assert wins[3]["start"] == 22 and wins[3]["end"] == 2 and wins[3]["share"] == pytest.approx(0.8)
    assert nt.window_text(wins[3]) == "10:00 pm to 2:00 am"
    assert [w["share"] for w in wins] == sorted(w["share"] for w in wins)
    assert wins[-1]["share"] == pytest.approx(1.0)


def test_recommendation_finds_the_busy_premarket_hours() -> None:
    rec = nt.recommend(nt.by_hour(fake_events()))
    w = rec["all"]["for"]["50"]
    assert 4 <= w["start"] <= 9 and w["hours"] <= 6
    assert "ET" in nt.alert_line(rec) and chr(0x2014) not in nt.alert_line(rec)


def test_last_price_before_looks_back_at_most_four_days() -> None:
    t0 = pd.Timestamp("2020-03-02 14:00", tz="UTC")
    minutes = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "B"],
            "t": [
                t0 - pd.Timedelta(days=5),
                t0 - pd.Timedelta(minutes=3),
                t0 + pd.Timedelta(minutes=1),
                t0,
            ],
            "c": [1.0, 2.0, 3.0, 9.0],
        }
    )
    ev = pd.DataFrame(
        {"symbol": ["A", "B", "A"], "at": [t0, t0, t0 - pd.Timedelta(days=4, minutes=1)]}
    )
    got = nt.last_price_before(minutes, ev)
    assert got[0] == 2.0 and got[1] == 9.0  # never a later trade; same-minute is known
    assert got[2] == 1.0  # just under 4 days back
    far = pd.DataFrame({"symbol": ["A"], "at": [t0]})
    assert np.isnan(nt.last_price_before(minutes.iloc[[0]], far)[0])  # 5 days back: too old


def test_charts_and_doc_are_valid_and_plain(tmp_path: Path) -> None:
    ev = fake_events()
    tab = nt.by_hour(ev)
    rec = nt.recommend(tab)
    nt.bars_grid(tab, tmp_path / "a.svg")
    series = {
        b: tab[tab["bucket"] == b].groupby("hour")["move_share_5"].sum().to_numpy() for b in nt.CAPS
    }
    nt.lines(
        series, tmp_path / "b.svg", "t", "s", "x", np.arange(24), str, lambda y: f"{y:.0%}", 0.2
    )
    for name in ("a.svg", "b.svg"):
        root = ET.fromstring((tmp_path / name).read_text(encoding="utf-8"))
        assert root.tag.endswith("svg") and float(root.get("width", 0)) > 0
        texts = [e for e in root.iter() if e.tag.endswith("text")]
        assert all(float(e.get("x", 0)) < float(root.get("width", 0)) for e in texts)
    md = nt.markdown(tab, rec, len(ev))
    assert chr(0x2014) not in md and "news_timing/coverage.svg" in md
    assert "| 4:00 am |" in md and "## Recommendation" in md
