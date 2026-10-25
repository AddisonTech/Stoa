from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stoa import clock
from stoa.newsbrain import panel, trial


def test_news_counts_for_the_first_session_whose_9am_cutoff_follows_it() -> None:
    days, cut = panel.calendar(date(2025, 3, 6), date(2025, 3, 11))
    assert days[:3] == [date(2025, 3, 6), date(2025, 3, 7), date(2025, 3, 10)]
    stamps = pd.Series(
        [
            datetime(2025, 3, 6, 13, 59, tzinfo=UTC),  # 8:59 am New York: same session
            datetime(2025, 3, 6, 14, 1, tzinfo=UTC),  # 9:01 am: the next session
            datetime(2025, 3, 6, 21, 30, tzinfo=UTC),  # after the close: the next session
            datetime(2025, 3, 8, 15, 0, tzinfo=UTC),  # Saturday: Monday
        ]
    )
    idx = panel.to_session(stamps, cut)
    assert [days[i] for i in idx] == [
        date(2025, 3, 6),
        date(2025, 3, 7),
        date(2025, 3, 7),
        date(2025, 3, 10),
    ]


def test_rewrites_of_the_same_story_are_not_first_breaks() -> None:
    t0 = pd.Timestamp("2025-03-06 12:00", tz="UTC")
    news = pd.DataFrame(
        {
            "symbol": ["ACME"] * 4 + ["ZZZ"],
            "created_at": [
                t0,
                t0 + pd.Timedelta(hours=2),
                t0 + pd.Timedelta(days=1),
                t0 + pd.Timedelta(days=5),
                t0 + pd.Timedelta(hours=1),
            ],
            "headline": [
                "Acme raises full year guidance after strong quarter",
                "Acme Raises Full-Year Guidance After Strong Quarter, Shares Rise",
                "Acme announces new chief financial officer",
                "Acme raises full year guidance after strong quarter",  # 5 days later: new
                "Acme raises full year guidance after strong quarter",  # other symbol
            ],
        }
    )
    assert list(panel.first_breaks(news)) == [True, False, True, True, True]


def test_daily_news_weights_rewrites_and_measures_quiet_spikes() -> None:
    t0 = pd.Timestamp("2025-03-06 12:00", tz="UTC")
    news = pd.DataFrame(
        {
            "symbol": ["ACME", "ACME", "ACME"],
            "created_at": [t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(days=30)],
            "headline": ["Acme wins big contract", "Acme wins big contract today", "Acme recall"],
            "net": np.array([0.8, -0.4, -0.9], dtype="float32"),
            "sidx": [70, 70, 90],
        }
    )
    day = panel.daily_news(news, n_sessions=100)
    first = day.iloc[0]
    assert first["n"] == 2 and first["first_break"] == 0.5
    assert first["sent"] == pytest.approx((0.8 - 0.25 * 0.4) / 1.25, rel=1e-5)
    later = day.iloc[1]
    # 2 articles in the previous 60 sessions: an average of 2/60 a day, a big spike today.
    assert later["coverage"] == pytest.approx(np.log1p(2 / 60))
    assert later["quiet_spike"] == pytest.approx(np.log(2 / (1 + 2 / 60)))
    assert later["surprise"] == 0.0  # fewer than 5 earlier news days: no baseline


@pytest.mark.parametrize(
    ("text", "guidance", "beat"),
    [
        ("The company raised its full-year outlook and revenue exceeded consensus.", 1, 1),
        ("We are lowering our fiscal 2024 guidance. Results were below expectations.", -1, -1),
        ("Guidance was reduced. Earnings beat analyst estimates.", -1, 1),
        ("Revenue grew 4% to $1.2 billion.", 0, 0),
    ],
)
def test_filing_guidance_and_beat_or_miss(text: str, guidance: int, beat: int) -> None:
    assert panel.direction(text, panel._RAISED, panel._CUT) == guidance
    assert panel.direction(text, panel._BEAT, panel._MISS) == beat


def test_customer_names_match_whole_words_only() -> None:
    names = {
        panel.norm_name("WALMART INC."): 1,
        panel.norm_name("Home Depot, Inc."): 2,
        panel.norm_name("Target Corp"): 3,
    }
    text = panel.norm_name(
        "Sales to Walmart Inc. and The Home Depot accounted for 31% of net sales; "
        "our targets for 2025 remain unchanged."
    )
    longest = max(len(n.split()) for n in names)
    assert panel.mentioned(text, names, longest) == {1, 2}


def test_forward_returns_use_the_next_open_or_the_last_close() -> None:
    idx = pd.date_range("2025-01-02", periods=5, freq="B")
    df = pd.DataFrame(
        {"open": [10.0, 11.0, np.nan, 12.0, 13.0], "close": [10.5, 11.5, 11.8, 12.5, 13.5]},
        index=idx,
    )
    r = panel.forward_returns(df, 2)
    assert r.iloc[0] == pytest.approx(11.8 / 10 - 1)  # open missing on day 3: its close
    assert r.iloc[1] == pytest.approx(12 / 11 - 1)
    assert r.iloc[3] == pytest.approx(13.5 / 12 - 1)  # past the data: the last close
    assert np.isnan(r.iloc[2])  # no entry open


def test_signals_are_filed_under_the_session_before_and_capped() -> None:
    session = date(2025, 3, 10)
    rows = pd.DataFrame(
        {
            "symbol": [f"S{i:02d}" for i in range(14)],
            "session": [session] * 14,
            "sidx": [5] * 14,
            "sent": 0.5,
            "surprise": 1.0,
            "n": 2,
            "filing_tone": 0.0,
            "ripple": 0.0,
        }
    )
    p = np.linspace(0.5, 0.9, 14)
    trial.register_kind(5)
    sigs = trial.signals_for(rows, p, 5)
    assert len(sigs) == trial.TOP_N
    assert all(s.score >= trial.THRESHOLD for s in sigs)
    assert sigs[0].score == pytest.approx(0.9) and sigs[0].session == date(2025, 3, 7)
    assert sigs[0].kind == "news_h5"


def synthetic_rows(days: list[date], seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_sym = 6
    sidx = np.repeat(np.arange(len(days)), n_sym)
    rows = pd.DataFrame(
        {"sidx": sidx, "symbol": np.tile([f"S{i}" for i in range(n_sym)], len(days))}
    )
    for f in panel.FEATURES:
        rows[f] = rng.normal(size=len(rows))
    rows["session"] = [days[i] for i in rows["sidx"]]
    edge = 0.02 * rows["sent"] + rng.normal(0, 0.03, len(rows))
    for h in panel.HORIZONS:
        rows[f"excess_{h}"] = edge
        rows[f"y_{h}"] = (edge > 0).astype("float32")
    return rows


def test_walk_forward_finds_a_planted_edge_and_freeze_ends_in_2022(tmp_path: Path) -> None:
    days = clock.sessions_between(date(2016, 1, 4), date(2023, 3, 31))
    rows = synthetic_rows(days)
    wf = trial.walk_forward(rows, days, 5)
    assert wf["auc"] > 0.6 and wf["log_loss"] < wf["base_log_loss"]
    assert wf["first_test_session"] >= "2018-01-01"
    frozen = trial.freeze(rows, days, 5, tmp_path)
    assert frozen.last_label_session <= "2022-12-30"
    assert (tmp_path / "model_h5.pkl").exists()
    p = frozen.predict(rows[rows["session"] >= date(2023, 1, 1)])
    assert ((p >= 0) & (p <= 1)).all()


def test_labelled_rows_never_cross_the_cut() -> None:
    days = clock.sessions_between(date(2022, 12, 1), date(2023, 1, 31))
    rows = synthetic_rows(days)
    last = days.index(date(2022, 12, 30))
    kept = trial.labelled(rows, 5, last)
    assert (kept["sidx"] + 5 <= last).all()


def test_entry_prices_take_the_first_bar_at_or_after_945_and_cache(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    class Client:
        def get(self, path: str, params: dict[str, Any]) -> Any:
            calls.append(params)

            class R:
                status_code = 200

                @staticmethod
                def json() -> dict[str, Any]:
                    return {"bars": {"AAA": [{"o": 10.5}, {"o": 10.7}], "BBB": []}}

            return R()

    class Pace:
        def wait(self) -> None:
            pass

    ep = trial.EntryPrices(Client(), Pace(), tmp_path)
    day = date(2025, 3, 10)
    got = ep(day, {"AAA", "BBB"})
    assert {k: float(v) for k, v in got.items()} == {"AAA": 10.5}
    assert calls[0]["start"] == "2025-03-10T13:45:00Z"  # 9:45 am New York
    assert ep(day, {"AAA"}) == got and len(calls) == 1  # cached
