"""The rolling state gives trial 9's rows bit for bit, without the history.

A small world on disk (news with rewrites and a late update, earnings 8-Ks, a 10-K
supplier link, daily bars with gaps) is read by the full rebuild (``panel.build``) and by
the rolling state seeded earlier and rolled forward one session at a time.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stoa import clock
from stoa.newsbrain import panel, rolling

SEED = date(2015, 5, 1)
CHECKS = (date(2015, 5, 6), date(2015, 5, 12), date(2015, 5, 20))
LAST = CHECKS[-1]
SYMS = ("AAA", "BBB", "CCC", "DDD", "SPY")
CIKS = {"AAA": 101, "BBB": 202, "CCC": 303, "DDD": 404, "SPY": 505}
NAMES = {
    "AAA": "Alpha Widgets Inc",
    "BBB": "Bravo Industries Corp",
    "CCC": "Charlie Semiconductor Ltd",
    "DDD": "Delta Logistics Holdings",
    "SPY": "Spider Trust Fund",
}


def ny(d: date, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(clock.at_ny(d, hhmm)).tz_convert("UTC")


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def world(root: Path) -> None:
    """Bars, news, filings, a 10-K and the fixed inputs, laid out as on the Dell."""
    rng = np.random.default_rng(7)
    days = clock.sessions_between(panel.PANEL_START, LAST)
    for k, sym in enumerate(SYMS):
        px = 40.0 * (1 + k) * np.cumprod(1 + rng.normal(0, 0.02, len(days)))
        vol = rng.integers(400_000, 2_000_000, len(days)).astype(float)
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days], name="date")
        df = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": vol}, idx)
        if sym == "DDD":
            df = df.drop(idx[40:43])  # a gap: carried closes, a missing dollar volume
        if sym == "CCC":
            df.loc[idx[60:85], "close"] = 30.0  # a flat run: pandas' cancellation path
        (root / "data/bars/wide").mkdir(parents=True, exist_ok=True)
        df.tz_localize("UTC").to_parquet(root / "data/bars/wide" / f"{sym}.parquet")
    heads = [
        "Alpha Widgets beats estimates as margins expand",
        "Alpha Widgets beats estimates as margins expand sharply",  # a rewrite
        "Bravo Industries wins large defense contract",
        "Charlie Semiconductor guidance raised on demand",
        "Delta Logistics shipping volumes slip",
        "Bravo Industries contract award confirmed by officials",
    ]
    rows = []
    aid = 1
    for d in clock.sessions_between(date(2015, 1, 5), LAST):
        for j in range(int(rng.integers(0, 4))):
            sym = SYMS[int(rng.integers(0, 4))]
            created = ny(d, "07:00") + pd.Timedelta(minutes=int(rng.integers(0, 600)), seconds=aid)
            rows.append(
                {
                    "id": aid,
                    "created_at": created,
                    "updated_at": created,
                    "symbols": [sym] if j % 3 else [sym, "BBB"],
                    "headline": heads[int(rng.integers(0, len(heads)))],
                    "summary": f"story {aid}",
                    "net": float(rng.uniform(-0.9, 0.9)),
                }
            )
            aid += 1
    # Published before the seed, updated after it: the late update reaches back.
    rows.append(
        {
            "id": aid,
            "created_at": ny(date(2015, 4, 29), "10:00"),
            "updated_at": ny(date(2015, 5, 8), "08:00"),
            "symbols": ["AAA"],
            "headline": "Alpha Widgets beats estimates as margins expand",
            "summary": f"story {aid}",
            "net": 0.5,
        }
    )
    news = pd.DataFrame(rows)
    news["month"] = news["updated_at"].dt.strftime("%Y-%m")
    for m, month in news.groupby("month"):
        g = month.sort_values("created_at")
        write(g.drop(columns=["net", "month"]), root / "data/newsbrain/news" / f"{m}.parquet")
        sc = pd.DataFrame(
            {
                "id": g["id"],
                "positive": ((1 + g["net"]) / 2).astype("float32"),
                "negative": ((1 - g["net"]) / 2).astype("float32"),
                "neutral": np.zeros(len(g), dtype="float32"),
            }
        )
        write(sc, root / "data/newsbrain/scores/news" / f"{m}.parquet")
    ciks = pd.DataFrame({"cik": list(CIKS.values()), "symbol": list(CIKS)})
    write(ciks, root / "data/newsbrain/sec/ciks.parquet")
    for sym, d, accn in (
        ("DDD", date(2015, 4, 30), "d-1"),
        ("CCC", date(2015, 5, 4), "c-1"),
        ("AAA", date(2015, 5, 11), "a-1"),
    ):
        cik = CIKS[sym]
        doc = pd.DataFrame(
            {
                "cik": [cik, cik],
                "accn": [accn, accn],
                "form": ["8-K", "8-K"],
                "items": ["2.02,9.01", "2.02,9.01"],
                "filed": [str(d), str(d)],
                "accepted": [ny(d, "07:30"), ny(d, "07:30")],
                "kind": ["main", "ex99"],
                "doc": ["m.htm", "ex99.htm"],
                "text": ["Cover.", "Results above expectations; we raised guidance for the year."],
            }
        )
        write(doc, root / "data/newsbrain/sec/earnings" / f"{cik}.parquet")
        sc = pd.DataFrame(
            {
                "accn": [accn, accn],
                "kind": ["main", "ex99"],
                "doc": ["m.htm", "ex99.htm"],
                "positive": np.array([0.2, 0.8], dtype="float32"),
                "negative": np.array([0.1, 0.1], dtype="float32"),
                "neutral": np.array([0.7, 0.1], dtype="float32"),
            }
        )
        write(sc, root / "data/newsbrain/scores/sec/earnings" / f"{cik}.parquet")
    for cik, d, text in (
        (101, date(2015, 3, 2), "Bravo Industries accounted for 22% of net sales."),
        (101, date(2015, 5, 14), "Delta Logistics accounted for 15% of revenues."),
        (303, date(2015, 5, 8), "Bravo Industries and Alpha Widgets were 30% of sales."),
    ):
        path = root / "data/newsbrain/sec/tenk" / f"{cik}.parquet"
        old = pd.read_parquet(path) if path.exists() else None
        new = pd.DataFrame(
            {"cik": [cik], "accn": [f"k-{cik}-{d}"], "accepted": [ny(d, "16:30")], "text": [text]}
        )
        write(new if old is None else pd.concat([old, new]), path)
    plan = pd.DataFrame(
        [
            {"symbol": s, "month": f"2015-{m:02d}", "days": 20, "dollars": 1e8}
            for s in SYMS
            for m in range(1, 7)
        ]
    )
    write(plan, root / "data/newsbrain/minute_plan.parquet")
    tick = {
        str(i): {"cik_str": CIKS[s], "ticker": s, "title": NAMES[s]} for i, s in enumerate(SYMS)
    }
    (root / "data/sec").mkdir(parents=True, exist_ok=True)
    (root / "data/sec/company_tickers.json").write_text(json.dumps(tick), encoding="utf-8")


@pytest.fixture()
def dell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    world(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def same(a: pd.Series, b: pd.Series) -> np.ndarray:
    x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    out: np.ndarray = (x == y) | (np.isnan(x) & np.isnan(y))
    return out


def test_the_rolling_state_matches_the_full_rebuild_bit_for_bit(dell: Path) -> None:
    hist = rolling.History.read(LAST, live=False)
    st = hist.seed(dell / "state", SEED)
    st.save()
    day = SEED
    seen = 0
    while day < LAST:
        day = clock.next_session(day)
        st.update(day, **hist.day(day))
        if day not in CHECKS:
            continue
        st.save()
        st = rolling.State.load(dell / "state")  # what the next morning reads back
        got = st.rows(today_bars=hist.bars_on(day))
        ref = panel.build(panel.PANEL_START, day)
        ref = ref[ref["session"] == day].sort_values("symbol").reset_index(drop=True)
        assert list(got["symbol"]) == list(ref["symbol"]), day
        for f in panel.FEATURES:
            assert same(got[f], ref[f]).all(), (day, f)
        seen += len(ref)
    assert seen > 0
    links = rolling.State.load(dell / "state").links
    assert set(links["cik"]) == {101, 303}  # AAA's second 10-K ended its first links


def test_pandas_running_sums_bit_for_bit() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 600) * np.exp(rng.normal(0, 3, 600))
    x[50:80] = np.nan
    x[200:240] = 3.0  # a constant run
    x[300] = 1e12  # a spike: catastrophic cancellation once it leaves the window
    var, mean = rolling.RollVar(1), rolling.RollMean(1)
    std, avg = [], []
    for v in x:
        var.push(np.array([v]))
        mean.push(np.array([v]))
        std.append(var.std()[0])
        avg.append(mean.value()[0])
    s = pd.Series(x)
    exp_std = s.rolling(20, min_periods=15).std().to_numpy()
    exp_avg = s.rolling(20, min_periods=20).mean().to_numpy()
    assert np.array_equal(np.array(std), exp_std, equal_nan=True)
    assert np.array_equal(np.array(avg), exp_avg, equal_nan=True)


def test_the_bar_state_saves_and_loads_exactly(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    close = 20 * np.cumprod(1 + rng.normal(0, 0.03, (3, 120)), axis=1)
    close[1, 30:35] = np.nan
    st = rolling.bars_from_history(["A", "B", "C"], close, np.full((3, 120), 1e6))
    st.save(tmp_path / "bars.npz")
    back = rolling.BarState.load(tmp_path / "bars.npz")
    pd.testing.assert_frame_equal(st.features(), back.features())
    back.push(close[:, -1], np.full(3, 1e6))
    st.push(close[:, -1], np.full(3, 1e6))
    pd.testing.assert_frame_equal(st.features(), back.features())


def test_an_update_must_be_the_next_session(dell: Path) -> None:
    hist = rolling.History.read(LAST, live=False)
    st = hist.seed(dell / "state", SEED)
    with pytest.raises(ValueError, match="not the next session"):
        st.update(CHECKS[0], **hist.day(CHECKS[0]))
