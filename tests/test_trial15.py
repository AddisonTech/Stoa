"""Trial 15: the top 5 by conviction every day at equal weight, 2% in cash, no SPY, no 0.55
line, no regime filter; the kill switch and the price sanity check stay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stoa import bigpaper, clock
from stoa.broker.base import BrokerPosition
from stoa.event_signals import Signal
from stoa.models import Side
from stoa.newsbrain import trial15
from stoa.state import KillSwitch, StateStore
from tests.test_bigpaper import SESSION, FakeBroker, at, runtime

D = Decimal
DECIDED = clock.previous_session(SESSION)
SIC = {
    "AAPL": "3571",
    "MSFT": "7372",
    "NVDA": "3674",
    "JPM": "6021",
    "XOM": "2911",
    "KO": "2080",
    "GLD": "6221",
    "QQQ": None,
    "SPY": None,
}


def test_the_rules_are_the_registered_ones() -> None:
    # Changing a rule is a new trial: the spec in the registry is 557c95176a137379.
    assert trial15.spec_hash() == "557c95176a137379"
    rules = trial15.spec()
    assert "no 0.55 line, no SPY, no regime filter" in rules["trading"]
    assert "25% kill switch" in rules["rails"] and "price sanity" in rules["rails"]
    assert "Claude never reads historical news" in rules["reader"]


def test_funds_and_spy_are_never_picks() -> None:
    assert trial15.is_company("AAPL", SIC)
    for fund in ("SPY", "QQQ", "GLD", "EWZ"):  # no code, a fund's code, or unknown
        assert not trial15.is_company(fund, SIC)


def test_the_top_5_by_conviction_ties_broken_by_symbol() -> None:
    rows = pd.DataFrame(
        {
            "symbol": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "XOM", "KO", "GLD"],
            "conviction": [0.99, 0.98, 0.60, 0.70, 0.70, 0.50, 0.40, 0.65, 0.97],
        }
    )
    got = trial15.choose(rows, SIC)
    assert list(got["symbol"]) == ["MSFT", "NVDA", "KO", "AAPL", "JPM"]
    assert trial15.choose(rows.iloc[:0], SIC).empty


@dataclass
class Calibrated:
    """A model whose calibrated p is flat over a step, like trial 9's."""

    features: tuple[str, ...] = ("x",)

    @property
    def clf(self) -> Any:
        outer = self

        class Clf:
            def predict_proba(self, x: np.ndarray) -> np.ndarray:
                p = x[:, 0] / 10
                return np.column_stack([1 - p, p])

        del outer
        return Clf()

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.full(len(rows), 0.49)


def test_conviction_is_the_score_before_calibration_so_ties_break() -> None:
    rows = pd.DataFrame({"x": [1.0, 3.0, 2.0]})
    assert list(trial15.conviction(Calibrated(), rows)) == pytest.approx([0.1, 0.3, 0.2])


def pos(sym: str, qty: str, px: str) -> BrokerPosition:
    return BrokerPosition(symbol=sym, qty=D(qty), avg_entry_price=D(px), current_price=D(px))


def sig(sym: str, score: float = 0.6) -> Signal:
    return Signal(sym, DECIDED, trial15.KIND, score, f"conviction {score:.3f}")


def test_the_plan_sells_the_old_buys_the_new_and_resizes_a_repeat() -> None:
    held = {"AAPL": pos("AAPL", "100", "100"), "MSFT": pos("MSFT", "50", "200")}
    picks = [sig(s) for s in ("MSFT", "NVDA", "KO", "JPM", "XOM")]
    ref = {s: D("100") for s in ("MSFT", "NVDA", "KO", "JPM", "XOM")}
    orders = trial15.plan_orders(picks, held, D(100000), ref, dict.fromkeys(ref, True))
    by = {(o.symbol, o.side): o for o in orders}
    assert by[("AAPL", Side.SELL)].qty == D(100) and by[("AAPL", Side.SELL)].reason == (
        "hold_period"
    )
    target = D("19600.00")  # (100,000 x 0.98) / 5
    assert trial15.target_notional(D(100000)) == target
    for s in ("NVDA", "KO", "JPM", "XOM"):
        assert by[(s, Side.BUY)].notional == target and by[(s, Side.BUY)].reason == "trial15_pick"
    # MSFT is held at $10,000, picked again: topped up by $9,600, not sold and bought back.
    assert by[("MSFT", Side.BUY)].notional == D("9600.00")
    assert by[("MSFT", Side.BUY)].reason == "resize"
    assert ("MSFT", Side.SELL) not in by
    assert orders[0].side is Side.SELL  # sells first


def test_a_repeat_pick_above_its_weight_is_trimmed() -> None:
    held = {"MSFT": pos("MSFT", "150", "200")}  # $30,000
    orders = trial15.plan_orders([sig("MSFT")], held, D(100000), {"MSFT": D(200)}, {})
    (o,) = orders
    assert o.side is Side.SELL and o.reason == "resize" and o.qty == D(52)  # whole shares


def test_grading_is_against_spy_after_costs() -> None:
    entry = {"SPY": 100.0, "AAPL": 100.0, "MSFT": 100.0, "KO": 100.0}
    exit_ = {"SPY": 101.0, "AAPL": 103.0, "MSFT": 101.0, "KO": 99.0}
    adv = {"AAPL": 5e9, "MSFT": 5e9, "KO": 5e9}
    g = {x["symbol"]: x for x in trial15.grade(["AAPL", "MSFT", "KO", "NOPE"], entry, exit_, adv)}
    assert g["AAPL"]["hit"] and 0.017 < g["AAPL"]["net"] < 0.02  # 2% over SPY less costs
    assert not g["MSFT"]["hit"]  # level with SPY loses to the costs
    assert not g["KO"]["hit"] and g["NOPE"]["hit"] is None
    t = trial15.tally([{"picks": list(g.values())}])
    assert t["picks"] == 3 and t["hits"] == 1 and t["hit_rate"] == pytest.approx(1 / 3)
    assert t["target"] == 0.75 and t["days"] == 1 and t["days_beating_spy"] == 0


# ---- the big paper account under trial 15 ----------------------------------------------------


PRICES = {
    "SPY": D(400),
    "AAPL": D(100),
    "MSFT": D(200),
    "NVDA": D(50),
    "JPM": D(150),
    "XOM": D(100),
    "KO": D(60),
    "CRM": D(250),
}


def live(tmp_path: Path, fb: FakeBroker, sigs: list[Signal], hhmm: str = "09:00") -> Any:
    return runtime(tmp_path, fb, "trial15", now=lambda: at(SESSION, hhmm), signals=lambda d: sigs)


def test_trial_15_buys_5_at_equal_weight_keeps_2_percent_and_never_spy(tmp_path: Path) -> None:
    fb = FakeBroker(
        D(4000), dict(PRICES), held={"SPY": D(10), "CRM": D(368)}, last_equity=D(100000)
    )
    sigs = [sig(s, 0.7 - i / 100) for i, s in enumerate(("AAPL", "MSFT", "NVDA", "JPM", "XOM"))]
    rt, rec = live(tmp_path, fb, sigs)
    assert rt.top5 and not rt.core and rt.label == "Trial 15, top 5 a day, idle money in cash"
    assert bigpaper.picks(rt).status == "picked"
    plan = StateStore(tmp_path / "big").load().pending_plan
    assert plan is not None and plan.strategy == "trial15"
    sells = {o.symbol for o in plan.orders if o.side is Side.SELL}
    buys = {o.symbol: o for o in plan.orders if o.side is Side.BUY}
    assert sells == {"SPY", "CRM"}  # yesterday's pick and any SPY go
    assert set(buys) == {"AAPL", "MSFT", "NVDA", "JPM", "XOM"} and "SPY" not in buys
    assert all(o.notional == D("19600.00") for o in buys.values())
    msg = rec.alerts[-1].message
    assert msg.startswith("Trial 15, top 5 a day, idle money in cash.")
    assert "conviction 0.700" in msg
    rt.now = lambda: at(SESSION, "09:46")
    assert bigpaper.execute(rt).status == "executed"
    held = fb.positions()
    assert set(held) == {"AAPL", "MSFT", "NVDA", "JPM", "XOM"}
    for p in held.values():
        assert p.market_value == pytest.approx(D(19600), abs=D("0.01"))
    assert fb.cash == pytest.approx(D(2000), abs=D("0.05"))  # the 2% reserve
    assert not any(o.symbol == "SPY" and o.side is Side.BUY for o in fb.orders.values())


def test_no_sector_cap_and_no_daily_loss_stop_under_trial_15(tmp_path: Path) -> None:
    # Five banks the same morning, after a day down 5%: all five are bought.
    prices = {"SPY": D(400), **{s: D(100) for s in ("JPM", "BAC", "WFC", "C", "GS")}}
    fb = FakeBroker(D(95000), prices, last_equity=D(100000))
    sigs = [sig(s) for s in ("JPM", "BAC", "WFC", "C", "GS")]
    rt, _ = live(tmp_path, fb, sigs)
    assert bigpaper.picks(rt).status == "picked"
    store = StateStore(tmp_path / "big")
    st = store.load()
    st.last_close_equity = D(100000)
    store.save(st)
    rt.now = lambda: at(SESSION, "09:46")
    bigpaper.execute(rt)
    assert set(fb.positions()) == {"JPM", "BAC", "WFC", "C", "GS"}


def test_the_price_sanity_check_still_blocks_a_buy(tmp_path: Path) -> None:
    fb = FakeBroker(D(100000), dict(PRICES), last_equity=D(100000))
    rt, rec = live(tmp_path, fb, [sig("AAPL"), sig("MSFT")])
    bigpaper.picks(rt)
    fb.prices["AAPL"] = D(115)  # 15% above the 9:00 am price by 9:45 am
    rt.now = lambda: at(SESSION, "09:46")
    bigpaper.execute(rt)
    assert set(fb.positions()) == {"MSFT"}
    assert "price sanity" in rec.alerts[-1].message.lower().replace("_", " ")


def test_the_kill_switch_still_sells_everything(tmp_path: Path) -> None:
    fb = FakeBroker(D(1000), dict(PRICES), held={"AAPL": D(100)}, last_equity=D(11000))
    rt, _ = live(tmp_path, fb, [sig("MSFT")])
    bigpaper.picks(rt)
    store = StateStore(tmp_path / "big")
    st = store.load()
    st.kill_switch = KillSwitch(active=True, tripped_session=date(2021, 6, 1), reason="test")
    store.save(st)
    rt.now = lambda: at(SESSION, "09:46")
    bigpaper.execute(rt)
    assert fb.positions() == {}


def test_the_summary_grades_yesterdays_picks_and_keeps_the_running_hit_rate(
    tmp_path: Path,
) -> None:
    fb = FakeBroker(D(100000), dict(PRICES), last_equity=D(100000))
    nxt = clock.next_session(SESSION)
    entry = {"SPY": 400.0, "AAPL": 100.0, "MSFT": 200.0}
    exit_ = {"SPY": 404.0, "AAPL": 104.0, "MSFT": 200.0}
    rt, _ = live(tmp_path, fb, [sig("AAPL"), sig("MSFT")])
    rt.prices_945 = lambda day, syms: entry if day == SESSION else exit_
    rt.bars = lambda syms, through: {}
    rt.chosen = [{"symbol": "AAPL", "adv": 5e9}, {"symbol": "MSFT", "adv": 5e9}]
    bigpaper.picks(rt)
    saved = bigpaper.picks15(rt, SESSION)
    assert [p["symbol"] for p in saved] == ["AAPL", "MSFT"] and saved[0]["saw"]["adv"] == 5e9
    lines = bigpaper.summary15_lines(rt, nxt)
    text = "\n".join(lines)
    assert f"Picks of {SESSION}" in text and "AAPL beat SPY by 2.9%" in text
    assert "MSFT lost to SPY by 1.1%" in text and "Day hit rate 1 of 2 (50%)" in text
    assert "Running: 1 of 2 picks beat SPY (50%) over 1 days, target 75%" in text
    # Graded once: the next evening does not count the same picks again.
    bigpaper.summary15_lines(rt, nxt)
    assert len(bigpaper.graded15(rt)) == 1


# ---- the local event reader in the feed --------------------------------------------------------


def articles(ids: list[int], hour: int = 12) -> pd.DataFrame:
    t = pd.Timestamp(f"2026-10-29 {hour:02d}:00", tz="UTC")
    return pd.DataFrame(
        {
            "id": ids,
            "symbols": [["AAPL"] for _ in ids],
            "headline": [f"news {i}" for i in ids],
            "summary": ["" for _ in ids],
            "created_at": [t for _ in ids],
            "updated_at": [t for _ in ids],
        }
    )


def test_the_feed_labels_new_articles_and_keeps_a_backlog(tmp_path: Path) -> None:
    from stoa.newsbrain import feed as nf

    def reader(arts: pd.DataFrame, seconds: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert seconds == 60.0
        done = arts.head(2)  # the budget runs out after two articles
        labels = pd.DataFrame(
            {
                "id": done["id"],
                "symbol": "AAPL",
                "event": "earnings_beat",
                "p_event": 0.9,
                "direction": 0.8,
                "size": 2.0,
            }
        )
        return labels, arts.iloc[2:]

    src = nf.Sources(
        news=None,
        filings=None,
        bars=None,
        scorer=None,  # type: ignore[arg-type]
        reader=reader,
        read_seconds=60.0,
    )
    fd = nf.Feed(tmp_path)
    assert nf.read_events(fd, src, articles([1, 2, 3])) == 2
    labels = pd.read_parquet(tmp_path / "events" / "labels.parquet")
    back = pd.read_parquet(tmp_path / "events" / "backlog.parquet")
    assert sorted(labels["id"]) == [1, 2] and list(back["id"]) == [3]
    assert labels["known_at"].notna().all()
    # The next run reads the backlog first, then what is new.
    assert nf.read_events(fd, src, articles([4], hour=13)) == 2
    labels = pd.read_parquet(tmp_path / "events" / "labels.parquet")
    assert sorted(labels["id"]) == [1, 2, 3, 4]
    assert pd.read_parquet(tmp_path / "events" / "backlog.parquet").empty
    assert "events/**" in nf.STATE_FILES


def test_a_reader_failure_never_stops_the_feed(tmp_path: Path) -> None:
    from stoa.newsbrain import feed as nf

    def broken(arts: pd.DataFrame, seconds: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise RuntimeError("out of memory")

    src = nf.Sources(
        news=None,
        filings=None,
        bars=None,
        scorer=None,  # type: ignore[arg-type]
        reader=broken,
        read_seconds=60.0,
    )
    fd = nf.Feed(tmp_path)
    assert nf.read_events(fd, src, articles([1, 2])) == 0
    assert sorted(pd.read_parquet(tmp_path / "events" / "backlog.parquet")["id"]) == [1, 2]
