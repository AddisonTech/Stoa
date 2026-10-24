"""Trial 8: the event signals and the event strategy, as pre-registered."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from stoa import clock, event_run, event_signals, wide
from stoa.broker.sim import SimBroker
from stoa.config import Settings
from stoa.data.base import Bars
from stoa.earnings import EarningsCalendar, empty_calendar
from stoa.event_signals import EARNINGS, INSIDER, Signal
from stoa.event_strategy import EventStrategy, diagnostics
from stoa.features import build_features, slice_features
from stoa.models import ClosedTrade, Holding, PlannedOrder, PositionMeta, Side
from stoa.planner import DecisionInput
from stoa.portfolio import Portfolio
from stoa.trader import whole_shares
from stoa.universe import AssetInfo, Universe
from tests.conftest import SMALL_WATCHLIST, make_settings

D = Decimal


def ny(day: str, hh: int, mm: int = 0) -> datetime:
    return datetime.fromisoformat(f"{day}T{hh:02d}:{mm:02d}").replace(tzinfo=clock.NY)


# ---- earnings drift: timing and no lookahead ------------------------------------------------


def test_release_sessions_follow_the_release_time() -> None:
    # Before the open: the close before is the day before, the first full session is that day.
    assert event_signals.release_sessions(ny("2024-07-25", 7)) == (
        date(2024, 7, 24),
        date(2024, 7, 25),
    )
    # After the close: that day's close, then the next session.
    assert event_signals.release_sessions(ny("2024-07-25", 16, 5)) == (
        date(2024, 7, 25),
        date(2024, 7, 26),
    )
    # During the session: the day is not full, so the next one counts.
    assert event_signals.release_sessions(ny("2024-07-25", 12)) == (
        date(2024, 7, 24),
        date(2024, 7, 26),
    )
    # On a weekend: the last session before, the next one after.
    assert event_signals.release_sessions(ny("2024-07-27", 10)) == (
        date(2024, 7, 26),
        date(2024, 7, 29),
    )


def drift_bars(jump: float, vol_mult: float, release_day: date) -> Bars:
    days = clock.sessions_between(date(2024, 5, 1), date(2024, 8, 30))
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    close = pd.Series(100.0, index=idx)
    vol = pd.Series(1e6, index=idx)
    first = pd.Timestamp(clock.next_session(release_day))
    close.loc[first:] = 100.0 * (1 + jump)
    vol.loc[first] = 1e6 * vol_mult
    stock = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": vol}
    )
    spy_c = pd.Series(400.0, index=idx)
    spy = pd.DataFrame({"open": spy_c, "high": spy_c, "low": spy_c, "close": spy_c, "volume": 1e8})
    return {"ACME": stock, "SPY": spy}


def releases(when: datetime) -> pd.DataFrame:
    t = pd.Timestamp(when).tz_convert(UTC)
    return pd.DataFrame(
        [{"symbol": "ACME", "event_time": t, "known_at": t, "source": "sec_8k_2.02"}]
    )


def test_drift_signal_is_known_at_the_first_full_session_close() -> None:
    day = date(2024, 7, 25)
    bars = drift_bars(0.06, 2.5, day)
    sigs = event_signals.earnings_drift(releases(ny("2024-07-25", 16, 5)), bars, "SPY")
    assert [(s.symbol, s.session, s.kind) for s in sigs] == [("ACME", date(2024, 7, 26), EARNINGS)]
    assert sigs[0].score == pytest.approx(0.06)
    # Nothing after the first full session changes it: a later crash is invisible.
    later = {k: v.copy() for k, v in bars.items()}
    later["ACME"].loc[pd.Timestamp("2024-07-29") :, "close"] = 1.0
    again = event_signals.earnings_drift(releases(ny("2024-07-25", 16, 5)), later, "SPY")
    assert [(s.session, s.score) for s in again] == [(s.session, s.score) for s in sigs]


@pytest.mark.parametrize(("jump", "mult"), [(0.049, 3.0), (0.08, 1.9)])
def test_drift_needs_five_percent_and_double_volume(jump: float, mult: float) -> None:
    bars = drift_bars(jump, mult, date(2024, 7, 25))
    assert event_signals.earnings_drift(releases(ny("2024-07-25", 16, 5)), bars, "SPY") == []


def test_untimed_releases_never_signal() -> None:
    bars = drift_bars(0.10, 3.0, date(2024, 7, 25))
    r = releases(ny("2024-07-25", 16, 5)).assign(source="volume_proxy")
    assert event_signals.earnings_drift(r, bars, "SPY") == []


# ---- insider clusters --------------------------------------------------------------------


SESSIONS = clock.sessions_between(date(2024, 1, 2), date(2024, 12, 31))


def buy(
    owner: int, trans: str, filed: str, value: float, plan: bool | None = False
) -> dict[str, Any]:
    return {
        "accession": f"a{owner}{trans}",
        "issuer_cik": 1,
        "symbol": "ACME",
        "owner_cik": owner,
        "relationship": "Director",
        "trans_date": date.fromisoformat(trans),
        "filing_date": date.fromisoformat(filed),
        "shares": value / 10,
        "price": 10.0,
        "value": value,
        "plan_10b5_1": plan,
        "form": "4",
    }


def clusters(rows: list[dict[str, Any]]) -> list[Signal]:
    return event_signals.insider_clusters(pd.DataFrame(rows), SESSIONS)


def test_two_insiders_within_ten_sessions_and_100k_make_a_cluster() -> None:
    out = clusters(
        [buy(1, "2024-03-04", "2024-03-05", 60_000), buy(2, "2024-03-08", "2024-03-11", 50_000)]
    )
    assert len(out) == 1 and out[0].kind == INSIDER and out[0].score == 110_000
    # Known at the close of the session after the filing date that completed it.
    assert out[0].session == date(2024, 3, 12)


@pytest.mark.parametrize(
    "rows",
    [
        [buy(1, "2024-03-04", "2024-03-05", 80_000), buy(1, "2024-03-06", "2024-03-07", 80_000)],
        [buy(1, "2024-03-04", "2024-03-05", 60_000), buy(2, "2024-03-19", "2024-03-20", 60_000)],
        [buy(1, "2024-03-04", "2024-03-05", 50_000), buy(2, "2024-03-05", "2024-03-06", 49_999)],
    ],
    ids=["one insider twice", "eleven sessions apart", "under 100k"],
)
def test_not_a_cluster(rows: list[dict[str, Any]]) -> None:
    assert clusters(rows) == []


def test_10b5_1_purchases_are_excluded_and_missing_flags_count() -> None:
    assert (
        clusters(
            [
                buy(1, "2024-03-04", "2024-03-05", 60_000),
                buy(2, "2024-03-05", "2024-03-06", 60_000, plan=True),
            ]
        )
        == []
    )
    assert (
        len(
            clusters(
                [
                    buy(1, "2024-03-04", "2024-03-05", 60_000),
                    buy(2, "2024-03-05", "2024-03-06", 60_000, plan=None),
                ]
            )
        )
        == 1
    )


def test_a_cluster_is_invisible_before_its_last_filing_is_known() -> None:
    # The second purchase is filed late: the cluster exists only once that filing is known.
    out = clusters(
        [buy(1, "2024-03-04", "2024-03-05", 60_000), buy(2, "2024-03-05", "2024-04-15", 60_000)]
    )
    assert [s.session for s in out] == [date(2024, 4, 16)]


def test_one_cluster_signal_per_issuer_per_forty_sessions() -> None:
    rows = [
        buy(1, "2024-03-04", "2024-03-05", 60_000),
        buy(2, "2024-03-05", "2024-03-06", 60_000),
        buy(3, "2024-03-25", "2024-03-26", 60_000),
        buy(4, "2024-03-26", "2024-03-27", 60_000),
        buy(5, "2024-06-03", "2024-06-04", 60_000),
        buy(6, "2024-06-04", "2024-06-05", 60_000),
    ]
    assert [s.session for s in clusters(rows)] == [date(2024, 3, 7), date(2024, 6, 6)]


def test_slots_go_to_insider_clusters_first_then_the_biggest_drift() -> None:
    sigs = [
        Signal("A", date(2024, 3, 1), EARNINGS, 0.06, ""),
        Signal("B", date(2024, 3, 1), EARNINGS, 0.12, ""),
        Signal("C", date(2024, 3, 1), INSIDER, 150_000, ""),
        Signal("D", date(2024, 3, 1), INSIDER, 900_000, ""),
    ]
    assert [s.symbol for s in event_signals.ranked(sigs)] == ["D", "C", "B", "A"]


# ---- the strategy --------------------------------------------------------------------------


def settings() -> Settings:
    return make_settings(
        watchlist=",".join(SMALL_WATCHLIST),
        earnings__block_when_unknown="false",
        risk__block_entries_when_risk_off="false",
        risk__max_pair_correlation="1.0",
    )


def decide(
    strat: EventStrategy,
    bars: Bars,
    day: date,
    *,
    holdings: dict[str, Holding] | None = None,
    equity: str = "100",
    universe: Universe | None = None,
) -> Any:
    feats = build_features(bars, "SPY")
    syms = tuple(sorted(set(SMALL_WATCHLIST)))
    uni = universe or Universe(
        watchlist=syms, benchmark="SPY", assets={s: AssetInfo(s, True, True) for s in syms}
    )
    return strat.decide(
        DecisionInput(
            session=day,
            execute_session=clock.next_session(day),
            features=slice_features(feats, day),
            holdings=holdings or {},
            equity=D(equity),
            cash=D(equity),
            prev_equity=D(equity),
            high_water_mark=D(equity),
            kill_switch_active=False,
            universe=uni,
        )
    )


def liquid_day(bars: Bars) -> date:
    f = build_features(bars, "SPY")
    for ts in reversed(f["SPY"].index[:-60]):
        d = ts.date()
        ok = all(
            f[s].loc[ts, "close"] >= 5 and f[s].loc[ts, "dollar_vol_20"] >= 5e6
            for s in ("AAPL", "MSFT", "NVDA", "JPM")
            if ts in f[s].index
        )
        if ok:
            return d
    raise AssertionError("no liquid day")


def test_strategy_fills_slots_in_the_registered_order(bars: Bars) -> None:
    day = liquid_day(bars)
    sigs = [
        Signal("AAPL", day, EARNINGS, 0.06, "small drift"),
        Signal("MSFT", day, EARNINGS, 0.20, "big drift"),
        Signal("NVDA", day, INSIDER, 200_000, "cluster"),
    ]
    s = settings().model_copy(
        update={
            "risk": settings().risk.model_copy(update={"max_positions": 2, "max_orders_per_day": 2})
        }
    )
    strat = EventStrategy(s, {day: event_signals.ranked(sigs)}, EarningsCalendar(empty_calendar()))
    d = decide(strat, bars, day)
    assert [o.symbol for o in d.orders if o.side is Side.BUY] == ["NVDA", "MSFT"]
    assert {e.symbol: e.kind for e in strat.entries} == {"NVDA": INSIDER, "MSFT": EARNINGS}
    assert all(o.notional is not None and o.notional <= D(20) for o in d.orders)


def test_signals_from_other_days_are_ignored(bars: Bars) -> None:
    day = liquid_day(bars)
    strat = EventStrategy(
        settings(),
        {clock.previous_session(day): [Signal("AAPL", day, EARNINGS, 0.1, "")]},
        EarningsCalendar(empty_calendar()),
    )
    assert decide(strat, bars, day).orders == ()


@pytest.mark.parametrize(
    ("kind", "held", "exits"), [(EARNINGS, 20, True), (INSIDER, 20, False), (INSIDER, 40, True)]
)
def test_hold_periods(bars: Bars, kind: str, held: int, exits: bool) -> None:
    day = liquid_day(bars)
    entry = clock.sessions_between(day - timedelta(days=90), day)[-(held + 1)]
    close = float(build_features(bars, "SPY")["AAPL"].loc[pd.Timestamp(entry), "close"])
    meta = PositionMeta(
        entry_session=entry, entry_price=D(str(close)), entry_atr=D(str(close))
    )  # stops far away
    holding = {
        "AAPL": Holding(symbol="AAPL", qty=D("0.1"), avg_entry_price=D(str(close)), meta=meta)
    }
    strat = EventStrategy(
        settings(), {}, EarningsCalendar(empty_calendar()), kind_of={"AAPL": kind}
    )
    d = decide(strat, bars, day, holdings=holding)
    reasons = {e.reason for e in d.exits if e.symbol == "AAPL"}
    assert ("hold_period" in reasons) is exits


def test_account_fit_and_whole_shares(bars: Bars) -> None:
    day = liquid_day(bars)
    f = build_features(bars, "SPY")
    price = float(f["AAPL"].loc[pd.Timestamp(day), "close"])
    syms = tuple(sorted(set(SMALL_WATCHLIST)))
    uni = Universe(
        watchlist=syms,
        benchmark="SPY",
        assets={s: AssetInfo(s, True, s != "AAPL") for s in syms},
        whole_shares=frozenset({"AAPL"}),
    )
    sig = {day: [Signal("AAPL", day, EARNINGS, 0.1, "")]}
    # Not fractionable and above the position size (20% of equity): it does not fit.
    small = EventStrategy(settings(), sig, EarningsCalendar(empty_calendar()))
    eq_small = str(round(price * 4, 2))  # 20% of this is 0.8 shares
    assert decide(small, bars, day, equity=eq_small, universe=uni).orders == ()
    # With enough equity for one whole share, it fits and is bought in whole shares.
    big = EventStrategy(settings(), sig, EarningsCalendar(empty_calendar()))
    eq_big = str(round(price * 12, 2))  # 20% buys 2.4 shares
    d = decide(big, bars, day, equity=eq_big, universe=uni)
    buys = [o for o in d.orders if o.side is Side.BUY]
    assert [o.symbol for o in buys] == ["AAPL"]
    ok, rej = whole_shares(buys, {"AAPL": D(str(price))}, uni)
    assert rej == [] and ok[0].qty == 2 and ok[0].notional is None


def test_whole_shares_never_exceed_the_notional() -> None:
    uni = Universe(("X",), "SPY", {"X": AssetInfo("X", True, False)}, whole_shares=frozenset({"X"}))
    o = PlannedOrder(
        symbol="X", side=Side.BUY, notional=D("19.99"), ref_price=D(10), reason="entry"
    )
    ok, rej = whole_shares([o], {"X": D("10.00")}, uni)
    assert ok[0].qty == 1
    ok, rej = whole_shares([o], {"X": D("25.00")}, uni)
    assert ok == [] and rej[0].reason == "whole_shares"


# ---- costs -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dv", "bps"),
    [
        (1e9, 5.0),
        (1e8, 25 * (0.1**0.5)),
        (1e7, 25.0),
        (5e6, 25 * (2**0.5)),
        (1e5, 75.0),
        (None, 75.0),
    ],
)
def test_liquidity_scaled_costs(dv: float | None, bps: float) -> None:
    assert wide.slippage_bps(dv) == pytest.approx(bps)


def test_the_sim_broker_charges_per_symbol_costs() -> None:
    from stoa.broker.base import OrderRequest

    uni = Universe(("A", "B"), "SPY", {k: AssetInfo(k, True, True) for k in ("A", "B")})
    s = settings()
    broker = SimBroker(
        Portfolio(cash=D(1000)), s.costs, uni, lambda sym, _d: 5.0 if sym == "A" else 50.0
    )
    broker.begin_session(date(2024, 3, 4), {"A": D(100), "B": D(100)})
    a = broker.submit(OrderRequest(symbol="A", side=Side.BUY, client_order_id="a", notional=D(100)))
    b = broker.submit(OrderRequest(symbol="B", side=Side.BUY, client_order_id="b", notional=D(100)))
    assert a.filled_avg_price == D("100.05") and b.filled_avg_price == D("100.5")


def test_liquidity_universe_has_no_lookahead() -> None:
    idx = pd.DatetimeIndex(
        [pd.Timestamp(d) for d in clock.sessions_between(date(2024, 1, 2), date(2024, 3, 29))]
    )
    df = pd.DataFrame({"close": 10.0, "volume": 1e6}, index=idx)  # $10M a day
    ok = wide.liquid(df)
    assert not ok.iloc[:19].any() and ok.iloc[19:].all()  # needs 20 sessions of history
    later = df.copy()
    later.iloc[40:, later.columns.get_loc("volume")] = 1.0
    assert wide.liquid(later).iloc[:40].equals(ok.iloc[:40])


# ---- holdout and diagnostics ------------------------------------------------------------


def test_event_data_stops_before_the_holdout(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(event_run, "EVENTS_DIR", tmp_path)
    sigs = [
        Signal("A", date(2026, 6, 30), EARNINGS, 0.1, ""),
        Signal("B", date(2026, 7, 1), INSIDER, 1e6, ""),  # inside the holdout
        Signal("C", date(2026, 10, 26), INSIDER, 1e6, ""),
    ]
    event_signals.to_frame(sigs).to_parquet(tmp_path / "signals.parquet")
    end = max(w.end for w in make_settings().backtest.windows.values())
    assert end < make_settings().backtest.holdout_start
    assert [s.symbol for s in event_run.load_signals(end)] == ["A"]


def test_diagnostics_match_trades_to_the_signal_that_opened_them() -> None:
    days = clock.sessions_between(date(2024, 3, 1), date(2024, 5, 31))
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    spy = pd.DataFrame({"open": 100.0, "close": 100.0}, index=idx)
    entries = [
        event_strategy_entry(date(2024, 3, 4), "A", INSIDER),
        event_strategy_entry(date(2024, 3, 4), "B", EARNINGS),
    ]
    trades = [
        trade("A", date(2024, 3, 5), date(2024, 5, 1), 10.0, 11.0),
        trade("B", date(2024, 3, 5), date(2024, 4, 2), 10.0, 9.5),
    ]
    out = {s.kind: s for s in diagnostics(entries, trades, spy, (INSIDER, EARNINGS))}
    assert out[INSIDER].trades == 1 and out[INSIDER].hit_rate == 1.0
    assert out[INSIDER].avg_excess == pytest.approx(0.10)
    assert out[EARNINGS].hit_rate == 0.0 and out[EARNINGS].avg_excess == pytest.approx(-0.05)


def event_strategy_entry(day: date, sym: str, kind: str) -> Any:
    from stoa.event_strategy import Entry

    return Entry(day, sym, kind, 1.0)


def trade(sym: str, a: date, b: date, px_in: float, px_out: float) -> ClosedTrade:
    return ClosedTrade(
        symbol=sym,
        entry_session=a,
        exit_session=b,
        qty=D(1),
        entry_price=D(str(px_in)),
        exit_price=D(str(px_out)),
        fees=D(0),
        pnl=D(str(px_out - px_in)),
        reason="hold_period",
    )
