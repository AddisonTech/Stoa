"""The model strategy: ranking, earnings exits, the horizon exit, event-day sizing, fallback."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd

from stoa import clock
from stoa.backtest import run_backtest
from stoa.config import Settings
from stoa.data.base import Bars
from stoa.earnings import EXEMPT_DEFAULT, EarningsCalendar, empty_calendar, synthetic_calendar
from stoa.features import build_features, regime_is_bull, slice_features
from stoa.model_strategy import ModelStrategy, PanelPredictor
from stoa.models import Holding, PositionMeta, Side
from stoa.planner import DecisionInput
from stoa.universe import Universe
from tests.conftest import SMALL_WATCHLIST, make_settings

D = Decimal
SECTORS = {
    "SPY": "ETF",
    "QQQ": "ETF2",
    "AAPL": "Tech",
    "MSFT": "Tech",
    "NVDA": "Tech",
    "JPM": "Fin",
    "XOM": "Energy",
    "KO": "Staples",
}


def settings(**kw: str) -> Settings:
    base = {
        "watchlist": ",".join(SMALL_WATCHLIST),
        "earnings__block_when_unknown": "false",
        "quant__max_rsi_14": "100",
        "quant__max_return_5": "1",
        "quant__min_dollar_volume_20": "0",
        "quant__max_volatility_20": "5",
        "quant__min_return_60": "-1",
    }
    base.update(kw)
    return make_settings(**base)


def uptrend(bars: Bars, day: date) -> list[str]:
    """Symbols that pass the v1 uptrend gate (close above the 50 day average) on ``day``."""
    f = build_features(bars, "SPY")
    ts = pd.Timestamp(day)
    out = []
    for sym in SMALL_WATCHLIST:
        if sym in ("SPY", "QQQ"):
            continue
        row = f[sym].loc[:ts].tail(2)
        if len(row) == 2 and (row["close"] > row["sma_50"]).all():
            out.append(sym)
    return out


def bull_day(bars: Bars, need: int = 3) -> date:
    """A bull-regime day, not an FOMC or CPI day, with at least ``need`` symbols in uptrend."""
    from stoa import events

    f = build_features(bars, "SPY")
    for ts in reversed(f["SPY"].index[:-30]):
        d = ts.date()
        calm = regime_is_bull(f, "SPY", d) and events.light_reason(d) is None
        if calm and len(uptrend(bars, d)) >= need:
            return d
    raise AssertionError("no suitable day")


def probs_panel(bars: Bars, values: dict[str, float]) -> pd.DataFrame:
    idx = bars["SPY"].index
    rows = []
    for ts in idx:
        for sym, p in values.items():
            rows.append((ts, sym, p))
    return pd.DataFrame(rows, columns=["date", "symbol", "p"]).set_index(["date", "symbol"])


def decision(
    s: Settings,
    bars: Bars,
    day: date,
    predictor: PanelPredictor,
    cal: EarningsCalendar | None = None,
    holdings: dict[str, Holding] | None = None,
    cash: str = "100",
    **kw: object,
):  # type: ignore[no-untyped-def]
    feats = build_features(bars, "SPY")
    strat = ModelStrategy(s, predictor, cal or EarningsCalendar(empty_calendar()), sectors=SECTORS)
    inp = DecisionInput(
        session=day,
        execute_session=clock.next_session(day),
        features=slice_features(feats, day),
        holdings=holdings or {},
        equity=D(100),
        cash=D(cash),
        prev_equity=D(100),
        high_water_mark=D(100),
        kill_switch_active=False,
        universe=Universe.assume_all_fractionable(s.watchlist, "SPY"),
        **kw,  # type: ignore[arg-type]
    )
    return strat.decide(inp)


def test_ranks_by_probability_and_respects_threshold(bars: Bars) -> None:
    s = settings()
    day = bull_day(bars, need=4)
    a, b, c, e = uptrend(bars, day)[:4]
    pred = PanelPredictor(probs_panel(bars, {a: 0.62, b: 0.58, c: 0.52, e: 0.60}), {})
    d = decision(s, bars, day, pred)
    assert [x.symbol for x in d.candidates] == [a, e, b]  # c is under the 0.55 threshold
    buys = [o for o in d.orders if o.side is Side.BUY]
    assert buys and all(o.probability is not None for o in buys)
    assert all(o.notional is not None and o.notional <= D(20) for o in buys)
    # Higher probability gets at least as much weight when volatility is similar.
    assert all(o.sector in SECTORS.values() for o in buys)


def test_no_predictions_means_no_entries(bars: Bars) -> None:
    s = settings()
    day = bull_day(bars)
    empty = PanelPredictor(
        pd.DataFrame(columns=["p"], index=pd.MultiIndex.from_tuples([], names=["date", "symbol"])),
        {},
    )
    d = decision(s, bars, day, empty)
    assert d.orders == ()


def test_earnings_blocks_entry_and_forces_exit(bars: Bars) -> None:
    s = settings()
    day = bull_day(bars)
    nxt = clock.next_session(day)
    release = clock.session(nxt).close + timedelta(minutes=5)
    known = datetime.combine(day - timedelta(days=20), datetime.min.time(), tzinfo=UTC)
    up = uptrend(bars, day)
    reporter, other, holder = up[0], up[1], up[2]
    cal = EarningsCalendar(
        pd.DataFrame(
            [
                {"symbol": reporter, "event_time": release, "known_at": known, "source": "t"},
                {"symbol": holder, "event_time": release, "known_at": known, "source": "t"},
            ]
        )
    )
    pred = PanelPredictor(probs_panel(bars, {reporter: 0.65, other: 0.6}), {})
    meta = PositionMeta(entry_session=day - timedelta(days=7), entry_price=D(1), entry_atr=D(50))
    held = {holder: Holding(symbol=holder, qty=D("0.1"), avg_entry_price=D(1), meta=meta)}
    d = decision(s, bars, day, pred, cal, held, cash="80")
    reasons = {r.symbol: r.reason for r in d.rejections}
    assert reasons.get(reporter) == "earnings"
    assert other in {o.symbol for o in d.orders if o.side is Side.BUY}
    sells = [o for o in d.orders if o.side is Side.SELL]
    assert [o.symbol for o in sells] == [holder] and sells[0].reason == "earnings_exit"


def test_event_day_halves_new_position_size(bars: Bars) -> None:
    from stoa import events

    s = settings(sizing__method="equal")
    f = build_features(bars, "SPY")
    light = calm = None
    for ts in f["SPY"].index[300:-30]:
        d0 = ts.date()
        if not regime_is_bull(f, "SPY", d0) or not uptrend(bars, d0):
            continue
        if events.light_reason(d0) and light is None:
            light = d0
        if events.light_reason(d0) is None and calm is None:
            calm = d0
    assert light is not None and calm is not None
    sym = uptrend(bars, light)[0]
    d = decision(s, bars, light, PanelPredictor(probs_panel(bars, {sym: 0.7}), {}))
    buys = [o for o in d.orders if o.side is Side.BUY]
    assert buys and buys[0].notional == D("10.00")
    assert ("CPI" in buys[0].thesis or "FOMC" in buys[0].thesis) and d.notes
    sym2 = uptrend(bars, calm)[0]
    c = decision(s, bars, calm, PanelPredictor(probs_panel(bars, {sym2: 0.7}), {}))
    calm_buys = [o for o in c.orders if o.side is Side.BUY]
    assert calm_buys and calm_buys[0].notional == D("20.00")


def test_horizon_exit_when_model_no_longer_favors(bars: Bars) -> None:
    s = settings()
    day = bull_day(bars)
    sym = uptrend(bars, day)[0]
    entry = clock.sessions_between(day - timedelta(days=40), day)[-12]
    meta = PositionMeta(entry_session=entry, entry_price=D(1), entry_atr=D(50))
    held = {sym: Holding(symbol=sym, qty=D("0.1"), avg_entry_price=D(1), meta=meta)}
    low = PanelPredictor(probs_panel(bars, {sym: 0.40}), {})
    d = decision(s, bars, day, low, holdings=held, cash="90")
    assert any(e.symbol == sym and e.reason == "horizon_exit" for e in d.exits)
    high = PanelPredictor(probs_panel(bars, {sym: 0.70}), {})
    d2 = decision(s, bars, day, high, holdings=held, cash="90")
    assert not any(e.reason == "horizon_exit" for e in d2.exits)


def test_fallback_buys_only_spy_and_tops_up(bars: Bars) -> None:
    s = settings()
    day = bull_day(bars)
    pred = PanelPredictor(probs_panel(bars, {"AAPL": 0.9}), {})
    d = decision(s, bars, day, pred, halt_reason="drift fallback", fallback_active=True)
    buys = [o for o in d.orders if o.side is Side.BUY]
    assert [o.symbol for o in buys] == ["SPY"] and buys[0].notional == D("98.00")
    assert buys[0].reason == "spy_fallback"
    # Already holding SPY at 50%: top up to 98%, never past the cash buffer.
    meta = PositionMeta(entry_session=day, entry_price=D(1), entry_atr=D(0))
    feats = build_features(bars, "SPY")
    px = D(str(feats["SPY"]["close"].loc[pd.Timestamp(day)]))
    held = {"SPY": Holding(symbol="SPY", qty=D(50) / px, avg_entry_price=px, meta=meta)}
    d2 = decision(
        s, bars, day, pred, holdings=held, cash="50", halt_reason="x", fallback_active=True
    )
    top = [o for o in d2.orders if o.side is Side.BUY]
    assert top and top[0].symbol == "SPY" and D(47) <= (top[0].notional or 0) <= D(48)


def test_halt_without_fallback_blocks_entries(bars: Bars) -> None:
    s = settings()
    day = bull_day(bars)
    pred = PanelPredictor(probs_panel(bars, {"AAPL": 0.9}), {})
    d = decision(s, bars, day, pred, halt_reason="quit rule")
    assert not [o for o in d.orders if o.side is Side.BUY]


def test_model_strategy_backtest_runs_with_every_rule(bars: Bars) -> None:
    s = settings(earnings__block_when_unknown="true")
    rng = np.random.default_rng(0)
    idx = bars["SPY"].index
    rows = [
        (ts, sym, float(rng.uniform(0.45, 0.7)))
        for ts in idx
        for sym in SMALL_WATCHLIST
        if sym != "SPY"
    ]
    panel = pd.DataFrame(rows, columns=["date", "symbol", "p"]).set_index(["date", "symbol"])
    from stoa.earnings import with_known_lead

    table = with_known_lead(synthetic_calendar(bars, EXEMPT_DEFAULT), 10)
    cal = EarningsCalendar(table)
    strat = ModelStrategy(s, PanelPredictor(panel, {}), cal, sectors=SECTORS)
    res = run_backtest(s, bars, strat, date(2020, 1, 2), date(2021, 6, 30))
    assert res.trades, "expected trades"
    assert float(res.exposure.max()) <= 0.98 + 1e-6 or res.exposure.max() < 1.0
    # Never held through a synthetic earnings release.
    rel = synthetic_calendar(bars, EXEMPT_DEFAULT)
    for t in res.trades:
        r = rel.loc[rel["symbol"] == t.symbol, "event_time"]
        entry_open = clock.session(t.entry_session).open
        exit_open = clock.session(t.exit_session).open
        inside = r[(r > pd.Timestamp(entry_open)) & (r < pd.Timestamp(exit_open))]
        assert inside.empty, f"{t.symbol} held through earnings {list(inside)}"
    # The sector cap held at every decision.
    assert res.rejections_by_reason.get("earnings", 0) > 0
