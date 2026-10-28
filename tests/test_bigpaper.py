"""The big paper account: kept apart from the $100 account, trial 12 as registered (trial 13's
overlay behind the setting), the SPY core at a broker, auto-approved picks, tagged alerts, and
a kill switch that sells everything."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from pydantic import SecretStr

from stoa import alerts, bigpaper, clock
from stoa.broker.base import Account, BrokerPosition, OrderRequest, OrderStatus
from stoa.broker.core import CoreBroker
from stoa.config import Secrets, Settings
from stoa.data.synthetic import synthetic_bars
from stoa.earnings import EarningsCalendar, empty_calendar
from stoa.event_signals import Signal
from stoa.models import PlannedOrder, PositionMeta, Side
from stoa.newsbrain import trial as t9
from stoa.planner import plan_id_for
from stoa.state import KillSwitch, Plan, StateStore
from stoa.universe import AssetInfo
from tests.conftest import ROOT, make_settings

D = Decimal
SESSION = date(2021, 6, 15)  # a Tuesday
BIG = {"ALPACA_BIG_PAPER_KEY_ID": "big-key", "ALPACA_BIG_PAPER_SECRET": "big-secret"}
SMALL = {"ALPACA_PAPER_KEY_ID": "small-key", "ALPACA_PAPER_SECRET": "small-secret"}


@dataclass
class FakeBroker:
    """An account that fills every market order at once at its price."""

    cash: Decimal
    prices: dict[str, Decimal]
    held: dict[str, Decimal] = field(default_factory=dict)
    orders: dict[str, OrderStatus] = field(default_factory=dict)
    session: date = SESSION
    last_equity: Decimal | None = None

    def equity(self) -> Decimal:
        return self.cash + sum((q * self.prices[s] for s, q in self.held.items()), D(0))

    def account(self) -> Account:
        return Account(equity=self.equity(), cash=self.cash, last_equity=self.last_equity)

    def positions(self) -> dict[str, BrokerPosition]:
        return {
            s: BrokerPosition(
                symbol=s, qty=q, avg_entry_price=self.prices[s], current_price=self.prices[s]
            )
            for s, q in self.held.items()
            if q > 0
        }

    def asset(self, symbol: str) -> AssetInfo:
        return AssetInfo(symbol, tradable=True, fractionable=True)

    def latest_prices(self, symbols: list[str]) -> dict[str, Decimal]:
        return {s: self.prices[s] for s in symbols if s in self.prices}

    def find_order(self, client_order_id: str) -> OrderStatus | None:
        return self.orders.get(client_order_id)

    def submit(self, request: OrderRequest) -> OrderStatus:
        if request.client_order_id in self.orders:
            return self.orders[request.client_order_id]
        px = self.prices[request.symbol]
        qty = request.qty if request.qty is not None else (request.notional or D(0)) / px
        if request.side is Side.BUY:
            if qty * px > self.cash + D("0.01"):
                st = self._status(request, "rejected", D(0), None)
                self.orders[request.client_order_id] = st
                return st
            self.cash -= qty * px
            self.held[request.symbol] = self.held.get(request.symbol, D(0)) + qty
        else:
            qty = min(qty, self.held.get(request.symbol, D(0)))
            self.cash += qty * px
            self.held[request.symbol] -= qty
        st = self._status(request, "filled", qty, px)
        self.orders[request.client_order_id] = st
        return st

    def _status(
        self, r: OrderRequest, status: str, qty: Decimal, px: Decimal | None
    ) -> OrderStatus:
        return OrderStatus(
            client_order_id=r.client_order_id,
            broker_order_id=r.client_order_id,
            symbol=r.symbol,
            side=r.side,
            status=status,
            filled_qty=qty,
            filled_avg_price=px,
            session=self.session,
        )

    def await_terminal(self, client_order_ids: list[str]) -> list[OrderStatus]:
        return [self.orders[c] for c in client_order_ids if c in self.orders]

    def orders_for_session(self, session: date) -> list[OrderStatus]:
        return [o for o in self.orders.values() if o.session == session]


def settings(tmp_path: Path, enabled: bool = True, strategy: str = "trial12") -> Settings:
    return make_settings(
        big_paper__enabled="true" if enabled else "false",
        big_paper__state_dir=str(tmp_path / "big"),
        big_paper__strategy=strategy,
        earnings__block_when_unknown="false",
        risk__block_entries_when_risk_off="false",
        risk__max_pair_correlation="1.0",
    )


def runtime(
    tmp_path: Path, broker: FakeBroker, strategy: str = "trial12", **kw: object
) -> tuple[bigpaper.BigRuntime, alerts.RecordingNotifier]:
    rec = alerts.RecordingNotifier()
    rt = bigpaper.BigRuntime(
        base=settings(tmp_path, strategy=strategy),
        secrets=Secrets.from_env({**SMALL, **BIG}),
        notifier=rec,
        broker=broker,  # type: ignore[arg-type]
        **kw,  # type: ignore[arg-type]
    )
    return rt, rec


# ---- kept apart ---------------------------------------------------------------------------------


def test_off_until_enabled_and_keyed_and_never_on_the_100_dollar_keys(tmp_path: Path) -> None:
    on = settings(tmp_path)
    assert bigpaper.off_reason(settings(tmp_path, enabled=False), Secrets.from_env(BIG))
    assert "ALPACA_BIG_PAPER_KEY_ID" in str(bigpaper.off_reason(on, Secrets.from_env(SMALL)))
    same = {**SMALL, "ALPACA_BIG_PAPER_KEY_ID": "small-key", "ALPACA_BIG_PAPER_SECRET": "x"}
    assert "must be different" in str(bigpaper.off_reason(on, Secrets.from_env(same)))
    assert bigpaper.off_reason(on, Secrets.from_env({**SMALL, **BIG})) is None
    assert make_settings().big_paper.enabled is False  # shipped off


def test_the_two_accounts_never_share_keys_or_state(tmp_path: Path) -> None:
    only_big = Secrets.from_env(BIG)
    assert not only_big.has_paper_keys()  # the $100 account cannot run on the big keys
    data = bigpaper.data_secrets(Secrets.from_env({**SMALL, **BIG}))
    assert data.alpaca_paper_key_id == SecretStr("big-key")
    assert data.finnhub_key is None and data.alpaca_live_key_id is None
    rt, _ = runtime(tmp_path, FakeBroker(D(100), {"SPY": D(400)}))
    assert rt.settings.state.dir == tmp_path / "big"
    assert make_settings().state.dir != make_settings().big_paper.state_dir
    assert make_settings().big_paper.strategy == "trial12"  # shipped on trial 12
    assert rt.settings.risk == rt.base.risk  # trial 12's rails, as registered
    assert rt.settings.backtest.starting_equity == D(100000)
    t13, _ = runtime(tmp_path, FakeBroker(D(100), {"SPY": D(400)}), strategy="trial13")
    assert t13.settings.risk.max_position_pct == D("0.05")  # trial 13's cap, only by setting
    assert make_settings().risk.max_position_pct == D("0.20")  # the $100 account's, unchanged
    assert (
        make_settings().approval.buys == "mode"
    )  # no tap on paper for either account (2026-10-28)


def test_a_disabled_account_does_nothing_and_sends_nothing(tmp_path: Path) -> None:
    rec = alerts.RecordingNotifier()
    rt = bigpaper.BigRuntime(
        base=settings(tmp_path, enabled=False), secrets=Secrets.from_env(BIG), notifier=rec
    )
    for job in (bigpaper.picks, bigpaper.execute, bigpaper.summary):
        r = job(rt, True)
        assert r.code == 0 and r.status.startswith("off")
    assert rec.alerts == [] and not (tmp_path / "big").exists()


def test_every_alert_is_tagged_big_paper() -> None:
    rec = alerts.RecordingNotifier()
    tagged = bigpaper.Tagged(rec, "BIG PAPER")
    tagged.send(alerts.Alert(title="Fills 2021-06-15", message="x"))
    tagged.send(alerts.Alert(title="BIG PAPER: already", message="x"))
    assert [a.title for a in rec.alerts] == ["BIG PAPER: Fills 2021-06-15", "BIG PAPER: already"]


# ---- the SPY core at a broker --------------------------------------------------------------------


def core(cash: str, spy_qty: str = "0") -> tuple[CoreBroker, FakeBroker]:
    fb = FakeBroker(D(cash), {"SPY": D(400), "AAA": D(50), "ZYXI": D(5)})
    if D(spy_qty):
        fb.held["SPY"] = D(spy_qty)
    return CoreBroker(fb, symbol="SPY", reserve_pct=D("0.02"), session=SESSION), fb  # type: ignore[arg-type]


def test_spy_counts_as_cash_and_is_never_a_position() -> None:
    cb, _ = core("1000", "10")
    assert cb.account().cash == D(5000) and "SPY" not in cb.positions()


def test_a_buy_short_of_cash_sells_spy_first() -> None:
    cb, fb = core("100", "10")
    st = cb.submit(
        OrderRequest(symbol="AAA", side=Side.BUY, notional=D(1000), client_order_id="b1")
    )
    assert st.filled and fb.held["AAA"] == D(20)
    assert fb.held["SPY"] < D(10) and fb.cash >= 0
    with pytest.raises(ValueError):
        cb.submit(OrderRequest(symbol="SPY", side=Side.BUY, notional=D(10), client_order_id="s"))


def test_the_sweep_fills_spy_to_the_cap_and_trims_above_it() -> None:
    cb, fb = core("10000")
    cb.sweep(kill_switch=False)
    assert fb.held["SPY"] * 400 == pytest.approx(D(9800), abs=D(1))  # all but the 2% reserve
    cb2, fb2 = core("0", "25")  # $10,000 all in SPY, on a new session
    cb2.sweep(kill_switch=False, cap=D("0.50"))
    assert fb2.held["SPY"] * 400 == pytest.approx(D(5000), abs=D(5))
    cb.sweep(kill_switch=True)
    assert fb.held["SPY"] == 0


# ---- the jobs ------------------------------------------------------------------------------


def at(day: date, hhmm: str) -> datetime:
    return clock.at_ny(day, hhmm).astimezone(UTC)


def bars_for(symbols: list[str], through: date) -> dict[str, pd.DataFrame]:
    out = synthetic_bars(
        sorted({*symbols, "SPY", "AAPL", "MSFT"}), date(2019, 1, 2), through, seed=3
    )
    return {k: v.loc[: pd.Timestamp(through)] for k, v in out.items()}


def no_earnings(bars: object, symbols: object, session: object) -> EarningsCalendar:
    return EarningsCalendar(empty_calendar())


@pytest.mark.parametrize(("strategy", "size"), [("trial12", D(20000)), ("trial13", None)])
def test_picks_are_sized_by_the_trial_auto_approved_and_tagged(
    tmp_path: Path, strategy: str, size: Decimal | None
) -> None:
    fb = FakeBroker(
        D(100000), {"SPY": D(400), "AAPL": D(100), "MSFT": D(200)}, last_equity=D(100000)
    )
    decided = clock.previous_session(SESSION)
    sigs = [
        Signal("AAPL", decided, t9.kind_for(1), 0.70, "p 0.70"),
        Signal("MSFT", decided, t9.kind_for(1), 0.60, "p 0.60"),
    ]
    rt, rec = runtime(
        tmp_path,
        fb,
        strategy,
        now=lambda: at(SESSION, "09:00"),
        bars=bars_for,
        signals=lambda d: sigs,
        calendar=no_earnings,
        sectors={},
    )
    r = bigpaper.picks(rt)
    assert r.code == 0 and r.status == "picked", r
    plan = StateStore(tmp_path / "big").load().pending_plan
    assert plan is not None and plan.execute_session == SESSION
    buys = [o for o in plan.orders if o.side is Side.BUY]
    assert {o.symbol for o in buys} == {"AAPL", "MSFT"}
    if size is not None:  # trial 12: 20% of the account each, as its backtest sized them
        assert all(o.notional == size for o in buys)
        assert not (tmp_path / "big" / "overlay.json").exists()
    else:  # trial 13: ATR sizing under its 5% cap
        assert all(o.notional is not None and o.notional <= D(5000) for o in buys)
    assert plan.strategy == strategy
    assert len(rec.alerts) == 1 and rec.alerts[0].title == f"BIG PAPER: Picks {SESSION}"
    assert "auto-approved" in rec.alerts[0].message and not rec.alerts[0].actions  # no tap buttons


def test_a_dry_run_needs_no_switch_and_saves_and_sends_nothing(tmp_path: Path) -> None:
    fb = FakeBroker(D(100000), {"SPY": D(400), "AAPL": D(100)}, last_equity=D(100000))
    sigs = [Signal("AAPL", clock.previous_session(SESSION), t9.kind_for(1), 0.70, "p 0.70")]
    rec = alerts.RecordingNotifier()
    off = settings(tmp_path, enabled=False)
    keys = Secrets.from_env({**SMALL, **BIG})
    assert bigpaper.off_reason(off, keys) and bigpaper.off_reason(off, keys, dry=True) is None
    assert bigpaper.off_reason(off, Secrets.from_env(SMALL), dry=True)  # still needs its keys
    rt = bigpaper.BigRuntime(
        base=off,
        secrets=keys,
        notifier=rec,
        broker=fb,  # type: ignore[arg-type]
        now=lambda: at(SESSION, "16:30"),  # after hours: a dry run still runs
        bars=bars_for,
        signals=lambda d: sigs,
        calendar=no_earnings,
        sectors={},
        dry=True,
    )
    r = bigpaper.picks(rt)
    assert r.status == "dry" and r.session == SESSION and r.sent == []
    assert rec.alerts == [] and fb.orders == {}
    assert not (tmp_path / "big").exists()  # no plan, no journal, no state


def test_picks_without_the_news_brain_still_sell_and_say_why(tmp_path: Path) -> None:
    fb = FakeBroker(D(95000), {"SPY": D(400), "AAPL": D(100)}, held={"AAPL": D(50)})
    rt, rec = runtime(
        tmp_path,
        fb,
        now=lambda: at(SESSION, "09:00"),
        bars=bars_for,
        signals=lambda d: (_ for _ in ()).throw(
            RuntimeError("the news for 2021-06 is not scored yet")
        ),
        calendar=no_earnings,
        sectors={},
    )
    assert bigpaper.picks(rt).status == "picked"
    plan = StateStore(tmp_path / "big").load().pending_plan
    assert plan is not None and [(o.symbol, o.side) for o in plan.orders] == [("AAPL", Side.SELL)]
    msg = rec.alerts[0].message
    assert "No buys today; the account stays in SPY" in msg and "not scored yet" in msg
    assert rec.alerts[0].priority == alerts.HIGH and rec.alerts[0].title.startswith("BIG PAPER")


def plan_for(orders: list[PlannedOrder]) -> Plan:
    pid = plan_id_for("trial12", "paper", clock.previous_session(SESSION))
    return Plan(
        plan_id=pid,
        strategy="trial12",
        mode="paper",
        decided_session=clock.previous_session(SESSION),
        execute_session=SESSION,
        created_at=at(SESSION, "09:00"),
        orders=tuple(
            o.model_copy(update={"client_order_id": f"stoa-{o.symbol}-{o.side.value}-{pid}"})
            for o in orders
        ),
        equity_at_decide=D(100000),
    )


def test_execute_sells_yesterdays_picks_buys_todays_and_sweeps_the_rest_into_spy(
    tmp_path: Path,
) -> None:
    fb = FakeBroker(
        D(0), {"SPY": D(400), "AAPL": D(100), "MSFT": D(200)}, held={"SPY": D(200), "AAPL": D(50)}
    )
    rt, rec = runtime(tmp_path, fb, now=lambda: at(SESSION, "09:46"))
    store = StateStore(tmp_path / "big")
    st = store.load()
    st.high_water_mark = D(85000)
    st.last_close_equity = D(85000)
    st.positions_meta = {
        "AAPL": PositionMeta(
            entry_session=clock.previous_session(SESSION), entry_price=D(100), entry_atr=D(2)
        )
    }
    st.pending_plan = plan_for(
        [
            PlannedOrder(
                symbol="AAPL", side=Side.SELL, qty=D(50), ref_price=D(100), reason="hold_period"
            ),
            PlannedOrder(
                symbol="MSFT",
                side=Side.BUY,
                notional=D(4000),
                ref_price=D(200),
                reason="entry",
                entry_atr=D(4),
                stop_price=D(190),
            ),
        ]
    )
    store.save(st)
    r = bigpaper.execute(rt)
    assert r.status == "executed", r
    assert "AAPL" not in fb.positions() and fb.held["MSFT"] == D(20)
    assert fb.cash < D("0.03") * fb.equity()  # idle money went back into SPY
    assert rec.alerts[-1].title == f"BIG PAPER: Fills {SESSION}"
    assert StateStore(tmp_path / "big").load().pending_plan is None


def test_the_kill_switch_sells_every_position_in_the_account(tmp_path: Path) -> None:
    held = {"SPY": D(100), "AAPL": D(50), "ZYXI": D(400)}  # ZYXI: a wide-universe name
    fb = FakeBroker(
        D(1000), {"SPY": D(400), "AAPL": D(100), "ZYXI": D(5), "MSFT": D(200)}, held=held
    )
    rt, rec = runtime(tmp_path, fb, now=lambda: at(SESSION, "09:46"))
    store = StateStore(tmp_path / "big")
    st = store.load()
    st.kill_switch = KillSwitch(
        active=True, tripped_session=clock.previous_session(SESSION), reason="test"
    )
    st.high_water_mark = D(100000)
    st.pending_plan = plan_for(
        [
            PlannedOrder(
                symbol="MSFT", side=Side.BUY, notional=D(4000), ref_price=D(200), reason="entry"
            ),
        ]
    )
    store.save(st)
    assert bigpaper.execute(rt).status == "executed"
    assert fb.positions() == {}  # SPY and the wide-universe name too
    assert "MSFT" not in fb.held
    assert (
        rec.alerts[-1].message.startswith("KILL SWITCH")
        and rec.alerts[-1].priority == alerts.URGENT
    )


def test_summary_records_the_day_against_spy(tmp_path: Path) -> None:
    fb = FakeBroker(D(1000), {"SPY": D(400)}, held={"SPY": D(200)})  # $81,000
    rt, rec = runtime(tmp_path, fb, now=lambda: at(SESSION, "17:16"), bars=bars_for)
    store = StateStore(tmp_path / "big")
    st = store.load()
    st.high_water_mark = D(100000)
    store.save(st)
    assert bigpaper.summary(rt).status == "summarized"
    curve = bigpaper.read_curve(tmp_path / "big")
    assert curve[-1]["session"] == SESSION.isoformat() and curve[-1]["equity"] == 81000.0
    assert not (tmp_path / "big" / "overlay.json").exists()  # trial 12 has no brake
    assert rec.alerts[-1].title == f"BIG PAPER: Day summary {SESSION}"
    assert "Drawdown 19.0% from the high." in rec.alerts[-1].message
    assert "Brake" not in rec.alerts[-1].message


def test_summary_sets_trial_13s_brake_when_set_to_trial_13(tmp_path: Path) -> None:
    fb = FakeBroker(D(1000), {"SPY": D(400)}, held={"SPY": D(200)})  # $81,000
    rt, rec = runtime(tmp_path, fb, "trial13", now=lambda: at(SESSION, "17:16"), bars=bars_for)
    store = StateStore(tmp_path / "big")
    st = store.load()
    st.high_water_mark = D(100000)
    store.save(st)
    r = bigpaper.summary(rt)
    assert r.status == "summarized", r
    curve = bigpaper.read_curve(tmp_path / "big")
    assert curve[-1]["session"] == SESSION.isoformat() and curve[-1]["equity"] == 81000.0
    ov = json.loads((tmp_path / "big" / "overlay.json").read_text(encoding="utf-8"))
    assert ov["halted"] is True  # 19% off the high: no new positions
    assert rec.alerts[-1].title == f"BIG PAPER: Day summary {SESSION}"
    assert "no new positions" in rec.alerts[-1].message


def test_the_workflow_reads_only_the_big_keys() -> None:
    text = (ROOT / ".github" / "workflows" / "bigpaper.yml").read_text(encoding="utf-8")
    assert "secrets.ALPACA_BIG_PAPER_KEY_ID" in text and "secrets.ALPACA_BIG_PAPER_SECRET" in text
    assert "secrets.ALPACA_PAPER_KEY_ID" not in text and "ALPACA_LIVE" not in text
    daily = (ROOT / ".github" / "workflows" / "daily.yml").read_text(encoding="utf-8")
    assert "BIG_PAPER" not in daily and "HF_TOKEN" not in daily


def test_the_feed_state_lives_only_in_the_private_dataset() -> None:
    import yaml

    text = (ROOT / ".github" / "workflows" / "bigpaper.yml").read_text(encoding="utf-8")
    wf = yaml.safe_load(text)
    job = wf["jobs"]["run"]
    assert "HF_TOKEN" not in job["env"]  # only the step that runs Stoa gets it
    with_token = [s["name"] for s in job["steps"] if "HF_TOKEN" in (s.get("env") or {})]
    assert with_token == ["Run"]
    run = next(s for s in job["steps"] if s["name"] == "Run")
    assert "runner.temp" in run["env"]["STOA__BIG_PAPER__FEED_DIR"]  # not under state-branch
    assert "login" not in text and "upload-artifact" not in text
    assert make_settings().big_paper.feed_repo == "ASmith91/stoa-feed"
    assert "feed" in str(wf[True]["workflow_dispatch"]["inputs"]["job"]["options"])


# ---- the public dashboard ------------------------------------------------------------------------


def dash_settings(tmp_path: Path) -> Settings:
    return make_settings(
        state__dir=str(tmp_path / "state"), big_paper__state_dir=str(tmp_path / "big")
    )


def test_the_dashboard_shows_the_big_account_from_day_one(tmp_path: Path) -> None:
    from stoa import dashboard

    page = dashboard.build(dash_settings(tmp_path), tmp_path / "site", ROOT).read_text(
        encoding="utf-8"
    )
    assert "Big paper account (trial 12)" in page and "Not trading yet" in page
    assert "trial 13" not in page.lower()


def test_the_dashboard_draws_the_big_equity_curve_and_trade_log(tmp_path: Path) -> None:
    from stoa import dashboard
    from stoa.journal import Journal

    root = tmp_path / "big"
    for day, eq, spy in ((date(2026, 11, 2), 100000, 500.0), (date(2026, 11, 3), 101500, 502.5)):
        bigpaper.record_close(root, day, D(eq), spy)
    Journal(root / "journal").write(
        at(date(2026, 11, 3), "09:46"),
        "execute",
        date(2026, 11, 3),
        "execution",
        fills=[{"symbol": "NBIS", "side": "buy", "filled_qty": "40", "filled_avg_price": "120.5"}],
        closed=[
            {
                "symbol": "MU",
                "entry_session": "2026-11-02",
                "exit_session": "2026-11-03",
                "qty": "20",
                "entry_price": "100",
                "exit_price": "103",
                "pnl": "60",
                "reason": "hold_period",
            }
        ],
    )
    s = dash_settings(tmp_path)
    index = dashboard.build(s, tmp_path / "site", ROOT)
    page = index.read_text(encoding="utf-8")
    data = json.loads((index.parent / "data.json").read_text(encoding="utf-8"))
    big = data["big_paper"]
    assert big["curve"]["series"]["Big paper"] == pytest.approx([100.0, 101.5])
    assert big["curve"]["series"]["SPY"] == pytest.approx([100.0, 100.5])
    assert "NBIS" in page and "MU" in page and "auto-approved" in page
    assert "Not trading yet" not in page


def test_the_kill_switch_is_reset_only_by_hand(tmp_path: Path) -> None:
    fb = FakeBroker(D(70000), {"SPY": D(400)})
    rt, rec = runtime(tmp_path, fb)
    store = StateStore(tmp_path / "big")
    st = store.load()
    st.kill_switch = KillSwitch(active=True, tripped_session=SESSION, reason="drawdown limit")
    st.high_water_mark = D(100000)
    store.save(st)
    assert bigpaper.reset_kill_switch(rt).status == "reset"
    st = store.load()
    assert not st.kill_switch.active and st.high_water_mark == D(70000)
    assert rec.alerts[-1].title == "BIG PAPER: Kill switch reset"
