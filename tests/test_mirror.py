"""The $100 account as a paper mirror of trial 15 (usd100.strategy: mirror_trial15, 2026-11-03):
at 9:45 am it sells yesterday's 5 and buys the big account's 5 for today at equal weight with
2% in cash. Kill switch and price check on; no SPY, no taps, never live."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from stoa import alerts, dashboard, jobs, mirror
from stoa.broker.sim import SimBroker
from stoa.config import Secrets, Settings, load_settings
from stoa.data.base import Bars
from stoa.experiments import Registry
from stoa.models import PlannedOrder, PositionMeta, Side
from stoa.planner import Decision, make_plan
from stoa.portfolio import Lot, Portfolio
from stoa.state import KillSwitch, StateStore
from stoa.universe import Universe
from tests.conftest import ROOT
from tests.test_jobs import DECIDE_AT, DECIDE_DAY, EXEC_AT, EXEC_DAY, LIVE, PAPER, settings_for

D = Decimal
PICKS = ["MSFT", "NVDA", "JPM", "XOM", "QQQ"]
PRICES = {
    "SPY": D("400"),
    "QQQ": D("300"),
    "AAPL": D("150"),
    "MSFT": D("300"),
    "NVDA": D("200"),
    "JPM": D("160"),
    "XOM": D("60"),
    "KO": D("55"),
}
PICKS_FILE = f"big-paper/trial15/picks/{EXEC_DAY.isoformat()}.json"
PLAN_FILE = f"big-paper/plans/{DECIDE_DAY.isoformat()}-abc1234567.json"


class DictFiles:
    """The state branch as a dict of path to text."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.refreshed = 0

    def refresh(self) -> str:
        self.refreshed += 1
        return "the test's copy"

    def read(self, path: str) -> str | None:
        return self.files.get(path)

    def listdir(self, path: str) -> list[str]:
        pre = path.rstrip("/") + "/"
        return sorted(
            k[len(pre) :] for k in self.files if k.startswith(pre) and "/" not in k[len(pre) :]
        )


def branch(
    symbols: list[str] = PICKS,
    session: date = EXEC_DAY,
    refs: dict[str, Decimal] | None = None,
) -> DictFiles:
    body = {
        "session": session.isoformat(),
        "model": "t9-frozen",
        "picks": [
            {"rank": i + 1, "symbol": s, "conviction": 0.6 - i / 100, "detail": f"pick {s}"}
            for i, s in enumerate(symbols)
        ],
    }
    ref = refs if refs is not None else PRICES
    plan = {
        "execute_session": session.isoformat(),
        "created_at": f"{session.isoformat()}T14:03:00Z",
        "orders": [
            {"symbol": s, "side": "buy", "ref_price": str(ref[s])} for s in symbols if s in ref
        ],
    }
    return DictFiles({PICKS_FILE: json.dumps(body), PLAN_FILE: json.dumps(plan)})


def mirror_settings(tmp: Path, mode: str = "paper") -> Settings:
    return settings_for(
        tmp,
        mode=mode,
        usd100__strategy="mirror_trial15",
        big_paper__state_dir=str(tmp / "big-paper"),
        state__shared_dir=str(tmp / "shared"),
    )


def account(s: Settings, held: dict[str, Decimal] | None = None, cash: str = "100") -> SimBroker:
    """A sim account at the 9:45 am prices, holding ``held`` (symbol to shares)."""
    positions = {
        sym: Lot(qty=q, avg_cost=PRICES[sym], entry_session=DECIDE_DAY)
        for sym, q in (held or {}).items()
    }
    b = SimBroker(
        Portfolio(cash=D(cash), positions=positions),
        s.costs,
        Universe.assume_all_fractionable(s.watchlist, "SPY"),
    )
    b.begin_session(EXEC_DAY, PRICES)
    store = StateStore(s.state.dir)
    state = store.load()
    state.positions_meta = {
        sym: PositionMeta(entry_session=DECIDE_DAY, entry_price=PRICES[sym], entry_atr=D(0))
        for sym in held or {}
    }
    state.high_water_mark = b.account().equity
    state.last_close_equity = b.account().equity
    store.save(state)
    return b


def run_execute(
    s: Settings,
    broker: SimBroker,
    files: DictFiles | None,
    env: dict[str, str] = PAPER,
    now: datetime = EXEC_AT,
) -> tuple[jobs.JobResult, alerts.RecordingNotifier]:
    rec = alerts.RecordingNotifier()
    rt = jobs.Runtime(
        settings=s,
        secrets=Secrets.from_env(env),
        notifier=rec,
        now=lambda: now,
        broker=broker,
        use_external=False,
        sleep=lambda _s: None,
        mirror_files=files,
    )
    return jobs.execute(rt), rec


def values(b: SimBroker) -> dict[str, Decimal]:
    return {k: p.qty * PRICES[k] for k, p in b.positions().items() if p.qty > 0}


# ---- the shipped config ----------------------------------------------------------------------


def test_the_shipped_config_mirrors_trial_15_on_paper() -> None:
    s = load_settings(ROOT / "config" / "default.yaml")
    assert s.usd100.mirror and not s.stop.usd100 and s.mode == "paper"
    assert s.usd100.picks_path == "big-paper/trial15/picks"
    assert s.usd100.branch == "stoa-state" and s.usd100.state_branch_dir == Path("state-branch")
    # trial 15 and the big account are untouched
    assert s.big_paper.strategy == "trial15" and not s.stop.big_paper and s.stop.shadow


# ---- sells, then buys, at equal weight with 2% in cash ----------------------------------------


def test_sells_yesterdays_then_buys_todays_5_at_equal_weight_with_2pct_cash(
    tmp_path: Path,
) -> None:
    s = mirror_settings(tmp_path)
    broker = account(s, {"AAPL": D("0.1"), "KO": D("0.1")}, cash="79.50")
    files = branch()
    res, rec = run_execute(s, broker, files)
    assert (res.code, res.status) == (0, "executed")
    assert files.refreshed == 1  # a fresh copy right before reading
    sides = [f.side for f in broker.fills]
    assert sides == [Side.SELL, Side.SELL] + [Side.BUY] * 5  # sells first
    assert {f.symbol for f in broker.fills if f.side is Side.SELL} == {"AAPL", "KO"}
    held = values(broker)
    assert set(held) == set(PICKS)
    eq = broker.account().equity
    for v in held.values():
        assert D("19.40") <= v <= D("19.60")  # (100 x 0.98) / 5, less slippage
    assert broker.account().cash >= eq * D("0.02") - D("0.02")
    assert sum(held.values()) <= eq * D("0.98")
    orders = [o for o in broker.orders.values() if o.side is Side.BUY]
    assert all(o.symbol != "SPY" for o in orders)
    # one alert for the run, no taps
    assert [a.title for a in rec.alerts] == [f"Trial 15 mirror {EXEC_DAY}"]
    state = StateStore(s.state.dir).load()
    assert EXEC_DAY in state.executed_sessions and not state.approval_required
    assert {t.symbol for t in state.recent_trades if t.exit_session == EXEC_DAY} == {"AAPL", "KO"}
    days = mirror.read_days(s.state.dir)
    assert days[-1]["held_after"] == sorted(PICKS) and days[-1]["reason"] is None


def test_target_is_98_percent_of_equity_over_5() -> None:
    assert mirror.target_notional(D("100")) == D("19.60")
    assert mirror.target_notional(D("101.27")) == D("19.84")


# ---- a pick held again ends at the same weight ------------------------------------------------


def test_a_repeated_pick_is_resized_to_equal_weight_not_sold_and_bought_back(
    tmp_path: Path,
) -> None:
    s = mirror_settings(tmp_path)
    # MSFT is worth $30 (too much), NVDA $10 (too little); both are picked again.
    broker = account(s, {"MSFT": D("0.1"), "NVDA": D("0.05")}, cash="60")
    res, _ = run_execute(s, broker, branch())
    assert res.status == "executed"
    by_sym: dict[str, set[Side]] = {}
    for f in broker.fills:
        by_sym.setdefault(f.symbol, set()).add(f.side)
    assert by_sym["MSFT"] == {Side.SELL} and by_sym["NVDA"] == {Side.BUY}  # never both
    held = values(broker)
    assert set(held) == set(PICKS)
    for v in held.values():
        assert abs(v - D("19.60")) < D("0.25"), held
    msft = broker.positions()["MSFT"].qty
    assert D("0.06") < msft < D("0.07")  # trimmed, not sold out


def test_a_held_pick_within_a_dollar_is_left_alone() -> None:
    picks = mirror.Picks(
        EXEC_DAY,
        tuple(mirror.Pick(i + 1, s, 0.5, "", PRICES[s]) for i, s in enumerate(PICKS)),
        "test",
    )
    from stoa.broker.base import BrokerPosition

    held = {
        "MSFT": BrokerPosition(
            symbol="MSFT", qty=D("0.065"), avg_entry_price=D(300), current_price=D(300)
        )
    }  # $19.50, target $19.60
    orders, _ = mirror.plan_orders(picks, held, D("100"), PRICES, {}, "SPY")
    assert "MSFT" not in {o.symbol for o in orders}
    assert len([o for o in orders if o.side is Side.BUY]) == 4


# ---- no picks: buy nothing, keep what is held, say why ----------------------------------------


def test_missing_picks_buy_nothing_keep_holdings_and_say_why(tmp_path: Path) -> None:
    s = mirror_settings(tmp_path)
    broker = account(s, {"AAPL": D("0.1")}, cash="85")
    res, rec = run_execute(s, broker, DictFiles({}))
    assert res.status == "no_picks" and res.code == 0
    assert broker.fills == [] and broker.positions()["AAPL"].qty == D("0.1")
    assert len(rec.alerts) == 1  # one alert
    a = rec.alerts[0]
    assert a.title == f"No trial 15 mirror orders {EXEC_DAY}" and a.priority == alerts.HIGH
    assert PICKS_FILE in a.message and "is missing" in a.message
    day = mirror.read_days(s.state.dir)[-1]
    assert PICKS_FILE in day["reason"] and day["held_after"] == ["AAPL"]
    records = [
        json.loads(x)
        for f in (s.state.dir / "journal").glob("*.jsonl")
        for x in f.read_text(encoding="utf-8").splitlines()
    ]
    ex = [r for r in records if r["job"] == "execute"][-1]["data"]
    assert ex["reason"] and "is missing" in ex["reason"] and ex["fills"] == []


def test_picks_for_another_day_are_not_used(tmp_path: Path) -> None:
    s = mirror_settings(tmp_path)
    broker = account(s, {"AAPL": D("0.1")}, cash="85")
    stale = branch(session=DECIDE_DAY)
    files = DictFiles({PICKS_FILE: stale.files[next(iter(stale.files))]})
    res, rec = run_execute(s, broker, files)
    assert res.status == "no_picks" and broker.fills == []
    assert f"is for '{DECIDE_DAY}', not {EXEC_DAY}" in rec.alerts[0].message


def test_a_pick_with_no_9am_price_is_not_bought(tmp_path: Path) -> None:
    s = mirror_settings(tmp_path)
    broker = account(s)
    refs = {k: v for k, v in PRICES.items() if k != "XOM"}
    res, rec = run_execute(s, broker, branch(refs=refs))
    assert res.status == "executed"
    assert set(values(broker)) == set(PICKS) - {"XOM"}
    assert "XOM not bought: no 9:00 am price" in rec.alerts[0].message


def test_the_price_check_against_9am_still_runs(tmp_path: Path) -> None:
    s = mirror_settings(tmp_path)
    broker = account(s)
    refs = {**PRICES, "JPM": D("140")}  # 9:00 am 140, 9:45 am 160: up 14%
    run_execute(s, broker, branch(refs=refs))
    assert "JPM" not in values(broker)
    day = mirror.read_days(s.state.dir)[-1]
    assert [(r["symbol"], r["reason"]) for r in day["rejected"]] == [("JPM", "price_sanity")]


# ---- the kill switch ------------------------------------------------------------------------


def test_the_kill_switch_sells_everything_and_buys_nothing(tmp_path: Path) -> None:
    s = mirror_settings(tmp_path)
    broker = account(s, {"AAPL": D("0.1"), "MSFT": D("0.05")}, cash="70")
    store = StateStore(s.state.dir)
    state = store.load()
    state.high_water_mark = D("200")  # equity 100 is 50% below the high
    store.save(state)
    res, rec = run_execute(s, broker, branch())
    assert res.status == "executed"
    assert values(broker) == {} and all(f.side is Side.SELL for f in broker.fills)
    assert StateStore(s.state.dir).load().kill_switch.active
    titles = [a.title for a in rec.alerts]
    assert any("KILL SWITCH" in t.upper() for t in titles)


def test_an_active_kill_switch_flattens_even_with_no_picks(tmp_path: Path) -> None:
    s = mirror_settings(tmp_path)
    broker = account(s, {"AAPL": D("0.1")}, cash="85")
    store = StateStore(s.state.dir)
    state = store.load()
    state.kill_switch = KillSwitch(active=True, tripped_session=DECIDE_DAY, reason="test")
    store.save(state)
    run_execute(s, broker, DictFiles({}))
    assert values(broker) == {}
    # and nothing is bought the next day either, picks or not
    assert not [f for f in broker.fills if f.side is Side.BUY]


# ---- never SPY ------------------------------------------------------------------------------


def test_never_buys_spy_even_if_it_is_picked(tmp_path: Path) -> None:
    s = mirror_settings(tmp_path)
    broker = account(s)
    run_execute(s, broker, branch(symbols=["SPY", *PICKS]))
    assert "SPY" not in values(broker) and set(values(broker)) == set(PICKS)
    assert not [o for o in broker.orders.values() if o.symbol == "SPY"]
    day = mirror.read_days(s.state.dir)[-1]
    assert "SPY" not in [p["symbol"] for p in day["picks"]]
    assert any("SPY is never bought" in n for n in day["notes"])


def test_a_plan_with_a_spy_buy_is_refused() -> None:
    o = PlannedOrder(
        symbol="SPY", side=Side.BUY, notional=D("19.60"), ref_price=D(400), reason="trial15_pick"
    )
    plan = make_plan(
        Decision(orders=(o,)),
        strategy=mirror.STRATEGY,
        mode="paper",
        session=DECIDE_DAY,
        execute_session=EXEC_DAY,
        created_at=EXEC_AT,
        equity=D(100),
    )
    with pytest.raises(ValueError, match="never buys SPY"):
        mirror.assert_no_benchmark(plan, "SPY")


# ---- paper only -----------------------------------------------------------------------------


def test_mirror_mode_refuses_live_and_says_why(tmp_path: Path) -> None:
    s = mirror_settings(tmp_path, mode="live")
    broker = account(s)
    res, rec = run_execute(s, broker, branch(), env={**LIVE, **PAPER})
    assert (res.code, res.status) == (jobs.EXIT_REFUSED, "refused")
    assert broker.fills == []
    assert "paper only" in rec.alerts[0].message and "mirror" in rec.alerts[0].message
    with pytest.raises(jobs.Refusal, match="paper only"):
        jobs.check_mode(s, Secrets.from_env({**LIVE, **PAPER}))
    with pytest.raises(mirror.LiveRefused):
        mirror.check_paper(s)


def test_rules_mode_in_live_is_not_affected(tmp_path: Path) -> None:
    s = settings_for(tmp_path, mode="live")
    mirror.check_paper(s)  # no refusal from the mirror for the v1 rules


# ---- the evening, the morning sweep and the override ------------------------------------------


def test_decide_in_mirror_mode_plans_nothing_and_says_what_comes(
    tmp_path: Path, bars: Bars
) -> None:
    from tests.test_jobs import runtime, sim

    s = mirror_settings(tmp_path)
    broker = sim(s, bars)
    rt, rec = runtime(s, bars, PAPER, DECIDE_AT, broker)
    res = jobs.decide(rt)
    assert res.status == "decided"
    state = StateStore(s.state.dir).load()
    assert state.pending_plan is None and state.track.last_strategy == mirror.STRATEGY
    assert "Tomorrow at 9:45 am: today's 5 are sold" in rec.alerts[-1].message
    assert "Trial 15 mirror: no run yet" in rec.alerts[-1].message
    data = dashboard.collect(s, tmp_path)
    d = data["decisions"][0]
    assert d["orders"] == [] and "evening plans nothing" in d["note"]


def test_the_9am_sweep_is_quiet_in_mirror_mode(tmp_path: Path, bars: Bars) -> None:
    from tests.test_jobs import runtime, sim

    s = mirror_settings(tmp_path)
    broker = sim(s, bars)
    jobs.decide(runtime(s, bars, PAPER, DECIDE_AT, broker)[0])
    rt, rec = runtime(s, bars, PAPER, datetime(2021, 10, 13, 13, 0, tzinfo=UTC), broker)
    assert jobs.sweep_job(rt).status == "mirror" and rec.alerts == []
    rt, _ = runtime(s, bars, PAPER, datetime(2021, 10, 13, 14, 30, tzinfo=UTC), broker)
    assert jobs.manual_override(rt).status == "refused"


# ---- the record: registry and dashboard -------------------------------------------------------


def test_registered_once_as_a_case_study_not_a_trial(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry.jsonl")
    assert mirror.register(reg) and not mirror.register(reg)
    (e,) = reg.load()
    assert (e.kind, e.name, e.metrics) == ("case_study", mirror.NAME, {})
    assert reg.trials() == []  # never counted in the trial penalty
    assert "no pass or fail" in e.params["kind"]


def test_the_dashboard_shows_picks_fills_reasons_and_no_tap_line(tmp_path: Path) -> None:
    s = mirror_settings(tmp_path)
    run_execute(s, account(s, {"AAPL": D("0.1")}, cash="85"), DictFiles({}))
    page = dashboard.render(dashboard.collect(s, tmp_path))
    assert "$100 account: trial 15 mirror" in page
    assert "buys need a tap" not in page
    assert "No orders:" in page and "is missing" in page
    s2 = mirror_settings(tmp_path / "b")
    run_execute(s2, account(s2), branch())
    page = dashboard.render(dashboard.collect(s2, tmp_path))
    assert "Picks: MSFT, NVDA, JPM, XOM, QQQ." in page and "Bought MSFT" in page


def test_graded_uses_trial_15_grades_for_the_picks_held() -> None:
    days = [
        {"session": "2026-11-04", "picks": [{"symbol": "A"}, {"symbol": "B"}], "held_after": ["A"]}
    ]
    big = [
        {
            "session": "2026-11-04",
            "picks": [
                {"symbol": "A", "net": 0.01, "hit": True},
                {"symbol": "B", "net": -0.02, "hit": False},
            ],
        }
    ]
    (g,) = mirror.graded(days, big)
    assert [p["symbol"] for p in g["picks"]] == ["A"]


# ---- the fresh copy of the state branch -------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def test_branch_files_read_a_fresh_copy_without_touching_the_checkout(tmp_path: Path) -> None:
    remote, writer, runner = tmp_path / "remote.git", tmp_path / "w", tmp_path / "r"
    _git(tmp_path, "init", "--bare", "-b", "stoa-state", str(remote))
    _git(tmp_path, "clone", str(remote), str(writer))
    _git(writer, "checkout", "-b", "stoa-state")
    (writer / "README.md").write_text("state\n", encoding="utf-8")
    _git(writer, "add", ".")
    _git(writer, "commit", "-m", "start")
    _git(writer, "push", "origin", "stoa-state")
    _git(tmp_path, "clone", "--branch", "stoa-state", str(remote), str(runner))
    # The picks land after the run checked the branch out, as at 9:10 am.
    path = "big-paper/trial15/picks/2026-11-04.json"
    (writer / path).parent.mkdir(parents=True)
    (writer / path).write_text('{"session": "2026-11-04", "picks": []}\n', encoding="utf-8")
    _git(writer, "add", ".")
    _git(writer, "commit", "-m", "picks")
    _git(writer, "push", "origin", "stoa-state")
    files = mirror.BranchFiles(runner, "stoa-state")
    assert not (runner / path).exists()
    assert files.refresh().startswith("a fresh copy of stoa-state")
    text = files.read(path)
    assert text is not None and "2026-11-04" in text
    assert files.listdir("big-paper/trial15/picks") == ["2026-11-04.json"]
    assert not (runner / path).exists()  # the checkout is untouched


def test_branch_files_fall_back_to_the_checkout_when_fetch_fails(tmp_path: Path) -> None:
    (tmp_path / "big-paper").mkdir()
    (tmp_path / "big-paper" / "x.json").write_text("{}", encoding="utf-8")
    files = mirror.BranchFiles(tmp_path, "stoa-state")
    note = files.refresh()
    assert "fetch failed" in note and files.read("big-paper/x.json") == "{}"
    assert files.listdir("big-paper") == ["x.json"]
