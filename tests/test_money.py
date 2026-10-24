from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stoa import money
from stoa.config import MoneyConfig
from stoa.state import Inception, KillSwitch, State

D = Decimal
CFG = MoneyConfig()


def started(equity: str = "100") -> State:
    return State(
        high_water_mark=D(equity),
        inception=Inception(session=date(2026, 7, 1), equity=D(equity), benchmark_close=D(500)),
    )


def test_milestone_alert_fires_once() -> None:
    s = started()
    assert money.check_milestone(s, D("199.99"), CFG) is None
    a = money.check_milestone(s, D("200"), CFG)
    assert a is not None and a.title == "Withdraw $100 now" and a.priority == 4
    assert "confirm-withdrawal" in a.message
    assert money.check_milestone(s, D("250"), CFG) is None
    assert s.milestone_alerted


def test_milestone_ignores_outside_deposits() -> None:
    s = started()
    s.net_flows = D(60)
    assert money.check_milestone(s, D(210), CFG) is None
    assert money.check_milestone(s, D(260), CFG) is not None


def test_milestone_off_when_account_size_differs() -> None:
    s = started("100000")
    assert not money.plan_active(s, CFG)
    assert money.check_milestone(s, D("250000"), CFG) is None
    assert money.deposit_mismatch_note(D(100000), CFG) is not None
    assert money.deposit_mismatch_note(D(100), CFG) is None


def test_confirm_withdrawal_rebases_high_water_mark() -> None:
    s = started()
    s.high_water_mark = D(210)
    s.milestone_alerted = True
    old = money.confirm_withdrawal(s, D(108), CFG.deposit)
    assert old == D(210)
    assert s.principal_withdrawn and s.high_water_mark == D(108)
    with pytest.raises(ValueError):
        money.confirm_withdrawal(s, D(108), CFG.deposit)


def test_confirmed_withdrawal_not_double_counted_from_activities() -> None:
    s = started()
    s.milestone_alerted = True
    money.confirm_withdrawal(s, D(105), CFG.deposit)
    assert s.principal_flow_pending
    flows = money.parse_flows(
        [{"id": "w1", "activity_type": "CSW", "net_amount": "-100", "date": "2026-09-02"}]
    )
    money.apply_flows(s, flows, date(2026, 7, 1))
    assert s.high_water_mark == D(105)
    assert not s.principal_flow_pending
    assert s.net_flows == D(-100)


def test_detected_withdrawal_moves_hwm_so_kill_switch_cannot_trip() -> None:
    s = started()
    s.high_water_mark = D(205)
    s.milestone_alerted = True
    flows = money.parse_flows(
        [{"id": "w1", "activity_type": "CSW", "net_amount": "-100", "date": "2026-09-02"}]
    )
    applied = money.apply_flows(s, flows, date(2026, 7, 1))
    assert len(applied) == 1 and s.high_water_mark == D(105)
    assert s.withdrawn_since_milestone == D(100)
    assert money.apply_flows(s, flows, date(2026, 7, 1)) == []
    alert = money.flows_alert(applied, s.principal_withdrawn)
    assert alert is not None and "confirm-withdrawal" in alert.message
    money.confirm_withdrawal(s, D(105), CFG.deposit)
    assert not s.principal_flow_pending
    assert s.high_water_mark == D(105)


def test_flows_before_inception_are_ignored() -> None:
    s = started()
    flows = money.parse_flows(
        [{"id": "d0", "activity_type": "CSD", "net_amount": "100", "date": "2026-06-20"}]
    )
    assert money.apply_flows(s, flows, date(2026, 7, 1)) == []
    assert s.high_water_mark == D(100) and s.net_flows == 0
    assert "d0" in s.processed_activity_ids


def test_deposit_raises_hwm() -> None:
    s = started()
    flows = money.parse_flows(
        [{"id": "d1", "activity_type": "CSD", "net_amount": "50", "date": "2026-08-01"}]
    )
    money.apply_flows(s, flows, date(2026, 7, 1))
    assert s.high_water_mark == D(150)


@pytest.mark.parametrize(
    ("realized", "rate", "expected"),
    [
        ("100", "0.25", "25.00"),
        ("0.05", "0.25", "0.02"),
        ("-40", "0.25", "0"),
        ("33.33", "0.30", "10.00"),
    ],
)
def test_tax_reserve(realized: str, rate: str, expected: str) -> None:
    assert money.tax_reserve(D(realized), D(rate)) == D(expected)


def test_reset_kill_switch() -> None:
    s = started()
    s.kill_switch = KillSwitch(active=True, reason="x")
    s.high_water_mark = D(200)
    assert money.reset_kill_switch(s, D(140)) == D(200)
    assert not s.kill_switch.active and s.high_water_mark == D(140)
    with pytest.raises(ValueError):
        money.reset_kill_switch(s, D(140))


def test_realized_tracked_by_year() -> None:
    s = State()
    s.add_realized(2026, D("12.5"))
    s.add_realized(2026, D("-2.5"))
    s.add_realized(2027, D("1"))
    assert s.realized_for_year(2026) == D(10) and s.realized_for_year(2027) == D(1)


def test_trans_mirroring_csw_counted_once() -> None:
    acts = [
        {"id": "t1", "activity_type": "TRANS", "net_amount": "-100", "date": "2026-09-02"},
        {"id": "w1", "activity_type": "CSW", "net_amount": "-100", "date": "2026-09-02"},
        {"id": "t2", "activity_type": "TRANS", "net_amount": "25", "date": "2026-09-03"},
    ]
    flows = money.parse_flows(acts)
    assert [f.activity_id for f in flows] == ["w1", "t2"]


def test_inception_day_flow_is_part_of_start() -> None:
    s = started()
    flows = money.parse_flows(
        [{"id": "d0", "activity_type": "CSD", "net_amount": "100", "date": "2026-07-01"}]
    )
    assert money.apply_flows(s, flows, date(2026, 7, 1)) == []


def test_signal_mode_withdraw_shadow() -> None:
    s = started()
    s.sim_cash = D(150)
    s.last_close_equity = D(205)
    assert money.withdraw_shadow(s, D(100)) == D(105)
    assert s.sim_cash == D(50) and s.net_flows == D(-100)
    with pytest.raises(ValueError, match="below"):
        money.withdraw_shadow(s, D(100))
