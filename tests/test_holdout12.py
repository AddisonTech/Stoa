from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace as Ns

import pandas as pd
import pytest

from stoa.newsbrain import holdout12 as ho
from stoa.newsbrain import sealed
from tests.conftest import ROOT


def test_the_wall_opens_only_for_trial_12s_key_and_only_through_the_holdout(tmp_path: Path) -> None:
    marker = tmp_path / "spent.json"
    with pytest.raises(sealed.HoldoutBreach):
        sealed.check(date(2026, 8, 3), "a read")
    with pytest.raises(sealed.HoldoutBreach), sealed.opened("someone else", marker):
        pass
    assert not marker.exists()
    with sealed.opened(sealed.HOLDOUT_KEY, marker):
        assert sealed.is_open()
        sealed.check(date(2026, 8, 3), "a read")
        sealed.check(date(2026, 10, 23), "a read")
        with pytest.raises(sealed.HoldoutBreach):
            sealed.check(date(2026, 10, 26), "after the holdout")
        assert sealed.month_allowed("2026-10") and not sealed.month_allowed("2026-11")
        df = pd.DataFrame({"t": pd.to_datetime(["2026-10-23 20:00", "2026-10-24 05:00"], utc=True)})
        assert len(sealed.before_holdout(df, "t")) == 1
    assert not sealed.is_open()
    with pytest.raises(sealed.HoldoutBreach):
        sealed.check(date(2026, 8, 3), "a read after the run")
    assert not sealed.month_allowed("2026-07")


def test_one_run_ever(tmp_path: Path) -> None:
    marker = tmp_path / "spent.json"
    with pytest.raises(RuntimeError), sealed.opened(sealed.HOLDOUT_KEY, marker):
        raise RuntimeError("stopped part way")
    assert not sealed.is_open()
    with pytest.raises(sealed.HoldoutBreach), sealed.opened(sealed.HOLDOUT_KEY, marker):
        pass  # a second run without the owner's word
    with sealed.opened(sealed.HOLDOUT_KEY, marker, owner_rerun=True):  # the owner's word
        pass
    sealed.mark_spent({"verdict": "FAIL"}, marker)
    with (
        pytest.raises(sealed.HoldoutBreach),
        sealed.opened(sealed.HOLDOUT_KEY, marker, owner_rerun=True),
    ):
        pass  # finished: spent for good


def m(ret: float, sharpe: float, mdd: float) -> Ns:
    return Ns(total_return=ret, sharpe=sharpe, max_drawdown=mdd)


def test_expected_trades_and_the_holdout_bar() -> None:
    assert ho.expected_trades() == pytest.approx(193 / 875 * 81)
    spy = m(0.10, 1.5, 0.05)
    assert ho.grade(m(0.12, 1.6, 0.04), spy, 15)[0] == "PASS"
    assert ho.grade(m(0.12, 1.6, 0.04), spy, 15)[1][-1]["passed"]
    assert ho.grade(m(0.09, 1.6, 0.04), spy, 15)[0] == "FAIL"  # return under SPY's
    assert ho.grade(m(0.12, 1.4, 0.04), spy, 15)[0] == "FAIL"  # Sharpe under SPY's
    assert ho.grade(m(0.12, 1.6, 0.06), spy, 15)[0] == "FAIL"  # deeper drawdown
    assert ho.grade(m(0.12, 1.6, 0.04), spy, 4)[0] == "STARVED"  # under 17.9 / 4
    assert ho.grade(m(0.12, 1.6, 0.04), spy, 5)[0] == "PASS"
    spec = ho.spec()
    assert spec["expected_trades"] == 17.9 and "too short" in spec["reported_not_graded"]


def test_decisions_md_records_the_owners_decision_and_the_spec() -> None:
    text = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    sec = text[text.index("## Trial 12 on the holdout") :]
    assert "Owner opens the holdout (2026-07-01 to 2026-10-23) once" in sec
    assert ho.spec_hash() in sec and "17.9" in sec
