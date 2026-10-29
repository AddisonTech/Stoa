"""Trial 14 (stoa/newsbrain/trial14.py): trial 12 exactly, idle money in cash, registered
before it can run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stoa.experiments import Registry
from stoa.newsbrain import spycore as sc
from stoa.newsbrain import trial14
from tests.conftest import ROOT, make_settings


def test_trial_14_is_registered_before_it_can_run(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry.jsonl")
    with pytest.raises(RuntimeError, match="not pre-registered"):
        trial14.run_trial14(make_settings(), pd.DataFrame(), lambda d, s: {}, reg, tmp_path)
    trial14.preregister(reg)
    trial14.preregister(reg)  # once only
    entries = reg.load()
    assert len(entries) == 1 and entries[0].kind == "preregistration"
    assert entries[0].config_hash == trial14.spec_hash() == "f385bc9d5040ef7b"
    assert trial14.preregistered(reg)
    spec = entries[0].params
    assert sc.spec_hash() in spec["same_as_trial_12"] and "cash" in spec["idle_money"]
    assert "brain's picks only" in spec["reason"] and "in-sample" in spec["in_sample"]


def test_decisions_md_records_trial_14() -> None:
    text = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    assert "## Trial 14: trial 12 with idle money in cash" in text
    assert trial14.spec_hash() in text and "## The big paper account runs trial 14" in text
