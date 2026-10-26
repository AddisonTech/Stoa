from __future__ import annotations

import re
import shutil
import subprocess

import pytest
import yaml

from tests.conftest import ROOT

DAILY = ROOT / ".github" / "workflows" / "daily.yml"


def cron_slots() -> list[tuple[str, str]]:
    """Each schedule entry with the job its comment names (DECIDE, SWEEP or EXECUTE)."""
    found = re.findall(r'- cron: "([^"]+)" # (DECIDE|SWEEP|EXECUTE)', DAILY.read_text("utf-8"))
    return [(cron, job.lower()) for cron, job in found]


def test_every_cron_slot_is_labelled_and_paired() -> None:
    daily = yaml.safe_load(DAILY.read_text(encoding="utf-8"))
    crons = [entry["cron"] for entry in daily[True]["schedule"]]  # "on" parses as True
    slots = cron_slots()
    assert sorted(c for c, _ in slots) == sorted(crons)
    for job in ("decide", "sweep", "execute"):
        assert sum(1 for _, j in slots if j == job) == 2  # the EDT and the EST slot


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(("cron", "job"), cron_slots())
def test_each_cron_slot_runs_its_own_job(cron: str, job: str) -> None:
    daily = yaml.safe_load(DAILY.read_text(encoding="utf-8"))
    steps = daily["jobs"]["run"]["steps"]
    pick = next(st for st in steps if st.get("id") == "pick")["run"]
    # Inputs are set in the script and it goes in on stdin as bytes: on Windows, quotes
    # in a command line argument do not survive the trip to bash, and text mode adds \r.
    script = f"JOB_INPUT=''\nSCHEDULE='{cron}'\n" + pick.replace('>> "$GITHUB_OUTPUT"', "")
    out = subprocess.run(
        ["bash"], input=script.encode(), capture_output=True, check=True
    ).stdout.decode()
    assert f"job={job}" in out.split()
