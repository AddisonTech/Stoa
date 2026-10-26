from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import yaml

from stoa import alerts, clock
from stoa import triggers as t
from tests.conftest import ROOT
from tests.test_schedule import cron_slots

WF = ROOT / ".github" / "workflows"
MON = date(2026, 11, 2)  # a Monday, after CHECKS_FROM


def ny(day: date, hhmm: str, seconds: int = 0) -> datetime:
    return clock.at_ny(day, hhmm) + timedelta(seconds=seconds)


def run(workflow: str, title: str, at: datetime) -> t.Run:
    return t.Run(workflow, title, at.astimezone(UTC))


# ---- the slots ------------------------------------------------------------------------


def test_slots_match_the_dell_tasks() -> None:
    text = (ROOT / "scripts" / "install-dispatch.ps1").read_text(encoding="utf-8")
    dell = dict(re.findall(r'Job = "(\w+)"; At = "(\d+:\d\d[ap]m)"', text))
    for job, hhmm, _ in t.SLOTS:
        h, m = map(int, hhmm.split(":"))
        assert dell[job] == f"{h % 12 or 12}:{m:02d}{'am' if h < 12 else 'pm'}"
    assert set(dell) == {job for job, _, _ in t.SLOTS}


def test_daily_crons_match_the_labelled_backups() -> None:
    assert dict(cron_slots()) == t.DAILY_CRONS


@pytest.mark.parametrize(
    ("job", "started", "expected"),
    [
        ("sweep", ny(MON + timedelta(days=1), "08:57", 3), ("analyst", MON, "19:45")),
        ("sweep", ny(MON, "08:57", 3), ("analyst", MON - timedelta(days=3), "19:45")),  # Friday
        ("close", ny(MON, "09:39", 2), ("sweep", MON, "08:57")),
        ("execute", ny(MON, "09:45", 2), ("close", MON, "09:39")),
        ("execute", ny(MON, "09:50", 40), ("close", MON, "09:39")),  # the GitHub cron backup
        ("decide", ny(MON, "17:15", 1), ("execute", MON, "09:45")),
        ("analyst", ny(MON, "19:45", 1), ("decide", MON, "17:15")),
    ],
)
def test_each_run_checks_the_slot_before_its_own(
    job: str, started: datetime, expected: tuple[str, date, str]
) -> None:
    slot = t.slot_to_check(job, started)
    assert slot is not None
    assert (slot.job, slot.at.date(), slot.at.strftime("%H:%M")) == expected


def test_no_check_for_a_run_far_from_its_slot_or_before_the_switch() -> None:
    assert t.slot_to_check("decide", ny(MON, "14:00")) is None  # a hand run at 2 pm
    assert t.slot_to_check("sweep", ny(date(2026, 10, 26), "08:57")) is None
    # Tonight's DECIDE is the first run to check anything: the 9:45 EXECUTE predates
    # the switch, so it is not checked; the 7:45 pm analyst checks the 5:15 DECIDE.
    assert t.slot_to_check("decide", ny(date(2026, 10, 26), "17:15")) is None
    slot = t.slot_to_check("analyst", ny(date(2026, 10, 26), "19:45"))
    assert slot is not None and slot.job == "decide"


# ---- run names ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("workflow", "title", "source", "job"),
    [
        ("daily.yml", "sweep via cron-job.org", "cron-job.org", "sweep"),
        ("daily.yml", "execute via dell", "dell", "execute"),
        ("daily.yml", "alert-test via manual", "manual", "alert-test"),
        ("daily.yml", "schedule 5 13 * * 1-5", "schedule", "sweep"),
        ("daily.yml", "schedule 20 22 * * 1-5", "schedule", "decide"),
        ("daily.yml", "daily", "unknown", "daily"),  # a run from before run names
        ("close.yml", "close via cron-job.org", "cron-job.org", "close"),
        ("close.yml", "close schedule 41 13 * * 1-5", "schedule", "close"),
        ("analyst.yml", "analyst via dell", "dell", "analyst"),
    ],
)
def test_source_and_job_from_the_run_name(workflow: str, title: str, source: str, job: str) -> None:
    r = run(workflow, title, ny(MON, "09:00"))
    assert t.source_of(r) == source and t.job_of(r) == job


def test_workflows_name_runs_the_way_the_check_reads_them() -> None:
    daily = yaml.safe_load((WF / "daily.yml").read_text(encoding="utf-8"))["run-name"]
    assert "format('schedule {0}', github.event.schedule)" in daily
    assert "format('{0} via {1}', inputs.job, inputs.source || 'manual')" in daily
    for name in ("close", "analyst"):
        rn = yaml.safe_load((WF / f"{name}.yml").read_text(encoding="utf-8"))["run-name"]
        assert f"format('{name} schedule {{0}}', github.event.schedule)" in rn
        assert f"format('{name} via {{0}}', inputs.source || 'manual')" in rn


def test_the_dell_marks_its_runs() -> None:
    text = (ROOT / "scripts" / "dispatch.ps1").read_text(encoding="utf-8")
    calls = re.findall(r"gh workflow run .*", text)
    assert len(calls) == 3 and all("-f source=dell" in c for c in calls)


def test_the_check_step_cannot_hold_up_or_fail_a_job() -> None:
    for name, job_key in (("daily", "run"), ("close", "close"), ("analyst", "shadow")):
        wf = yaml.safe_load((WF / f"{name}.yml").read_text(encoding="utf-8"))
        assert wf["permissions"]["actions"] == "read"
        steps = wf["jobs"][job_key]["steps"]
        idx = [i for i, st in enumerate(steps) if st.get("name") == "Check the last trigger"]
        assert len(idx) == 1
        step = steps[idx[0]]
        assert step["continue-on-error"] is True
        assert "cancelled()" in step["if"]
        assert step["run"].startswith("stoa triggers check --job ")
        runs_before = " ".join(str(st.get("run", "")) for st in steps[: idx[0]])
        assert "stoa" in runs_before  # the job itself runs first
        assert "source" in wf[True]["workflow_dispatch"]["inputs"]


# ---- the alert ------------------------------------------------------------------------

SWEEP = t.Slot("sweep", "daily.yml", ny(MON, "08:57"))


def test_no_alert_when_cron_job_org_started_it() -> None:
    runs = [run("daily.yml", "sweep via cron-job.org", ny(MON, "08:57", 4))]
    assert t.missed_alert(SWEEP, runs) is None
    early = [run("daily.yml", "sweep via cron-job.org", ny(MON, "08:56", 40))]
    assert t.missed_alert(SWEEP, early) is None


def test_a_backup_start_is_a_high_alert_naming_the_trigger() -> None:
    runs = [
        run("daily.yml", "sweep via dell", ny(MON, "08:57", 3)),
        run("daily.yml", "execute via cron-job.org", ny(MON, "08:58")),  # another job
    ]
    a = t.missed_alert(SWEEP, runs)
    assert a is not None and a.priority == alerts.HIGH
    assert a.title == "Trigger missed: 8:57 am sweep Mon Nov 2"
    assert "cron-job.org did not start the 8:57 am sweep job" in a.message
    assert "the Dell started it at 8:57 am, so the job ran" in a.message
    assert chr(0x2014) not in a.title + a.message  # no em dash


def test_a_late_primary_counts_as_missed() -> None:
    runs = [run("daily.yml", "sweep via cron-job.org", ny(MON, "09:03"))]
    a = t.missed_alert(SWEEP, runs)
    assert a is not None and a.priority == alerts.URGENT
    assert "The first start came at 9:03 am, from cron-job.org." in a.message


def test_nothing_started_is_urgent() -> None:
    runs = [run("daily.yml", "schedule 5 13 * * 1-5", ny(MON, "09:21"))]
    a = t.missed_alert(SWEEP, runs)
    assert a is not None and a.priority == alerts.URGENT
    assert "Neither did a backup. The first start came at 9:21 am, from GitHub cron." in a.message
    b = t.missed_alert(SWEEP, [])
    assert b is not None and "No run has started since." in b.message


# ---- the check, end to end ------------------------------------------------------------


class FakeGitHub(t.GitHubRuns):
    def __init__(self, started: datetime, runs: list[t.Run]) -> None:
        self._started, self._runs = started, runs
        self.asked: list[tuple[str, datetime]] = []

    def created(self, run_id: str) -> datetime:
        return self._started

    def runs(self, workflow: str, since: datetime) -> list[t.Run]:
        self.asked.append((workflow, since))
        return [r for r in self._runs if r.workflow == workflow and r.created >= since]


def test_check_sends_once_and_skips_a_miss_already_reported() -> None:
    gh = FakeGitHub(ny(MON, "09:39", 2), [run("daily.yml", "sweep via dell", ny(MON, "08:57"))])
    rec = alerts.RecordingNotifier()
    assert t.check("close", "1", gh, rec, lambda since: []) == "8:57 am sweep: missed, alert sent"
    assert rec.titles() == ["Trigger missed: 8:57 am sweep Mon Nov 2"]
    assert gh.asked == [("daily.yml", ny(MON, "08:56"))]
    seen = rec.titles()
    assert (
        t.check("close", "2", gh, rec, lambda since: seen)
        == "8:57 am sweep: missed, already reported"
    )
    assert len(rec.alerts) == 1
    # An unreadable topic sends rather than stays quiet.
    assert t.check("close", "3", gh, rec, lambda since: None).endswith("alert sent")


def test_check_is_quiet_when_on_time_or_nothing_to_check() -> None:
    rec = alerts.RecordingNotifier()
    ok = FakeGitHub(
        ny(MON, "09:45"), [run("close.yml", "close via cron-job.org", ny(MON, "09:39"))]
    )
    assert t.check("execute", "1", ok, rec) == "9:39 am close: started by cron-job.org"
    odd = FakeGitHub(ny(MON, "13:00"), [])
    assert t.check("execute", "1", odd, rec) == "nothing to check"
    assert rec.alerts == []


def test_github_runs_reads_the_api_with_the_run_token() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/actions/runs/42"):
            return httpx.Response(200, json={"created_at": "2026-11-02T13:57:03Z"})
        body = {
            "workflow_runs": [
                {"display_title": "sweep via cron-job.org", "created_at": "2026-11-02T13:57:02Z"}
            ]
        }
        return httpx.Response(200, json=body)

    gh = t.GitHubRuns(
        "AddisonTech/Stoa",
        "ghs_secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert "ghs_secret" not in repr(gh)
    assert gh.created("42") == datetime(2026, 11, 2, 13, 57, 3, tzinfo=UTC)
    runs = gh.runs("daily.yml", ny(MON, "08:56"))
    assert runs == [
        t.Run("daily.yml", "sweep via cron-job.org", datetime(2026, 11, 2, 13, 57, 2, tzinfo=UTC))
    ]
    req = seen[-1]
    assert req.method == "GET" and req.headers["Authorization"] == "Bearer ghs_secret"
    assert req.url.path == "/repos/AddisonTech/Stoa/actions/workflows/daily.yml/runs"
    assert parse_qs(urlparse(str(req.url)).query)["created"] == [">=2026-11-02T13:56:00Z"]
    assert all(r.method == "GET" for r in seen)


def test_github_errors_raise_and_the_cli_swallows_them(monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from stoa.cli import main

    def boom(self: t.GitHubRuns, run_id: str) -> datetime:
        raise RuntimeError("GitHub API /actions/runs/1: HTTP 500")

    monkeypatch.setattr(t.GitHubRuns, "created", boom)
    for k, v in {
        "GITHUB_REPOSITORY": "AddisonTech/Stoa",
        "GITHUB_TOKEN": "x",
        "GITHUB_RUN_ID": "1",
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("STOA_NTFY_TOPIC", raising=False)
    res = CliRunner().invoke(
        main,
        ["--config", str(ROOT / "config" / "default.yaml"), "triggers", "check", "--job", "sweep"],
    )
    assert res.exit_code == 0 and "Trigger check skipped: RuntimeError" in res.output
    for k in ("GITHUB_REPOSITORY", "GITHUB_TOKEN", "GITHUB_RUN_ID"):
        monkeypatch.delenv(k)
    res = CliRunner().invoke(
        main,
        ["--config", str(ROOT / "config" / "default.yaml"), "triggers", "check", "--job", "sweep"],
    )
    assert res.exit_code == 0 and "Not on GitHub Actions" in res.output
