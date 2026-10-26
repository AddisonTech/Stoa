"""Did the last scheduled job start on time?

cron-job.org is the primary clock: at each slot it starts the job on GitHub by
workflow_dispatch with ``source=cron-job.org``. The Dell's Task Scheduler
(``source=dell``) and GitHub cron are backups. Every run names its trigger in its
run name ("sweep via cron-job.org", "schedule 5 13 * * 1-5"), which the GitHub API
returns as ``display_title``.

At the end of each scheduled job's run, ``check`` looks at the slot before it. If
no run started by cron-job.org was created within GRACE of that slot, it sends one
alert naming the slot and which triggers did or did not start it. A backup start
keeps the job running, so that alert is HIGH; nothing at all within GRACE is URGENT.
It only reads: run listings from GitHub and recent titles on the alert topic, so the
same miss is reported once however many runs check it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from stoa import alerts, clock

log = logging.getLogger(__name__)

PRIMARY = "cron-job.org"
GRACE = timedelta(minutes=5)
EARLY = timedelta(minutes=1)  # a trigger a little before the minute still counts
# A run checks only when it belongs to a slot it started near, so a manual run at an
# odd hour does not dig up old slots.
OWN_SLOT_WINDOW = timedelta(hours=2)
# cron-job.org became the primary clock on this afternoon; earlier slots had no
# source in their run names and are not checked.
CHECKS_FROM = datetime(2026, 10, 26, 17, 0, tzinfo=clock.NY)

# (job, New York time, workflow file), weekdays, in the order of the day.
SLOTS: tuple[tuple[str, str, str], ...] = (
    ("sweep", "08:57", "daily.yml"),
    ("close", "09:39", "close.yml"),
    ("execute", "09:45", "daily.yml"),
    ("decide", "17:15", "daily.yml"),
    ("analyst", "19:45", "analyst.yml"),
)
WORKFLOW_JOBS = {"close.yml": "close", "analyst.yml": "analyst"}
# The daily workflow's backup crons (UTC) and the job each one runs; a test holds
# this to the labels in daily.yml.
DAILY_CRONS = {
    "20 21 * * 1-5": "decide",
    "20 22 * * 1-5": "decide",
    "5 13 * * 1-5": "sweep",
    "5 14 * * 1-5": "sweep",
    "50 13 * * 1-5": "execute",
    "50 14 * * 1-5": "execute",
}
SOURCE_NAMES = {PRIMARY: "cron-job.org", "dell": "the Dell", "schedule": "GitHub cron"}


@dataclass(frozen=True)
class Slot:
    job: str
    workflow: str
    at: datetime  # New York time

    @property
    def label(self) -> str:
        return f"{_clock_time(self.at)} {self.job}"

    @property
    def key(self) -> str:
        return f"{self.label} {self.at:%a %b} {self.at.day}"


@dataclass(frozen=True)
class Run:
    workflow: str  # file name, e.g. daily.yml
    title: str  # the run name (display_title)
    created: datetime


def _clock_time(ts: datetime) -> str:
    ny = ts.astimezone(clock.NY)
    return f"{ny.hour % 12 or 12}:{ny.minute:02d} {'am' if ny.hour < 12 else 'pm'}"


def slots_on(day: date) -> list[Slot]:
    if day.weekday() >= 5:
        return []
    return [Slot(job, wf, clock.at_ny(day, hhmm).astimezone(clock.NY)) for job, hhmm, wf in SLOTS]


def recent_slots(now: datetime, days: int = 7) -> list[Slot]:
    today = now.astimezone(clock.NY).date()
    out: list[Slot] = []
    for back in range(days, -1, -1):
        out.extend(slots_on(today - timedelta(days=back)))
    return out


def own_slot(job: str, started: datetime) -> Slot | None:
    """The slot this run belongs to: its job's latest slot at or before the start."""
    mine = [s for s in recent_slots(started) if s.job == job and s.at <= started + EARLY]
    if not mine or started - mine[-1].at > OWN_SLOT_WINDOW:
        return None
    return mine[-1]


def slot_to_check(job: str, started: datetime) -> Slot | None:
    """The slot just before this run's own slot, if it is one cron-job.org owns."""
    mine = own_slot(job, started)
    if mine is None:
        return None
    before = [s for s in recent_slots(started) if s.at < mine.at]
    if not before or before[-1].at < CHECKS_FROM:
        return None
    return before[-1]


def source_of(run: Run) -> str:
    """cron-job.org, dell, manual, schedule, or unknown (a run from before run names)."""
    title = run.title.strip()
    if title.startswith("schedule ") or " schedule " in title:
        return "schedule"
    found = re.search(r" via (\S+)$", title)
    return found.group(1) if found else "unknown"


def job_of(run: Run) -> str | None:
    if run.workflow in WORKFLOW_JOBS:
        return WORKFLOW_JOBS[run.workflow]
    title = run.title.strip()
    if title.startswith("schedule "):
        return DAILY_CRONS.get(title.removeprefix("schedule ").strip())
    return title.split(" ", 1)[0] or None


def starts(slot: Slot, runs: Iterable[Run]) -> list[Run]:
    """Runs of the slot's job created from EARLY before it to GRACE after it."""
    lo, hi = slot.at - EARLY, slot.at + GRACE
    found = [
        r
        for r in runs
        if r.workflow == slot.workflow and job_of(r) == slot.job and lo <= r.created <= hi
    ]
    return sorted(found, key=lambda r: r.created)


def title_for(slot: Slot) -> str:
    return f"Trigger missed: {slot.key}"


def missed_alert(slot: Slot, runs: Sequence[Run]) -> alerts.Alert | None:
    """The alert for this slot, or None when cron-job.org started it in time."""
    on_time = starts(slot, runs)
    if any(source_of(r) == PRIMARY for r in on_time):
        return None
    day = f"{slot.at:%a %b} {slot.at.day}"
    head = f"cron-job.org did not start the {slot.label} job ({day}) within 5 minutes."
    fix = " Check the job's history in cron-job.org and the GitHub token (it expires 2027-10-26)."
    if on_time:
        first = on_time[0]
        who = SOURCE_NAMES.get(source_of(first), source_of(first))
        body = f"{head} {who} started it at {_clock_time(first.created)}, so the job ran.{fix}"
        return alerts.Alert(title_for(slot), body, priority=alerts.HIGH, tags=("alarm_clock",))
    later = sorted(
        (
            r
            for r in runs
            if r.workflow == slot.workflow and job_of(r) == slot.job and r.created > slot.at + GRACE
        ),
        key=lambda r: r.created,
    )
    if later:
        who = SOURCE_NAMES.get(source_of(later[0]), source_of(later[0]))
        after = f" The first start came at {_clock_time(later[0].created)}, from {who}."
    else:
        after = " No run has started since."
    body = f"{head} Neither did a backup.{after}{fix}"
    return alerts.Alert(title_for(slot), body, priority=alerts.URGENT, tags=("rotating_light",))


# ---- I/O ------------------------------------------------------------------------------


class GitHubRuns:
    """Reads workflow runs with the run's own GITHUB_TOKEN (actions: read)."""

    def __init__(
        self,
        repo: str,
        token: str,
        *,
        api: str = "https://api.github.com",
        client: httpx.Client | None = None,
    ) -> None:
        self.repo = repo
        self.api = api.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self.client = client or httpx.Client(timeout=20.0)

    def __repr__(self) -> str:
        return f"GitHubRuns(repo={self.repo!r}, token=<hidden>)"

    def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        resp = self.client.get(
            f"{self.api}/repos/{self.repo}{path}", params=params, headers=self._headers
        )
        if resp.status_code != 200:
            raise RuntimeError(f"GitHub API {path}: HTTP {resp.status_code}")
        return resp.json()

    def created(self, run_id: str) -> datetime:
        return _ts(self._get(f"/actions/runs/{run_id}")["created_at"])

    def runs(self, workflow: str, since: datetime) -> list[Run]:
        stamp = since.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        data = self._get(
            f"/actions/workflows/{workflow}/runs",
            {"created": f">={stamp}", "per_page": "100"},
        )
        return [
            Run(workflow, str(r.get("display_title") or ""), _ts(r["created_at"]))
            for r in data.get("workflow_runs", [])
        ]


def _ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def already_sent(titles: Iterable[str], slot: Slot) -> bool:
    return title_for(slot) in set(titles)


def check(
    job: str,
    run_id: str,
    github: GitHubRuns,
    notifier: alerts.Notifier,
    sent_titles: Callable[[datetime], list[str] | None] | None = None,
) -> str:
    """Check the slot before this run's own; send at most one alert. Returns what it did.

    ``sent_titles(since)`` returns the titles already on the alert topic since a
    time, or None when the topic cannot be read (then it sends anyway).
    """
    started = github.created(run_id)
    slot = slot_to_check(job, started)
    if slot is None:
        return "nothing to check"
    alert = missed_alert(slot, github.runs(slot.workflow, slot.at - EARLY))
    if alert is None:
        return f"{slot.label}: started by cron-job.org"
    if sent_titles is not None:
        titles = sent_titles(slot.at - EARLY)
        if titles is not None and already_sent(titles, slot):
            return f"{slot.label}: missed, already reported"
    alerts.safe_send(notifier, alert)
    return f"{slot.label}: missed, alert sent"
