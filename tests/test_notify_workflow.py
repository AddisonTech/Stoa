from __future__ import annotations

import yaml

from tests.conftest import ROOT

NOTIFY = ROOT / ".github" / "workflows" / "notify.yml"


def test_notify_sends_only_to_the_alert_topic() -> None:
    text = NOTIFY.read_text(encoding="utf-8")
    wf = yaml.safe_load(text)
    assert wf["permissions"] == {"contents": "read"}
    assert set(wf[True]["workflow_dispatch"]["inputs"]) == {"title", "message", "priority"}
    secrets = set(__import__("re").findall(r"secrets\.(\w+)", text))
    assert secrets == {"STOA_NTFY_TOPIC", "STOA_NTFY_TOKEN"}  # no broker or model keys
    run = " ".join(st["run"] for st in wf["jobs"]["send"]["steps"])
    assert "${{" not in run  # inputs reach the script through env, never pasted into it
