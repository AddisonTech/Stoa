from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from stoa import alerts
from stoa.models import PlannedOrder, Side

D = Decimal


def transport(status: int, seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status)

    return httpx.MockTransport(handler)


def notifier(status: int, seen: list[httpx.Request], **kw: object) -> alerts.NtfyNotifier:
    return alerts.NtfyNotifier(
        "https://ntfy.example/",
        "secret-topic-123",
        client=httpx.Client(transport=transport(status, seen)),
        sleep=lambda _s: None,
        **kw,  # type: ignore[arg-type]
    )


def test_ntfy_payload_fields_and_priority() -> None:
    seen: list[httpx.Request] = []
    n = notifier(200, seen, token="tok", default_click="https://app.example")
    alert = alerts.Alert(
        title="BUY NVDA", message="go", priority=alerts.HIGH, tags=("chart",), markdown=True
    )
    assert n.send(alert)
    req = seen[0]
    assert str(req.url) == "https://ntfy.example/"
    assert req.headers["Authorization"] == "Bearer tok"
    body = json.loads(req.content)
    assert body == {
        "topic": "secret-topic-123",
        "title": "BUY NVDA",
        "message": "go",
        "priority": 4,
        "tags": ["chart"],
        "click": "https://app.example",
        "markdown": True,
    }


@pytest.mark.parametrize(
    ("prio", "name"), [(5, "URGENT"), (4, "HIGH"), (3, "DEFAULT"), (2, "LOW"), (1, "MIN")]
)
def test_priority_constants(prio: int, name: str) -> None:
    assert getattr(alerts, name) == prio


def test_priority_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        alerts.Alert(title="x", message="y", priority=6)


def test_failure_retries_and_never_raises() -> None:
    seen: list[httpx.Request] = []
    n = notifier(500, seen, retries=2)
    assert n.send(alerts.Alert(title="t", message="m")) is False
    assert len(seen) == 3


def test_transport_error_never_raises() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    n = alerts.NtfyNotifier(
        "https://ntfy.example",
        "t",
        client=httpx.Client(transport=httpx.MockTransport(boom)),
        sleep=lambda _s: None,
    )
    assert n.send(alerts.Alert(title="t", message="m")) is False


def test_topic_hidden_from_repr() -> None:
    n = notifier(200, [])
    assert "secret-topic-123" not in repr(n)


def test_long_message_truncated_under_limit() -> None:
    seen: list[httpx.Request] = []
    n = notifier(200, seen, max_message_bytes=100)
    n.send(alerts.Alert(title="t", message="é" * 500))
    msg = json.loads(seen[0].content)["message"]
    assert len(msg.encode()) <= 100 and msg.endswith("[truncated]")


def test_safe_send_swallows_exceptions() -> None:
    class Broken:
        def send(self, alert: alerts.Alert) -> bool:
            raise RuntimeError("x")

    assert alerts.safe_send(Broken(), alerts.hello_alert()) is False


def test_signal_alert_text() -> None:
    order = PlannedOrder(
        symbol="NVDA",
        side=Side.BUY,
        notional=D("19.40"),
        ref_price=D(100),
        stop_price=D("91.25"),
        reason="entry",
        thesis="Strong trend.",
    )
    a = alerts.signal_alert(order, D("102.10"), "paper")
    assert (
        a.message == "BUY NVDA $19.40 (19% of account) at the open. Stop $91.25. Why: Strong trend."
    )
    assert a.priority == alerts.HIGH
    sell = PlannedOrder(
        symbol="KO", side=Side.SELL, qty=D("0.123"), ref_price=D(60), reason="stop_loss"
    )
    s = alerts.signal_alert(sell, D(100), "paper")
    assert s.message.startswith("SELL KO all 0.123 sh at the open. Why: stop loss")


def test_daily_summary_always_has_core_lines() -> None:
    s = alerts.Summary(
        session="2026-03-02",
        mode="paper",
        equity=D("101.5"),
        day_change=0.004,
        since_start=0.015,
        spy_since_start=0.01,
        positions=[alerts.PositionLine("NVDA", D("20.1"), 0.03)],
        cash=D("80"),
        next_actions=[],
        api_spend=0.12,
        api_cap=5.0,
        realized_ytd=D("4"),
        tax_reserve=D("1"),
        kill_switch=False,
    )
    a = alerts.daily_summary_alert(s)
    assert a.priority == alerts.DEFAULT
    for part in (
        "Equity $101.50",
        "vs SPY",
        "NVDA",
        "no trades at the open",
        "API spend $0.12 of $5.00",
        "short-term",
    ):
        assert part in a.message


def test_urgent_alert_types() -> None:
    assert alerts.kill_switch_alert(D(75), D(100), 0.25).priority == alerts.URGENT
    assert alerts.job_failure_alert("decide", "x").priority == alerts.URGENT
    assert alerts.data_failure_alert("decide", "x").priority == alerts.URGENT
    assert alerts.rejected_order_alert("A", Side.BUY, "x", live=True).priority == alerts.URGENT
    assert alerts.rejected_order_alert("A", Side.BUY, "x", live=False).priority == alerts.HIGH


def test_console_and_recording_notifiers(capsys: pytest.CaptureFixture[str]) -> None:
    assert alerts.ConsoleNotifier().send(alerts.hello_alert())
    assert "Stoa test alert" in capsys.readouterr().out
    rec = alerts.RecordingNotifier()
    rec.send(alerts.hello_alert())
    assert rec.titles() == ["Stoa test alert"]
