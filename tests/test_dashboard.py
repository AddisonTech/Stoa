from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

from stoa import dashboard
from stoa.cli import main
from stoa.config import SECRET_ENV, Settings
from stoa.journal import Journal
from stoa.models import ClosedTrade, PositionMeta
from stoa.state import CapRaise, HeldAtClose, State, StateStore, StopOut
from stoa.tracking import DailyRecord
from tests.conftest import ROOT, SMALL_WATCHLIST, make_settings

D = Decimal
NOW = datetime(2021, 10, 13, 22, 30, tzinfo=UTC)
FAKE_SECRET = "pk-live-value-that-must-never-leak"
SESSIONS = [date(2021, 10, d) for d in (6, 7, 8, 11, 12, 13)]


def settings_for(tmp: Path, live_url: str = "") -> Settings:
    return make_settings(
        watchlist=",".join(SMALL_WATCHLIST),
        mode="paper",
        state__dir=str(tmp / "state"),
        dashboard__live_url=live_url,
        big_paper__state_dir=str(tmp / "big-paper"),
    )


def write_fixture(s: Settings) -> None:
    days = []
    for i, day in enumerate(SESSIONS):
        days.append(
            DailyRecord(
                session=day,
                mode="paper",
                broker_equity=D(100 + i),
                adjusted_equity=D(100 + i),
                benchmark_close=D(400 + 2 * i),
                reference_closes={"MTUM": D(150 + i)} if i >= 1 else {},
            )
        )
    state = State(
        last_close_session=SESSIONS[-1],
        decided_sessions=SESSIONS,
        executed_sessions=SESSIONS[1:],
        positions_meta={
            "AAPL": PositionMeta(
                entry_session=date(2021, 10, 7),
                entry_price=D(140),
                entry_atr=D(3),
                stop_price=D("134.25"),
            )
        },
        holdings_at_close={"AAPL": HeldAtClose(qty=D("0.35"), close=D(145))},
        stop_outs=[
            StopOut(
                symbol="KO",
                client_order_id="stoa-stop-20211012-KO-1",
                session=date(2021, 10, 12),
                qty=D(1),
                stop_price=D(54),
                fill_price=D("53.90"),
                filled_at=datetime(2021, 10, 12, 15, 1, tzinfo=UTC),
            )
        ],
        recent_trades=[
            ClosedTrade(
                symbol="KO",
                entry_session=date(2021, 10, 6),
                exit_session=date(2021, 10, 12),
                qty=D(1),
                entry_price=D(55),
                exit_price=D("53.90"),
                fees=D(0),
                pnl=D("-1.10"),
                reason="stop_out",
            )
        ],
    )
    state.track.days = days
    state.llm_spend_by_month = {"2021-10": D("35.00")}
    state.cap_raises = [
        CapRaise(month="2021-10", amount=D(10), at=NOW, nonce="raise-nonce-never-shown")
    ]
    StateStore(s.state.dir).save(state)
    j = Journal(s.state.dir / "journal")
    for day in SESSIONS:
        ts = datetime.combine(day, datetime.min.time(), tzinfo=UTC).replace(hour=22, minute=20)
        j.write(
            ts,
            "decide",
            day,
            "decision",
            strategy="rule",
            candidates=[{"symbol": "AAPL"}],
            quant_kills=[],
            exits=[{"symbol": "KO", "reason": "stop_loss", "detail": "close 53 <= stop 54"}],
            rejections=[],
            regime_bull=True,
        )
        j.write(
            ts,
            "decide",
            day,
            "plan",
            plan={
                "orders": [
                    {
                        "symbol": "AAPL",
                        "side": "buy",
                        "notional": "20.00",
                        "reason": "entry",
                        "thesis": (
                            f"Strong 60 day trend <script>x</script> {FAKE_SECRET} "
                            "ALPACA_PAPER_SECRET"
                        ),
                        "probability": 0.61,
                    }
                ]
            },
        )
    (s.state.dir / "healthcheck.json").write_text(
        json.dumps(
            {
                "at": NOW.isoformat(),
                "session": "2021-10-13",
                "signal": "success",
                "configured": True,
                "sent": True,
            }
        ),
        encoding="utf-8",
    )
    report = s.shared_dir / "results" / "20211001T120000-model"
    report.mkdir(parents=True)
    (report / "calibration.json").write_text(
        json.dumps(
            {
                "chosen": "gbm",
                "horizon": 10,
                "walkforward_passed": True,
                "pooled": {"gbm": {"ece": 0.021, "brier": 0.24, "auc": 0.56, "n": 900}},
                "reliability": [
                    {"lo": 0.3, "hi": 0.4, "count": 200, "mean_p": 0.35, "hit_rate": 0.33},
                    {"lo": 0.5, "hi": 0.6, "count": 500, "mean_p": 0.55, "hit_rate": 0.57},
                    {
                        "lo": 0.9,
                        "hi": 1.0,
                        "count": 0,
                        "mean_p": float("nan"),
                        "hit_rate": float("nan"),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def build(tmp: Path, s: Settings) -> tuple[str, dict[str, object]]:
    index = dashboard.build(s, tmp / "site", ROOT, now=NOW)
    page = index.read_text(encoding="utf-8")
    data = json.loads((index.parent / "data.json").read_text(encoding="utf-8"))
    return page, data


def test_builds_from_fixture_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_PAPER_SECRET", FAKE_SECRET)
    s = settings_for(tmp_path)
    write_fixture(s)
    page, data = build(tmp_path, s)
    assert 'class="badge b-paper">PAPER<' in page
    assert "<polyline" in page and 'class="s-ref line"' in page  # Stoa, SPY and MTUM lines
    curves = data["curves"]
    assert isinstance(curves, dict) and curves["dates"][0] == "2021-10-07"  # first day with MTUM
    assert set(curves["series"]) == {"Stoa", "SPY", "MTUM"}
    assert all(v[0] == 100 for v in curves["series"].values())
    assert "AAPL" in page and "$134.25" in page and "0.35" in page
    assert "KO stopped out 2021-10-12 at $53.90" in page
    assert "Strong 60 day trend" in page and "&lt;script&gt;" in page and "<script" not in page
    assert page.count('class="decision"') == len(SESSIONS)
    assert "Report 20211001T120000-model" in page and 'class="chart cal"' in page
    assert "success ping sent" in page
    assert "$35.00 of $35.00 in 2021-10 (raised $10.00 this month)" in page
    assert "paused at the cap" in page and "raise-nonce-never-shown" not in page
    assert "raise-nonce-never-shown" not in json.dumps(data)
    # Six records are five graded sessions (returns between closes) against the minimum.
    assert "<span>5 of 42</span>" in page


def test_builds_cleanly_from_empty_state(tmp_path: Path) -> None:
    page, data = build(tmp_path, settings_for(tmp_path))
    assert "No trading days recorded yet" in page
    assert "No open positions." in page
    assert "No decisions yet" in page
    assert "No model report yet" in page
    assert "no check-in recorded yet" in page
    assert "$0.00 of $25.00 in 2021-10" in page and "paused at the cap" not in page
    assert "0 of 210 sessions scored since 2026-10-26, shadow only, never traded" in page
    assert "<polyline" not in page and 'class="badge b-paper">PAPER<' in page
    assert data["positions"] == [] and data["decisions"] == []


def test_no_secret_or_key_name_in_site_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in SECRET_ENV.values():
        monkeypatch.setenv(name, f"value-of-{name.lower()}-123")
    monkeypatch.setenv("ALPACA_PAPER_SECRET", FAKE_SECRET)
    s = settings_for(tmp_path)
    write_fixture(s)
    dashboard.build(s, tmp_path / "site", ROOT, now=NOW)
    for f in (tmp_path / "site").iterdir():
        text = f.read_text(encoding="utf-8")
        assert FAKE_SECRET not in text
        for name in SECRET_ENV.values():
            assert name not in text, (f.name, name)
            assert f"value-of-{name.lower()}-123" not in text


def test_read_only_page_and_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*_a: object, **_k: object) -> httpx.Response:
        raise AssertionError("the dashboard build must not use the network")

    monkeypatch.setattr(httpx.Client, "send", no_network)
    s = settings_for(tmp_path)
    write_fixture(s)
    page, _ = build(tmp_path, s)
    # With the live panel off (the default), the page has no script at all.
    for banned in ("<form", "<button", "<input", "<script", "ntfy", "approve?", "onclick"):
        assert banned not in page.lower()
    links = re.findall(r'href="([^"]+)"', page)
    assert links == ["https://github.com/AddisonTech/Stoa"]
    assert "<link" not in page and "http://" not in page  # nothing external is loaded


def test_cli_builds_into_the_folder(tmp_path: Path) -> None:
    out = tmp_path / "site"
    res = CliRunner().invoke(
        main,
        [
            "--config",
            str(ROOT / "config" / "default.yaml"),
            "dashboard",
            "build",
            "--out",
            str(out),
        ],
        env={"STOA_STATE_DIR": str(tmp_path / "state"), "STOA_MODE": "paper"},
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    assert (out / "index.html").exists() and (out / "data.json").exists()


LIVE = "https://stoa-live.vercel.app/api/positions"


def test_live_panel_is_hidden_until_the_function_is_set_up(tmp_path: Path) -> None:
    page, data = build(tmp_path, settings_for(tmp_path))
    assert data["live_url"] == ""
    assert "Live positions are not set up yet" in page and "<script" not in page


def test_live_panel_polls_only_the_live_function_and_writes_text_only(tmp_path: Path) -> None:
    s = make_settings(
        watchlist=",".join(SMALL_WATCHLIST),
        mode="paper",
        state__dir=str(tmp_path / "state"),
        dashboard__live_url=LIVE,
    )
    write_fixture(s)
    page, _ = build(tmp_path, s)
    # The Live card comes first, before the tiles and every other section.
    assert page.index("<h2>Live</h2>") < page.index('class="tiles"') < page.index("Trade log")
    scripts = re.findall(r"<script>(.*?)</script>", page, flags=re.S)
    assert len(scripts) == 1
    js = scripts[0]
    assert f'var URL="{LIVE}"' in js
    assert js.count("fetch(") == 1 and "credentials:'omit'" in js
    for banned in ("innerHTML", "eval(", "POST", "method:", "ntfy", "approve", "localStorage"):
        assert banned not in js
    assert "EVERY=30*1000" in js  # every 30 seconds while the market is open
    assert "Market closed" in js and "Last updated" in js
    assert "Live data unavailable right now" in js
    for label, _ in dashboard.LIVE_COLUMNS:
        assert f"<th>{label}</th>" in page
    # The rest of the page is still built from committed state, with or without the function.
    assert "Trade log" in page and "Positions" in page


def test_live_url_must_be_https() -> None:
    with pytest.raises(ValueError, match="https"):
        make_settings(dashboard__live_url="http://example.com/api")
