"""Exam order, enforced in code: walk-forward, then the sealed holdout, then paper, then live.

- The holdout cannot be sealed until a walk-forward result exists and passed
  (``state/exam/walkforward.json``, written by `stoa model train`).
- Live trading is refused (on top of the triple lock) until the paper exam has
  run long enough, traded enough and passed on the cost-adjusted record.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from stoa.config import Settings
from stoa.state import StateStore
from stoa.tracking import ExamStatus, Track, paper_exam

EXAM_DIR = "exam"


@dataclass(frozen=True)
class WalkForwardRecord:
    created: str
    passed: bool
    reasons: list[str]
    checks: dict[str, Any]
    report_path: str
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_path(state_dir: Path) -> Path:
    return state_dir / EXAM_DIR / "walkforward.json"


def write_walkforward(state_dir: Path, rec: WalkForwardRecord) -> Path:
    p = record_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")
    return p


def read_walkforward(state_dir: Path) -> WalkForwardRecord | None:
    p = record_path(state_dir)
    if not p.exists():
        return None
    return WalkForwardRecord(**json.loads(p.read_text(encoding="utf-8")))


def load_track(state_dir: Path) -> Track:
    return StateStore(state_dir).load().track


def paper_track(settings: Settings) -> Track:
    """Where live mode reads the paper record from (the renamed paper state, by default)."""
    if settings.exam.paper_state_dir is not None:
        return load_track(settings.exam.paper_state_dir)
    return load_track(settings.state.dir)


def status(settings: Settings, holdout_lock: Any | None = None) -> ExamStatus:
    wf = read_walkforward(settings.shared_dir)
    track = paper_track(settings)
    strategy = "model" if settings.strategy.kind == "model" else "agents"
    g, passed = paper_exam(track, settings.exam, strategy)
    lock = holdout_lock
    return ExamStatus(
        walkforward_done=wf is not None,
        walkforward_passed=bool(wf and wf.passed),
        holdout_sealed=lock is not None,
        holdout_scored=bool(lock is not None and getattr(lock, "scored_at", None)),
        holdout_passed=bool(
            lock is not None and getattr(lock, "score", None) and lock.score.get("passed")
        ),
        paper=g,
        paper_sessions_needed=settings.exam.paper_min_sessions,
        paper_trades_needed=settings.exam.paper_min_trades,
        paper_passed=passed,
    )


def live_problems(settings: Settings) -> list[str]:
    """Empty when the exams allow live trading."""
    st = status(settings)
    out = []
    if settings.exam.require_walkforward_pass and not st.walkforward_passed:
        out.append(
            "walk-forward has not passed"
            if st.walkforward_done
            else "walk-forward has not been run (`stoa model train`)"
        )
    if not st.paper_passed:
        out += [p for p in st.problems() if p.startswith("paper")]
    return out


def render(st: ExamStatus, today: date) -> str:
    def mark(ok: bool) -> str:
        return "pass" if ok else "not yet"

    g = st.paper
    lines = [
        f"Exam status on {today}",
        "",
        f"1. Walk-forward        {mark(st.walkforward_passed)}"
        + ("" if st.walkforward_done else " (not run)"),
        f"2. Holdout sealed      {mark(st.holdout_sealed)}",
        f"   Holdout scored      {mark(st.holdout_scored)}"
        + (f", {'PASSED' if st.holdout_passed else 'FAILED'}" if st.holdout_scored else ""),
        f"3. Paper exam          {mark(st.paper_passed)}: {g.sessions} of "
        f"{st.paper_sessions_needed} sessions, {g.trades} of {st.paper_trades_needed} trades",
        f"   after costs         return {g.total_return:+.1%} vs SPY {g.spy_return:+.1%}, "
        f"drawdown {g.max_drawdown:.1%} vs SPY {g.spy_drawdown:.1%}, "
        f"Sharpe {g.sharpe:.2f} vs SPY {g.spy_sharpe:.2f}",
        f"   raw broker fills    return {g.raw_total_return:+.1%}",
        f"4. Live allowed        {'yes' if st.live_allowed else 'no'}",
    ]
    return "\n".join(lines)
