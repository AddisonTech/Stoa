"""Trial 7: the Claude analyst in shadow mode, scored on fresh news only, evaluated once.

Pre-registered in docs/DECISIONS.md ("Trial 7: analyst forward test"):

- Only news published on or after ``analyst.shadow_start`` (2026-10-26, New York
  time) is scored, so the analyst model cannot already know how it played out.
- Every session, after DECIDE, a separate workflow fetches the session's news for
  the point-in-time S&P 500 from Alpaca, scores it through the Batch API and
  commits the scores to ``shared/shadow/`` on the state branch.
- Shadow scores never reach DECIDE, EXECUTE or an order. ``model.use_text`` stays
  false. DECIDE only reads the progress line for its daily summary.
- After ``analyst.shadow_sessions`` (210) scored sessions, ``stoa analyst
  evaluate`` runs the pre-registered comparison once and logs exactly one trial.

The job keeps its own ledger (``ledger.json``) and never writes ``state.json``,
so its commits never collide with DECIDE's. The spend cap still covers every
Claude call: the cap check sees this ledger's spend plus whatever DECIDE spent.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from stoa import alerts, clock, spendcap
from stoa.approvals import ApprovalClient
from stoa.config import Settings
from stoa.files import replace_file
from stoa.models import ZERO
from stoa.state import CapAsk, CapRaise, State

log = logging.getLogger(__name__)

SHADOW_DIR = "shadow"
EVAL_NAME = "analyst-evaluate:price+quality+text:logistic:h10"


class ShadowError(RuntimeError):
    """The shadow test refused an action (too early, already evaluated)."""


class Skip(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    session: date
    reason: str


class ShadowLedger(BaseModel):
    """Everything the shadow job remembers. Lives in shared/shadow/ledger.json."""

    model_config = ConfigDict(extra="forbid")

    sessions: list[date] = Field(default_factory=list)  # fully scored sessions, in order
    skipped: list[Skip] = Field(default_factory=list)
    items_by_session: dict[str, int] = Field(default_factory=dict)
    spend_by_month: dict[str, Decimal] = Field(default_factory=dict)
    cap_raises: list[CapRaise] = Field(default_factory=list)
    cap_asks: list[CapAsk] = Field(default_factory=list)
    cap_warned_months: list[str] = Field(default_factory=list)


def shadow_dir(settings: Settings) -> Path:
    return settings.shared_dir / SHADOW_DIR


def ledger_path(settings: Settings) -> Path:
    return shadow_dir(settings) / "ledger.json"


def scores_path(settings: Settings) -> Path:
    return shadow_dir(settings) / "scores.parquet"


def evaluation_path(settings: Settings) -> Path:
    return shadow_dir(settings) / "evaluation.json"


def load_ledger(settings: Settings) -> ShadowLedger:
    p = ledger_path(settings)
    if not p.exists():
        return ShadowLedger()
    return ShadowLedger.model_validate_json(p.read_text(encoding="utf-8"))


def save_ledger(settings: Settings, ledger: ShadowLedger) -> None:
    p = ledger_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(ledger.model_dump_json(indent=2) + "\n", encoding="utf-8")
    replace_file(tmp, p)


# ---- spend: one cap over every Claude call ------------------------------------------------


def merged_view(
    state: State, ledger: ShadowLedger, extra: Mapping[str, Decimal] | None = None
) -> State:
    """A throwaway State whose spend and cap records cover DECIDE, the shadow job and
    ``extra`` (trial 15's live Claude reader, ``stoa/newsbrain/claude15.py``)."""
    extra = extra or {}
    months = set(state.llm_spend_by_month) | set(ledger.spend_by_month) | set(extra)
    spend = {
        m: state.llm_spend_by_month.get(m, ZERO)
        + ledger.spend_by_month.get(m, ZERO)
        + extra.get(m, ZERO)
        for m in months
    }
    return State(
        llm_spend_by_month=spend,
        cap_raises=[*state.cap_raises, *ledger.cap_raises],
        cap_asks=[*state.cap_asks, *ledger.cap_asks],
        cap_warned_months=sorted(set(state.cap_warned_months) | set(ledger.cap_warned_months)),
    )


def big_reader_spend(settings: Settings) -> dict[str, Decimal]:
    """Trial 15's live Claude reader's spend, so the one monthly cap covers it too."""
    from stoa.newsbrain import claude15

    return claude15.spend(settings)


def keep_new_cap_records(view: State, before: State, ledger: ShadowLedger) -> None:
    """Copy what the cap gate added to the view into the shadow ledger."""
    ledger.cap_raises = [*ledger.cap_raises, *view.cap_raises[len(before.cap_raises) :]]
    ledger.cap_asks = [*ledger.cap_asks, *view.cap_asks[len(before.cap_asks) :]]
    new_months = set(view.cap_warned_months) - set(before.cap_warned_months)
    ledger.cap_warned_months = sorted(set(ledger.cap_warned_months) | new_months)


# ---- one session --------------------------------------------------------------------------


def window(session: date, start: date) -> tuple[datetime, datetime]:
    """The news that belongs to ``session``: from the previous close to this close."""
    close = clock.session(session).close
    prev_close = clock.session(clock.previous_session(session)).close
    first = datetime.combine(start, time(0), tzinfo=clock.NY).astimezone(UTC)
    return max(prev_close, first), close


def select(news: pd.DataFrame, lo: datetime, hi: datetime, start: date) -> pd.DataFrame:
    """Rows published in [lo, hi), and never before the forward start."""
    if news.empty:
        return news
    known = pd.to_datetime(news["known_at"], utc=True)
    floor = pd.Timestamp(datetime.combine(start, time(0), tzinfo=clock.NY))
    keep = (known >= pd.Timestamp(lo)) & (known < pd.Timestamp(hi)) & (known >= floor)
    return news.loc[keep.to_numpy()]


class NewsSource(Protocol):
    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame: ...


@dataclass
class ShadowRun:
    session: date | None
    status: str  # not_started | already_scored | paused | scored | incomplete | over_budget
    items: int = 0
    scored: int = 0
    failed: int = 0
    spent_usd: float = 0.0
    note: str = ""


def run_session(
    settings: Settings,
    state: State,
    ledger: ShadowLedger,
    *,
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
    send: Callable[[alerts.Alert], object],
    news: NewsSource,
    client: Any,
    approvals: ApprovalClient | None,
    approval_topic: str | None,
    members: Callable[[date], set[str]],
    progress: Callable[[str], None] | None = None,
) -> ShadowRun:
    """Score the last closed session's news into the shadow store. Changes only ``ledger``."""
    from stoa import analyst as an

    s = settings
    cfg = s.analyst
    session = clock.last_closed_session(now())
    if session < cfg.shadow_start:
        return ShadowRun(session, "not_started", note=f"the forward test starts {cfg.shadow_start}")
    if session in ledger.sessions:
        return ShadowRun(session, "already_scored")

    view = merged_view(state, ledger, big_reader_spend(s))
    before = view.model_copy(deep=True)
    g = spendcap.gate(
        view,
        s.budget,
        s.approval,
        now=now,
        sleep=sleep,
        send=send,
        client=approvals,
        server=s.alerts.server,
        approval_topic=approval_topic,
    )
    keep_new_cap_records(view, before, ledger)
    if g.paused:
        ledger.skipped = [*ledger.skipped, Skip(session=session, reason="paused at the cap")]
        return ShadowRun(
            session, "paused", note=f"{alerts.money(g.spent)} of {alerts.money(g.cap)}"
        )

    lo, hi = window(session, cfg.shadow_start)
    symbols = sorted(members(session))
    raw = news.fetch(symbols, lo.astimezone(UTC).date(), hi.astimezone(UTC).date())
    df = select(raw, lo, hi, cfg.shadow_start)
    items = an.items_from_news(df, cfg.text_chars, cfg.max_items_per_symbol_per_day)
    other = g.spent - ledger.spend_by_month.get(g.month, ZERO)  # what DECIDE spent this month
    budget = an.Ledger(float(g.cap - other), ledger.spend_by_month, clock=now)
    a = an.Analyst(
        cfg, s.budget, s.llm.prices, client, an.ScoreCache(cfg.cache_dir), budget, sleep=sleep
    )
    run = ShadowRun(session, "scored", items=len(items))
    if items:
        try:
            rep = a.run_batches(
                items,
                an.BatchJournal(shadow_dir(s) / "batches.json"),
                allow_spend=False,
                progress=progress,
            )
        except an.BudgetExceeded as exc:
            ledger.skipped = [*ledger.skipped, Skip(session=session, reason="over budget")]
            return ShadowRun(session, "over_budget", items=len(items), note=str(exc))
        run.scored, run.failed, run.spent_usd = rep.scored + rep.cached, rep.failed, rep.spent_usd
        if rep.failures:
            run.note = f"Failed items: {an.summarize_failures(rep.failures)}."
        an.merge_scores(scores_path(s), a.rows(items))
    heads_up = spendcap.warn_if_due(
        merged_view(state, ledger, big_reader_spend(s)), s.budget, now()
    )
    if heads_up is not None:
        ledger.cap_warned_months = sorted({*ledger.cap_warned_months, g.month})
        send(heads_up)
    if run.failed:
        reason = f"items failed: {an.summarize_failures(rep.failures)}" if items else "items failed"
        ledger.skipped = [*ledger.skipped, Skip(session=session, reason=reason[:300])]
        run.status = "incomplete"
        return run
    ledger.sessions = sorted({*ledger.sessions, session})
    ledger.items_by_session = {**ledger.items_by_session, session.isoformat(): len(items)}
    return run


# ---- progress for the summary and the dashboard ------------------------------------------


def progress(
    settings: Settings, state: State, ledger: ShadowLedger, now: datetime
) -> dict[str, Any]:
    month = spendcap.month_of(now)
    view = merged_view(state, ledger, big_reader_spend(settings))
    return {
        "start": settings.analyst.shadow_start.isoformat(),
        "sessions": len(ledger.sessions),
        "needed": settings.analyst.shadow_sessions,
        "last": ledger.sessions[-1].isoformat() if ledger.sessions else None,
        "skipped": len(ledger.skipped),
        "evaluated": evaluation_path(settings).exists(),
        "month": month,
        "spent": float(spendcap.spent(view, month)),
        "cap": float(spendcap.cap_for(view, settings.budget, month)),
    }


def summary_line(settings: Settings, state: State, now: datetime) -> str:
    """One line for the daily summary. Never raises: DECIDE must not care about the shadow."""
    try:
        p = progress(settings, state, load_ledger(settings), now)
    except Exception as exc:
        return f"Analyst shadow: progress unavailable ({type(exc).__name__})"
    return (
        f"Analyst shadow: {p['sessions']} of {p['needed']} sessions scored; "
        f"Claude spend {alerts.money(p['spent'])} of {alerts.money(p['cap'])} in {p['month']}"
    )


# ---- the one evaluation ------------------------------------------------------------------


def eval_settings(settings: Settings) -> Settings:
    """Trial 1's model as pre-registered: logistic, 10 sessions, all price and quality features."""
    m = settings.model.model_copy(
        update={
            "horizon": 10,
            "test_horizon": None,
            "candidates": ("logistic",),
            "price_features": None,
            "normalize": "zscore",
            "use_quality": True,
            "use_text": False,
            "embargo_sessions": None,
        }
    )
    return settings.model_copy(update={"model": m})


Runner = Callable[[Settings, Any, bool], pd.DataFrame]  # (settings, training inputs, text) -> oos


def _walkforward(settings: Settings, inputs: Any, text: bool) -> pd.DataFrame:
    from stoa.dataset import FeatureSet
    from stoa.training import run_walkforward

    fs = FeatureSet(quality=True, text=text)
    return run_walkforward(settings, inputs, fs, kinds=("logistic",)).scored("logistic")


def evaluate(
    settings: Settings,
    repo: Path,
    now: datetime,
    *,
    runner: Runner = _walkforward,
    inputs: Any = None,
) -> dict[str, Any]:
    """Run the pre-registered comparison once. Refuses early and refuses a second run."""
    from stoa import calibration as cal
    from stoa import holdout
    from stoa.experiments import Registry, deflate
    from stoa.training import long_short_returns

    s = settings
    ledger = load_ledger(s)
    need = s.analyst.shadow_sessions
    if len(ledger.sessions) < need:
        raise ShadowError(
            f"only {len(ledger.sessions)} of {need} sessions are scored; the evaluation is "
            "pre-registered for after all of them and runs once"
        )
    registry = Registry(s.registry_path)
    if evaluation_path(s).exists() or any(e.name == EVAL_NAME for e in registry.load()):
        raise ShadowError("the analyst forward test was already evaluated; it runs once")
    se = eval_settings(s)
    if inputs is None:
        from stoa.pipeline import load_inputs

        last = clock.last_closed_session(now)
        loaded = load_inputs(se, last, blocked=holdout.blocked_window(repo, se)).training
        scores = pd.read_parquet(scores_path(s)) if scores_path(s).exists() else None
        inputs = dataclasses.replace(loaded, text_scores=scores)
    start = pd.Timestamp(s.analyst.shadow_start)
    results: dict[str, Any] = {}
    oos_by: dict[str, pd.DataFrame] = {}
    for label, text in (("without_text", False), ("with_text", True)):
        o = runner(se, inputs, text)
        fwd = o.loc[o.index.get_level_values("date") >= start]
        oos_by[label] = fwd
        sc = cal.score(fwd["p"].to_numpy(), fwd["y"].to_numpy()) if len(fwd) else None
        results[label] = sc.to_dict() if sc is not None else None
    with_text = oos_by["with_text"].copy()
    with_text.attrs["horizon"] = 10
    rets = long_short_returns(with_text, se.model.max_candidates, se.model.threshold)
    w, wo = results["with_text"], results["without_text"]
    base = _base_log_loss(w["base_rate"]) if w else math.nan
    registry.append(
        kind="train",
        name=EVAL_NAME,
        config_hash=se.strategy_hash(),
        data_hash=f"shadow:{len(ledger.sessions)}:{ledger.sessions[-1].isoformat()}",
        params={"trial": 7, "horizon": 10, "kind": "logistic", "features": "price+quality+text"},
        metrics={
            "with_text": w,
            "without_text": wo,
            "sharpe_daily": _per_session_sharpe(rets),
        },
        returns=rets,
    )
    d = deflate(registry, list(rets), period=int(rets.attrs.get("period", 1)))
    checks = {
        "calibrated": bool(w and w["ece"] < s.model.max_ece),
        "beats_base_rate": bool(w and w["log_loss"] < base),
        "beats_without_text": bool(w and wo and w["log_loss"] < wo["log_loss"]),
        "deflated_sharpe": bool(math.isfinite(d.get("dsr", math.nan)) and d["dsr"] >= 0.95),
    }
    out = {
        "evaluated_at": now.isoformat(),
        "sessions": len(ledger.sessions),
        "first_session": ledger.sessions[0].isoformat(),
        "last_session": ledger.sessions[-1].isoformat(),
        "with_text": w,
        "without_text": wo,
        "base_rate_log_loss": base,
        "deflated": d,
        "checks": checks,
        "passed": all(checks.values()),
    }
    p = evaluation_path(s)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def _base_log_loss(rate: float) -> float:
    b = min(max(rate, 1e-9), 1 - 1e-9)
    return -(b * math.log(b) + (1 - b) * math.log(1 - b))


def _per_session_sharpe(rets: pd.Series) -> float:
    period = int(rets.attrs.get("period", 1))
    if len(rets) <= 2 or rets.std() <= 0:
        return 0.0
    return float(rets.mean() / rets.std(ddof=1)) / math.sqrt(period)
