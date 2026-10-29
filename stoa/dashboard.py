"""A read-only status page built from committed state. `stoa dashboard build`.

Reads only files the daily and monthly workflows commit to the stoa-state
branch (plus HOLDOUT.lock from the repo) and writes a static site: one
index.html with inline SVG charts and inline CSS, plus data.json with the
same numbers. It makes no network calls, never builds a broker, and never
reads a secret. The page has no buttons, forms or links that could trade or
approve anything.

Files read, relative to the state directory (``state.dir``) and the shared
directory (``state.shared_dir``, the same as the state directory when unset):

- ``<state>/state.json``             equity history, positions, stops, kill switch
- ``<state>/journal/*.jsonl``        decisions, picks, fills and closed trades (the trade log)
- ``<state>/healthcheck.json``       whether the last dead man's switch ping went out
- ``<shared>/exam/walkforward.json`` walk-forward result for the exam
- ``<shared>/results/*-model/calibration.json``  the latest model report
- ``<repo>/HOLDOUT.lock``            holdout seal and score
"""

from __future__ import annotations

import html
import itertools
import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from stoa import __version__, clock, exam, holdout, redact, shadow, shadowboard, spendcap
from stoa.config import SECRET_ENV, Settings
from stoa.journal import read_journal
from stoa.state import State, StateStore

DECISIONS_SHOWN = 20
TRADES_SHOWN = 250
CHECKIN_FILE = "healthcheck.json"
HIDDEN = "[hidden]"


# ---- reading committed state ------------------------------------------------------------


def latest_model_report(results: Path) -> tuple[str, dict[str, Any]] | None:
    """The newest ``*-model/calibration.json`` under ``results``, by its timestamped name."""
    if not results.exists():
        return None
    reports = sorted(p for p in results.glob("*-model/calibration.json") if p.is_file())
    if not reports:
        return None
    path = reports[-1]
    try:
        return path.parent.name, json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_checkin(state_dir: Path) -> dict[str, Any] | None:
    p = state_dir / CHECKIN_FILE
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    keep = ("at", "session", "signal", "configured", "sent")
    return {k: data.get(k) for k in keep} if isinstance(data, dict) else None


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def equity_curves(state: State, references: Iterable[str]) -> dict[str, Any]:
    """Stoa, SPY and each reference fund, all rebased to 100 on the same start date.

    Stoa's line is the cost-adjusted record with deposits and withdrawals taken
    out of each day's return. The start is the first day every line has a
    value, so the comparison is always like for like.
    """
    days = state.track.days
    refs = [r for r in references if any(r in d.reference_closes for d in days)]
    start = next(
        (i for i, d in enumerate(days) if all(r in d.reference_closes for r in refs)), None
    )
    if start is None or len(days) - start < 1:
        return {"dates": [], "series": {}, "references_missing": list(references)}
    window = days[start:]
    stoa: list[float | None] = [100.0]
    level = 100.0
    for prev, cur in itertools.pairwise(window):
        base = float(prev.adjusted_equity)
        r = (float(cur.adjusted_equity) - float(cur.flow)) / base - 1 if base > 0 else 0.0
        level *= 1 + r
        stoa.append(level)
    series: dict[str, list[float | None]] = {"Stoa": stoa}
    spy0 = float(window[0].benchmark_close)
    series["SPY"] = [float(d.benchmark_close) / spy0 * 100 if spy0 else None for d in window]
    for ref in refs:
        r0 = float(window[0].reference_closes[ref])
        series[ref] = [
            float(d.reference_closes[ref]) / r0 * 100 if ref in d.reference_closes and r0 else None
            for d in window
        ]
    return {
        "dates": [d.session.isoformat() for d in window],
        "series": series,
        "references_missing": [r for r in references if r not in refs],
    }


def positions(state: State) -> list[dict[str, Any]]:
    last = state.last_close_session
    out = []
    for sym, m in sorted(state.positions_meta.items()):
        held = state.holdings_at_close.get(sym)
        qty = held.qty if held else None
        close = held.close if held else None
        held_days = clock.sessions_after(m.entry_session, last) if last else 0
        out.append(
            {
                "symbol": sym,
                "entry_session": m.entry_session.isoformat(),
                "entry_price": float(m.entry_price),
                "qty": float(qty) if qty is not None else None,
                "value": float(qty * close) if qty is not None and close is not None else None,
                "close": float(close) if close is not None else None,
                "gain": float(close / m.entry_price - 1) if close and m.entry_price > 0 else None,
                "stop_price": float(m.stop_price) if m.stop_price is not None else None,
                "days_held": max(held_days, 0),
            }
        )
    return out


def _orders(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for o in plan.get("orders") or []:
        out.append(
            {
                "side": str(o.get("side", "")),
                "symbol": str(o.get("symbol", "")),
                "notional": _num(o.get("notional")),
                "qty": _num(o.get("qty")),
                "reason": str(o.get("reason", "")),
                "thesis": str(o.get("thesis") or ""),
                "probability": _num(o.get("probability")),
                "conviction": o.get("conviction"),
            }
        )
    return out


def decisions(records: list[dict[str, Any]], limit: int = DECISIONS_SHOWN) -> list[dict[str, Any]]:
    """The latest DECIDE runs, newest first: what was planned and why, from the journal."""
    by_session: dict[str, dict[str, Any]] = {}
    for rec in records:
        if rec.get("job") != "decide" or not rec.get("session"):
            continue
        step, data = rec.get("step"), rec.get("data") or {}
        entry = by_session.setdefault(
            rec["session"], {"session": rec["session"], "at": rec.get("ts")}
        )
        entry["at"] = rec.get("ts") or entry["at"]
        if step == "decision":
            entry["strategy"] = str(data.get("strategy", ""))
            entry["regime_bull"] = data.get("regime_bull")
            entry["candidates"] = len(data.get("candidates") or [])
            entry["exits"] = [
                {k: str(e.get(k, "")) for k in ("symbol", "reason", "detail")}
                for e in data.get("exits") or []
            ]
            entry["rejections"] = [
                {k: str(r.get(k, "")) for k in ("symbol", "side", "reason", "detail")}
                for r in data.get("rejections") or []
            ]
            entry["quant_kills"] = len(data.get("quant_kills") or [])
        elif step == "plan":
            entry["orders"] = _orders(data.get("plan") or {})
    ordered = sorted(by_session.values(), key=lambda e: e["session"], reverse=True)
    return ordered[:limit]


def _size(notional: Any, qty: Any, price: Any = None) -> str:
    n, q, p = _num(notional), _num(qty), _num(price)
    if q is not None and p is not None:
        return f"{q:g} sh at {_money(p)} = {_money(q * p)}"
    if n is not None:
        return _money(n)
    return f"{q:g} sh" if q is not None else ""


def _row(
    session: Any,
    kind: str,
    symbol: Any,
    side: Any,
    size: str,
    reason: str,
    pnl: float | None = None,
) -> dict[str, Any]:
    return {
        "date": str(session or ""),
        "kind": kind,
        "symbol": str(symbol or ""),
        "side": str(side or "").lower(),
        "size": size,
        "reason": reason,
        "pnl": pnl,
    }


def trade_log(
    records: list[dict[str, Any]], state: State, limit: int = TRADES_SHOWN
) -> list[dict[str, Any]]:
    """Every decision and fill, newest first: picks, drops, fills, skips and closed trades.

    Picks are the final morning picks from the sweep (the evening plan when no
    sweep ran). P&L is realized on closed trades and unrealized, at the last
    close, on buys still held.
    """
    rows: list[dict[str, Any]] = []
    swept = {r.get("session") for r in records if r.get("job") == "sweep"}
    held = state.holdings_at_close
    for rec in records:
        job, step, data = rec.get("job"), rec.get("step"), rec.get("data") or {}
        session = rec.get("session")
        if job == "sweep" and step == "sweep":
            plan = data.get("plan") or {}
            for o in _orders(plan):
                why = o["thesis"] or o["reason"].replace("_", " ")
                rows.append(
                    _row(
                        session, "pick", o["symbol"], o["side"], _size(o["notional"], o["qty"]), why
                    )
                )
            for sym in data.get("dropped") or []:
                tone = (data.get("tones") or {}).get(sym) or {}
                why = f"dropped on overnight news: {tone.get('worst', '')}".strip()
                rows.append(_row(session, "dropped", sym, "buy", "", why))
        elif job == "decide" and step == "plan":
            plan = data.get("plan") or {}
            execute_on = plan.get("execute_session")
            if execute_on in swept:
                continue
            for o in _orders(plan):
                why = o["thesis"] or o["reason"].replace("_", " ")
                rows.append(
                    _row(
                        execute_on,
                        "pick",
                        o["symbol"],
                        o["side"],
                        _size(o["notional"], o["qty"]),
                        why,
                    )
                )
        elif job == "execute" and step in ("execution", "override"):
            manual = step == "override"
            for f in data.get("fills") or []:
                sym = f.get("symbol")
                pnl = None
                h = held.get(str(sym))
                meta = state.positions_meta.get(str(sym))
                price = _num(f.get("filled_avg_price"))
                if str(f.get("side")) == "buy" and h is not None and meta is not None and price:
                    pnl = float(h.qty) * (float(h.close) - price)
                rows.append(
                    _row(
                        session,
                        "filled",
                        sym,
                        f.get("side"),
                        _size(None, f.get("filled_qty"), f.get("filled_avg_price")),
                        ", ".join(
                            w
                            for w in (
                                "manual override" if manual else "",
                                "open position" if pnl is not None else "",
                            )
                            if w
                        ),
                        pnl,
                    )
                )
            for r in data.get("rejections") or []:
                why = f"{str(r.get('reason', '')).replace('_', ' ')}: {r.get('detail', '')}"
                rows.append(_row(session, "skipped", r.get("symbol"), r.get("side"), "", why))
        if step in ("execution", "stop_outs"):
            for t in data.get("closed") or []:
                qty, entry, exit_ = (_num(t.get(k)) for k in ("qty", "entry_price", "exit_price"))
                rows.append(
                    _row(
                        t.get("exit_session") or session,
                        "closed",
                        t.get("symbol"),
                        "sell",
                        _size(None, qty, exit_),
                        f"{str(t.get('reason', '')).replace('_', ' ')}, bought "
                        f"{t.get('entry_session', '')} at {_money(entry)}",
                        _num(t.get("pnl")),
                    )
                )
    order = {"pick": 0, "dropped": 1, "skipped": 2, "filled": 3, "closed": 4}
    rows.sort(key=lambda r: (r["date"], order.get(r["kind"], 9)), reverse=True)
    return rows[:limit]


def big_paper(settings: Settings) -> dict[str, Any]:
    """The big paper account (its trial, trial 12 by default): equity against SPY, trade log."""
    root = settings.big_paper.state_dir
    curve_path = root / "curve.json"
    rows: list[dict[str, Any]] = (
        json.loads(curve_path.read_text(encoding="utf-8")) if curve_path.exists() else []
    )
    dates = [str(r["session"]) for r in rows]
    series: dict[str, list[float | None]] = {}
    if rows:
        e0 = float(rows[0]["equity"])
        series["Big paper"] = [float(r["equity"]) / e0 * 100 if e0 else None for r in rows]
        s0 = _num(rows[0].get("spy_close"))
        series["SPY"] = [
            float(r["spy_close"]) / s0 * 100 if s0 and _num(r.get("spy_close")) else None
            for r in rows
        ]
    state = StateStore(root).load()
    # The 9:00 am picks are journaled as "picks"; the trade log reads them as the morning picks.
    records = [
        {**r, "job": "sweep", "step": "sweep"} if r.get("job") == "picks" else r
        for r in read_journal(root / "journal")
    ]
    return {
        "enabled": settings.big_paper.enabled,
        "strategy": settings.big_paper.strategy,
        "stopped": settings.stop.big_paper,
        "tag": settings.big_paper.tag,
        "equity": float(rows[-1]["equity"]) if rows else None,
        "curve": {"dates": dates, "series": series},
        "trades": trade_log(records, state),
        "kill_switch": state.kill_switch.active,
    }


def last_run(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    stamped = [r for r in records if r.get("ts")]
    if not stamped:
        return None
    rec = max(stamped, key=lambda r: str(r["ts"]))
    return {"at": rec["ts"], "job": rec.get("job"), "session": rec.get("session")}


def exam_progress(settings: Settings, repo: Path) -> dict[str, Any]:
    lock = holdout.read_lock(repo)
    st = exam.status(settings, lock)
    g = st.paper
    return {
        "walkforward_done": st.walkforward_done,
        "walkforward_passed": st.walkforward_passed,
        "holdout_sealed": st.holdout_sealed,
        "holdout_scored": st.holdout_scored,
        "holdout_passed": st.holdout_passed,
        "paper_sessions": g.sessions,
        "paper_sessions_needed": st.paper_sessions_needed,
        "paper_trades": g.trades,
        "paper_trades_needed": st.paper_trades_needed,
        "paper_passed": st.paper_passed,
        "live_allowed": st.live_allowed,
    }


def analyst_spend(state: State, settings: Settings, now: datetime) -> dict[str, Any]:
    """This month's analyst spend against its cap. Raise nonces stay in state, not here."""
    month = spendcap.month_of(now)
    cap = spendcap.cap_for(state, settings.budget, month)
    used = spendcap.spent(state, month)
    return {
        "month": month,
        "spent": float(used),
        "cap": float(cap),
        "base_cap": settings.budget.monthly_cap_usd,
        "ceiling": settings.budget.hard_ceiling_usd,
        "paused": cap > 0 and used >= cap,
        "raises": [
            {"amount": float(r.amount), "at": r.at.isoformat()}
            for r in state.cap_raises
            if r.month == month
        ],
    }


def calibration(report: tuple[str, dict[str, Any]] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    name, rep = report
    bins = [
        {
            "mean_p": _num(b.get("mean_p")),
            "hit_rate": _num(b.get("hit_rate")),
            "count": b.get("count", 0),
        }
        for b in rep.get("reliability") or []
    ]
    chosen = str(rep.get("chosen", ""))
    pooled = (rep.get("pooled") or {}).get(chosen) or {}
    return {
        "report": name,
        "chosen": chosen,
        "horizon": rep.get("horizon"),
        "walkforward_passed": rep.get("walkforward_passed"),
        "ece": _num(pooled.get("ece")),
        "brier": _num(pooled.get("brier")),
        "auc": _num(pooled.get("auc")),
        "n": pooled.get("n"),
        "bins": [b for b in bins if b["mean_p"] is not None and b["hit_rate"] is not None],
    }


def collect(
    settings: Settings, repo: Path, *, results: Path | None = None, now: datetime | None = None
) -> dict[str, Any]:
    """Everything the page shows, as plain JSON-ready data."""
    state_dir = settings.state.dir
    state = StateStore(state_dir).load()
    records = read_journal(state_dir / "journal")
    built = now or clock.now_utc()
    try:
        ledger = shadow.load_ledger(settings)
    except (OSError, ValueError):
        ledger = shadow.ShadowLedger()
    track = state.track
    mode = track.days[-1].mode if track.days else settings.mode
    equity = track.days[-1] if track.days else None
    return {
        "generated_at": built.isoformat(),
        "version": __version__,
        "mode": mode,
        "live_url": settings.dashboard.live_url,
        "benchmark": settings.benchmark,
        "equity": {
            "session": equity.session.isoformat() if equity else None,
            "broker": float(equity.broker_equity) if equity else None,
            "adjusted": float(equity.adjusted_equity) if equity else None,
        },
        "curves": equity_curves(state, settings.sources.reference_symbols),
        "positions": positions(state),
        "stop_outs": [
            {
                "symbol": s.symbol,
                "session": s.session.isoformat(),
                "fill_price": float(s.fill_price),
                "stop_price": float(s.stop_price),
                "filled_at": s.filled_at.isoformat() if s.filled_at else None,
            }
            for s in reversed(state.stop_outs[-10:])
        ],
        "decisions": decisions(records),
        "trades": trade_log(records, state),
        "exam": exam_progress(settings, repo),
        "calibration": calibration(latest_model_report(results or settings.shared_dir / "results")),
        "last_run": last_run(records),
        "last_decided": state.decided_sessions[-1].isoformat() if state.decided_sessions else None,
        "last_executed": (
            state.executed_sessions[-1].isoformat() if state.executed_sessions else None
        ),
        "checkin": read_checkin(state_dir),
        "kill_switch": {
            "active": state.kill_switch.active,
            "since": (
                state.kill_switch.tripped_session.isoformat()
                if state.kill_switch.tripped_session
                else None
            ),
        },
        "fallback": {"active": track.fallback_active, "reason": track.fallback_reason},
        "big_paper": big_paper(settings),
        "shadow_board": shadowboard.read_board(settings.big_paper.state_dir),
        "analyst": analyst_spend(shadow.merged_view(state, ledger), settings, built),
        "shadow": {
            k: v
            for k, v in shadow.progress(settings, state, ledger, built).items()
            if k in ("start", "sessions", "needed", "last", "skipped", "evaluated")
        },
    }


# ---- keeping secrets out ---------------------------------------------------------------


def scrub(text: str, env: Mapping[str, str] | None = None) -> str:
    """Remove any secret name or value that could have slipped into recorded text."""
    e = os.environ if env is None else env
    values = sorted(
        (v for n in SECRET_ENV.values() if len(v := e.get(n, "").strip()) >= redact.MIN_SECRET_LEN),
        key=len,
        reverse=True,
    )
    for v in values:
        text = text.replace(v, HIDDEN)
    for n in SECRET_ENV.values():
        text = text.replace(n, HIDDEN)
    return redact.redact(text)


# ---- rendering --------------------------------------------------------------------------


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _money(v: float | None) -> str:
    return "n/a" if v is None else f"${v:,.2f}"


def _pct(v: float | None, signed: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.1%}" if signed else f"{v:.1%}"


def _when(stamp: str | None) -> str:
    if not stamp:
        return "never"
    try:
        ts = datetime.fromisoformat(stamp)
    except ValueError:
        return _e(stamp)
    if ts.tzinfo is None:
        return ts.strftime("%Y-%m-%d %H:%M")
    return clock.to_ny(ts).strftime("%Y-%m-%d %H:%M ET")


@dataclass(frozen=True)
class _Line:
    name: str
    css: str
    values: list[float | None]


SERIES_CLASS = {"Stoa": "s-stoa", "SPY": "s-spy"}


def line_chart(dates: list[str], lines: list[_Line]) -> str:
    """Line chart, x by session, y rebased to 100.

    The SVG stretches to the card (lines keep their width at any size) and the
    axis labels are HTML, so they stay readable on a phone and on a desktop.
    """
    vals = [v for ln in lines for v in ln.values if v is not None]
    if not dates or not vals:
        return ""
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    n = len(dates)
    box = 1000

    def x(i: int) -> float:
        return box * i / (n - 1) if n > 1 else box / 2

    def frac(v: float) -> float:
        return (v - lo) / (hi - lo)

    places = 1 if hi - lo < 5 else 0
    ticks = [lo + (hi - lo) * k / 4 for k in range(5)]
    parts = [
        '<div class="plot"><div class="yaxis">',
        *(f'<span style="top:{(1 - frac(v)) * 100:.1f}%">{v:.{places}f}</span>' for v in ticks),
        f'</div><svg class="chart" viewBox="0 0 {box} {box}" preserveAspectRatio="none" '
        f'role="img" aria-label="Growth of 100 for {", ".join(ln.name for ln in lines)}">',
    ]
    for v in ticks:
        yy = box * (1 - frac(v))
        parts.append(f'<line class="grid" x1="0" x2="{box}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
    for ln in lines:
        pts = [(x(i), box * (1 - frac(v))) for i, v in enumerate(ln.values) if v is not None]
        if len(pts) == 1:
            pts = pts * 2  # one day: a round cap draws it as a dot
        if pts:
            coords = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
            parts.append(f'<polyline class="{ln.css} line" points="{coords}"/>')
    parts.append("</svg></div>")
    last = f"<span>{_e(dates[-1])}</span>" if n > 1 else ""
    parts.append(f'<div class="xaxis"><span>{_e(dates[0])}</span>{last}</div>')
    return "".join(parts)


def calibration_chart(bins: list[dict[str, Any]], size: int = 280) -> str:
    """Reliability diagram: predicted probability against the hit rate, per bin."""
    if not bins:
        return ""
    m = 32
    span = size - 2 * m

    def pos(p: float, q: float) -> tuple[float, float]:
        return m + span * p, m + span * (1 - q)

    biggest = max(int(b.get("count") or 0) for b in bins) or 1
    parts = [
        f'<svg class="chart cal" viewBox="0 0 {size} {size}" role="img" '
        'aria-label="Calibration: predicted probability against observed hit rate">'
    ]
    for k in range(6):
        t = k / 5
        gx, _ = pos(t, 0)
        _, gy = pos(0, t)
        parts.append(f'<line class="grid" x1="{gx:.1f}" x2="{gx:.1f}" y1="{m}" y2="{size - m}"/>')
        parts.append(f'<line class="grid" x1="{m}" x2="{size - m}" y1="{gy:.1f}" y2="{gy:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{gx:.1f}" y="{size - m + 14}" text-anchor="middle">'
            f"{t:.1f}</text>"
        )
        parts.append(
            f'<text class="tick" x="{m - 6}" y="{gy + 4:.1f}" text-anchor="end">{t:.1f}</text>'
        )
    x0, y0 = pos(0, 0)
    x1, y1 = pos(1, 1)
    parts.append(f'<line class="diag" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}"/>')
    pts = [pos(float(b["mean_p"]), float(b["hit_rate"])) for b in bins]
    if len(pts) > 1:
        coords = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        parts.append(f'<polyline class="s-stoa line" points="{coords}"/>')
    for (px, py), b in zip(pts, bins, strict=True):
        r = 2.5 + 5.5 * math.sqrt(int(b.get("count") or 0) / biggest)
        parts.append(f'<circle class="s-stoa dot" cx="{px:.1f}" cy="{py:.1f}" r="{r:.1f}"/>')
    parts.append(
        f'<text class="tick" x="{size / 2}" y="{size - 4}" text-anchor="middle">predicted</text>'
    )
    parts.append(
        f'<text class="tick" x="10" y="{size / 2}" text-anchor="middle" '
        f'transform="rotate(-90 10 {size / 2})">hit rate</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _empty(text: str) -> str:
    return f'<p class="empty">{_e(text)}</p>'


def _badge(mode: str) -> str:
    cls = {"paper": "b-paper", "live": "b-live"}.get(mode, "b-other")
    return f'<span class="badge {cls}">{_e(mode.upper())}</span>'


def _progress(done: int, need: int) -> str:
    frac = min(1.0, done / need) if need > 0 else 1.0
    return (
        f'<div class="bar" role="img" aria-label="{done} of {need}">'
        f'<span style="width:{frac * 100:.0f}%"></span></div>'
    )


def _check(ok: bool | None, yes: str = "done", no: str = "not yet") -> str:
    if ok:
        return f'<span class="ok">{_e(yes)}</span>'
    return f'<span class="muted">{_e(no)}</span>'


def render_curves(data: dict[str, Any]) -> str:
    c = data["curves"]
    if not c["dates"]:
        return _empty("No trading days recorded yet. The first DECIDE run starts the record.")
    lines = [
        _Line(name, SERIES_CLASS.get(name, "s-ref"), vals) for name, vals in c["series"].items()
    ]
    legend = "".join(
        f'<li><span class="key {ln.css}"></span>{_e(ln.name)} '
        f"{_pct((ln.values[-1] or 100) / 100 - 1) if ln.values and ln.values[-1] else ''}</li>"
        for ln in lines
    )
    note = ""
    if c["references_missing"]:
        note = (
            f'<p class="muted small">{_e(", ".join(c["references_missing"]))} not recorded yet; '
            "it is added from the first DECIDE that records it.</p>"
        )
    return (
        f'<ul class="legend">{legend}</ul>{line_chart(c["dates"], lines)}'
        f'<p class="muted small">Growth of 100 from {_e(c["dates"][0])}. Stoa is after its '
        "own cost model, with deposits and withdrawals taken out.</p>" + note
    )


def render_positions(data: dict[str, Any]) -> str:
    rows = data["positions"]
    if not rows:
        return _empty("No open positions.")
    head = "".join(
        f"<th>{h}</th>"
        for h in ("Symbol", "Entry", "Entry price", "Shares", "Value", "Gain", "Stop", "Days held")
    )
    body = []
    for p in rows:
        qty = "n/a" if p["qty"] is None else f"{p['qty']:.4f}".rstrip("0").rstrip(".")
        cells = [
            ("Symbol", f"<strong>{_e(p['symbol'])}</strong>"),
            ("Entry", _e(p["entry_session"])),
            ("Entry price", _money(p["entry_price"])),
            ("Shares", _e(qty)),
            ("Value", _money(p["value"])),
            ("Gain", _pct(p["gain"])),
            ("Stop", _money(p["stop_price"])),
            ("Days held", _e(p["days_held"])),
        ]
        body.append("<tr>" + "".join(f'<td data-label="{k}">{v}</td>' for k, v in cells) + "</tr>")
    table = (
        f'<table class="stack"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'
    )
    outs = data["stop_outs"]
    if outs:
        items = "".join(
            f"<li>{_e(o['symbol'])} stopped out {_e(o['session'])} at {_money(o['fill_price'])} "
            f"(stop {_money(o['stop_price'])})</li>"
            for o in outs
        )
        table += f'<h3>Recent stop-outs</h3><ul class="plain">{items}</ul>'
    return (
        table + '<p class="muted small">Stops are placed at the broker each morning for '
        "positions held from a prior day. They only ever move up.</p>"
    )


def _order_line(o: dict[str, Any]) -> str:
    amount = _money(o["notional"]) if o["notional"] is not None else f"{o['qty'] or 0:g} sh"
    bits = [f"<strong>{_e(o['side'].upper())} {_e(o['symbol'])}</strong> {_e(amount)}"]
    if o["probability"] is not None:
        bits.append(f"p={o['probability']:.2f}")
    bits.append(f"<em>{_e(o['reason'].replace('_', ' '))}</em>")
    line = " ".join(bits)
    if o["thesis"]:
        line += f'<div class="why">{_e(o["thesis"])}</div>'
    return f"<li>{line}</li>"


def render_decisions(data: dict[str, Any]) -> str:
    items = data["decisions"]
    if not items:
        return _empty("No decisions yet. Each DECIDE after the close adds one here.")
    out = []
    for d in items:
        orders = d.get("orders") or []
        exits = d.get("exits") or []
        rej = d.get("rejections") or []
        regime = d.get("regime_bull")
        meta = [f"strategy {_e(d.get('strategy') or 'n/a')}"]
        if regime is not None:
            meta.append("bull regime" if regime else "not a bull regime")
        n_cand = d.get("candidates", 0)
        meta.append(f"{n_cand} candidate{'' if n_cand == 1 else 's'}")
        if d.get("quant_kills"):
            meta.append(f"{d['quant_kills']} filtered out")
        body = ""
        if orders:
            body += '<ul class="plain">' + "".join(_order_line(o) for o in orders) + "</ul>"
        else:
            body += '<p class="muted">No trades planned.</p>'
        if exits:
            body += (
                '<p class="sub">Exit signals</p><ul class="plain">'
                + "".join(
                    f"<li>{_e(e['symbol'])}: {_e(e['reason'].replace('_', ' '))}, "
                    f"{_e(e['detail'])}</li>"
                    for e in exits
                )
                + "</ul>"
            )
        if rej:
            body += (
                '<p class="sub">Held back by the risk engine</p><ul class="plain">'
                + "".join(
                    f"<li>{_e(r['side'])} {_e(r['symbol'])}: {_e(r['reason'].replace('_', ' '))}, "
                    f"{_e(r['detail'])}</li>"
                    for r in rej
                )
                + "</ul>"
            )
        out.append(
            f'<details class="decision"{" open" if d is items[0] else ""}>'
            f"<summary><span>{_e(d['session'])}</span>"
            f'<span class="muted">{len(orders)} order{"s" if len(orders) != 1 else ""}</span>'
            f'</summary><p class="muted small">{" | ".join(meta)}</p>{body}</details>'
        )
    return "".join(out)


def render_trades(data: dict[str, Any], note: str | None = None) -> str:
    rows = data.get("trades") or []
    if not rows:
        return _empty(
            "No trades yet. Every morning pick, fill, skip and closed trade will be listed "
            "here, with its size, its reason and its profit or loss, from the next runs on."
        )
    labels = ("Date", "What", "Symbol", "Side", "Size", "Reason", "P&L")
    body = ""
    for r in rows:
        pnl = _money(r["pnl"]) if r["pnl"] is not None else ""
        cells = (r["date"], r["kind"], r["symbol"], r["side"], r["size"], r["reason"], pnl)
        body += (
            "<tr>"
            + "".join(
                f'<td data-label="{_e(k)}">{_e(v)}</td>' for k, v in zip(labels, cells, strict=True)
            )
            + "</tr>"
        )
    head = "".join(f"<th>{_e(k)}</th>" for k in labels)
    return f'<table class="stack"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>' + (
        f'<p class="muted small">{note}</p>'
        if note
        else '<p class="muted small">Picks are the 9:00 am picks; buys need a tap to be '
        "placed. P&amp;L is realized on closed trades and at the last close on positions "
        "still held. Paper money.</p>"
    )


def _rate(x: float | None) -> str:
    return "" if x is None else f"{x:.0%}"


def _vs(x: float | None) -> str:
    return "" if x is None else f"{x * 100:+.2f}%"


def _shadow_table(labels: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    head = "".join(f"<th>{_e(k)}</th>" for k in labels)
    body = "".join(
        "<tr>"
        + "".join(f'<td data-label="{_e(k)}">{_e(v)}</td>' for k, v in zip(labels, r, strict=True))
        + "</tr>"
        for r in rows
    )
    return f'<table class="stack"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _shadow_col(m: dict[str, Any]) -> str:
    role = "picks model" if m.get("role") == "picks" else "shadow, never trades"
    return f"{m.get('version')} ({role})"


def render_shadow(data: dict[str, Any]) -> str:
    """The shadow scoreboard: every stock the big account's brain scores, graded, one column
    per model (the picks' model and each shadow model), each with its own totals."""
    b = data.get("shadow_board") or {}
    models = [m for m in b.get("models") or [] if isinstance(m, dict)]
    line = float(b.get("line", 0.55))
    note = (
        '<p class="muted small">Every stock the big account&#39;s news brain scores each '
        "morning is locked before the 9:30 open, buys or not, and graded the next morning: in "
        "at 9:45 am, out at 9:45 am a session later, against SPY over the same span, the "
        "trades&#39; own rule. A hit beat SPY after costs. Shadow models score the same "
        "stocks the same morning and are locked and graded the same way, but never trade. "
        "Each column keeps its own totals. Backfilled days, scored after the fact, are kept "
        "apart and never counted in the locked numbers.</p>"
    )
    back = "".join(
        f'<p class="muted small">Backfill for {_e(m["version"])} (scored after the fact, '
        f"inside its training window): {m['backfill']['days']} mornings, above-line average "
        f"vs SPY {_vs(m['backfill'].get('above_avg_vs_spy')) or 'n/a'}, top 10 vs SPY "
        f"{_vs(m['backfill'].get('top_avg_vs_spy')) or 'n/a'}.</p>"
        for m in models
        if (m.get("backfill") or {}).get("days")
    )
    graded = [m for m in models if (m.get("locked") or {}).get("days")]
    if not graded:
        waiting = ", ".join(_shadow_col(m) for m in models)
        return (
            _empty(
                "No locked morning graded yet. Each 9:00 am picks locks the list it scored"
                + (f" ({waiting})" if waiting else "")
                + "; it is graded after 9:45 am the session after."
            )
            + back
            + note
        )
    cols = [_shadow_col(m) for m in models]
    t = [m.get("locked") or {} for m in models]

    def row(label: str, cell: Any) -> tuple[str, ...]:
        return (label, *(cell(x) if x.get("days") else "waiting" for x in t))

    tiles = _shadow_table(
        ("Locked mornings", *cols),
        [
            row("Graded since", lambda x: f"{x['first']} ({x['days']} mornings)"),
            row("Stocks scored", lambda x: str(x["scored"])),
            row(f"Above the {line:.2f} line", lambda x: str(x["above"])),
            row("Hit rate above the line", lambda x: _rate(x.get("hit_rate")) or "n/a"),
            row("Above-line average vs SPY", lambda x: _vs(x.get("above_avg_vs_spy")) or "n/a"),
            row("Top 10 vs SPY", lambda x: _vs(x.get("top_avg_vs_spy")) or "n/a"),
            row("All stocks beat SPY after costs", lambda x: _rate(x.get("all_hit_rate")) or "n/a"),
        ],
    )

    def cal_cell(c: dict[str, Any]) -> str:
        if not c["n"]:
            return "0"
        return f"{c['n']}, average p {c['mean_p']:.3f}, {_rate(c['hit_rate'])} beat SPY"

    buckets = [str(x) for x in b.get("buckets") or []]
    by = [{c["bucket"]: c for c in x.get("calibration", [])} for x in t]
    cal = _shadow_table(
        ("Bucket", *cols),
        [(k, *(cal_cell(c[k]) if k in c else "" for c in by)) for k in buckets],
    )
    recent = sorted(
        ((d, m) for m in models for d in (m.get("locked_days") or [])[:10]),
        key=lambda dm: (dm[0]["session"], dm[1].get("role") == "picks"),
        reverse=True,
    )
    days = _shadow_table(
        (
            "Morning",
            "Model",
            "Scored",
            "Above line",
            "Hit rate",
            "Above-line vs SPY",
            "Top 10 vs SPY",
        ),
        [
            (
                d["session"],
                str(m.get("version")),
                str(d["scored"]),
                str(d["above"]),
                _rate(d.get("hit_rate")),
                _vs(d.get("above_avg_vs_spy")),
                _vs(d.get("top_avg_vs_spy")),
            )
            for d, m in recent[:20]
        ],
    )
    return tiles + "<h3>Calibration</h3>" + cal + "<h3>Recent mornings</h3>" + days + back + note


INTRO_TRIAL12 = (
    '<p class="muted small">Trial 12 on its own $100,000 Alpaca paper account, exactly as '
    "registered: the news brain's picks, up to 5 at a time at 20% of the account each, every "
    "other dollar in SPY, sold after one session. Picks are auto-approved. Paper money; kept "
    "apart from the $100 account above.</p>"
)
INTRO_TRIAL13 = (
    '<p class="muted small">Trial 13 on its own $100,000 Alpaca paper account: the news '
    "brain's picks from trial 12, sized by the risk overlay (1% risk a position, 5% cap, "
    "half size and less SPY 10% off the high, no new positions 15% off). Picks are "
    "auto-approved. Paper money; kept apart from the $100 account above.</p>"
)

INTRO_TRIAL14 = (
    '<p class="muted small">Trial 14 on its own $100,000 Alpaca paper account: trial 12 '
    "exactly (the news brain's picks, up to 5 at a time at 20% of the account each, sold "
    "after one session), with every other dollar held as cash instead of SPY, so the account "
    "shows the picks alone. SPY is the line to beat. Picks are auto-approved. Paper money; "
    "kept apart from the $100 account above.</p>"
)
INTROS = {"trial13": INTRO_TRIAL13, "trial14": INTRO_TRIAL14}
INTRO_STOPPED = (
    '<p class="muted small">Stopped by the owner on 2026-10-29: everything held is sold and '
    "nothing is bought until the next trial starts. Its own $100,000 Alpaca paper account; "
    "kept apart from the $100 account above.</p>"
)


def big_paper_title(data: dict[str, Any]) -> str:
    b = data.get("big_paper") or {}
    if b.get("stopped"):
        return "Big paper account (stopped)"
    strategy = str(b.get("strategy") or "trial12")
    return f"Big paper account (trial {strategy.removeprefix('trial')})"


def render_big_paper(data: dict[str, Any]) -> str:
    b = data.get("big_paper") or {}
    c = b.get("curve") or {"dates": [], "series": {}}
    intro = INTRO_STOPPED if b.get("stopped") else INTROS.get(str(b.get("strategy")), INTRO_TRIAL12)
    if not c["dates"]:
        return intro + _empty("Not trading yet. Its first day summary starts this chart.")
    lines = [
        _Line(name, "s-stoa" if name == "Big paper" else "s-spy", vals)
        for name, vals in c["series"].items()
    ]
    legend = "".join(
        f'<li><span class="key {ln.css}"></span>{_e(ln.name)} '
        f"{_pct((ln.values[-1] or 100) / 100 - 1) if ln.values and ln.values[-1] else ''}</li>"
        for ln in lines
    )
    status = " Kill switch ON." if b.get("kill_switch") else ""
    head = (
        f"<p>Equity {_e(_money(b.get('equity')))}.{_e(status)}</p>"
        f'<ul class="legend">{legend}</ul>{line_chart(c["dates"], lines)}'
        f'<p class="muted small">Growth of 100 from {_e(c["dates"][0])}.</p>'
    )
    return (
        intro
        + head
        + "<h3>Trade log</h3>"
        + render_trades(
            {"trades": b.get("trades") or []},
            note="Picks are the 9:00 am picks, auto-approved. P&amp;L is realized on closed trades "
            "and at the last close on positions still held. Paper money.",
        )
    )


def render_exam(data: dict[str, Any]) -> str:
    x = data["exam"]
    holdout_text = (
        ("passed" if x["holdout_passed"] else "failed")
        if x["holdout_scored"]
        else ("sealed, not scored" if x["holdout_sealed"] else "not sealed")
    )
    wf_no = "run, not passed" if x["walkforward_done"] else "not run"
    sessions = (x["paper_sessions"], x["paper_sessions_needed"])
    trades = (x["paper_trades"], x["paper_trades_needed"])
    return (
        '<ol class="exam">'
        f"<li><span>Walk-forward</span>{_check(x['walkforward_passed'], 'passed', wf_no)}</li>"
        f'<li><span>Holdout</span><span class="{"ok" if x["holdout_passed"] else "muted"}">'
        f"{_e(holdout_text)}</span></li>"
        f"<li><span>Paper sessions</span><span>{sessions[0]} of {sessions[1]}</span>"
        f"{_progress(*sessions)}</li>"
        f"<li><span>Paper trades</span><span>{trades[0]} of {trades[1]}</span>"
        f"{_progress(*trades)}</li>"
        f"<li><span>Paper exam</span>{_check(x['paper_passed'], 'passed', 'not yet')}</li>"
        f"<li><span>Live allowed</span>{_check(x['live_allowed'], 'yes', 'no')}</li>"
        "</ol>"
    )


def render_calibration(data: dict[str, Any]) -> str:
    c = data["calibration"]
    if c is None:
        return _empty("No model report yet. The monthly retrain writes one.")
    stats = []
    for label, key, fmt in (
        ("ECE", "ece", "{:.3f}"),
        ("Brier", "brier", "{:.3f}"),
        ("AUC", "auc", "{:.3f}"),
    ):
        if c[key] is not None:
            stats.append(f"{label} {fmt.format(c[key])}")
    verdict = c["walkforward_passed"]
    head = (
        f'<p class="muted small">Report {_e(c["report"])}, model {_e(c["chosen"])}'
        + (f", {c['horizon']} session horizon" if c["horizon"] else "")
        + (f". {', '.join(stats)}" if stats else "")
        + (
            f". Walk-forward {'passed' if verdict else 'did not pass'}"
            if verdict is not None
            else ""
        )
        + ".</p>"
    )
    chart = calibration_chart(c["bins"]) or _empty("The report has no reliability bins.")
    return (
        head
        + chart
        + '<p class="muted small">On the diagonal means the probabilities are honest.</p>'
    )


def render_health(data: dict[str, Any]) -> str:
    run = data["last_run"]
    ck = data["checkin"]
    if ck is None:
        ping = '<span class="muted">no check-in recorded yet</span>'
    elif not ck.get("configured"):
        ping = '<span class="warn">not configured</span>'
    elif ck.get("sent"):
        ping = f'<span class="ok">{_e(ck.get("signal"))} ping sent</span> {_when(ck.get("at"))}'
    else:
        ping = (
            f'<span class="bad">{_e(ck.get("signal"))} ping did not go out</span> '
            f"{_when(ck.get('at'))}"
        )
    ks = data["kill_switch"]
    fb = data["fallback"]
    an = data["analyst"]
    sh = data["shadow"]
    shadow_line = (
        f"{sh['sessions']} of {sh['needed']} sessions scored since {_e(sh['start'])}"
        + (f", last {_e(sh['last'])}" if sh["last"] else "")
        + (" (evaluated)" if sh["evaluated"] else ", shadow only, never traded")
    )
    spend = f"{_money(an['spent'])} of {_money(an['cap'])} in {_e(an['month'])}"
    if an["raises"]:
        spend += f" (raised {_money(sum(r['amount'] for r in an['raises']))} this month)"
    if an["paused"]:
        spend += ' <span class="warn">paused at the cap</span>'
    rows = [
        ("Last run", f"{_when(run['at'])} ({_e(run['job'])})" if run else "never"),
        ("Last DECIDE", _e(data["last_decided"] or "never")),
        ("Last EXECUTE", _e(data["last_executed"] or "never")),
        ("Dead man's switch", ping),
        (
            "Kill switch",
            f'<span class="bad">ON since {_e(ks["since"])}</span>' if ks["active"] else "off",
        ),
        ("Analyst spend", spend),
        ("Analyst forward test", shadow_line),
        (
            "SPY fallback",
            f'<span class="warn">ON: {_e(fb["reason"])}</span>' if fb["active"] else "off",
        ),
    ]
    return '<dl class="kv">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>"


LIVE_POLL_SECONDS = 30
LIVE_CLOSED_POLL_SECONDS = 300
LIVE_COLUMNS = (
    ("Symbol", "symbol"),
    ("Qty", "qty"),
    ("Price", "price"),
    ("Market value", "market_value"),
    ("Today", "today_change"),
    ("Today %", "today_change_pct"),
    ("Total P/L", "total_pl"),
    ("Total P/L %", "total_pl_pct"),
)

LIVE_JS = """
(function(){
var URL=__URL__,EVERY=__EVERY__*1000,CLOSED=__CLOSED__*1000;
var $=function(id){return document.getElementById(id)};
var usd=new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'});
var num=new Intl.NumberFormat('en-US',{maximumFractionDigits:6});
function money(v,signed){if(v==null)return '';var t=usd.format(Math.abs(v));
return signed?(v>0?'+':v<0?'-':'')+t:(v<0?'-'+t:t)}
function pct(v){return v==null?'':(v>0?'+':'')+v.toFixed(2)+'%'}
function tone(el,v){el.className=v>0?'up':v<0?'down':''}
var timer=null,last=null;
function cell(tr,label,text,v){var td=document.createElement('td');
td.setAttribute('data-label',label);
td.textContent=text;if(v!==undefined)tone(td,v);tr.appendChild(td)}
function show(d){
$('live-equity').textContent=money(d.equity);$('live-cash').textContent=money(d.cash);
var c=$('live-change');c.textContent=money(d.today_change,true)+' ('+pct(d.today_change_pct)+')';
tone(c,d.today_change);
var body=$('live-rows');body.textContent='';
d.positions.forEach(function(p){var tr=document.createElement('tr');
cell(tr,'Symbol',p.symbol);cell(tr,'Qty',p.qty==null?'':num.format(p.qty));
cell(tr,'Price',money(p.price));cell(tr,'Market value',money(p.market_value));
cell(tr,'Today',money(p.today_change,true),p.today_change);
cell(tr,'Today %',pct(p.today_change_pct),p.today_change_pct);
cell(tr,'Total P/L',money(p.total_pl,true),p.total_pl);
cell(tr,'Total P/L %',pct(p.total_pl_pct),p.total_pl_pct);body.appendChild(tr)});
$('live-empty').hidden=d.positions.length>0;$('live-table').hidden=d.positions.length===0;
var when=new Date(d.as_of).toLocaleTimeString([],
{hour:'numeric',minute:'2-digit',second:'2-digit'});
$('live-status').textContent=(d.market_open?'Last updated ':'Market closed. Last updated ')+when;
$('live-body').hidden=false}
function load(){
fetch(URL,{cache:'no-store',credentials:'omit'}).then(function(r){
if(!r.ok)throw new Error(r.status);return r.json()}).then(function(d){last=d;show(d);plan()})
.catch(function(){$('live-status').textContent=last?'Live data unavailable right now; '+
'showing the last update.':'Live data unavailable right now. The rest of this page '+
'is from the last committed state.';plan()})}
function plan(){clearTimeout(timer);
if(document.visibilityState!=='visible')return;
timer=setTimeout(load,last&&last.market_open?EVERY:CLOSED)}
document.addEventListener('visibilitychange',function(){
if(document.visibilityState==='visible')load();else clearTimeout(timer)});
load()})();
"""


def render_live(data: dict[str, Any]) -> str:
    """The live paper account, filled in by the browser from the live function."""
    url = str(data.get("live_url") or "")
    if not url:
        return _empty("Live positions are not set up yet (dashboard.live_url is blank).")
    head = "".join(f"<th>{_e(label)}</th>" for label, _ in LIVE_COLUMNS)
    script = (
        LIVE_JS.replace("__URL__", json.dumps(url).replace("</", r"<\/"))
        .replace("__EVERY__", str(LIVE_POLL_SECONDS))
        .replace("__CLOSED__", str(LIVE_CLOSED_POLL_SECONDS))
    )
    return (
        '<p id="live-status" class="small muted">Loading live data...</p>'
        '<div id="live-body" hidden>'
        '<div class="tiles live-tiles">'
        '<div class="tile"><div class="k">Equity (paper)</div>'
        '<div class="v" id="live-equity"></div></div>'
        '<div class="tile"><div class="k">Today</div><div class="v" id="live-change"></div></div>'
        '<div class="tile"><div class="k">Cash</div><div class="v" id="live-cash"></div></div>'
        "</div>"
        f'<table class="stack" id="live-table"><thead><tr>{head}</tr></thead>'
        '<tbody id="live-rows"></tbody></table>'
        '<p class="empty" id="live-empty" hidden>No open positions.</p>'
        "</div>"
        '<noscript><p class="empty">Live data needs JavaScript.</p></noscript>'
        f"<script>{script}</script>"
    )


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#14181f;--muted:#5d6675;--border:#dfe3ea;
--stoa:#2563eb;--spy:#9a6700;--ref:#7c3aed;--ok:#1a7f37;--warn:#9a6700;--bad:#cf222e;
--paper:#0969da;--live:#cf222e;--grid:#e8ebf0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d1117;--card:#161b22;
--fg:#e6edf3;--muted:#8d96a0;--border:#30363d;--stoa:#58a6ff;--spy:#d29922;--ref:#bc8cff;
--ok:#3fb950;--warn:#d29922;--bad:#f85149;--paper:#58a6ff;--live:#f85149;--grid:#21262d}}
:root[data-theme="dark"]{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--muted:#8d96a0;
--border:#30363d;--stoa:#58a6ff;--spy:#d29922;--ref:#bc8cff;--ok:#3fb950;--warn:#d29922;
--bad:#f85149;--paper:#58a6ff;--live:#f85149;--grid:#21262d}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;overflow-wrap:anywhere}
main{max-width:980px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;margin:4px 0 16px}
h1{font-size:22px;margin:0}
h2{font-size:16px;margin:0 0 10px}
h3{font-size:14px;margin:14px 0 6px}
.badge{font-size:12px;font-weight:700;letter-spacing:.06em;padding:3px 8px;border-radius:6px;
color:#fff}
.b-paper{background:var(--paper)}.b-live{background:var(--live)}
.b-other{background:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;
margin-bottom:12px}
.tile,.card{background:var(--card);border:1px solid var(--border);border-radius:10px;
padding:12px}
.tile .k{color:var(--muted);font-size:12px}.tile .v{font-size:20px;font-weight:600}
.grid2{display:grid;grid-template-columns:1fr;gap:12px}
@media (min-width:760px){.grid2{grid-template-columns:1fr 1fr}}
.card{margin-bottom:12px}
.chart{width:100%;height:auto;display:block}
.chart.cal{max-width:320px;margin:0 auto}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .diag{stroke:var(--muted);stroke-dasharray:4 4;stroke-width:1}
.chart .tick{fill:var(--muted);font-size:11px}
.chart .line{fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;
vector-effect:non-scaling-stroke}
.chart .grid{vector-effect:non-scaling-stroke}
.plot{position:relative;height:220px;margin-left:40px}
.plot .chart{width:100%;height:100%}
.yaxis span{position:absolute;left:-40px;width:34px;text-align:right;transform:translateY(-50%);
font-size:11px;color:var(--muted)}
.xaxis{display:flex;justify-content:space-between;margin:4px 0 0 40px;font-size:11px;
color:var(--muted)}
.s-stoa{stroke:var(--stoa)}.s-spy{stroke:var(--spy)}.s-ref{stroke:var(--ref)}
.key.s-stoa{border-color:var(--stoa)}.key.s-spy{border-color:var(--spy)}
.key.s-ref{border-color:var(--ref)}
.dot.s-stoa{fill:var(--stoa)}.dot.s-spy{fill:var(--spy)}.dot.s-ref{fill:var(--ref)}
.legend{list-style:none;margin:0 0 8px;padding:0;display:flex;flex-wrap:wrap;gap:4px 14px;
font-size:13px}
.key{display:inline-block;width:14px;height:3px;border-top:3px solid;margin-right:6px;
vertical-align:middle}
.muted{color:var(--muted)}.small{font-size:12px}.ok{color:var(--ok)}.warn{color:var(--warn)}
.bad{color:var(--bad)}
.empty{color:var(--muted);margin:4px 0}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-weight:600;font-size:12px}
@media (max-width:600px){
table.stack thead{display:none}
table.stack tr{display:block;border:1px solid var(--border);border-radius:8px;margin-bottom:8px;
padding:4px 8px}
table.stack td{display:flex;justify-content:space-between;gap:12px;border:0;padding:3px 0}
table.stack td::before{content:attr(data-label);color:var(--muted);font-size:12px}}
.plain{list-style:none;margin:6px 0;padding:0}.plain li{margin:4px 0}
.why{color:var(--muted);font-size:13px;margin:2px 0 0 0}
.sub{font-size:12px;color:var(--muted);margin:10px 0 0}
details.decision{border-top:1px solid var(--border);padding:8px 0}
details.decision:first-of-type{border-top:0}
details.decision summary{cursor:pointer;display:flex;justify-content:space-between;gap:12px;
font-weight:600}
.exam{list-style:none;margin:0;padding:0}
.exam li{display:grid;grid-template-columns:1fr auto;gap:2px 12px;padding:6px 0;
border-bottom:1px solid var(--border)}
.exam li:last-child{border-bottom:0}
.bar{grid-column:1/-1;height:6px;background:var(--grid);border-radius:3px;overflow:hidden}
.bar span{display:block;height:100%;background:var(--stoa)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;margin:0}
.kv dt{color:var(--muted)}.kv dd{margin:0}
footer{color:var(--muted);font-size:12px;margin:8px 0 24px}
footer a{color:inherit}
.up{color:var(--ok)}.down{color:var(--bad)}
.live-tiles{margin:8px 0 10px}
"""


def render(data: dict[str, Any]) -> str:
    mode = str(data["mode"])
    eq = data["equity"]
    curves = data["curves"]
    since = vs_spy = None
    series = curves["series"]
    if curves["dates"] and series.get("Stoa"):
        since = (series["Stoa"][-1] or 100) / 100 - 1
        spy_last = (series.get("SPY") or [None])[-1]
        vs_spy = since - ((spy_last or 100) / 100 - 1) if spy_last else None
    tiles = [
        ("Equity", _money(eq["broker"])),
        ("Since start", _pct(since)),
        ("Versus SPY", _pct(vs_spy)),
        ("Open positions", str(len(data["positions"]))),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{_e(v)}</div></div>'
        for k, v in tiles
    )
    as_of = f"as of {_e(eq['session'])}" if eq["session"] else "no trading days yet"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Stoa Dashboard</title>"
        '<meta name="description" content="Read-only status of the Stoa trading system.">'
        f"<style>{CSS}</style></head><body><main>"
        f'<header><h1>Stoa</h1>{_badge(mode)}<span class="muted">{as_of}</span></header>'
        f'<section class="card"><h2>Live</h2>{render_live(data)}</section>'
        f'<section class="tiles">{tiles_html}</section>'
        f'<section class="card"><h2>Equity vs SPY and MTUM</h2>{render_curves(data)}</section>'
        f'<section class="card"><h2>Positions</h2>{render_positions(data)}</section>'
        '<div class="grid2">'
        f'<section class="card"><h2>Exam progress toward live</h2>{render_exam(data)}</section>'
        f'<section class="card"><h2>Runs and checks</h2>{render_health(data)}</section>'
        "</div>"
        f'<section class="card"><h2>Trade log</h2>{render_trades(data)}</section>'
        f'<section class="card"><h2>{_e(big_paper_title(data))}</h2>'
        f"{render_big_paper(data)}</section>"
        f'<section class="card"><h2>Shadow scoreboard</h2>{render_shadow(data)}</section>'
        f'<section class="card"><h2>Last {DECISIONS_SHOWN} decisions</h2>'
        f"{render_decisions(data)}</section>"
        f'<section class="card"><h2>Model calibration</h2>{render_calibration(data)}</section>'
        f"<footer>Read only: nothing is traded from this page. Built "
        f"{_when(data['generated_at'])} by Stoa {_e(data['version'])} from committed state. "
        '<a href="https://github.com/AddisonTech/Stoa">Source</a></footer>'
        "</main></body></html>"
    )


def _json_default(obj: object) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return str(obj)


def build(
    settings: Settings,
    out: Path,
    repo: Path,
    *,
    results: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Write ``out/index.html`` and ``out/data.json``. Returns the index path."""
    data = collect(settings, repo, results=results, now=now)
    out.mkdir(parents=True, exist_ok=True)
    page = scrub(render(data))
    blob = scrub(json.dumps(data, indent=1, default=_json_default, allow_nan=False))
    (out / "data.json").write_text(blob + "\n", encoding="utf-8")
    index = out / "index.html"
    index.write_text(page + "\n", encoding="utf-8")
    return index
