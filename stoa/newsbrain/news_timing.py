"""When news lands, and which hours' news moves prices (docs/DECISIONS.md, news timing).

Analysis only, on 2016 to 2022 (the training years): it reads nothing from 2023 on,
so trial 11's test window stays clean, and never the holdout. For each article on a
liquid name of trial 9's universe (at most 3 tickers): its hour of day in New York
time, weekday or weekend, the company's size, and its move against SPY from the
last trade before the article to the close 1 and 5 sessions later.

Writes docs/NEWS_TIMING.md with SVG charts in docs/news_timing/.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stoa import clock
from stoa.newsbrain import ROOT, sealed
from stoa.newsbrain import panel as pn
from stoa.newsbrain import trial as t9
from stoa.newsbrain.alpaca_pull import MINUTE_DIR
from stoa.newsbrain.premarket import SharesOutstanding

log = logging.getLogger(__name__)

START = date(2016, 1, 1)
END = date(2022, 12, 31)
CAPS = ("small", "mid", "large")
HOURS = tuple(range(24))
LOOKBACK = pd.Timedelta(days=4)  # the last trade before an article, within 4 days
DOC = Path("docs/NEWS_TIMING.md")
FIG_DIR = Path("docs/news_timing")
OUT = ROOT / "news_timing"

# Chart styling: the validated categorical slots 1 to 3 (small, mid, large) on a light surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
COLOR = {"small": "#2a78d6", "mid": "#eb6834", "large": "#1baf7a"}


# ---- the data ------------------------------------------------------------------------------


def last_price_before(minutes: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    """The close of each event's symbol's last minute bar at or before its time, within 4 days.

    ``minutes``: symbol, t, c. ``events``: symbol, at (both tz-aware UTC).
    """
    if events.empty:
        return np.array([], dtype=float)
    ev = events[["symbol", "at"]].reset_index(names="i").sort_values("at")
    mb = minutes[["symbol", "t", "c"]].sort_values("t")
    got = pd.merge_asof(
        ev, mb, left_on="at", right_on="t", by="symbol", direction="backward", tolerance=LOOKBACK
    )
    out = np.full(len(events), np.nan)
    out[got["i"].to_numpy()] = got["c"].to_numpy(dtype=float)
    return out


def month_minutes(stamp: str, symbols: set[str], root: Path = MINUTE_DIR) -> pd.DataFrame:
    parts = []
    want = sorted(symbols)
    for path in sorted((root / stamp).glob("*.parquet")):
        if pq.ParquetFile(path).metadata.num_rows == 0:
            continue  # an empty group file: no bars that month, and no text type to filter
        df = pd.read_parquet(path, columns=["symbol", "t", "c"], filters=[("symbol", "in", want)])
        if len(df):
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["symbol", "t", "c"])
    df = pd.concat(parts, ignore_index=True)
    df["symbol"] = df["symbol"].astype(str)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df


def events(root: Path = MINUTE_DIR) -> pd.DataFrame:
    """One row per (article, symbol) from 2016 to 2022 with hour, day type, size and moves."""
    days, cut = pn.calendar(pn.PANEL_START, pn.PANEL_END)
    days_2022 = [d for d in days if d <= END]
    sealed.check(END, "the news timing study")
    news = pn.load_news(cut)
    news["at"] = pd.to_datetime(news["known_at"], utc=True)
    ny = news["at"].dt.tz_convert(clock.NY)
    keep = (ny.dt.date >= START) & (ny.dt.date <= END)
    news = news[keep].copy()
    panel = pd.read_parquet(pn.PANEL, columns=["symbol", "sidx", "adv", "prev_close"])
    news = news.merge(panel, on=["symbol", "sidx"], how="inner")  # liquid names only
    ny = news["at"].dt.tz_convert(clock.NY)
    news["hour"] = ny.dt.hour
    news["weekend"] = ny.dt.weekday >= 5
    ciks = pn.load_ciks()
    cik_of = dict(zip(ciks["symbol"], ciks["cik"], strict=True))
    shares = SharesOutstanding()
    buckets = []
    for sym, sess, pc, a in zip(
        news["symbol"], news["sidx"], news["prev_close"], news["adv"], strict=True
    ):
        sh = shares(int(cik_of[sym]), days[int(sess)]) if sym in cik_of else None
        buckets.append(t9.cap_bucket(sh * pc if sh and math.isfinite(pc) else None, a))
    news["bucket"] = buckets
    log.info("%d article-symbol rows, 2016 to 2022", len(news))

    closes_at = np.array([pd.Timestamp(clock.at_ny(d, "16:00")).value for d in days_2022])
    news["e"] = np.searchsorted(closes_at, pn.nanos(news["at"]), side="left")
    news = news[news["e"] + 4 < len(days_2022)].reset_index(drop=True)

    # Prices at the article, month by month (with the previous month for early-month articles).
    p0 = np.full(len(news), np.nan)
    s0 = np.full(len(news), np.nan)
    stamps = news["at"].dt.tz_convert(clock.NY).dt.strftime("%Y-%m")
    prev: pd.DataFrame | None = None
    for stamp in sorted(stamps.unique()):
        idx = np.flatnonzero((stamps == stamp).to_numpy())
        syms = set(news.loc[idx, "symbol"]) | {"SPY"}
        cur = month_minutes(stamp, syms, root)
        both = cur if prev is None else pd.concat([prev, cur], ignore_index=True)
        ev = news.loc[idx, ["symbol", "at"]].reset_index(drop=True)
        p0[idx] = last_price_before(both, ev)
        s0[idx] = last_price_before(both, ev.assign(symbol="SPY"))
        tail_from = cur["t"].max() - LOOKBACK if len(cur) else None
        prev = cur[cur["t"] >= tail_from] if tail_from is not None else None
        log.info("news timing prices %s: %d articles", stamp, len(idx))
    news["p0"], news["s0"] = p0, s0

    spy = pn.bar_frame("SPY", days_2022)
    assert spy is not None
    spy_close = spy["close"].ffill().to_numpy(dtype=float)
    frames: dict[str, Any] = {}
    for h in (1, 5):
        out = np.full(len(news), np.nan)
        for sym, g in news.groupby("symbol"):
            if sym not in frames:
                df = pn.bar_frame(str(sym), days_2022)
                frames[str(sym)] = None if df is None else df["close"].ffill().to_numpy(dtype=float)
            c = frames[str(sym)]
            if c is None:
                continue
            j = g["e"].to_numpy() + h - 1
            stock = c[j] / g["p0"].to_numpy() - 1
            bench = spy_close[j] / g["s0"].to_numpy() - 1
            out[g.index.to_numpy()] = stock - bench
        news[f"excess_{h}"] = out
    cols = ["symbol", "at", "hour", "weekend", "bucket", "excess_1", "excess_5"]
    return pd.DataFrame(news[cols])


# ---- the tables -----------------------------------------------------------------------------


def by_hour(ev: pd.DataFrame) -> pd.DataFrame:
    """Per (bucket, day type, hour): articles, their share of the bucket's day type, and
    the share of the bucket's total absolute move (1 and 5 sessions) after that hour's news."""
    ev = ev.assign(day=np.where(ev["weekend"], "weekend", "weekday"))
    ev = ev.assign(abs_1=ev["excess_1"].abs(), abs_5=ev["excess_5"].abs())
    rows = []
    for b in (*CAPS, "all"):
        e = ev if b == "all" else ev[ev["bucket"] == b]
        tot1, tot5 = np.nansum(e["abs_1"]), np.nansum(e["abs_5"])
        for day in ("weekday", "weekend"):
            d = e[e["day"] == day]
            for hr in HOURS:
                x = d[d["hour"] == hr]
                rows.append(
                    {
                        "bucket": b,
                        "day": day,
                        "hour": hr,
                        "articles": len(x),
                        "article_share": len(x) / max(len(d), 1),
                        "move_share_1": float(np.nansum(x["abs_1"]) / tot1) if tot1 else 0.0,
                        "move_share_5": float(np.nansum(x["abs_5"]) / tot5) if tot5 else 0.0,
                        "mean_abs_5": float(np.nanmean(x["abs_5"])) if len(x) else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def best_windows(share: np.ndarray) -> list[dict[str, Any]]:
    """For k = 1 to 24 hours: the contiguous window (wrapping past midnight) with the most share."""
    n = len(share)
    doubled = np.concatenate([share, share])
    out = []
    for k in range(1, n + 1):
        sums = [doubled[s : s + k].sum() for s in range(n)]
        s = int(np.argmax(sums))
        out.append({"hours": k, "start": s, "end": (s + k) % n, "share": float(sums[s])})
    return out


def hhmm(h: int) -> str:
    h %= 24
    return f"{h % 12 or 12}:00 {'am' if h < 12 else 'pm'}"


def window_text(w: dict[str, Any]) -> str:
    return f"{hhmm(w['start'])} to {hhmm(w['end'])}"


def recommend(tab: pd.DataFrame) -> dict[str, Any]:
    """Scan windows that catch most of the 5 session move, weekday and weekend news together."""
    out: dict[str, Any] = {}
    for b in (*CAPS, "all"):
        t = tab[tab["bucket"] == b]
        share = t.groupby("hour")["move_share_5"].sum().reindex(HOURS, fill_value=0.0).to_numpy()
        wins = best_windows(share)
        need = {pct: next(w for w in wins if w["share"] >= pct / 100) for pct in (50, 80, 90)}
        out[b] = {
            "windows": wins,
            "for": {str(p): {**w, "text": window_text(w)} for p, w in need.items()},
            "weekend_share": float(t[t["day"] == "weekend"]["move_share_5"].sum()),
        }
    return out


# ---- charts (hand-written SVG) ----------------------------------------------------------------


def _svg(width: int, height: int, body: list[str], title: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}" '
            'font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">',
            f"<title>{escape(title)}</title>",
            f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
            *body,
            "</svg>",
        ]
    )


def _hour_axis(x0: float, x1: float, y: float, step: float) -> list[str]:
    out = []
    for hr in (0, 4, 8, 12, 16, 20):
        x = x0 + (hr + 0.5) * step
        out.append(
            f'<text x="{x:.1f}" y="{y + 14:.1f}" font-size="11" fill="{INK_2}" '
            f'text-anchor="middle">{hhmm(hr).replace(":00", "")}</text>'
        )
    out.append(f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{INK_2}"/>')
    return out


def bars_grid(tab: pd.DataFrame, path: Path) -> None:
    """Share of each size's articles by hour: weekdays on top, weekends below."""
    w, h = 960, 470
    pw, ph, left, top, gap = 280, 150, 60, 64, 36
    ymax = float(tab[tab["bucket"].isin(CAPS)]["article_share"].max()) or 1.0
    ymax = math.ceil(ymax * 100 / 2) * 2 / 100
    body = [
        f'<text x="{left}" y="28" font-size="16" font-weight="600" fill="{INK}">'
        "When news lands, by hour (New York time), 2016 to 2022</text>",
        f'<text x="{left}" y="46" font-size="12" fill="{INK_2}">Share of each size\'s articles '
        "in each hour. Weekdays on top, weekends below.</text>",
    ]
    for r, day in enumerate(("weekday", "weekend")):
        for c, b in enumerate(CAPS):
            x0 = left + c * (pw + gap)
            y0 = top + r * (ph + 70)
            base = y0 + ph
            step = pw / 24
            t = tab[(tab["bucket"] == b) & (tab["day"] == day)].sort_values("hour")
            n = int(t["articles"].sum())
            body.append(
                f'<text x="{x0}" y="{y0 - 8}" font-size="12" font-weight="600" fill="{INK}">'
                f"{b.capitalize()} caps, {day}s ({n:,} articles)</text>"
            )
            for frac in (0.5, 1.0):
                yy = base - frac * ph
                body.append(
                    f'<line x1="{x0}" x2="{x0 + pw}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{GRID}"/>'
                )
                if c == 0:
                    body.append(
                        f'<text x="{x0 - 6}" y="{yy + 4:.1f}" font-size="11" fill="{INK_2}" '
                        f'text-anchor="end">{frac * ymax:.0%}</text>'
                    )
            row: Any
            for row in t.itertuples():
                bh = float(row.article_share) / ymax * ph
                bx = x0 + int(row.hour) * step + 1
                body.append(
                    f'<rect x="{bx:.1f}" y="{base - bh:.1f}" width="{step - 2:.1f}" '
                    f'height="{max(bh, 0):.1f}" rx="2" fill="{COLOR[b]}">'
                    f"<title>{b} {day} {hhmm(int(row.hour))}: {int(row.articles):,} articles "
                    f"({float(row.article_share):.1%})</title></rect>"
                )
            body += _hour_axis(x0, x0 + pw, base, step)
    path.write_text(_svg(w, h, body, "When news lands, by hour"), encoding="utf-8", newline="\n")


def lines(
    series: dict[str, np.ndarray],
    path: Path,
    title: str,
    sub: str,
    x_label: str,
    xs: np.ndarray,
    x_fmt: Any,
    y_fmt: Any,
    ymax: float,
) -> None:
    """One chart, one axis, one series per size, labelled at the line's end."""
    w, h = 760, 380
    left, top, pw, ph = 70, 70, 560, 240
    x_min, x_max = float(xs.min()), float(xs.max())

    def px(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * pw

    def py(y: float) -> float:
        return top + ph - y / ymax * ph

    body = [
        f'<text x="{left}" y="28" font-size="16" font-weight="600" fill="{INK}">'
        f"{escape(title)}</text>",
        f'<text x="{left}" y="46" font-size="12" fill="{INK_2}">{escape(sub)}</text>',
    ]
    for k in range(5):
        yv = ymax * k / 4
        body.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{py(yv):.1f}" y2="{py(yv):.1f}" '
            f'stroke="{GRID if k else INK_2}"/>'
        )
        body.append(
            f'<text x="{left - 8}" y="{py(yv) + 4:.1f}" font-size="11" fill="{INK_2}" '
            f'text-anchor="end">{y_fmt(yv)}</text>'
        )
    ticks = xs if len(xs) <= 8 else xs[:: max(1, len(xs) // 6)]
    for xv in ticks:
        body.append(
            f'<text x="{px(float(xv)):.1f}" y="{top + ph + 18}" font-size="11" fill="{INK_2}" '
            f'text-anchor="middle">{x_fmt(xv)}</text>'
        )
    body.append(
        f'<text x="{left + pw / 2}" y="{top + ph + 38}" font-size="12" fill="{INK_2}" '
        f'text-anchor="middle">{escape(x_label)}</text>'
    )
    ends = []
    for name, ys in series.items():
        pts = " ".join(
            f"{px(float(x)):.1f},{py(float(y)):.1f}" for x, y in zip(xs, ys, strict=True)
        )
        body.append(
            f'<polyline points="{pts}" fill="none" stroke="{COLOR[name]}" stroke-width="2" '
            'stroke-linejoin="round"/>'
        )
        for x, y in zip(xs, ys, strict=True):
            body.append(
                f'<circle cx="{px(float(x)):.1f}" cy="{py(float(y)):.1f}" r="7" fill="transparent">'
                f"<title>{name}: {x_fmt(x)}, {y_fmt(float(y))}</title></circle>"
            )
        ends.append((py(float(ys[-1])), name, float(ys[-1])))
    # Direct labels at the right end, nudged apart so they never overlap.
    ends.sort()
    last = -1e9
    for y_end, name, val in ends:
        y = max(y_end, last + 15)
        last = y
        x_end = left + pw
        body.append(
            f'<circle cx="{x_end:.1f}" cy="{py(val):.1f}" r="4" fill="{COLOR[name]}" '
            f'stroke="{SURFACE}" stroke-width="2"/>'
        )
        body.append(
            f'<text x="{x_end + 10:.1f}" y="{y + 4:.1f}" font-size="12" fill="{INK}">'
            f"{name.capitalize()} caps</text>"
        )
    path.write_text(_svg(w, h, body, title), encoding="utf-8", newline="\n")


# ---- the report -------------------------------------------------------------------------------


def pct(x: float) -> str:
    return f"{x:.1%}"


def markdown(tab: pd.DataFrame, rec: dict[str, Any], n: int) -> str:
    allb = rec["all"]
    lines_md = [
        "# News timing",
        "",
        "When company news lands, and which hours' news is followed by the biggest moves. "
        "Analysis only, beside trial 11 (docs/DECISIONS.md); it changes none of trial 11's rules.",
        "",
        f"**Data:** {n:,} article and company pairs from 2016 to 2022, the training years, so "
        "the 2023 to 2026 test window stays clean; the holdout is never read. Alpaca (Benzinga) "
        "news tagging at most 3 tickers, on names in trial 9's liquid universe. Times are New York "
        "time, by when an article was known (its update time). Weekends are Saturday and Sunday; "
        "market holidays count as weekdays.",
        "",
        "**Move:** the company's return against SPY from the last trade before the article "
        "(pre-market and after-hours included) to the close 1 and 5 sessions later, counting the "
        "article's own session as the first if it lands before 4:00 pm. A size bucket's total "
        "move is the sum of these absolute moves; each hour's share is the part that came after "
        "that hour's news. Sizes as trial 9: small under $2 billion, mid $2 to $10 billion, "
        "large above.",
        "",
        "## When news lands",
        "",
        "![Share of articles by hour, weekday and weekend, by size](news_timing/news_by_hour.svg)",
        "",
        "## Which hours' news moves prices",
        "",
        "![Share of the 1 session move by hour of the news, by size](news_timing/move_share_1.svg)",
        "",
        "![Share of the 5 session move by hour of the news, by size](news_timing/move_share_5.svg)",
        "",
        "| Hour (ET) | Articles | Share of 1 session move | Share of 5 session move | "
        "Small: share of 5 session move | Mid | Large |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for hr in HOURS:

        def g(b: str, col: str, h: int = hr) -> float:
            t = tab[(tab["bucket"] == b) & (tab["hour"] == h)]
            return float(t[col].sum())

        lines_md.append(
            f"| {hhmm(hr)} | {int(g('all', 'articles')):,} | {pct(g('all', 'move_share_1'))} | "
            f"{pct(g('all', 'move_share_5'))} | {pct(g('small', 'move_share_5'))} | "
            f"{pct(g('mid', 'move_share_5'))} | {pct(g('large', 'move_share_5'))} |"
        )
    lines_md += [
        "",
        "Weekend news (Saturday and Sunday) is "
        + ", ".join(f"{pct(rec[b]['weekend_share'])} of the {b}-cap 5 session move" for b in CAPS)
        + ".",
        "",
        "## Scan windows for a live news watcher",
        "",
        "![Share of the 5 session move caught by the best window of each length, by size]"
        "(news_timing/coverage.svg)",
        "",
        "The best single window of each length, from weekday and weekend news together (a "
        "weekend article is caught the next time the window opens):",
        "",
        "| Catch | All sizes | Small caps | Mid caps | Large caps |",
        "| --- | --- | --- | --- | --- |",
    ]
    for p in ("50", "80", "90"):
        cells = [
            f"{rec[b]['for'][p]['hours']} h, {rec[b]['for'][p]['text']}" for b in ("all", *CAPS)
        ]
        lines_md.append(f"| {p}% of the move | " + " | ".join(cells) + " |")
    lines_md += [
        "",
        "What each extra hour of scanning adds (all sizes, best window of each length):",
        "",
        "| Hours scanned | Window (ET) | Share caught | Added by the last hour |",
        "| --- | --- | --- | --- |",
    ]
    prev = 0.0
    for w in allb["windows"][:16]:
        lines_md.append(
            f"| {w['hours']} | {window_text(w)} | {pct(w['share'])} | +{pct(w['share'] - prev)} |"
        )
        prev = w["share"]
    lines_md += [
        "",
        "## Recommendation",
        "",
        recommendation_text(rec),
        "",
        "## Limits",
        "",
        "- An article's known time can lag the event; some moves started before the article.",
        "- The move is measured to the close, so it includes whatever else happened that day; "
        "the share by hour is a guide to where news-driven moves cluster, not a causal estimate.",
        "- Prices are split and dividend adjusted; the last trade before an article can be up to "
        "4 days earlier for thinly traded names.",
        "- 2016 to 2022 only. Later years may differ.",
        "",
    ]
    return "\n".join(lines_md)


def recommendation_text(rec: dict[str, Any]) -> str:
    a, s = rec["all"]["for"], rec["small"]["for"]
    return (
        f"Scan {a['80']['text']} ET ({a['80']['hours']} hours) to catch about 80% of the 5 "
        f"session move behind all sizes' news, or {a['90']['text']} ({a['90']['hours']} hours) "
        f"for 90%. For small caps alone: {s['80']['text']} ({s['80']['hours']} hours) for 80%."
    )


def alert_line(rec: dict[str, Any]) -> str:
    a, s = rec["all"]["for"], rec["small"]["for"]
    return (
        f"News scan windows (2016 to 2022): {a['80']['text']} ET catches 80% of the 5 day move "
        f"({a['80']['hours']} h), {a['90']['text']} catches 90% ({a['90']['hours']} h); small caps "
        f"80%: {s['80']['text']} ({s['80']['hours']} h)."
    )


def run(root: Path = MINUTE_DIR, repo: Path = Path(".")) -> dict[str, Any]:
    ev = events(root)
    OUT.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(OUT / "events.parquet", index=False)
    tab = by_hour(ev)
    rec = recommend(tab)
    figs = repo / FIG_DIR
    figs.mkdir(parents=True, exist_ok=True)
    bars_grid(tab, figs / "news_by_hour.svg")
    hours = np.array(HOURS)
    for hzn in (1, 5):
        series = {
            b: tab[tab["bucket"] == b]
            .groupby("hour")[f"move_share_{hzn}"]
            .sum()
            .reindex(HOURS, fill_value=0.0)
            .to_numpy()
            for b in CAPS
        }
        top = max(float(v.max()) for v in series.values())
        lines(
            series,
            figs / f"move_share_{hzn}.svg",
            f"Share of the {hzn} session move by hour of the news, 2016 to 2022",
            "Absolute move against SPY after each hour's news, as a share of the size's total.",
            "Hour the article was known (New York time)",
            hours,
            lambda x: hhmm(int(x)).replace(":00", ""),
            lambda y: f"{y:.0%}",
            max(0.02, math.ceil(top * 50) / 50),
        )
    ks = np.arange(1, 25)
    lines(
        {b: np.array([w["share"] for w in rec[b]["windows"]]) for b in CAPS},
        figs / "coverage.svg",
        "Share of the 5 session move caught, by hours scanned",
        "The best single window of each length, weekday and weekend news together.",
        "Hours scanned a day",
        ks,
        lambda x: str(int(x)),
        lambda y: f"{y:.0%}",
        1.0,
    )
    (repo / DOC).write_text(markdown(tab, rec, len(ev)), encoding="utf-8", newline="\n")
    summary = {
        "articles": len(ev),
        "recommend": {b: rec[b]["for"] for b in rec},
        "alert": alert_line(rec),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    tab.to_csv(OUT / "by_hour.csv", index=False)
    return summary
