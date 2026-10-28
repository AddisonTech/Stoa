"""Counts by year and coverage for each pull, as markdown."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

from stoa.newsbrain import ROOT
from stoa.newsbrain.alpaca_pull import MINUTE_STORE_DIR, NEWS_DIR, universe
from stoa.newsbrain.sec_pull import SEC_DIR


def news_by_year() -> pd.DataFrame:
    articles: dict[int, int] = defaultdict(int)
    symbols: dict[int, set[str]] = defaultdict(set)
    for path in sorted(NEWS_DIR.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["symbols"])
        year = int(path.stem[:4])
        articles[year] += len(df)
        for syms in df["symbols"]:
            symbols[year].update(syms)
    wanted = set(universe())
    return pd.DataFrame(
        [
            {"year": y, "articles": n, "universe symbols with news": len(symbols[y] & wanted)}
            for y, n in sorted(articles.items())
        ]
    )


def minute_by_year() -> pd.DataFrame:
    months: dict[int, set[str]] = defaultdict(set)
    bars: dict[int, int] = defaultdict(int)
    symbols: dict[int, set[str]] = defaultdict(set)
    for path in sorted(MINUTE_STORE_DIR.glob("*/*.parquet")):
        year = int(path.parent.name[:4])
        months[year].add(path.parent.name)
        bars[year] += pq.read_metadata(path).num_rows
        symbols[year].update(pd.read_parquet(path, columns=["symbol"])["symbol"].unique())
    return pd.DataFrame(
        [
            {"year": y, "months": len(months[y]), "bars": bars[y], "symbols": len(symbols[y])}
            for y in sorted(bars)
        ]
    )


def sec_by_year() -> pd.DataFrame:
    parts = []
    for name in ("earnings", "tenk", "eightk"):
        for path in sorted((SEC_DIR / name).glob("*.parquet")):
            df = pd.read_parquet(path, columns=["cik", "filed", "kind", "text"])
            if len(df):
                df["pass"] = name
                df["has_text"] = df["text"].str.len() > 0
                parts.append(df.drop(columns="text"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["year"] = df["filed"].astype(str).str[:4]
    return (
        df.groupby(["pass", "year"])
        .agg(
            documents=("kind", "size"), with_text=("has_text", "sum"), companies=("cik", "nunique")
        )
        .reset_index()
    )


def markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        cells = [
            f"{v:,}"
            if isinstance(v, int | float) and not isinstance(v, bool) and c != "year"
            else str(v)
            for c, v in zip(cols, row, strict=True)
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report() -> str:
    out = ["# News brain data coverage", ""]
    for title, fn in (
        ("News", news_by_year),
        ("Minute bars", minute_by_year),
        ("SEC text", sec_by_year),
    ):
        df = fn()
        out += [f"## {title}", "", markdown(df) if len(df) else "(nothing yet)", ""]
    ciks = SEC_DIR / "ciks.parquet"
    if ciks.exists():
        c = pd.read_parquet(ciks)
        out.append(
            f"Universe symbols with an SEC CIK: {c['symbol'].nunique()} of {len(universe())} "
            f"({c['cik'].nunique()} companies)."
        )
    text = "\n".join(out)
    (ROOT / "coverage.md").write_text(text + "\n", encoding="utf-8")
    return text
