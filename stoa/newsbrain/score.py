"""Score every article and filing with FinBERT on the GPU, as the pulls land.

News: headline plus summary, 128 tokens, the same text and truncation the
9:00 am sweep uses. Filings: the first section (the first substantial
paragraphs, about 1,500 characters), 256 tokens. 10-K customer paragraphs are
for the supplier and customer map and are not scored.

One score file per input file; an input whose score file exists is done. The
loop waits for new files until every pull has logged that it is finished.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from stoa.newsbrain import ROOT
from stoa.newsbrain.alpaca_pull import NEWS_DIR
from stoa.newsbrain.pace import log_progress, write_parquet
from stoa.newsbrain.sec_pull import SEC_DIR
from stoa.reader import LABELS, FinBert, Reader, text_of

log = logging.getLogger(__name__)

SCORE_DIR = ROOT / "scores"
MODEL = "ProsusAI/finbert"
REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
SECTION_CHARS = 1500


def first_section(text: str, limit: int = SECTION_CHARS) -> str:
    """The opening of a filing: its first substantial paragraphs, without the cover lines."""
    keep: list[str] = []
    size = 0
    for para in (text or "").split("\n"):
        p = para.strip()
        if len(p) < 40:
            continue
        keep.append(p)
        size += len(p)
        if size >= limit:
            break
    return " ".join(keep)[:limit]


def score_frame(reader: Reader, texts: list[str]) -> pd.DataFrame:
    probs = reader.score(texts)
    return pd.DataFrame(probs, columns=list(LABELS))


def score_news_file(reader: Reader, src: Path, dst: Path) -> int:
    df = pd.read_parquet(src, columns=["id", "headline", "summary"])
    texts = [text_of(h, s) for h, s in zip(df["headline"], df["summary"], strict=True)]
    out = score_frame(reader, texts)
    out.insert(0, "id", df["id"].to_numpy())
    write_parquet(out, dst)
    return len(out)


def score_sec_file(reader: Reader, src: Path, dst: Path) -> int:
    df = pd.read_parquet(src, columns=["accn", "kind", "doc", "text"])
    texts = [first_section(t) for t in df["text"]]
    out = score_frame(reader, texts) if texts else pd.DataFrame(columns=list(LABELS))
    for i, col in enumerate(("accn", "kind", "doc")):
        out.insert(i, col, df[col].to_numpy())
    write_parquet(out, dst)
    return len(out)


def pending() -> list[tuple[str, Path, Path]]:
    jobs: list[tuple[str, Path, Path]] = []
    for src in sorted(NEWS_DIR.glob("*.parquet")):
        dst = SCORE_DIR / "news" / src.name
        if not dst.exists():
            jobs.append(("news", src, dst))
    for name in ("earnings", "eightk"):
        for src in sorted((SEC_DIR / name).glob("*.parquet")):
            dst = SCORE_DIR / "sec" / name / src.name
            if not dst.exists():
                jobs.append(("sec", src, dst))
    return jobs


def pulls_finished() -> bool:
    for job in ("news", "sec"):
        path = ROOT / "logs" / f"{job}.jsonl"
        if not path.exists():
            return False
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines or not json.loads(lines[-1]).get("done"):
            return False
    return True


def score_forever(batch_size: int = 256) -> None:
    news_reader = FinBert(MODEL, REVISION, device="cuda", batch_size=batch_size, max_length=128)
    sec_reader = FinBert(MODEL, REVISION, device="cuda", batch_size=batch_size // 2, max_length=256)
    progress = ROOT / "logs" / "score.jsonl"
    while True:
        jobs = pending()
        if not jobs:
            if pulls_finished():
                log_progress(progress, done=True)
                return
            time.sleep(120)
            continue
        for kind, src, dst in jobs:
            t0 = time.monotonic()
            if kind == "news":
                n = score_news_file(news_reader, src, dst)
            else:
                n = score_sec_file(sec_reader, src, dst)
            secs = time.monotonic() - t0
            if kind == "news" or n >= 50:
                log_progress(
                    progress,
                    kind=kind,
                    file=src.name,
                    items=n,
                    seconds=round(secs, 2),
                    per_second=round(n / secs, 1) if secs > 0 else None,
                )
