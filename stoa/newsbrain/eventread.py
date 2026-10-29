"""The event reader: a local open-source language model labels what each news item means.

For every (article, tagged company) pair with 1 to 3 tagged companies, the model reads the
headline and the start of the summary and gives three answers, each one token picked from a
fixed list, so it can never ramble or invent a format:

- the event type, one of ``EVENTS`` (contract won, guidance cut, FDA ruling, ...);
- the direction for that company: good (+), bad (-) or neutral (0);
- the size: 1 small, 2 medium, 3 large.

Each answer keeps its probabilities, so the labels carry how sure the model was:
``p_event`` (the chosen type), ``direction`` = p(+) - p(-) and ``size`` = the expected size.

It reads only the text. It never sees a price or a date, and it runs on the Dell's GPU
(history) or a CPU (the live runner). Claude is never used on historical news: it already
knows how those stories turned out (docs/DECISIONS.md, trial 15). An open model trained in
2024 knows some outcomes too; the labels are what the article says, the outcomes come only
from prices, and the leak is logged there as a known limit.

The instruction part of the prompt is the same for every item, so it is run through the
model once and its attention cache is reused for every batch: only the article's own tokens
are computed each time.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa.newsbrain import ROOT

log = logging.getLogger(__name__)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Apache 2.0
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
VERSION = "er1"  # changing the prompt, the model or the list is a new version
EVENT_DIR = ROOT / "events"
MAX_TAGS = 3
SUMMARY_CHARS = 120
HEADLINE_CHARS = 240

# (code, letter, what the model is told). The letter is the token the model answers with.
EVENTS: tuple[tuple[str, str, str], ...] = (
    (
        "other",
        "A",
        "none of the below: market recap, price move story, opinion, lifestyle, "
        "economy or general news",
    ),
    ("earnings_beat", "B", "the company's earnings or sales beat estimates, or record results"),
    ("earnings_miss", "C", "the company's earnings or sales missed estimates, or a loss"),
    ("guidance_up", "D", "the company raised its guidance or forecast"),
    ("guidance_down", "E", "the company cut its guidance or forecast, or warned on profit"),
    ("contract_won", "F", "the company won a contract, order, award or customer"),
    ("contract_lost", "G", "the company lost a contract or customer, or one was cancelled"),
    ("exec_change", "H", "a CEO, CFO, other executive or board member joins or leaves"),
    ("legal", "I", "lawsuit, investigation, fine, recall or regulator action against it"),
    ("fda", "J", "FDA or drug regulator decision, or a clinical trial result"),
    ("merger", "K", "merger, acquisition, buyout offer or asset sale"),
    ("offering", "L", "the company sells new stock or convertible debt"),
    ("analyst_up", "M", "an analyst upgrades it, starts it at buy, or raises its price target"),
    ("analyst_down", "N", "an analyst downgrades it, starts it at sell, or cuts its target"),
    ("buyback_dividend", "O", "buyback, dividend raise or special dividend"),
    ("product", "P", "product launch, new approval to sell, or partnership"),
    ("distress", "Q", "layoffs, restructuring, bankruptcy, default or going concern doubt"),
    ("insider", "R", "an insider or big investor buys or sells shares or discloses a stake"),
    ("dividend_cut", "S", "dividend cut or suspended"),
    ("index", "T", "added to or removed from a stock index"),
)
CODES = tuple(c for c, _, _ in EVENTS)
DIRECTIONS = ("+", "-", "0")
SIZES = ("1", "2", "3")

SYSTEM = (
    "You label financial news for one company. Read only what the text says. "
    "Do not use anything you know about what happened later."
)
# Made-up examples, one per common type, so the small model sees the format and the
# difference between a real company event and general news.
EXAMPLES = (
    ("XYZ", "XYZ Q3 EPS $1.12 Beats $0.98 Estimate, Sales $2.1B Beat $2.0B", "B+2"),
    ("XYZ", "Why XYZ Stock Is Trading Lower Today", "A-1"),
    ("XYZ", "XYZ Cuts FY Sales Outlook To $4.0B-$4.2B From $4.5B", "E-3"),
    ("XYZ", "Morgan Stanley Upgrades XYZ To Overweight, Raises Price Target To $80", "M+2"),
    ("XYZ", "XYZ Awarded $450M Navy Contract", "F+2"),
    ("XYZ", "XYZ Announces Proposed $300M Public Offering Of Common Stock", "L-2"),
    ("XYZ", "10 Stocks Moving In Thursday's Pre-Market Session", "A01"),
    ("XYZ", "XYZ CEO To Step Down; CFO Named Interim Chief", "H-2"),
    ("XYZ", "FDA Approves XYZ's Drug For Advanced Lung Cancer", "J+3"),
    ("XYZ", "ABC To Acquire XYZ For $42 Per Share In Cash", "K+3"),
    ("XYZ", "Billionaire Investor Says The Market Is In A Bubble", "A01"),
)
INSTRUCTIONS = (
    "Event types:\n"
    + "\n".join(f"{letter} {text}" for _, letter, text in EVENTS)
    + "\n\nAnswer with exactly three characters: the event letter; then + if the news is "
    "good for the named company, - if bad, 0 if neutral or unclear; then 1, 2 or 3 for how "
    "big it is for that company (1 small, 2 medium, 3 large). Use A when the item is not "
    "about a specific event at the named company.\n\nExamples:\n"
    + "".join(f"Company: {c}\nHeadline: {h}\nAnswer: {a}\n\n" for c, h, a in EXAMPLES)
)


def prompt_parts(symbol: str, headline: str, summary: str) -> tuple[str, str]:
    """The shared prefix and this item's own text, in the model's chat format."""
    prefix = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{INSTRUCTIONS}"
    head = " ".join(str(headline).split())[:HEADLINE_CHARS]
    summ = " ".join(str(summary or "").split())[:SUMMARY_CHARS]
    body = f"Company: {symbol}\nHeadline: {head}\n"
    if summ:
        body += f"Summary: {summ}\n"
    return prefix, body + "<|im_end|>\n<|im_start|>assistant\n"


@dataclass
class Labels:
    event: np.ndarray  # index into EVENTS
    p_event: np.ndarray
    direction: np.ndarray  # p(+) - p(-), -1 to 1
    size: np.ndarray  # expected size, 1 to 3


class Reader:
    """The model, its tokenizer and the cached instruction prefix."""

    def __init__(self, device: str | None = None, batch: int = 24) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = (
            AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.batch = batch
        prefix, _ = prompt_parts("X", "x", "")
        self.prefix_ids = self.tok(prefix, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            self.prefix_cache = self.model(self.prefix_ids, use_cache=True).past_key_values
        self.n_prefix = int(self.prefix_ids.shape[1])
        self.letter_ids = self._ids([" " + letter for _, letter, _ in EVENTS])
        self.dir_ids = self._ids(list(DIRECTIONS))
        self.size_ids = self._ids(list(SIZES))

    def _ids(self, chars: Sequence[str]) -> list[int]:
        out = []
        for ch in chars:
            ids = self.tok(ch, add_special_tokens=False).input_ids
            if len(ids) != 1:
                raise ValueError(f"{ch!r} is not one token")
            out.append(int(ids[0]))
        return out

    def _step(self, ids: Any, mask: Any, pos: Any, cache: Any) -> Any:
        out = self.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,  # only the last position's answer is read
        )
        return out.logits[:, -1, :].float()

    def label(self, bodies: Sequence[str]) -> Labels:
        """Labels for item texts (the part after the shared prefix), in order."""
        n = len(bodies)
        ev = np.zeros(n, dtype=np.int8)
        pe = np.zeros(n, dtype=np.float32)
        di = np.zeros(n, dtype=np.float32)
        sz = np.zeros(n, dtype=np.float32)
        enc = [self.tok(b, add_special_tokens=False).input_ids for b in bodies]
        order = np.argsort([len(e) for e in enc], kind="stable")
        t0 = time.time()
        for k, start in enumerate(range(0, n, self.batch)):
            idx = order[start : start + self.batch]
            res = self._batch([enc[i] for i in idx])
            ev[idx], pe[idx], di[idx], sz[idx] = res
            if k % 200 == 199:
                done = start + len(idx)
                log.info(
                    "event reader: %d of %d, %.0f a second", done, n, done / (time.time() - t0)
                )
        return Labels(ev, pe, di, sz)

    def _batch(self, seqs: list[list[int]]) -> tuple[np.ndarray, ...]:
        torch = self.torch
        b = len(seqs)
        width = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((b, width), pad, dtype=torch.long)
        body_mask = torch.zeros((b, width), dtype=torch.long)
        pos = torch.zeros((b, width), dtype=torch.long)
        for i, s in enumerate(seqs):
            k = len(s)
            ids[i, width - k :] = torch.tensor(s)
            body_mask[i, width - k :] = 1
            pos[i, width - k :] = torch.arange(self.n_prefix, self.n_prefix + k)
            pos[i, : width - k] = self.n_prefix
        lengths = torch.tensor([len(s) for s in seqs])
        ids, body_mask, pos = ids.to(self.device), body_mask.to(self.device), pos.to(self.device)
        mask = torch.cat(
            [torch.ones((b, self.n_prefix), dtype=torch.long, device=self.device), body_mask], 1
        )
        cache = _repeat(self.prefix_cache, b)
        nxt = (self.n_prefix + lengths).to(self.device).unsqueeze(1)
        with torch.no_grad():
            logits = self._step(ids, mask, pos, cache)
            p_ev = torch.softmax(logits[:, self.letter_ids], dim=1)
            ev = p_ev.argmax(1)
            chosen = torch.tensor(self.letter_ids, device=self.device)[ev].unsqueeze(1)
            mask = torch.cat([mask, torch.ones((b, 1), dtype=torch.long, device=self.device)], 1)
            logits = self._step(chosen, mask, nxt, cache)
            p_dir = torch.softmax(logits[:, self.dir_ids], dim=1)
            d = p_dir.argmax(1)
            chosen = torch.tensor(self.dir_ids, device=self.device)[d].unsqueeze(1)
            mask = torch.cat([mask, torch.ones((b, 1), dtype=torch.long, device=self.device)], 1)
            logits = self._step(chosen, mask, nxt + 1, cache)
            p_sz = torch.softmax(logits[:, self.size_ids], dim=1)
        sizes = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        return (
            ev.cpu().numpy().astype(np.int8),
            p_ev.max(1).values.cpu().numpy(),
            (p_dir[:, 0] - p_dir[:, 1]).cpu().numpy(),
            (p_sz * sizes).sum(1).cpu().numpy(),
        )


def _repeat(cache: Any, b: int) -> Any:
    import copy

    c = copy.deepcopy(cache)
    c.batch_repeat_interleave(b)
    return c


# ---- the history run ------------------------------------------------------------------------


def items(news: pd.DataFrame, symbols: frozenset[str] | None = None) -> pd.DataFrame:
    """One row per (article, tagged company) for articles tagging 1 to 3 companies, kept to
    ``symbols`` (SEC filers: no crypto, currencies or most funds) when given."""
    df = news[news["symbols"].map(len).between(1, MAX_TAGS)]
    df = df[["id", "symbols", "headline", "summary"]].explode("symbols")
    df = df.rename(columns={"symbols": "symbol"})
    if symbols is not None:
        df = df[df["symbol"].isin(symbols)]
    return df.reset_index(drop=True)


def frame(it: pd.DataFrame, lab: Labels) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": it["id"].to_numpy(),
            "symbol": it["symbol"].astype(str).to_numpy(),
            "event": pd.Categorical.from_codes(
                lab.event.astype(np.int16), categories=pd.Index(CODES)
            ),
            "p_event": lab.p_event.astype(np.float32),
            "direction": lab.direction.astype(np.float32),
            "size": lab.size.astype(np.float32),
        }
    )


def companies() -> frozenset[str]:
    """Symbols with an SEC filer behind them, the stocks the brain can pick."""
    from stoa.newsbrain.sec_pull import SEC_DIR

    return frozenset(pd.read_parquet(SEC_DIR / "ciks.parquet", columns=["symbol"])["symbol"])


def read(reader: Reader, news: pd.DataFrame, symbols: frozenset[str] | None = None) -> pd.DataFrame:
    """Labels for every (article, company) pair in ``news`` (columns id, symbols, headline,
    summary)."""
    it = items(news, symbols)
    if it.empty:
        return frame(it, Labels(*(np.zeros(0) for _ in range(4))))
    bodies = [
        prompt_parts(s, h, m)[1]
        for s, h, m in zip(it["symbol"], it["headline"], it["summary"], strict=True)
    ]
    return frame(it, reader.label(bodies))


def months(news_dir: Path) -> Iterator[Path]:
    yield from sorted(news_dir.glob("*.parquet"))


def run_history(news_dir: Path, out: Path = EVENT_DIR, batch: int = 24) -> None:
    """Label every month not yet done, newest first (the most recent data matters most).
    Resumable: a month is written whole or not at all."""
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"
    reader = Reader(batch=batch)
    symbols = companies()
    todo = [p for p in reversed(list(months(news_dir))) if not (out / p.name).exists()]
    log.info("event reader: %d months to label", len(todo))
    for path in todo:
        t0 = time.time()
        news = pd.read_parquet(path, columns=["id", "symbols", "headline", "summary"])
        labels = read(reader, news, symbols)
        tmp = out / (path.stem + ".tmp")
        labels.to_parquet(tmp, index=False)
        tmp.replace(out / path.name)
        rec = {
            "month": path.stem,
            "items": len(labels),
            "seconds": round(time.time() - t0, 1),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "version": VERSION,
        }
        with progress.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        log.info("event reader: %s", rec)


def load(event_dir: Path = EVENT_DIR) -> pd.DataFrame:
    """Every label written so far."""
    parts = [pd.read_parquet(p) for p in sorted(event_dir.glob("*.parquet"))]
    if not parts:
        return pd.DataFrame(columns=["id", "symbol", "event", "p_event", "direction", "size"])
    df = pd.concat(parts, ignore_index=True)
    df["event"] = pd.Categorical(df["event"].astype(str), categories=CODES)
    return df.drop_duplicates(["id", "symbol"], keep="last")
