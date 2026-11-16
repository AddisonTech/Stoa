"""Trial 17b's reader: a local open-source model reads what changed in a filing and tags it.

Rules find what is new; the model only judges it. For each filing and the same report a
year earlier (the pair the change scores use):

- **New paragraphs**: a paragraph of the filing (at least 60 characters) is new when no
  paragraph of last year's same section, or of last year's whole document for the
  questions that read the whole document, shares half or more of its distinct terms with it
  (Jaccard of term sets below 0.5; numbers do not count as terms, so a paragraph that only
  updates figures is not new). **Removed paragraphs** are the same test run backwards.
- Each question reads only the new paragraphs that can bear on it, in document order, up to
  ``CAP`` characters. A question with nothing to read is answered "no" without the model.

The five questions (the tag is "yes" when the model's probability of Yes over No is at
least 0.5):

- ``litigation``: new paragraphs of legal proceedings, plus new paragraphs of risk factors
  and MD&A that name a lawsuit, investigation, claim or court. "A new lawsuit, government
  investigation or other legal proceeding against the company, or a material expansion of
  one?"
- ``exec_change``: new paragraphs anywhere that name the CEO or CFO (or principal executive
  or financial officer) with a word of change. "The CEO or CFO left, resigned, retired, was
  removed or replaced, or a new or interim one was named?"
- ``new_risk``: new paragraphs of risk factors. "A specific new risk to the company, not
  only updated dates, figures or wording?"
- ``going_concern_added``: new paragraphs anywhere that mention a going concern. "Substantial
  doubt about the company's ability to continue as a going concern?"
- ``restatement``: new paragraphs anywhere that mention restating. "The company restated, or
  will restate, financial statements it had already issued, to correct an error?"

``going_concern_removed`` (reported, not used to drop anything): last year's removed
paragraphs that mention a going concern, the same question, and no paragraph of this year's
document names both "substantial doubt" and "going concern".

The model is Qwen2.5-1.5B-Instruct (Apache 2.0), pinned to one revision, run on the Dell's
GPU in float16, answers read from the next-token probabilities of "Yes" and "No". It
never sees a price, a date or a company name beyond what the paragraphs say. Claude never
reads a historical filing. An open model trained in 2024 may know some outcomes; the tags
are what the text says, the outcomes come only from prices.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoa.filings import LOGS, ROOT, TEXT
from stoa.filings.pull import write
from stoa.newsbrain.pace import log_progress

log = logging.getLogger(__name__)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Apache 2.0, the event reader's model
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
VERSION = "fr1"  # changing a prompt, a rule or the model is a new version
TAGS_DIR = ROOT / "tags"
MIN_PARA = 60
NOVEL_BELOW = 0.5
CAP = 2400  # characters of new text a question reads
THRESHOLD = 0.5
_TERM = re.compile(r"[a-z]{2,}")
_LIT = re.compile(
    r"lawsuit|litigat|complaint|class action|plaintiff|subpoena|investigat|indict|"
    r"arbitrat|\bcourt\b|alleg|\bsued\b|settlement|enforcement action",
    re.I,
)
_EXEC = re.compile(
    r"chief executive officer|chief financial officer|\bceo\b|\bcfo\b|"
    r"principal executive officer|principal financial officer",
    re.I,
)
_CHANGE = re.compile(
    r"resign|retire|appoint|\bnamed\b|succe|depart|terminat|step(ped|s)? down|"
    r"transition|interim|\bhired\b|separation|left the company|replac",
    re.I,
)
_GC = re.compile(r"going concern", re.I)
_DOUBT = re.compile(r"substantial doubt", re.I)
_RESTATE = re.compile(r"restat", re.I)

SYSTEM = (
    "You read passages that a U.S. public company added to its annual or quarterly report "
    "this year. Answer the question about the passages with Yes or No only."
)
QUESTIONS = {
    "litigation": "Do these passages report a new lawsuit, government investigation or other "
    "legal proceeding against the company, or a material expansion of an existing one (new "
    "claims, a larger amount sought, an adverse ruling)?",
    "exec_change": "Do these passages say the company's chief executive officer or chief "
    "financial officer left, resigned, retired, was removed or was replaced, or that a new or "
    "interim CEO or CFO was named?",
    "new_risk": "Do these passages warn about a specific new risk to the company's business? "
    "Answer No if they only state figures, introduce the section, or say there were no "
    "changes.",
    "going_concern": "Do these passages say there is substantial doubt about the company's "
    "ability to continue as a going concern?",
    "restatement": "Do these passages say the company restated, or will restate, financial "
    "statements it had already issued, to correct an error?",
}
TAGS = (
    "litigation",
    "exec_change",
    "new_risk",
    "going_concern_added",
    "restatement",
    "going_concern_removed",
)


def spec() -> dict[str, Any]:
    return {
        "version": VERSION,
        "model": f"{MODEL} at revision {REVISION} (Apache 2.0), float16 on the Dell's GPU",
        "what_it_reads": f"only paragraphs new this year (at least {MIN_PARA} characters, "
        f"term-set Jaccard below {NOVEL_BELOW} with every paragraph of last year's same "
        "report's same section, or whole document for the whole-document questions), the ones "
        f"that can bear on each question, in order, up to {CAP} characters",
        "questions": QUESTIONS,
        "inputs": {
            "litigation": "new legal proceedings paragraphs, plus new risk factor and MD&A "
            "paragraphs matching " + _LIT.pattern,
            "exec_change": "new paragraphs of the whole document matching "
            + _EXEC.pattern
            + " and "
            + _CHANGE.pattern,
            "new_risk": "new risk factor paragraphs",
            "going_concern_added": "new paragraphs of the whole document matching 'going concern'",
            "restatement": "new paragraphs of the whole document matching 'restat'",
            "going_concern_removed": "last year's paragraphs gone this year that match "
            "'going concern', the going concern question, and no paragraph this year naming "
            "both 'substantial doubt' and 'going concern'; reported only",
        },
        "answer": f"P(Yes) / (P(Yes) + P(No)) from the next-token probabilities; tag when at "
        f"least {THRESHOLD}; no new text to read is 'no' without the model",
        "negative_tags": [
            "litigation",
            "exec_change",
            "new_risk",
            "going_concern_added",
            "restatement",
        ],
        "claude": "never reads a historical filing",
    }


# ---- what is new ------------------------------------------------------------------------------


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n+", text or "") if len(p.strip()) >= MIN_PARA]


def _terms(p: str) -> frozenset[str]:
    return frozenset(_TERM.findall(p.lower()))


def novel(now: Sequence[str], before: Sequence[str]) -> list[str]:
    """Paragraphs of ``now`` whose terms overlap no paragraph of ``before`` by half or more."""
    old = [_terms(p) for p in before]
    exact = set(old)
    index: dict[str, list[int]] = {}
    for k, t in enumerate(old):
        for w in t:
            index.setdefault(w, []).append(k)
    out = []
    for p in now:
        t = _terms(p)
        if not t or t in exact:
            continue
        hits: dict[int, int] = {}
        for w in t:
            for k in index.get(w, ()):
                hits[k] = hits.get(k, 0) + 1
        best = max((n / (len(t) + len(old[k]) - n) for k, n in hits.items()), default=0.0)
        if best < NOVEL_BELOW:
            out.append(p)
    return out


def clip(paras: Sequence[str], cap: int = CAP) -> str:
    out: list[str] = []
    n = 0
    for p in paras:
        if n + len(p) > cap:
            if not out:
                out.append(p[:cap])
            break
        out.append(p)
        n += len(p) + 2
    return "\n\n".join(out)


def inputs(cur: dict[str, str], prev: dict[str, str]) -> dict[str, str]:
    """The text each question reads ("" when there is nothing new for it)."""
    full_now, full_before = paragraphs(cur["full"]), paragraphs(prev["full"])
    new = {s: novel(paragraphs(cur[s]), paragraphs(prev[s])) for s in ("rf", "mdna", "legal")}
    lit = new["legal"] + [p for s in ("rf", "mdna") for p in new[s] if _LIT.search(p)]
    pick = [p for p in full_now if _EXEC.search(p) and _CHANGE.search(p)]
    gc = [p for p in full_now if _GC.search(p)]
    rs = [p for p in full_now if _RESTATE.search(p)]
    gone_gc = [p for p in full_before if _GC.search(p)]
    still = any(_GC.search(p) and _DOUBT.search(p) for p in full_now)
    return {
        "litigation": clip(lit),
        "exec_change": clip(novel(pick, full_before) if pick else []),
        "new_risk": clip(new["rf"]),
        "going_concern_added": clip(novel(gc, full_before) if gc else []),
        "restatement": clip(novel(rs, full_before) if rs else []),
        "going_concern_removed": "" if still or not gone_gc else clip(novel(gone_gc, full_now)),
    }


QUESTION_OF = {
    "litigation": "litigation",
    "exec_change": "exec_change",
    "new_risk": "new_risk",
    "going_concern_added": "going_concern",
    "restatement": "restatement",
    "going_concern_removed": "going_concern",
}


def prompt(question: str, text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nQuestion: {question}\n\n"
        f"Passages:\n{text}\n\nAnswer Yes or No.<|im_end|>\n<|im_start|>assistant\n"
    )


# ---- the model ---------------------------------------------------------------------------------


class Reader:
    def __init__(self, batch_tokens: int = 6_000) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = (
            AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.batch_tokens = batch_tokens
        self.yes = self._one("Yes")
        self.no = self._one("No")

    def _one(self, word: str) -> int:
        ids = self.tok(word, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"{word!r} is not one token")
        return int(ids[0])

    def p_yes(self, prompts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        enc = [self.tok(p, add_special_tokens=False).input_ids for p in prompts]
        order = np.argsort([len(e) for e in enc], kind="stable")
        out = np.zeros(len(prompts), dtype=np.float32)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        k = 0
        while k < len(order):
            width = len(enc[order[min(len(order) - 1, k)]])
            n = max(1, self.batch_tokens // max(width, 1))
            idx = order[k : k + n]
            width = max(len(enc[i]) for i in idx)
            ids = torch.full((len(idx), width), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for r, i in enumerate(idx):
                s = enc[i]
                ids[r, width - len(s) :] = torch.tensor(s)
                mask[r, width - len(s) :] = 1
            pos = (mask.cumsum(1) - 1).clamp(min=0)
            with torch.no_grad():
                logits = (
                    self.model(
                        input_ids=ids.to(self.device),
                        attention_mask=mask.to(self.device),
                        position_ids=pos.to(self.device),
                        logits_to_keep=1,
                    )
                    .logits[:, -1, :]
                    .float()
                )
            two = torch.softmax(logits[:, [self.yes, self.no]], dim=1)[:, 0]
            out[idx] = two.cpu().numpy()
            k += len(idx)
        return out


# ---- the pass over filings ---------------------------------------------------------------------


def texts_for(cik: int, accns: set[str]) -> dict[str, dict[str, str]]:
    df = pd.read_parquet(TEXT / f"{cik}.parquet", columns=["accn", "full", "rf", "mdna", "legal"])
    df = df[df["accn"].isin(accns)]
    out: dict[str, dict[str, str]] = {}
    for a, full, rf, mdna, legal in zip(
        df["accn"], df["full"], df["rf"], df["mdna"], df["legal"], strict=True
    ):
        out[str(a)] = {"full": str(full), "rf": str(rf), "mdna": str(mdna), "legal": str(legal)}
    return out


def done_accns() -> set[str]:
    out: set[str] = set()
    for p in TAGS_DIR.glob("part-*.parquet"):
        out |= set(pd.read_parquet(p, columns=["accn"])["accn"])
    return out


def run(todo: pd.DataFrame, chunk: int = 400) -> None:
    """``todo``: rows with cik, accn, prior_accn, in the order to read them."""
    reader = Reader()
    done = done_accns()
    todo = todo[~todo["accn"].isin(done)].reset_index(drop=True)
    progress = LOGS / "tag.jsonl"
    log_progress(progress, todo=len(todo), already=len(done), version=VERSION)
    part = len(list(TAGS_DIR.glob("part-*.parquet")))
    t0 = time.time()
    for k in range(0, len(todo), chunk):
        block = todo.iloc[k : k + chunk]
        rows: list[dict[str, Any]] = []
        jobs: list[tuple[int, str, str]] = []
        for cik_key, g in block.groupby("cik", sort=False):
            cik = int(str(cik_key))
            tx = texts_for(cik, set(g["accn"]) | set(g["prior_accn"]))
            for accn, prior in zip(g["accn"].astype(str), g["prior_accn"].astype(str), strict=True):
                cur, prev = tx.get(accn), tx.get(prior)
                row: dict[str, Any] = {"accn": accn, "cik": cik, "version": VERSION}
                if cur is None or prev is None:
                    row["missing_text"] = True
                    rows.append(row)
                    continue
                row["missing_text"] = False
                ins = inputs(cur, prev)
                for t in TAGS:
                    row[f"chars_{t}"] = len(ins[t])
                    row[f"p_{t}"] = 0.0
                    if ins[t]:
                        jobs.append((len(rows), t, prompt(QUESTIONS[QUESTION_OF[t]], ins[t])))
                rows.append(row)
        if jobs:
            p = reader.p_yes([j[2] for j in jobs])
            for (ri, t, _), v in zip(jobs, p, strict=True):
                rows[ri][f"p_{t}"] = float(v)
        df = pd.DataFrame(rows)
        for t in TAGS:
            if f"p_{t}" in df:
                df[t] = df[f"p_{t}"] >= THRESHOLD
        write(df, TAGS_DIR / f"part-{part:05d}.parquet")
        part += 1
        log_progress(
            progress,
            done=k + len(block),
            of=len(todo),
            prompts=len(jobs),
            per_second=round((k + len(block)) / (time.time() - t0), 2),
        )


def load_tags() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(TAGS_DIR.glob("part-*.parquet"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main(argv: list[str]) -> None:
    """``tag <todo.parquet>``: read the filings listed (cik, accn, prior_accn), in order."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    todo = pd.read_parquet(Path(argv[0]))
    run(todo)
