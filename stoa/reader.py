"""FinBERT: a local, open-source reader that turns financial text into tone.

ProsusAI/finbert is a BERT model fine-tuned on financial news sentences. For
each text it gives three probabilities: positive, negative and neutral. It runs
on the CPU (the 9:00 am sweep in GitHub Actions) or a GPU (the historical
backfill), with the same tokenizer, the same truncation and the same pinned
revision, so a score means the same thing in research and live.

torch and transformers are optional (``pip install ".[reader]"``) and are
imported only when a reader is built. Nothing here calls a paid service.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

LABELS = ("positive", "negative", "neutral")


class Reader(Protocol):
    def score(self, texts: Sequence[str]) -> np.ndarray:
        """One row per text: probabilities in LABELS order, each row summing to 1."""
        ...


def text_of(headline: str, summary: str = "", limit: int = 1000) -> str:
    """What the reader sees for one article: the headline, then the summary."""
    head = (headline or "").strip()
    body = (summary or "").strip()
    if body and body != head:
        text = f"{head}. {body}" if head else body
    else:
        text = head
    return text[:limit]


def net(probs: np.ndarray) -> np.ndarray:
    """Positive minus negative, from -1 to 1."""
    out: np.ndarray = probs[:, 0] - probs[:, 1]
    return out


class FinBert:
    """FinBERT on one device. Batched; on CUDA it runs in half precision."""

    def __init__(
        self,
        model: str = "ProsusAI/finbert",
        revision: str | None = None,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 128,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch: Any = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model, revision=revision)
        net_: Any = AutoModelForSequenceClassification.from_pretrained(model, revision=revision)
        if device.startswith("cuda"):
            net_ = net_.half()
        self.model: Any = net_.to(device).eval()
        label2id = {k.lower(): int(v) for k, v in self.model.config.label2id.items()}
        self.order = [label2id[name] for name in LABELS]

    def score(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        n = len(texts)
        out = np.zeros((n, len(LABELS)), dtype=np.float32)
        if n == 0:
            return out
        # Similar lengths in one batch waste less padding; results go back in order.
        order = sorted(range(n), key=lambda i: len(texts[i]))
        with torch.inference_mode():
            for i in range(0, n, self.batch_size):
                idx = order[i : i + self.batch_size]
                enc = self.tokenizer(
                    [texts[j] or " " for j in idx],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.float()
                probs = torch.softmax(logits, dim=-1)[:, self.order].cpu().numpy()
                out[idx] = probs
        return out
