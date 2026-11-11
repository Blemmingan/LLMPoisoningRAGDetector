"""Advanced retrieval without an LLM, for the knowledge base of known attacks.

Three techniques, each behind its own setting so the ablation can measure them alone or
combined (query rewriting -> hybrid search -> reranking):

- Query rewriting (deterministic): a chunk is mostly benign text with the attack in one
  or two sentences, and its embedding is dominated by the benign part. The chunk is split
  into sentences (fenced code blocks kept whole), each scored with simple signals (the labeling patterns for instruction
  override, prompt leaking, role play, exfiltration, triggers and conditionals, plus URLs,
  base64, second-person address and code that reaches the network, files or processes), and the most suspicious sentences are searched
  separately. It only focuses the search: it never decides anything, and with no
  suspicious sentence the whole chunk is searched as before.
- Hybrid search: BM25 over the knowledge-base texts next to the embedding search, fused
  with Reciprocal Rank Fusion. Attacks share rare literal markers ("PWNED", "secret key",
  URLs, base64) that a lexical index catches and embeddings blur.
- Reranking: a multilingual cross-encoder reads the chunk and each candidate together and
  reorders the candidate pool; the top K go to the analyst.
"""

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

from corpus_forensics.knowledge_base.labeling import (
    B64,
    CONDITIONAL,
    EXFIL,
    JAILBREAK,
    OVERRIDE,
    PERSONA,
    PROMPT_LEAK,
    TRIGGER_KEYWORD,
)
from corpus_forensics.schemas import RetrievedAttack

TOKEN = re.compile(r"\w+", re.UNICODE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
URL = re.compile(r"https?://|www\.", re.I)
SECOND_PERSON = re.compile(r"\b(?:you|your|yourself|tú|usted|vos|du|dein\w*|ihr)\b", re.I)
# code that reaches the network, files, processes, the screen or the keyboard
RISKY_CODE = re.compile(
    r"\b(?:requests\.\w+|urllib|socket|subprocess|os\.system|os\.popen|open_connection|"
    r"start_server|\.connect\(|\.send(?:all)?\(|open\([^)]*\)|getpass|platform\.\w+|"
    r"pyautogui|pynput|keyboard|clipboard|pbpaste|scp|rsync|ssh|ftplib|smtplib|paramiko)\b",
    re.I,
)
SIGNALS: tuple[re.Pattern[str], ...] = (
    OVERRIDE,
    PROMPT_LEAK,
    JAILBREAK,
    PERSONA,
    EXFIL,
    TRIGGER_KEYWORD,
    CONDITIONAL,
    URL,
    B64,
    SECOND_PERSON,
    RISKY_CODE,
)


# --------------------------------------------------------------------------- rewriting


def sentence_score(sentence: str) -> int:
    """Number of suspicious signals present in a sentence."""
    return sum(1 for pattern in SIGNALS if pattern.search(sentence))


CODE_LINE = re.compile(
    r"^\s*(?:import |from \S+ import |def |class |for |while |if .*:$|with |return |"
    r"[\w.\[\]]+\s*=\s*\S|[\w.]+\(.*\)\s*$|[{}()\]]\s*$)"
)


def _looks_like_code(segment: str) -> bool:
    lines = [line for line in segment.splitlines() if line.strip()]
    return bool(lines) and sum(1 for line in lines if CODE_LINE.match(line)) >= max(
        1, len(lines) // 2
    )


def split_segments(text: str) -> list[str]:
    """Sentences of the prose, with each fenced code block kept whole.

    Splitting code into "sentences" breaks snippets apart (an exfiltration snippet is one
    unit), so text between ``` fences is one segment. A chunk can start or end inside a
    block: with an odd number of fences the side that looks like code is taken as code.
    """
    parts = text.split("```")
    code_first = len(parts) % 2 == 0 and _looks_like_code(parts[0])
    segments: list[str] = []
    for i, part in enumerate(parts):
        is_code = (i % 2 == 0) if code_first else (i % 2 == 1)
        if is_code:
            if part.strip():
                segments.append(part.strip())
        else:
            segments.extend(s.strip() for s in SENTENCE_SPLIT.split(part) if s and s.strip())
    return segments


def rewrite_query(text: str, max_sentences: int = 3) -> list[str]:
    """The most suspicious sentences (or code blocks) of a chunk, or the whole chunk."""
    sentences = split_segments(text)
    scored = [(sentence_score(s), i, s) for i, s in enumerate(sentences)]
    suspicious = sorted((x for x in scored if x[0] > 0), key=lambda x: (-x[0], x[1]))
    if not suspicious:
        return [text]
    chosen = sorted(suspicious[:max_sentences], key=lambda x: x[1])  # keep reading order
    return [s for _, _, s in chosen]


# --------------------------------------------------------------------------- BM25 + RRF


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25Index:
    """Okapi BM25 over the knowledge-base texts."""

    def __init__(self, entries: Sequence[RetrievedAttack], k1: float = 1.5, b: float = 0.75):
        self.entries = list(entries)
        self.k1 = k1
        self.b = b
        self.docs = [Counter(tokenize(e.text)) for e in self.entries]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int) -> list[tuple[RetrievedAttack, float]]:
        terms = set(tokenize(query)) & self.idf.keys()
        if not terms or not self.entries:
            return []
        scores: list[tuple[float, int]] = []
        for i, doc in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
            score = sum(
                self.idf[t] * doc[t] * (self.k1 + 1) / (doc[t] + norm) for t in terms if t in doc
            )
            if score > 0:
                scores.append((score, i))
        scores.sort(key=lambda x: -x[0])
        return [(self.entries[i], s) for s, i in scores[:top_k]]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Fuse ranked id lists: each id scores sum(1 / (k + rank)); ranks, not raw scores."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


# --------------------------------------------------------------------------- reranking


# Pair length cap: a 500-character chunk plus a long knowledge-base entry would reach the
# model's 512-token limit, and attention cost grows with the square of the length. The
# attack-bearing part of an entry is at its start, so 256 tokens keep it.
RERANK_MAX_TOKENS = 256


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    return CrossEncoder(model_name, device="cpu", max_length=RERANK_MAX_TOKENS)


class CrossEncoderReranker:
    """Scores (query, candidate) pairs with a local cross-encoder (loaded on first use)."""

    def __init__(
        self,
        model_name: str,
        predict: Callable[[list[tuple[str, str]]], Sequence[float]] | None = None,
    ) -> None:
        self.model_name = model_name
        self._predict = predict

    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, c) for c in candidates]
        if self._predict is not None:
            return [float(s) for s in self._predict(pairs)]
        model = _load_cross_encoder(self.model_name)
        return [float(s) for s in model.predict(pairs, batch_size=32, show_progress_bar=False)]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
