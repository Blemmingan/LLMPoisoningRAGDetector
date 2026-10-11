"""Loader stub for BIPIA (Benchmark for Indirect Prompt Injection Attacks)."""

from collections.abc import Iterable
from pathlib import Path

from corpus_forensics.knowledge_base.loaders.base import BaseKBLoader
from corpus_forensics.knowledge_base.schemas import AttackEntry


class BipiaLoader(BaseKBLoader):
    """Loader stub for BIPIA benchmark dataset.

    Intended behavior:
        Parses BIPIA indirect prompt injection test suites (QA, summarization, web search,
        code completion) and extracts contextual injection payloads embedded into synthetic documents.
    """

    @property
    def source_name(self) -> str:
        return "bipia"

    def load(self, source_path: Path | str | None = None) -> Iterable[AttackEntry]:
        """Load BIPIA indirect prompt injection benchmarks."""
        # TODO(team): Implement parser for BIPIA benchmark repo:
        # 1. Parse JSON files from data/knowledge_base/bipia/
        # 2. Extract injection payloads, targets, and simulated attack environments.
        # 3. Label concealment techniques (delimiters, fake formatting, markdown links).
        return []
