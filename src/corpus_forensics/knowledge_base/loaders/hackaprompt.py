"""Loader stub for HackAPrompt competition dataset."""

from collections.abc import Iterable
from pathlib import Path

from corpus_forensics.knowledge_base.loaders.base import BaseKBLoader
from corpus_forensics.knowledge_base.schemas import AttackEntry


class HackAPromptLoader(BaseKBLoader):
    """Loader stub for HackAPrompt dataset.

    Intended behavior:
        Parses successful jailbreak and injection submissions from the HackAPrompt competition,
        extracting clever obfuscation techniques, foreign language bypasses, and emoji/unicode tricks.
    """

    @property
    def source_name(self) -> str:
        return "hackaprompt"

    def load(self, source_path: Path | str | None = None) -> Iterable[AttackEntry]:
        """Load HackAPrompt competition payloads."""
        # TODO(team): Implement parser for HackAPrompt dataset:
        # 1. Load JSON records from data/knowledge_base/hackaprompt/
        # 2. Extract winning prompt injections and jailbreak prompts.
        # 3. Categorize concealment techniques (unicode, base64, foreign language, token splitting).
        return []
