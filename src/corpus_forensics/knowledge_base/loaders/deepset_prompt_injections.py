"""Loader stub for Deepset Prompt Injections dataset."""

from collections.abc import Iterable
from pathlib import Path

from corpus_forensics.knowledge_base.loaders.base import BaseKBLoader
from corpus_forensics.knowledge_base.schemas import AttackEntry


class DeepsetPromptInjectionsLoader(BaseKBLoader):
    """Loader stub for deepset/prompt-injections dataset.

    Intended behavior:
        Loads injection prompts from the deepset benchmark parquet/csv files,
        filtering for label == 1 (injection), classifying by attack category,
        and producing AttackEntry objects.
    """

    @property
    def source_name(self) -> str:
        return "deepset_prompt_injections"

    def load(self, source_path: Path | str | None = None) -> Iterable[AttackEntry]:
        """Load deepset prompt injection examples."""
        # TODO(team): Implement parser for deepset dataset:
        # 1. Read CSV/Parquet from data/knowledge_base/deepset/
        # 2. Filter for malicious entries (label=1).
        # 3. Classify attack intent and generate AttackEntry with unique hash IDs.
        return []
