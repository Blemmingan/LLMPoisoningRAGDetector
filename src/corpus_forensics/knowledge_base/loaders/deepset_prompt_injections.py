"""Loader for the deepset/prompt-injections dataset (Hugging Face, Apache-2.0)."""

from collections.abc import Iterable, Sequence
from pathlib import Path

import pyarrow.parquet as pq

from corpus_forensics.knowledge_base.loaders.base import (
    LABEL_PENDING,
    LABEL_STATUS_KEY,
    BaseKBLoader,
    make_entry_id,
    normalize_text,
    scalar_metadata,
)
from corpus_forensics.knowledge_base.schemas import AttackEntry
from corpus_forensics.schemas import TipoDeAtaque


class DeepsetPromptInjectionsLoader(BaseKBLoader):
    """Loads the injection examples (label == 1) of deepset/prompt-injections.

    Texts are in English and German. Only the requested splits are loaded: by default
    'train' goes to the knowledge base and 'test' is reserved for the evaluation dataset.
    The attack type is left pending; some rows are misinformation requests, which are
    out of scope and must be filtered at labeling time.
    """

    def __init__(self, splits: Sequence[str] = ("train",)) -> None:
        self.splits = tuple(splits)

    @property
    def source_name(self) -> str:
        return "deepset_prompt_injections"

    def load(self, source_path: Path | str | None = None) -> Iterable[AttackEntry]:
        """Load injection rows from the downloaded Parquet files."""
        if source_path is None:
            return
        root = Path(source_path)
        seen: set[str] = set()
        for split in self.splits:
            for file in sorted(root.glob(f"{split}-*.parquet")):
                table = pq.read_table(file, columns=["text", "label"])
                for row_idx, (text, label) in enumerate(
                    zip(
                        table.column("text").to_pylist(),
                        table.column("label").to_pylist(),
                        strict=True,
                    )
                ):
                    if label != 1 or not text or not text.strip():
                        continue
                    key = normalize_text(text)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield AttackEntry(
                        id=make_entry_id(self.source_name, split, text),
                        text=text.strip(),
                        source=self.source_name,
                        attack_type=TipoDeAtaque.OTRO,
                        metadata=scalar_metadata(
                            {
                                "split": split,
                                "fila_origen": row_idx,
                                LABEL_STATUS_KEY: LABEL_PENDING,
                            }
                        ),
                    )
