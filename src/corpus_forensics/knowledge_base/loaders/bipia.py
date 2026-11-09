"""Loader for BIPIA, Microsoft's Benchmark for Indirect Prompt Injection Attacks (MIT)."""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

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

MODALITIES = ("text", "code")


class BipiaLoader(BaseKBLoader):
    """Loads BIPIA attack instructions from `{text,code}_attack_{train,test}.json`.

    Each file maps a category (e.g. 'Base Encoding', 'Cookie Theft') to a list of attack
    instructions. BIPIA's train and test splits use disjoint categories, so loading only
    'train' into the knowledge base and keeping 'test' for evaluation avoids leakage.
    """

    def __init__(self, splits: Sequence[str] = ("train",)) -> None:
        self.splits = tuple(splits)

    @property
    def source_name(self) -> str:
        return "bipia"

    def load(self, source_path: Path | str | None = None) -> Iterable[AttackEntry]:
        """Load attack instructions for the configured splits."""
        if source_path is None:
            return
        root = Path(source_path)
        seen: set[str] = set()
        for split in self.splits:
            for modality in MODALITIES:
                file = root / f"{modality}_attack_{split}.json"
                if not file.exists():
                    continue
                data: dict[str, list[str]] = json.loads(file.read_text(encoding="utf-8"))
                for category, attacks in data.items():
                    for variant, text in enumerate(attacks):
                        if not text or not text.strip():
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
                                    "modalidad": modality,
                                    "categoria_origen": category,
                                    "variante": variant,
                                    LABEL_STATUS_KEY: LABEL_PENDING,
                                }
                            ),
                        )
