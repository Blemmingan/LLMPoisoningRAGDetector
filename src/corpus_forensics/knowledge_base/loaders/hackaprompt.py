"""Loader for the HackAPrompt competition dataset (Hugging Face, MIT, gated)."""

import random
from collections import defaultdict
from collections.abc import Iterable
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


class HackAPromptLoader(BaseKBLoader):
    """Loads a deterministic, level-stratified sample of successful HackAPrompt attacks.

    The dataset has ~600k submissions (~78k successful, ~19k unique). We keep only
    successful ones (`correct == True`), drop duplicates and extreme lengths, and sample
    `max_entries` spread evenly over the competition levels, with a fixed seed.
    """

    def __init__(
        self,
        max_entries: int = 400,
        min_chars: int = 20,
        max_chars: int = 1500,
        seed: int = 42,
    ) -> None:
        self.max_entries = max_entries
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.seed = seed

    @property
    def source_name(self) -> str:
        return "hackaprompt"

    def load(self, source_path: Path | str | None = None) -> Iterable[AttackEntry]:
        """Load the sampled successful attacks from the downloaded Parquet file(s)."""
        if source_path is None:
            return
        by_level: dict[int, dict[str, str]] = defaultdict(dict)
        for file in sorted(Path(source_path).glob("*.parquet")):
            table = pq.read_table(file, columns=["user_input", "correct", "level"])
            for text, correct, level in zip(
                table.column("user_input").to_pylist(),
                table.column("correct").to_pylist(),
                table.column("level").to_pylist(),
                strict=True,
            ):
                if not correct or not text:
                    continue
                text = text.strip()
                if not (self.min_chars <= len(text) <= self.max_chars):
                    continue
                by_level[int(level)].setdefault(normalize_text(text), text)

        if not by_level:
            return

        rng = random.Random(self.seed)
        levels = sorted(by_level)
        quota = self.max_entries // len(levels)
        selected: list[tuple[int, str]] = []
        leftovers: list[tuple[int, str]] = []
        for level in levels:
            texts = sorted(by_level[level].values())
            rng.shuffle(texts)
            selected.extend((level, t) for t in texts[:quota])
            leftovers.extend((level, t) for t in texts[quota:])
        rng.shuffle(leftovers)
        selected.extend(leftovers[: self.max_entries - len(selected)])

        seen: set[str] = set()
        for level, text in selected:
            key = normalize_text(text)
            if key in seen:
                continue
            seen.add(key)
            yield AttackEntry(
                id=make_entry_id(self.source_name, "sample", text),
                text=text,
                source=self.source_name,
                attack_type=TipoDeAtaque.OTRO,
                metadata=scalar_metadata(
                    {"nivel": level, "split": "sample", LABEL_STATUS_KEY: LABEL_PENDING}
                ),
            )
