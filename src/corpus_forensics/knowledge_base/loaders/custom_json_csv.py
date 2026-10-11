"""Functional generic loader for custom JSON and CSV attack knowledge base files."""

import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from corpus_forensics.knowledge_base.loaders.base import BaseKBLoader
from corpus_forensics.knowledge_base.schemas import AttackEntry
from corpus_forensics.schemas import TecnicaDeOcultamiento, TipoDeAtaque


class CustomJsonCsvLoader(BaseKBLoader):
    """Loads attack entries from custom JSON or CSV files."""

    @property
    def source_name(self) -> str:
        return "custom"

    def load(self, source_path: Path | str | None = None) -> Iterable[AttackEntry]:
        """Load attack entries from JSON or CSV.

        Args:
            source_path: Path to JSON/CSV file or directory containing them.

        Yields:
            AttackEntry models.
        """
        if source_path is None:
            return

        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source path does not exist: {path}")

        files = [path] if path.is_file() else list(path.glob("*.*"))

        for file in files:
            if file.suffix.lower() == ".json":
                yield from self._load_json(file)
            elif file.suffix.lower() == ".csv":
                yield from self._load_csv(file)

    def _load_json(self, path: Path) -> Iterable[AttackEntry]:
        """Parse JSON file containing list of attack entries or wrapped dict."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        items = data if isinstance(data, list) else data.get("entries", [data])
        for idx, item in enumerate(items):
            yield self._parse_item(item, default_id=f"{path.stem}_{idx}")

    def _load_csv(self, path: Path) -> Iterable[AttackEntry]:
        """Parse CSV file containing attack records."""
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                yield self._parse_item(dict(row), default_id=f"{path.stem}_{idx}")

    def _parse_item(self, item: dict[str, Any], default_id: str) -> AttackEntry:
        """Parse dictionary record into AttackEntry model."""
        text = str(item.get("text") or item.get("payload") or item.get("prompt") or "")
        entry_id = str(item.get("id") or "")
        if not entry_id:
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            entry_id = f"{default_id}_{h}"

        attack_type_raw = str(item.get("attack_type") or item.get("tipo_de_ataque") or "otro")
        try:
            attack_type = TipoDeAtaque(attack_type_raw)
        except ValueError:
            attack_type = TipoDeAtaque.OTRO

        concealment_raw = str(
            item.get("concealment_technique") or item.get("tecnica_de_ocultamiento") or "ninguna"
        )
        try:
            concealment = TecnicaDeOcultamiento(concealment_raw)
        except ValueError:
            concealment = TecnicaDeOcultamiento.NINGUNA

        return AttackEntry(
            id=entry_id,
            text=text,
            source=str(item.get("source", "custom")),
            attack_type=attack_type,
            concealment_technique=concealment,
            attacker_objective=str(
                item.get("attacker_objective") or item.get("objective") or "custom"
            ),
            mitigation=str(item.get("mitigation", "")),
            is_variant=bool(item.get("is_variant", False)),
            metadata={
                k: v
                for k, v in item.items()
                if k
                not in {
                    "id",
                    "text",
                    "payload",
                    "source",
                    "attack_type",
                    "tipo_de_ataque",
                    "concealment_technique",
                    "tecnica_de_ocultamiento",
                    "attacker_objective",
                    "mitigation",
                    "is_variant",
                }
            },
        )
