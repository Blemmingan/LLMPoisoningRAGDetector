"""Loader for MITRE ATLAS (Adversarial Threat Landscape for AI Systems, Apache-2.0)."""

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from corpus_forensics.knowledge_base.loaders.base import (
    LABEL_PENDING,
    LABEL_STATUS_KEY,
    BaseKBLoader,
    make_entry_id,
    scalar_metadata,
)
from corpus_forensics.knowledge_base.schemas import AttackEntry
from corpus_forensics.schemas import TipoDeAtaque

# ATLAS techniques within our scope (prompt injection and its outcomes in RAG systems).
RELEVANT_TECHNIQUES: dict[str, str] = {
    "AML.T0051": "LLM Prompt Injection",
    "AML.T0051.000": "LLM Prompt Injection: Direct",
    "AML.T0051.001": "LLM Prompt Injection: Indirect",
    "AML.T0051.002": "LLM Prompt Injection: Triggered",
    "AML.T0054": "LLM Jailbreak",
    "AML.T0056": "Extract LLM System Prompt",
    "AML.T0057": "LLM Data Leakage",
    "AML.T0068": "LLM Prompt Obfuscation",
    "AML.T0070": "RAG Poisoning",
    "AML.T0071": "False RAG Entry Injection",
    "AML.T0077": "LLM Response Rendering",
}

_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def clean_markdown(text: str) -> str:
    """Drop markdown link targets and collapse whitespace."""
    return " ".join(_MD_LINK.sub(r"\1", text or "").split())


class MitreAtlasLoader(BaseKBLoader):
    """Loads ATLAS techniques and real case-study steps related to prompt injection.

    Produces two kinds of entries (metadata `tipo_entrada`):
    - 'tecnica': the description of each relevant technique, with its official mitigations.
    - 'caso': each step of a real case study that uses a relevant technique.
    Descriptions are prose rather than payloads; ingestion can filter them by kind.
    """

    def __init__(self, include_techniques: bool = True, include_case_studies: bool = True) -> None:
        self.include_techniques = include_techniques
        self.include_case_studies = include_case_studies

    @property
    def source_name(self) -> str:
        return "mitre_atlas"

    def load(self, source_path: Path | str | None = None) -> Iterable[AttackEntry]:
        """Load entries from `ATLAS.yaml`."""
        if source_path is None:
            return
        path = Path(source_path)
        file = path / "ATLAS.yaml" if path.is_dir() else path
        if not file.exists():
            return
        data: dict[str, Any] = yaml.safe_load(file.read_text(encoding="utf-8"))
        matrix = data["matrices"][0]
        version = str(data.get("version", ""))

        mitigations_by_technique: dict[str, list[str]] = {}
        for mitigation in matrix.get("mitigations", []):
            for ref in mitigation.get("techniques", []):
                tid = ref.get("id") if isinstance(ref, dict) else ref
                mitigations_by_technique.setdefault(str(tid), []).append(
                    f"{mitigation['id']} {mitigation['name']}"
                )

        def mitigations_for(tid: str) -> str:
            parent = tid.split(".")[0] + "." + tid.split(".")[1]
            names = mitigations_by_technique.get(tid) or mitigations_by_technique.get(parent, [])
            return "; ".join(dict.fromkeys(names))

        if self.include_techniques:
            for technique in matrix.get("techniques", []):
                tid = technique["id"]
                if tid not in RELEVANT_TECHNIQUES:
                    continue
                text = clean_markdown(technique.get("description", ""))
                if not text:
                    continue
                yield AttackEntry(
                    id=make_entry_id(self.source_name, "tecnica", tid),
                    text=text,
                    source=self.source_name,
                    attack_type=TipoDeAtaque.OTRO,
                    attacker_objective=RELEVANT_TECHNIQUES[tid],
                    mitigation=mitigations_for(tid),
                    metadata=scalar_metadata(
                        {
                            "tipo_entrada": "tecnica",
                            "tecnica_atlas": tid,
                            "nombre_atlas": RELEVANT_TECHNIQUES[tid],
                            "version_atlas": version,
                            LABEL_STATUS_KEY: LABEL_PENDING,
                        }
                    ),
                )

        if self.include_case_studies:
            for case in data.get("case-studies", []):
                for step_idx, step in enumerate(case.get("procedure", [])):
                    tid = str(step.get("technique", ""))
                    if tid not in RELEVANT_TECHNIQUES:
                        continue
                    text = clean_markdown(step.get("description", ""))
                    if not text:
                        continue
                    yield AttackEntry(
                        id=make_entry_id(self.source_name, "caso", f"{case['id']}-{step_idx}"),
                        text=text,
                        source=self.source_name,
                        attack_type=TipoDeAtaque.OTRO,
                        attacker_objective=RELEVANT_TECHNIQUES[tid],
                        mitigation=mitigations_for(tid),
                        metadata=scalar_metadata(
                            {
                                "tipo_entrada": "caso",
                                "tecnica_atlas": tid,
                                "caso_atlas": case["id"],
                                "nombre_caso": case.get("name", ""),
                                "version_atlas": version,
                                LABEL_STATUS_KEY: LABEL_PENDING,
                            }
                        ),
                    )
