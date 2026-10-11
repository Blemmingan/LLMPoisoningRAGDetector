"""Loader stub for OWASP Top 10 for LLM Applications attack patterns."""

from collections.abc import Iterable
from pathlib import Path

from corpus_forensics.knowledge_base.loaders.base import BaseKBLoader
from corpus_forensics.knowledge_base.schemas import AttackEntry


class OwaspLlmTop10Loader(BaseKBLoader):
    """Loader stub for OWASP LLM Top 10 (LLM01 Prompt Injection, etc.).

    Intended behavior:
        Parses official OWASP Top 10 for LLMs markdown/yaml/json datasets,
        extracting prompt injection scenarios, indirect injection vectors,
        and associated mitigations into AttackEntry objects.
    """

    @property
    def source_name(self) -> str:
        return "owasp_llm_top10"

    def load(self, source_path: Path | str | None = None) -> Iterable[AttackEntry]:
        """Load OWASP LLM Top 10 attack definitions."""
        # TODO(team): Implement parser for OWASP LLM Top 10 repository artifacts:
        # 1. Read YAML/JSON vulnerability definitions from data/knowledge_base/owasp/
        # 2. Extract example attack prompts under LLM01: Prompt Injection.
        # 3. Map taxonomy entries to TipoDeAtaque (inyeccion_directa, prompt_leaking, etc.).
        # 4. Extract official OWASP mitigations and attach to mitigation field.
        return []
