"""Knowledge base loaders registry."""

from corpus_forensics.knowledge_base.loaders.base import BaseKBLoader
from corpus_forensics.knowledge_base.loaders.bipia import BipiaLoader
from corpus_forensics.knowledge_base.loaders.custom_json_csv import CustomJsonCsvLoader
from corpus_forensics.knowledge_base.loaders.deepset_prompt_injections import (
    DeepsetPromptInjectionsLoader,
)
from corpus_forensics.knowledge_base.loaders.hackaprompt import HackAPromptLoader
from corpus_forensics.knowledge_base.loaders.mitre_atlas import MitreAtlasLoader
from corpus_forensics.knowledge_base.loaders.owasp_llm_top10 import OwaspLlmTop10Loader

_KB_LOADERS: dict[str, type[BaseKBLoader]] = {
    "custom": CustomJsonCsvLoader,
    "owasp_llm_top10": OwaspLlmTop10Loader,
    "mitre_atlas": MitreAtlasLoader,
    "deepset_prompt_injections": DeepsetPromptInjectionsLoader,
    "bipia": BipiaLoader,
    "hackaprompt": HackAPromptLoader,
}


def get_kb_loader(source_name: str) -> BaseKBLoader:
    """Retrieve loader instance by source name."""
    name = source_name.lower()
    if name not in _KB_LOADERS:
        raise ValueError(
            f"Unknown KB source '{source_name}'. Available sources: {list(_KB_LOADERS.keys())}"
        )
    return _KB_LOADERS[name]()


def list_available_sources() -> list[str]:
    """Return list of all registered KB source names."""
    return list(_KB_LOADERS.keys())


__all__ = [
    "BaseKBLoader",
    "CustomJsonCsvLoader",
    "OwaspLlmTop10Loader",
    "MitreAtlasLoader",
    "DeepsetPromptInjectionsLoader",
    "BipiaLoader",
    "HackAPromptLoader",
    "get_kb_loader",
    "list_available_sources",
]
