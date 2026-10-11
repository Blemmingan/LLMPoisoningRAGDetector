"""Knowledge base module."""

from corpus_forensics.knowledge_base.ingest import ingest_knowledge_base
from corpus_forensics.knowledge_base.schemas import AttackEntry

__all__ = [
    "AttackEntry",
    "ingest_knowledge_base",
]
