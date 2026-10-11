"""Pipeline for ingesting known attack entries into the vector store."""

from pathlib import Path
from typing import Any

from corpus_forensics.config import get_settings
from corpus_forensics.embeddings.base import Embedder
from corpus_forensics.knowledge_base.loaders import get_kb_loader, list_available_sources
from corpus_forensics.knowledge_base.schemas import AttackEntry
from corpus_forensics.vectorstore.base import VectorStore


def ingest_knowledge_base(
    vector_store: VectorStore,
    embedder: Embedder,
    source: str = "all",
    source_path: Path | str | None = None,
    reset: bool = False,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Ingest attack knowledge base entries into the vector store.

    Idempotent by entry ID.

    Args:
        vector_store: Target vector store.
        embedder: Embedding model instance.
        source: Specific source name or 'all'.
        source_path: Path to source data. If None, defaults to configured paths.
        reset: If True, resets vector store before ingestion.
        batch_size: Embedding batch size.

    Returns:
        Summary dictionary with counts of ingested entries.
    """
    settings = get_settings()

    if reset:
        vector_store.reset()

    sources_to_run = list_available_sources() if source.lower() == "all" else [source.lower()]

    ingested_counts: dict[str, int] = {}
    total_ingested = 0

    for src_name in sources_to_run:
        loader = get_kb_loader(src_name)

        # Resolve path
        path_to_use: Path | str | None = source_path
        if path_to_use is None:
            if src_name == "custom":
                # Default custom path to samples directory
                path_to_use = settings.samples_dir
            else:
                path_to_use = settings.kb_dir / src_name

        entries: list[AttackEntry] = list(loader.load(path_to_use))
        if not entries:
            ingested_counts[src_name] = 0
            continue

        # Process in batches
        for i in range(0, len(entries), batch_size):
            batch = entries[i : i + batch_size]
            ids = [e.id for e in batch]
            texts = [e.text for e in batch]
            metadatas = [
                {
                    "source": e.source,
                    "attack_type": e.attack_type.value,
                    "concealment_technique": e.concealment_technique.value,
                    "attacker_objective": e.attacker_objective,
                    "mitigation": e.mitigation,
                    "is_variant": e.is_variant,
                    **e.metadata,
                }
                for e in batch
            ]

            embeddings = embedder.embed_documents(texts)
            vector_store.add(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

        ingested_counts[src_name] = len(entries)
        total_ingested += len(entries)

    return {
        "status": "success",
        "total_ingested": total_ingested,
        "by_source": ingested_counts,
        "store_count": vector_store.count(),
    }
