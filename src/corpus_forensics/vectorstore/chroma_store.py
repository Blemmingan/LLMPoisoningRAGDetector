"""Persistent ChromaDB implementation of the VectorStore interface."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from corpus_forensics.schemas import RetrievedAttack, TecnicaDeOcultamiento, TipoDeAtaque
from corpus_forensics.vectorstore.base import VectorStore

DEFAULT_COLLECTION_NAME = "known_attacks"


class ChromaVectorStore(VectorStore):
    """Local, persistent ChromaDB vector store for known attack payloads.

    Operates without a server and requires explicit embeddings (no default embedding function).
    """

    def __init__(
        self,
        persist_dir: Path | str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Initialize local persistent Chroma store.

        Args:
            persist_dir: Directory where Chroma will persist data on disk.
            collection_name: Name of the vector collection.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add or upsert documents with explicit pre-computed embeddings."""
        if not ids:
            return

        # Sanitize metadatas for Chroma (primitives only)
        clean_metadatas: list[dict[str, str | int | float | bool]] = []
        for m in metadatas:
            clean_m: dict[str, str | int | float | bool] = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_m[k] = v
                elif isinstance(v, (list, tuple)):
                    clean_m[k] = ",".join(str(item) for item in v)
                elif v is None:
                    clean_m[k] = ""
                else:
                    clean_m[k] = str(v)
            clean_metadatas.append(clean_m)

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=clean_metadatas,  # type: ignore[arg-type]
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedAttack]:
        """Query nearest known attacks using explicit cosine distance vector search."""
        if self.count() == 0:
            return []

        # Chroma query accepts where filter if provided
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        retrieved: list[RetrievedAttack] = []
        ids_raw = results.get("ids")
        docs_raw = results.get("documents")
        metas_raw = results.get("metadatas")
        distances_raw = results.get("distances")

        ids_list = ids_raw[0] if ids_raw else []
        docs_list = docs_raw[0] if docs_raw else []
        metas_list = metas_raw[0] if metas_raw else []
        distances_list = distances_raw[0] if distances_raw else []

        for i in range(len(ids_list)):
            entry_id = ids_list[i]
            doc_text = str(docs_list[i]) if i < len(docs_list) else ""
            meta_item = metas_list[i] if i < len(metas_list) else {}
            meta = dict(meta_item) if meta_item else {}
            dist = float(distances_list[i]) if i < len(distances_list) else 1.0

            # Cosine distance to similarity score: similarity = 1 - (dist / 2) or max(0, 1 - dist)
            # In Chroma hnsw:space cosine, distance = 1 - cosine_similarity
            similarity = max(0.0, min(1.0, 1.0 - dist))

            attack_type_str = str(meta.get("attack_type", "otro"))
            try:
                attack_type = TipoDeAtaque(attack_type_str)
            except ValueError:
                attack_type = TipoDeAtaque.OTRO

            concealment_str = str(meta.get("concealment_technique", "ninguna"))
            try:
                concealment = TecnicaDeOcultamiento(concealment_str)
            except ValueError:
                concealment = TecnicaDeOcultamiento.NINGUNA

            retrieved.append(
                RetrievedAttack(
                    attack_id=entry_id,
                    text=doc_text,
                    source=str(meta.get("source", "unknown")),
                    attack_type=attack_type,
                    concealment_technique=concealment,
                    attacker_objective=str(meta.get("attacker_objective", "unknown")),
                    mitigation=str(meta.get("mitigation", "")),
                    similarity_score=round(similarity, 4),
                    metadata=dict(meta),
                )
            )

        return retrieved

    def count(self) -> int:
        """Return total entries in collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Reset the collection, purging all persisted entries."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_stats(self) -> dict[str, Any]:
        """Aggregate counts by source and attack type."""
        total = self.count()
        if total == 0:
            return {"total_entries": 0, "by_source": {}, "by_attack_type": {}}

        # Fetch all metadata records to compute breakdown
        all_data = self._collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas") or []

        by_source: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for m in metadatas:
            if not m:
                continue
            src = str(m.get("source", "unknown"))
            at = str(m.get("attack_type", "otro"))
            by_source[src] = by_source.get(src, 0) + 1
            by_type[at] = by_type.get(at, 0) + 1

        return {
            "total_entries": total,
            "by_source": by_source,
            "by_attack_type": by_type,
        }
