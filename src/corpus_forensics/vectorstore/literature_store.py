"""Second Chroma collection holding literature passages (papers and technical reports).

Kept separate from the short attack examples: long prose and short prompts end up in
different regions of the embedding space, so mixing them skews similarity.
"""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel, Field

LITERATURE_COLLECTION = "literatura"


class LiteraturePassage(BaseModel):
    """A literature passage returned by a similarity query."""

    id: str
    texto: str
    titulo: str
    seccion: str
    fuente_id: str
    url: str
    similarity_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class LiteratureStore:
    """Persistent Chroma collection for literature passages."""

    def __init__(self, persist_dir: Path | str, collection_name: str = LITERATURE_COLLECTION):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert passages with pre-computed embeddings."""
        if ids:
            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,  # type: ignore[arg-type]
                metadatas=metadatas,  # type: ignore[arg-type]
            )

    def query(
        self, query_embedding: list[float], top_k: int = 3, where: dict[str, Any] | None = None
    ) -> list[LiteraturePassage]:
        """Return the `top_k` most similar passages, optionally filtered by metadata."""
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=min(top_k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        passages: list[LiteraturePassage] = []
        ids = result["ids"][0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for pid, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
            m = dict(meta or {})
            passages.append(
                LiteraturePassage(
                    id=pid,
                    texto=doc or "",
                    titulo=str(m.get("titulo", "")),
                    seccion=str(m.get("seccion", "")),
                    fuente_id=str(m.get("fuente_id", "")),
                    url=str(m.get("url", "")),
                    similarity_score=max(0.0, min(1.0, 1.0 - float(dist))),
                    metadata=m,
                )
            )
        return passages

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def stats(self) -> dict[str, Any]:
        """Passage counts by source."""
        data = self._collection.get(include=["metadatas"])
        by_source: dict[str, int] = {}
        for meta in data.get("metadatas") or []:
            key = str((meta or {}).get("fuente_id", "?"))
            by_source[key] = by_source.get(key, 0) + 1
        return {"total": self.count(), "by_source": dict(sorted(by_source.items()))}
