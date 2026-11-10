"""What the retriever returned for each chunk, kept for evaluation.

The trace is an internal record read by the evaluation harness (retriever metrics,
faithfulness of the cited example). It is not part of the forensic report.
"""

from pydantic import BaseModel, Field

from corpus_forensics.schemas import Chunk, RetrievedAttack
from corpus_forensics.vectorstore.literature_store import LiteraturePassage


class RetrievedExampleTrace(BaseModel):
    """One knowledge-base example returned for a chunk."""

    id: str
    rank: int
    similarity: float
    attack_type: str
    concealment: str


class RetrievedPassageTrace(BaseModel):
    """One literature passage returned for a chunk, with its topic labels."""

    id: str
    rank: int
    similarity: float
    labels: list[str] = Field(default_factory=list)


class ChunkTrace(BaseModel):
    """Retrieval results for one chunk."""

    index: int
    start: int
    end: int
    text: str
    examples: list[RetrievedExampleTrace] = Field(default_factory=list)
    passages: list[RetrievedPassageTrace] = Field(default_factory=list)


def build_chunk_trace(
    chunk: Chunk, attacks: list[RetrievedAttack], passages: list[LiteraturePassage]
) -> ChunkTrace:
    """Record what was retrieved for a chunk, in rank order."""
    return ChunkTrace(
        index=chunk.index,
        start=chunk.start,
        end=chunk.end,
        text=chunk.text,
        examples=[
            RetrievedExampleTrace(
                id=a.attack_id,
                rank=rank,
                similarity=round(a.similarity_score, 4),
                attack_type=a.attack_type.value,
                concealment=a.concealment_technique.value,
            )
            for rank, a in enumerate(attacks, start=1)
        ],
        passages=[
            RetrievedPassageTrace(
                id=p.id,
                rank=rank,
                similarity=round(p.similarity_score, 4),
                labels=sorted(k for k, v in p.metadata.items() if k.startswith("cubre_") and v),
            )
            for rank, p in enumerate(passages, start=1)
        ],
    )
