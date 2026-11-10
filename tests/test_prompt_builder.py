"""Tests for the analyst prompt builder and literature grounding."""

from pathlib import Path

from corpus_forensics.analysis.prompt_builder import PromptBuilder
from corpus_forensics.analysis.retriever import AttackRetriever
from corpus_forensics.embeddings.mock import MockEmbedder
from corpus_forensics.schemas import Chunk
from corpus_forensics.vectorstore.chroma_store import ChromaVectorStore
from corpus_forensics.vectorstore.literature_store import LiteraturePassage, LiteratureStore


def _chunk() -> Chunk:
    return Chunk(id="d_chunk_0", document_id="d", text="Hola mundo", start=0, end=10, index=0)


def test_prompt_requests_configured_language() -> None:
    prompt = PromptBuilder(output_language="es").build_prompt(_chunk(), [])
    assert "in Spanish" in prompt
    assert "Never translate `fragmento_exacto`" in prompt


def test_prompt_language_is_configurable() -> None:
    assert "in English" in PromptBuilder(output_language="en").build_prompt(_chunk(), [])
    assert "in Italian" in PromptBuilder(output_language="Italian").build_prompt(_chunk(), [])


def test_prompt_includes_literature_context() -> None:
    """Verify that literature passages are rendered into the prompt template."""
    passage = LiteraturePassage(
        id="lit_001",
        texto="NIST guidelines suggest filtering untrusted RAG documents.",
        titulo="NIST AI 100-2",
        seccion="4.1 Mitigation Strategies",
        fuente_id="nist_ai_100_2",
        url="https://doi.org/nist",
        similarity_score=0.92,
        metadata={},
    )
    prompt = PromptBuilder().build_prompt(_chunk(), [], retrieved_literature=[passage])

    assert "RETRIEVED SECURITY LITERATURE & RESEARCH CONTEXT" in prompt
    assert "NIST AI 100-2" in prompt
    assert "4.1 Mitigation Strategies" in prompt
    assert "0.92" in prompt
    assert "NIST guidelines suggest filtering untrusted RAG documents." in prompt


def test_prompt_literature_empty_fallback() -> None:
    """Verify clean fallback when no literature passages are returned."""
    prompt = PromptBuilder().build_prompt(_chunk(), [], retrieved_literature=[])
    assert "No relevant security research literature found in Knowledge Base." in prompt


def test_prompt_contains_hard_negative_few_shot_guidance() -> None:
    """Verify that prompt contains hard negative few-shot distinction."""
    prompt = PromptBuilder().build_prompt(_chunk(), [])
    assert "HARD NEGATIVES" in prompt
    assert "Example HARD NEGATIVE" in prompt
    assert "Example TRUE POSITIVE" in prompt


def test_retriever_literature_query(tmp_path: Path) -> None:
    """Verify AttackRetriever retrieves from LiteratureStore."""
    vector_store = ChromaVectorStore(persist_dir=tmp_path / "attacks")
    lit_store = LiteratureStore(persist_dir=tmp_path / "literature")
    embedder = MockEmbedder(dimension=32)

    retriever = AttackRetriever(
        vector_store=vector_store,
        embedder=embedder,
        literature_store=lit_store,
        literature_top_k=2,
    )

    chunk = _chunk()
    # Empty store returns empty list without error
    assert retriever.retrieve_literature_for_chunk(chunk) == []

    # Add passage to literature store
    emb = embedder.embed_query("sample literature passage")
    lit_store.add(
        ids=["lit_1"],
        texts=["sample literature passage"],
        embeddings=[emb],
        metadatas=[{"titulo": "Title", "seccion": "1.0", "fuente_id": "src", "url": ""}],
    )

    passages = retriever.retrieve_literature_for_chunk(chunk)
    assert len(passages) == 1
    assert passages[0].id == "lit_1"
    assert passages[0].titulo == "Title"
