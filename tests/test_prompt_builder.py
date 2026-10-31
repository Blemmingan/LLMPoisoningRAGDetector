"""Tests for the analyst prompt builder."""

from corpus_forensics.analysis.prompt_builder import PromptBuilder
from corpus_forensics.schemas import Chunk


def _chunk() -> Chunk:
    return Chunk(id="d_chunk_0", document_id="d", text="Hola mundo", start=0, end=10, index=0)


def test_prompt_requests_configured_language() -> None:
    prompt = PromptBuilder(output_language="es").build_prompt(_chunk(), [])
    assert "in Spanish" in prompt
    assert "Never translate `fragmento_exacto`" in prompt


def test_prompt_language_is_configurable() -> None:
    assert "in English" in PromptBuilder(output_language="en").build_prompt(_chunk(), [])
    assert "in Italian" in PromptBuilder(output_language="Italian").build_prompt(_chunk(), [])
