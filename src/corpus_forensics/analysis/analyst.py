"""Per-chunk forensic analyst invoking the LLM client."""

import logging

from corpus_forensics.analysis.prompt_builder import PromptBuilder
from corpus_forensics.llm.base import BudgetExceededError, LLMClient, LLMError
from corpus_forensics.schemas import (
    AnalisisChunkLLM,
    Chunk,
    ChunkFinding,
    Hallazgo,
    RetrievedAttack,
)

logger = logging.getLogger(__name__)

LLM_ERROR_PREFIX = "LLM_ERROR:"


class ChunkAnalyst:
    """Performs forensic analysis on individual document chunks."""

    def __init__(self, llm_client: LLMClient, prompt_builder: PromptBuilder) -> None:
        """Initialize chunk analyst.

        Args:
            llm_client: LLM client supporting structured generation.
            prompt_builder: Prompt builder composing prompt templates.
        """
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder

    def analyze_chunk(self, chunk: Chunk, retrieved_attacks: list[RetrievedAttack]) -> ChunkFinding:
        """Analyze a chunk in context with retrieved attacks and return structured findings.

        The LLM answers with `AnalisisChunkLLM`; pipeline-owned fields (chunk index,
        offsets, channel, verification) are filled in here and by the verifier.
        A failed call is logged and returned as a clean finding flagged with
        LLM_ERROR_PREFIX, except when the spending cap is reached, which stops the run.

        Args:
            chunk: Text chunk under analysis.
            retrieved_attacks: Most similar attacks from the knowledge base.

        Returns:
            ChunkFinding for the chunk.
        """
        prompt = self.prompt_builder.build_prompt(chunk, retrieved_attacks)
        try:
            output = self.llm_client.generate_structured(prompt, AnalisisChunkLLM)
        except BudgetExceededError:
            raise
        except LLMError as exc:
            logger.warning("LLM analysis failed for chunk %s: %s", chunk.id, exc)
            return ChunkFinding(
                chunk_index=chunk.index,
                malicioso=False,
                hallazgos=[],
                analyst_reasoning=f"{LLM_ERROR_PREFIX} {exc}",
            )

        hallazgos = [
            Hallazgo(
                **{**h.model_dump(), "confianza": min(1.0, max(0.0, h.confianza))},
                indice_chunk=chunk.index,
            )
            for h in output.hallazgos
        ]
        return ChunkFinding(
            chunk_index=chunk.index,
            malicioso=output.malicioso,
            hallazgos=hallazgos,
            analyst_reasoning=output.razonamiento,
        )
