"""Per-chunk forensic analyst invoking the LLM client."""

from corpus_forensics.analysis.prompt_builder import PromptBuilder
from corpus_forensics.llm.base import LLMClient
from corpus_forensics.schemas import Chunk, ChunkFinding, RetrievedAttack


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

        Args:
            chunk: Text chunk under analysis.
            retrieved_attacks: Most similar attacks from the knowledge base.

        Returns:
            Structured ChunkFinding emitted by the LLM.
        """
        prompt = self.prompt_builder.build_prompt(chunk, retrieved_attacks)
        finding = self.llm_client.generate_structured(prompt, ChunkFinding)

        # Ensure chunk index is populated
        finding.chunk_index = chunk.index
        for h in finding.hallazgos:
            h.indice_chunk = chunk.index

        return finding
