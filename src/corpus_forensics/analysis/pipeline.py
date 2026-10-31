"""End-to-end Forensic Analysis Pipeline.

Orchestrates:
document -> preprocessing -> chunking -> vector retrieval ->
analyst LLM -> aggregation -> verification -> forensic report.

NOTE: Analyzed documents are transient in-memory models and are NEVER
persisted into the vector database or knowledge base.
"""

from collections.abc import Sequence
from pathlib import Path

from corpus_forensics.analysis.aggregator import FindingAggregator
from corpus_forensics.analysis.analyst import LLM_ERROR_PREFIX, ChunkAnalyst
from corpus_forensics.analysis.prompt_builder import PromptBuilder
from corpus_forensics.analysis.retriever import AttackRetriever
from corpus_forensics.analysis.verifier import FindingVerifier
from corpus_forensics.config import Settings, get_settings
from corpus_forensics.embeddings.base import Embedder
from corpus_forensics.embeddings.factory import get_embedder
from corpus_forensics.ingestion.base import get_loader_for_path
from corpus_forensics.llm.base import LLMClient
from corpus_forensics.llm.factory import get_llm_client
from corpus_forensics.preprocessing.chunker import Chunker
from corpus_forensics.schemas import AnalisisMetadata, DocumentReport
from corpus_forensics.vectorstore.base import VectorStore
from corpus_forensics.vectorstore.chroma_store import ChromaVectorStore


class ForensicPipeline:
    """Orchestrates forensic corpus scanning against indirect prompt injections."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Initialize pipeline components with dependency injection.

        Args:
            settings: Application configuration.
            llm_client: LLM client (defaults to configured provider).
            embedder: Embedding model (defaults to configured provider).
            vector_store: Vector store (defaults to Chroma persistent store).
        """
        self.settings = settings or get_settings()

        self.embedder = embedder or get_embedder(
            provider=self.settings.embedder_provider,
            dimension=self.settings.embedding_dimension,
        )

        self.vector_store = vector_store or ChromaVectorStore(
            persist_dir=self.settings.vectorstore_dir
        )

        self.llm_client = llm_client or get_llm_client(provider=self.settings.llm_provider)

        self.chunker = Chunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            unit=self.settings.chunk_unit,  # type: ignore[arg-type]
        )

        self.retriever = AttackRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=self.settings.top_k,
        )

        self.prompt_builder = PromptBuilder(output_language=self.settings.report_language)
        self.analyst = ChunkAnalyst(
            llm_client=self.llm_client,
            prompt_builder=self.prompt_builder,
        )

        self.aggregator = FindingAggregator()
        self.verifier = FindingVerifier()

    def analyze_document(self, file_path: Path | str) -> DocumentReport:
        """Analyze a single document and produce a structured forensic report.

        Args:
            file_path: Path to the target document.

        Returns:
            DocumentReport containing verdict, findings, and metadata.
        """
        path = Path(file_path)

        # 1. Ingestion
        loader = get_loader_for_path(path)
        doc = loader.load(path)

        # 2. Preprocessing & Chunking (with overlap and offsets)
        chunks = self.chunker.chunk_document(doc)

        chunk_findings = []
        # 3 & 4. Per-chunk retrieval and LLM analysis
        for chunk in chunks:
            retrieved_attacks = self.retriever.retrieve_for_chunk(chunk)
            finding = self.analyst.analyze_chunk(chunk, retrieved_attacks)
            chunk_findings.append(finding)

        # 5. Cross-chunk Aggregation & Deduplication
        aggregated_findings = self.aggregator.aggregate(chunk_findings, chunks)

        # 6. Anti-hallucination Verification (literal substring checks)
        verified_findings = self.verifier.verify(aggregated_findings, doc)

        # Determine document-level verdict
        # Document is malicious if any verified finding has confidence >= 0.5
        is_malicious = any(h.confianza >= 0.5 for h in verified_findings)

        # 7. Construct Report
        metadata = AnalisisMetadata(
            n_chunks=len(chunks),
            loader_used=loader.name,
            llm_is_mock=self.llm_client.is_mock,
            embedder_is_mock=self.embedder.is_mock,
            llm_provider=self.llm_client.provider_name,
            llm_model=getattr(self.llm_client, "model", None),
            n_chunks_failed=sum(
                1
                for cf in chunk_findings
                if (cf.analyst_reasoning or "").startswith(LLM_ERROR_PREFIX)
            ),
        )

        return DocumentReport(
            documento=str(path),
            malicioso=is_malicious,
            hallazgos=verified_findings,
            metadata_de_analisis=metadata,
        )

    def analyze_many(self, paths: Sequence[Path | str]) -> list[DocumentReport]:
        """Analyze a batch of documents sequentially."""
        return [self.analyze_document(p) for p in paths]
