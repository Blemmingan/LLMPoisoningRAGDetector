"""End-to-end Forensic Analysis Pipeline.

Orchestrates:
document -> preprocessing -> chunking -> vector retrieval ->
analyst LLM -> aggregation -> verification -> forensic report.

NOTE: Analyzed documents are transient in-memory models and are NEVER
persisted into the vector database or knowledge base.
"""

from collections.abc import Sequence
from pathlib import Path

from corpus_forensics.analysis.advanced_retrieval import CrossEncoderReranker
from corpus_forensics.analysis.aggregator import FindingAggregator
from corpus_forensics.analysis.analyst import LLM_ERROR_PREFIX, ChunkAnalyst
from corpus_forensics.analysis.prompt_builder import PromptBuilder, resolve_prompt_path
from corpus_forensics.analysis.retriever import AttackRetriever
from corpus_forensics.analysis.trace import ChunkTrace, build_chunk_trace
from corpus_forensics.analysis.verdict import VerdictPolicy
from corpus_forensics.analysis.verifier import FindingVerifier
from corpus_forensics.config import Settings, get_settings
from corpus_forensics.embeddings.base import Embedder
from corpus_forensics.embeddings.factory import get_embedder
from corpus_forensics.ingestion.base import get_loader_for_path
from corpus_forensics.llm.base import LLMClient
from corpus_forensics.llm.factory import get_llm_client
from corpus_forensics.preprocessing.chunker import Chunker
from corpus_forensics.schemas import AnalisisMetadata, Chunk, DocumentReport, RetrievedAttack
from corpus_forensics.vectorstore.base import VectorStore
from corpus_forensics.vectorstore.chroma_store import ChromaVectorStore
from corpus_forensics.vectorstore.literature_store import LiteraturePassage, LiteratureStore


class ForensicPipeline:
    """Orchestrates forensic corpus scanning against indirect prompt injections."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        literature_store: LiteratureStore | None = None,
    ) -> None:
        """Initialize pipeline components with dependency injection.

        Args:
            settings: Application configuration.
            llm_client: LLM client (defaults to configured provider).
            embedder: Embedding model (defaults to configured provider).
            vector_store: Vector store (defaults to Chroma persistent store).
            literature_store: Literature store for academic context (defaults to Chroma persistent store).
        """
        self.settings = settings or get_settings()

        self.embedder = embedder or get_embedder(
            provider=self.settings.embedder_provider,
            dimension=self.settings.embedding_dimension,
        )

        self.vector_store = vector_store or ChromaVectorStore(
            persist_dir=self.settings.vectorstore_dir
        )

        self.literature_store = literature_store or LiteratureStore(
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
            literature_store=self.literature_store,
            literature_top_k=self.settings.literature_top_k,
            use_normalized_text=self.settings.normalize_for_retrieval,
            query_rewrite=self.settings.query_rewrite,
            query_rewrite_max_sentences=self.settings.query_rewrite_max_sentences,
            hybrid_search=self.settings.hybrid_search,
            rrf_k=self.settings.rrf_k,
            reranker=CrossEncoderReranker(self.settings.rerank_model)
            if self.settings.rerank
            else None,
            rerank_candidates=self.settings.rerank_candidates,
        )

        self.prompt_builder = PromptBuilder(
            template_path=resolve_prompt_path(self.settings.analyst_prompt),
            output_language=self.settings.report_language,
        )
        self.analyst = ChunkAnalyst(
            llm_client=self.llm_client,
            prompt_builder=self.prompt_builder,
        )

        self.aggregator = FindingAggregator()
        self.verifier = FindingVerifier(penalize=self.settings.verifier_penalize)
        self.verdict_policy = VerdictPolicy.from_settings(self.settings)
        self.last_trace: list[ChunkTrace] = []

    def _retrieve(self, chunk: Chunk) -> tuple[list[RetrievedAttack], list[LiteraturePassage]]:
        """Retrieve examples and literature for a chunk, embedding it once."""
        query_embedding = self.retriever.embed_chunk(chunk)
        attacks = self.retriever.retrieve_for_chunk(chunk, query_embedding)
        passages = self.retriever.retrieve_literature_for_chunk(chunk, query_embedding)
        return attacks, passages

    def retrieve_document(self, file_path: Path | str) -> list[ChunkTrace]:
        """Load, chunk and retrieve only (no LLM call): used to evaluate the retriever."""
        path = Path(file_path)
        doc = get_loader_for_path(path).load(path)
        traces = [
            build_chunk_trace(c, *self._retrieve(c)) for c in self.chunker.chunk_document(doc)
        ]
        self.last_trace = traces
        return traces

    def analyze_document(self, file_path: Path | str) -> DocumentReport:
        """Analyze a single document and produce a structured forensic report.

        What was retrieved for each chunk is left in `self.last_trace` for evaluation.

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
        traces: list[ChunkTrace] = []
        # 3 & 4. Per-chunk retrieval and LLM analysis
        for chunk in chunks:
            retrieved_attacks, retrieved_literature = self._retrieve(chunk)
            traces.append(build_chunk_trace(chunk, retrieved_attacks, retrieved_literature))
            finding = self.analyst.analyze_chunk(
                chunk, retrieved_attacks, retrieved_literature=retrieved_literature
            )
            chunk_findings.append(finding)
        self.last_trace = traces

        # 5. Cross-chunk Aggregation & Deduplication
        aggregated_findings = self.aggregator.aggregate(chunk_findings, chunks)

        # 6. Anti-hallucination Verification (literal substring checks)
        verified_findings = self.verifier.verify(aggregated_findings, doc)

        # Document-level verdict (configurable policy: threshold, verified, severity)
        is_malicious = self.verdict_policy.is_malicious(verified_findings)

        # 7. Construct Report
        metadata = AnalisisMetadata(
            n_chunks=len(chunks),
            loader_used=loader.name,
            llm_is_mock=self.llm_client.is_mock,
            embedder_is_mock=self.embedder.is_mock,
            llm_provider=self.llm_client.provider_name,
            llm_model=getattr(self.llm_client, "model", None),
            verdict_policy=self.verdict_policy.describe(),
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
