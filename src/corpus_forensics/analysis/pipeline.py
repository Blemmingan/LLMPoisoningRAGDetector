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
from corpus_forensics.analysis.timing import StageTimer
from corpus_forensics.analysis.trace import ChunkTrace, build_chunk_trace
from corpus_forensics.analysis.verdict import VerdictPolicy
from corpus_forensics.analysis.verifier import FindingVerifier
from corpus_forensics.config import Settings, get_settings
from corpus_forensics.embeddings.base import Embedder
from corpus_forensics.embeddings.factory import get_embedder
from corpus_forensics.ingestion.base import get_loader_for_path
from corpus_forensics.llm.base import LLMClient
from corpus_forensics.llm.factory import get_llm_client
from corpus_forensics.llm.tokens import count_tokens, schema_tokens
from corpus_forensics.preprocessing.chunker import Chunker
from corpus_forensics.schemas import (
    AnalisisMetadata,
    Chunk,
    Document,
    DocumentReport,
    RetrievedAttack,
)
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
        # computational cost of the last document: seconds per stage, estimated tokens
        self.timer = StageTimer()
        self.last_timings: dict[str, float] = {}
        self.last_token_estimate: dict[str, int] = {}
        self.last_document: Document | None = None  # loaded channels, for the UI

    def _retrieve(self, chunk: Chunk) -> tuple[list[RetrievedAttack], list[LiteraturePassage]]:
        """Retrieve examples and literature for a chunk, embedding it once."""
        with self.timer.stage("embedding_chunk"):
            query_embedding = self.retriever.embed_chunk(chunk)
        with self.timer.stage("recuperacion_ejemplos"):
            attacks = self.retriever.retrieve_for_chunk(chunk, query_embedding)
        with self.timer.stage("recuperacion_literatura"):
            passages = self.retriever.retrieve_literature_for_chunk(chunk, query_embedding)
        return attacks, passages

    def _start_document(self) -> None:
        self.timer.reset()
        self.retriever.timer.reset()
        self.last_token_estimate = {}

    def _finish_document(self) -> None:
        timings = self.timer.snapshot()
        # sub-stages of the example retrieval, reported apart (already inside its total)
        timings.update({f"ejemplos.{k}": v for k, v in self.retriever.timer.snapshot().items()})
        self.last_timings = timings

    def _estimate_tokens(
        self, chunk: Chunk, attacks: list[RetrievedAttack], passages: list[LiteraturePassage]
    ) -> None:
        """Add the analyst prompt's tokens for this chunk, split by section."""
        model = getattr(self.llm_client, "model", None) or self.settings.llm_model
        builder = self.prompt_builder
        prompt = builder.build_prompt(chunk, attacks, retrieved_literature=passages)
        parts = {
            "chunk": count_tokens(chunk.text, model),
            "vista_normalizada": count_tokens(builder._format_normalized_view(chunk), model),
            "ejemplos": count_tokens(builder._format_retrieved_attacks(attacks), model),
            "literatura": count_tokens(builder._format_retrieved_literature(passages), model),
        }
        total = count_tokens(prompt, model)
        parts["plantilla"] = max(0, total - sum(parts.values()))
        parts["schema"] = schema_tokens(model)
        parts["total_entrada"] = total + parts["schema"]
        parts["llamadas"] = 1
        for key, value in parts.items():
            self.last_token_estimate[key] = self.last_token_estimate.get(key, 0) + value

    def retrieve_document(self, file_path: Path | str) -> list[ChunkTrace]:
        """Load, chunk and retrieve only (no LLM call): used to evaluate the retriever.

        The analyst prompt is still built to estimate its tokens (no call is made).
        """
        self._start_document()
        path = Path(file_path)
        with self.timer.stage("carga"):
            doc = get_loader_for_path(path).load(path)
        self.last_document = doc
        with self.timer.stage("chunking_normalizacion"):
            chunks = self.chunker.chunk_document(doc)
        traces = []
        for chunk in chunks:
            attacks, passages = self._retrieve(chunk)
            traces.append(build_chunk_trace(chunk, attacks, passages))
            self._estimate_tokens(chunk, attacks, passages)
        self.last_trace = traces
        self._finish_document()
        return traces

    def analyze_document(self, file_path: Path | str) -> DocumentReport:
        """Analyze a single document and produce a structured forensic report.

        What was retrieved for each chunk is left in `self.last_trace` for evaluation.

        Args:
            file_path: Path to the target document.

        Returns:
            DocumentReport containing verdict, findings, and metadata.
        """
        self._start_document()
        path = Path(file_path)

        # 1. Ingestion
        with self.timer.stage("carga"):
            loader = get_loader_for_path(path)
            doc = loader.load(path)
        self.last_document = doc

        # 2. Preprocessing & Chunking (with overlap and offsets)
        with self.timer.stage("chunking_normalizacion"):
            chunks = self.chunker.chunk_document(doc)

        chunk_findings = []
        traces: list[ChunkTrace] = []
        # 3 & 4. Per-chunk retrieval and LLM analysis
        for chunk in chunks:
            retrieved_attacks, retrieved_literature = self._retrieve(chunk)
            traces.append(build_chunk_trace(chunk, retrieved_attacks, retrieved_literature))
            self._estimate_tokens(chunk, retrieved_attacks, retrieved_literature)
            with self.timer.stage("llm_analista"):
                finding = self.analyst.analyze_chunk(
                    chunk, retrieved_attacks, retrieved_literature=retrieved_literature
                )
            chunk_findings.append(finding)
        self.last_trace = traces

        # 5 & 6. Cross-chunk aggregation and anti-hallucination verification
        with self.timer.stage("agregacion_verificacion"):
            aggregated_findings = self.aggregator.aggregate(chunk_findings, chunks)
            verified_findings = self.verifier.verify(aggregated_findings, doc)
        self._finish_document()

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
