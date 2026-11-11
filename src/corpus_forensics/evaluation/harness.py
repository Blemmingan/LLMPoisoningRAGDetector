"""Evaluation harness running the pipeline across the benchmark dataset."""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from corpus_forensics.analysis.pipeline import ForensicPipeline
from corpus_forensics.config import Settings
from corpus_forensics.evaluation.dataset import EvalDatasetLoader
from corpus_forensics.evaluation.metrics import EvaluationMetrics, compute_all_metrics
from corpus_forensics.evaluation.records import (
    PredictionRecord,
    UsageDelta,
    read_records,
    write_records,
)
from corpus_forensics.evaluation.report import render_markdown
from corpus_forensics.llm.usage import UsageRecord

PREDICTIONS_FILE = "predictions.jsonl"
PARTIAL_FILE = "predictions.partial.jsonl"
PARTIAL_CONFIG_FILE = "run_config.partial.json"


def _ledger_records(path: Path, skip: int) -> list[UsageRecord]:
    """Ledger records after the first `skip` lines."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()][skip:]
    return [UsageRecord.model_validate(json.loads(line)) for line in lines]


def _ledger_size(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _usage_delta(records: list[UsageRecord]) -> UsageDelta:
    return UsageDelta(
        calls=len(records),
        cached_calls=sum(1 for r in records if r.cached),
        prompt_tokens=sum(r.prompt_tokens for r in records),
        completion_tokens=sum(r.completion_tokens for r in records),
        cost_usd=round(sum(r.cost_usd for r in records), 6),
    )


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def _dir_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1e6, 1)


def _model_size_mb(name_or_path: str) -> float | None:
    """Size of a local model folder, or of its Hugging Face cache entry."""
    local = Path(name_or_path).expanduser()
    if local.exists():
        return _dir_size_mb(local)
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return _dir_size_mb(cache / f"models--{name_or_path.replace('/', '--')}")


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process (ru_maxrss: KB on Linux, bytes on macOS)."""
    try:
        import resource  # noqa: PLC0415
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1e6 if sys.platform == "darwin" else 1e3), 1)


def resource_usage(settings: Settings) -> dict[str, Any]:
    """Memory and disk footprint of the run (computational cost analysis)."""
    return {
        "memoria_pico_mb": peak_memory_mb(),
        "indice_vectorial_mb": _dir_size_mb(Path(settings.vectorstore_dir)),
        "modelo_embeddings_mb": _model_size_mb(settings.local_embedding_model)
        if settings.embedder_provider == "local"
        else None,
        "modelo_reranker_mb": _model_size_mb(settings.rerank_model) if settings.rerank else None,
        "cpu_logicos": os.cpu_count(),
        "plataforma": f"{platform.system()} {platform.machine()}",
    }


def run_config(
    settings: Settings, pipeline: ForensicPipeline, dataset_dir: Path, retrieval_only: bool
) -> dict[str, Any]:
    """Everything needed to reproduce a run (class 11: reproducibility)."""
    try:
        kb_by_type = pipeline.vector_store.get_stats().get("by_attack_type", {})
    except Exception:  # stats are informative only
        kb_by_type = {}
    card_path = dataset_dir / "dataset_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
    return {
        "dataset": str(dataset_dir),
        "dataset_seed": card.get("seed"),
        "commit": _git_commit(),
        "solo_retrieval": retrieval_only,
        "llm_provider": pipeline.llm_client.provider_name,
        "llm_model": getattr(pipeline.llm_client, "model", None),
        "llm_is_mock": pipeline.llm_client.is_mock,
        "llm_temperature": settings.llm_temperature,
        "embedder_provider": settings.embedder_provider,
        "embedder_model": (
            settings.local_embedding_model
            if settings.embedder_provider == "local"
            else settings.embedding_model
        ),
        "embedder_is_mock": pipeline.embedder.is_mock,
        "top_k": settings.top_k,
        "literature_top_k": settings.literature_top_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunk_unit": settings.chunk_unit,
        "normalize_for_retrieval": settings.normalize_for_retrieval,
        "query_rewrite": settings.query_rewrite,
        "hybrid_search": settings.hybrid_search,
        "rerank": settings.rerank,
        "rerank_model": settings.rerank_model if settings.rerank else None,
        "rerank_candidates": settings.rerank_candidates,
        "analyst_prompt": settings.analyst_prompt,
        "verifier_penalize": settings.verifier_penalize,
        "verdict_threshold": settings.verdict_threshold,
        "verdict_require_verified": settings.verdict_require_verified,
        "verdict_min_severity": settings.verdict_min_severity,
        "kb_por_tipo": kb_by_type,
    }


class EvaluationHarness:
    """Orchestrates benchmark evaluation runs over annotated document datasets."""

    def __init__(self, pipeline: ForensicPipeline | None = None) -> None:
        self.pipeline = pipeline or ForensicPipeline()

    def run(
        self,
        dataset_dir: Path | str,
        output_dir: Path | str | None = None,
        retrieval_only: bool = False,
        extra_config: dict[str, Any] | None = None,
        sample_size: int | None = None,
    ) -> dict[str, Any]:
        """Run the pipeline over the dataset, then write predictions and metrics.

        Args:
            dataset_dir: Directory containing documents and labels.jsonl.
            output_dir: Where predictions.jsonl, eval_summary.json and eval_summary.md go.
            retrieval_only: Only load, chunk and retrieve (no LLM call, no cost): measures
                the retriever.
            extra_config: Extra entries for run_config.json (e.g. the ablation variant).
            sample_size: Evaluate only this many documents, evenly spaced over the dataset
                (labels are grouped by category, so every category is represented).
        """
        settings = self.pipeline.settings
        dataset_path = Path(dataset_dir)
        out_path = Path(output_dir or settings.output_dir / "eval")
        samples = EvalDatasetLoader(dataset_path).load()
        if not samples:
            raise ValueError(f"No evaluation samples loaded from {dataset_dir}")
        if sample_size is not None and 0 < sample_size < len(samples):
            step = len(samples) / sample_size
            samples = [samples[int(i * step)] for i in range(sample_size)]

        config = run_config(settings, self.pipeline, dataset_path, retrieval_only)
        config.update(extra_config or {})
        if sample_size is not None:
            config["muestra"] = len(samples)
        done = self._resume(out_path, config)

        ledger = settings.usage_ledger_path
        partial = out_path / PARTIAL_FILE
        for sample in samples:
            if sample.label.documento in done:
                continue
            before = _ledger_size(ledger)
            start = time.perf_counter()
            if retrieval_only:
                traces = self.pipeline.retrieve_document(sample.file_path)
                report = None
            else:
                report = self.pipeline.analyze_document(sample.file_path)
                traces = self.pipeline.last_trace
            record = PredictionRecord(
                documento=sample.label.documento,
                label=sample.label,
                report=report,
                traces=traces,
                latency_s=round(time.perf_counter() - start, 4),
                usage=_usage_delta(_ledger_records(ledger, before)),
                timings=dict(getattr(self.pipeline, "last_timings", {}) or {}),
                tokens_estimados=dict(getattr(self.pipeline, "last_token_estimate", {}) or {}),
            )
            # appended as soon as it is ready: an interrupted run resumes from here
            with open(partial, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
            done[record.documento] = record

        records = [done[s.label.documento] for s in samples]
        config["recursos"] = resource_usage(settings)
        result = write_outputs(records, config, out_path)
        partial.unlink(missing_ok=True)
        (out_path / PARTIAL_CONFIG_FILE).unlink(missing_ok=True)
        return result

    @staticmethod
    def _resume(out_path: Path, config: dict[str, Any]) -> dict[str, PredictionRecord]:
        """Records of an interrupted run with the same configuration (empty otherwise)."""
        out_path.mkdir(parents=True, exist_ok=True)
        partial = out_path / PARTIAL_FILE
        config_path = out_path / PARTIAL_CONFIG_FILE
        comparable = json.loads(
            json.dumps({k: v for k, v in config.items() if k != "commit"}, default=str)
        )
        if partial.exists() and config_path.exists():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous != comparable:
                raise ValueError(
                    f"{partial} belongs to a run with another configuration; "
                    "delete it or use another output folder."
                )
            return {r.documento: r for r in read_records(partial)}
        partial.unlink(missing_ok=True)
        config_path.write_text(
            json.dumps(comparable, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return {}


def write_outputs(
    records: list[PredictionRecord], config: dict[str, Any], out_path: Path
) -> dict[str, Any]:
    """Compute metrics from records and write predictions, JSON and Markdown summaries."""
    out_path.mkdir(parents=True, exist_ok=True)
    predictions_path = out_path / PREDICTIONS_FILE
    write_records(records, predictions_path)
    (out_path / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics = compute_all_metrics(records, config)
    json_path = out_path / "eval_summary.json"
    json_path.write_text(
        json.dumps(metrics.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path = out_path / "eval_summary.md"
    md_path.write_text(render_markdown(metrics), encoding="utf-8")
    return {
        "metrics": metrics,
        "predictions": predictions_path,
        "json_summary": json_path,
        "markdown_summary": md_path,
        "total_evaluated": len(records),
    }


def recompute(predictions_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Recompute metrics from a saved predictions.jsonl (no pipeline run, no cost)."""
    records = read_records(predictions_path)
    config_path = predictions_path.parent / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return write_outputs(records, config, output_dir or predictions_path.parent)


__all__ = ["EvaluationHarness", "EvaluationMetrics", "recompute", "write_outputs"]
