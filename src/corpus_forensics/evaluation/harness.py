"""Evaluation harness running the pipeline across the benchmark dataset."""

import json
import subprocess
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
    ) -> dict[str, Any]:
        """Run the pipeline over the dataset, then write predictions and metrics.

        Args:
            dataset_dir: Directory containing documents and labels.jsonl.
            output_dir: Where predictions.jsonl, eval_summary.json and eval_summary.md go.
            retrieval_only: Only load, chunk and retrieve (no LLM call, no cost): measures
                the retriever.
            extra_config: Extra entries for run_config.json (e.g. the ablation variant).
        """
        settings = self.pipeline.settings
        dataset_path = Path(dataset_dir)
        out_path = Path(output_dir or settings.output_dir / "eval")
        samples = EvalDatasetLoader(dataset_path).load()
        if not samples:
            raise ValueError(f"No evaluation samples loaded from {dataset_dir}")

        ledger = settings.usage_ledger_path
        records: list[PredictionRecord] = []
        for sample in samples:
            before = _ledger_size(ledger)
            start = time.perf_counter()
            if retrieval_only:
                traces = self.pipeline.retrieve_document(sample.file_path)
                report = None
            else:
                report = self.pipeline.analyze_document(sample.file_path)
                traces = self.pipeline.last_trace
            records.append(
                PredictionRecord(
                    documento=sample.label.documento,
                    label=sample.label,
                    report=report,
                    traces=traces,
                    latency_s=round(time.perf_counter() - start, 4),
                    usage=_usage_delta(_ledger_records(ledger, before)),
                )
            )

        config = run_config(settings, self.pipeline, dataset_path, retrieval_only)
        config.update(extra_config or {})
        return write_outputs(records, config, out_path)


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
