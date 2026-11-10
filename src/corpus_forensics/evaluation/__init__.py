"""Evaluation harness, metrics, and dataset loading module."""

from corpus_forensics.evaluation.dataset import EvalDatasetLoader, EvalLabel, EvalSample
from corpus_forensics.evaluation.harness import EvaluationHarness
from corpus_forensics.evaluation.metrics import EvaluationMetrics, compute_all_metrics
from corpus_forensics.evaluation.records import PredictionRecord, read_records, write_records

__all__ = [
    "EvalDatasetLoader",
    "EvalLabel",
    "EvalSample",
    "EvaluationHarness",
    "EvaluationMetrics",
    "PredictionRecord",
    "compute_all_metrics",
    "read_records",
    "write_records",
]
