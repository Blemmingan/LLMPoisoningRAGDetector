"""Evaluation harness, metrics, and dataset loading module."""

from corpus_forensics.evaluation.dataset import EvalDatasetLoader, EvalLabel, EvalSample
from corpus_forensics.evaluation.harness import EvaluationHarness
from corpus_forensics.evaluation.metrics import (
    EvaluationMetrics,
    compute_all_metrics,
    compute_attack_type_accuracy,
    compute_classification_metrics,
    compute_concealment_accuracy,
    compute_fragment_match_rate,
    compute_hard_negative_fpr,
)

__all__ = [
    "EvalDatasetLoader",
    "EvalLabel",
    "EvalSample",
    "EvaluationMetrics",
    "EvaluationHarness",
    "compute_all_metrics",
    "compute_classification_metrics",
    "compute_hard_negative_fpr",
    "compute_attack_type_accuracy",
    "compute_concealment_accuracy",
    "compute_fragment_match_rate",
]
