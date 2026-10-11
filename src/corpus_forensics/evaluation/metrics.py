"""Evaluation metrics for forensic detector benchmark evaluation."""

from typing import Any

from pydantic import BaseModel, Field

from corpus_forensics.evaluation.dataset import EvalSample
from corpus_forensics.schemas import CategoriaDocumento, DocumentReport


class EvaluationMetrics(BaseModel):
    """Aggregate evaluation metrics report."""

    total_samples: int
    precision: float = Field(description="Precision for 'malicioso' classification")
    recall: float = Field(description="Recall for 'malicioso' classification")
    f1: float = Field(description="F1 score for 'malicioso' classification")
    accuracy: float = Field(description="Accuracy for 'malicioso' classification")
    fpr_hard_negatives: float = Field(
        description="False positive rate on hard negative (security/academic) documents"
    )
    attack_type_accuracy: float = Field(
        description="Accuracy in classifying tipo_de_ataque for true malicious cases"
    )
    concealment_technique_accuracy: float = Field(
        description="Accuracy in classifying tecnica_de_ocultamiento"
    )
    fragment_match_rate: float = Field(
        description="Rate at which predicted fragment overlaps ground truth fragment"
    )
    sliced_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Metrics sliced by category, attack type, subtlety, or position",
    )


def compute_classification_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    """Compute binary classification precision, recall, F1, and accuracy."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if not yt and yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt and not yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if not yt and not yp)
    total = len(y_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def compute_hard_negative_fpr(
    samples: list[EvalSample], predictions: list[DocumentReport]
) -> float:
    """Calculate False Positive Rate specifically on hard negative documents."""
    hard_negatives = [
        (s, p)
        for s, p in zip(samples, predictions, strict=False)
        if s.label.categoria == CategoriaDocumento.HARD_NEGATIVE
    ]
    if not hard_negatives:
        return 0.0

    false_positives = sum(1 for s, p in hard_negatives if p.malicioso)
    return round(false_positives / len(hard_negatives), 4)


def compute_attack_type_accuracy(
    samples: list[EvalSample], predictions: list[DocumentReport]
) -> float:
    """Calculate accuracy of tipo_de_ataque identification for true positive samples."""
    evaluable = [
        (s, p)
        for s, p in zip(samples, predictions, strict=False)
        if s.label.malicioso and s.label.tipo_de_ataque is not None
    ]
    if not evaluable:
        return 0.0

    correct = 0
    for s, p in evaluable:
        gt_type = s.label.tipo_de_ataque
        # Check if any finding emitted by model matched the ground truth attack type
        if any(h.tipo_de_ataque == gt_type for h in p.hallazgos):
            correct += 1

    return round(correct / len(evaluable), 4)


def compute_concealment_accuracy(
    samples: list[EvalSample], predictions: list[DocumentReport]
) -> float:
    """Calculate accuracy of tecnica_de_ocultamiento identification."""
    evaluable = [
        (s, p)
        for s, p in zip(samples, predictions, strict=False)
        if s.label.malicioso and s.label.tecnica_de_ocultamiento
    ]
    if not evaluable:
        return 0.0

    correct = 0
    for s, p in evaluable:
        gt_techniques = set(s.label.tecnica_de_ocultamiento)
        pred_techniques = {t for h in p.hallazgos for t in h.tecnica_de_ocultamiento}
        # Overlap match
        if gt_techniques & pred_techniques:
            correct += 1

    return round(correct / len(evaluable), 4)


def compute_fragment_match_rate(
    samples: list[EvalSample], predictions: list[DocumentReport]
) -> float:
    """Calculate rate at which predicted verbatim fragment overlaps ground truth fragment."""
    evaluable = [
        (s, p)
        for s, p in zip(samples, predictions, strict=False)
        if s.label.malicioso and s.label.fragmento_exacto
    ]
    if not evaluable:
        return 0.0

    matches = 0
    for s, p in evaluable:
        if not s.label.fragmento_exacto:
            continue
        gt_frag = s.label.fragmento_exacto.strip().lower()
        matched = False
        for h in p.hallazgos:
            pred_frag = h.fragmento_exacto.strip().lower()
            if gt_frag in pred_frag or pred_frag in gt_frag:
                matched = True
                break
        if matched:
            matches += 1

    return round(matches / len(evaluable), 4)


def compute_all_metrics(
    samples: list[EvalSample], predictions: list[DocumentReport]
) -> EvaluationMetrics:
    """Calculate overall evaluation metrics along with category/subtlety slices."""
    y_true = [s.label.malicioso for s in samples]
    y_pred = [p.malicioso for p in predictions]

    cls_metrics = compute_classification_metrics(y_true, y_pred)
    fpr_hn = compute_hard_negative_fpr(samples, predictions)
    at_acc = compute_attack_type_accuracy(samples, predictions)
    ct_acc = compute_concealment_accuracy(samples, predictions)
    frag_match = compute_fragment_match_rate(samples, predictions)

    # TODO(team): Complete detailed metric slicing:
    # 1. Slice by 'categoria' (limpio vs envenenado vs hard_negative)
    # 2. Slice by 'tipo_de_ataque' (inyeccion_directa, jailbreak, exfiltracion, etc.)
    # 3. Slice by 'posicion' (inicio vs medio vs fin)
    # 4. Slice by 'sutileza' (obvio vs parafraseado vs disfrazado)
    sliced_metrics = {
        "categoria": {
            "limpio_count": sum(
                1 for s in samples if s.label.categoria == CategoriaDocumento.LIMPIO
            ),
            "envenenado_count": sum(
                1 for s in samples if s.label.categoria == CategoriaDocumento.ENVENENADO
            ),
            "hard_negative_count": sum(
                1 for s in samples if s.label.categoria == CategoriaDocumento.HARD_NEGATIVE
            ),
        },
        "slicing_stub": "TODO(team): Implement multi-dimensional slice breakdowns for benchmark reporting.",
    }

    return EvaluationMetrics(
        total_samples=len(samples),
        precision=cls_metrics["precision"],
        recall=cls_metrics["recall"],
        f1=cls_metrics["f1"],
        accuracy=cls_metrics["accuracy"],
        fpr_hard_negatives=fpr_hn,
        attack_type_accuracy=at_acc,
        concealment_technique_accuracy=ct_acc,
        fragment_match_rate=frag_match,
        sliced_metrics=sliced_metrics,
    )
