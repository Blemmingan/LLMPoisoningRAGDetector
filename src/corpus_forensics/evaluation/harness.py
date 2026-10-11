"""Evaluation harness running the pipeline across the benchmark dataset."""

import json
from pathlib import Path
from typing import Any

from corpus_forensics.analysis.pipeline import ForensicPipeline
from corpus_forensics.config import get_settings
from corpus_forensics.evaluation.dataset import EvalDatasetLoader
from corpus_forensics.evaluation.metrics import EvaluationMetrics, compute_all_metrics


class EvaluationHarness:
    """Orchestrates benchmark evaluation runs over annotated document datasets."""

    def __init__(self, pipeline: ForensicPipeline | None = None) -> None:
        """Initialize evaluation harness.

        Args:
            pipeline: ForensicPipeline instance (defaults to standard pipeline).
        """
        self.pipeline = pipeline or ForensicPipeline()

    def run(
        self,
        dataset_dir: Path | str,
        output_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Run evaluation over the dataset and write summary reports.

        Args:
            dataset_dir: Directory containing documents and labels.jsonl.
            output_dir: Directory where evaluation summaries will be written.

        Returns:
            Dictionary containing metrics and output file paths.
        """
        settings = get_settings()
        out_path = Path(output_dir or settings.output_dir / "eval")
        out_path.mkdir(parents=True, exist_ok=True)

        loader = EvalDatasetLoader(dataset_dir)
        samples = loader.load()

        if not samples:
            raise ValueError(f"No evaluation samples loaded from {dataset_dir}")

        predictions = []
        for sample in samples:
            report = self.pipeline.analyze_document(sample.file_path)
            predictions.append(report)

        metrics = compute_all_metrics(samples, predictions)

        # Write summary JSON
        json_summary_path = out_path / "eval_summary.json"
        with open(json_summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics.model_dump(), f, indent=2)

        # Write summary Markdown
        md_summary_path = out_path / "eval_summary.md"
        md_content = self._render_metrics_markdown(metrics, dataset_dir)
        md_summary_path.write_text(md_content, encoding="utf-8")

        return {
            "metrics": metrics,
            "json_summary": json_summary_path,
            "markdown_summary": md_summary_path,
            "total_evaluated": len(samples),
        }

    def _render_metrics_markdown(self, metrics: EvaluationMetrics, dataset_dir: Path | str) -> str:
        """Render evaluation metrics into Markdown summary table."""
        return f"""# Forensic Detector Evaluation Summary

**Dataset Directory:** `{dataset_dir}`
**Total Samples Evaluated:** {metrics.total_samples}

## Document-Level Classification Metrics (Target: 'malicioso')
| Métrica | Valor |
|---|---|
| **Precisión (Precision)** | {metrics.precision:.4f} |
| **Exhaustividad (Recall)** | {metrics.recall:.4f} |
| **F1-Score** | {metrics.f1:.4f} |
| **Exactitud (Accuracy)** | {metrics.accuracy:.4f} |
| **FPR en Hard Negatives** | {metrics.fpr_hard_negatives:.4f} |

## Granular Attack Detection Metrics
| Métrica | Valor |
|---|---|
| **Tipo de Ataque (Accuracy)** | {metrics.attack_type_accuracy:.4f} |
| **Técnica de Ocultamiento (Accuracy)** | {metrics.concealment_technique_accuracy:.4f} |
| **Tasa de Coincidencia de Fragmento** | {metrics.fragment_match_rate:.4f} |

## Dataset Category Breakdown
```json
{json.dumps(metrics.sliced_metrics, indent=2)}
```
"""
