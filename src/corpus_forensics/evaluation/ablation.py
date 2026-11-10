"""Ablation study: run the evaluation once per configuration variant and compare.

Variants live in a YAML file (default `data/eval/ablations.yaml`): a name, a description,
the question it answers and setting overrides. Each variant writes its own run folder
(predictions.jsonl, metrics), so a paid run is never repeated: existing folders are
skipped unless `force` is set, and the comparison table is rebuilt from the folders.
"""

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from corpus_forensics.analysis.pipeline import ForensicPipeline
from corpus_forensics.config import Settings, get_settings
from corpus_forensics.evaluation.harness import PREDICTIONS_FILE, EvaluationHarness

BASE_VARIANT = "base"
SUMMARY_JSON = "ablation_summary.json"
SUMMARY_MD = "ablation_summary.md"
# overrides that change shared components: a variant with any of them builds its own
COMPONENT_SETTINGS = {
    "embedder_provider",
    "embedding_dimension",
    "local_embedding_model",
    "embedding_model",
    "vectorstore_dir",
    "llm_provider",
    "llm_model",
}


class Variant(BaseModel):
    """One ablation configuration."""

    name: str
    description: str = ""
    question: str | None = None
    requires_llm: bool = False
    overrides: dict[str, Any] = Field(default_factory=dict)


def load_variants(path: Path) -> list[Variant]:
    """Read and validate the variants file (unknown setting names are an error)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    variants = [Variant.model_validate(v) for v in data.get("variants", [])]
    names = [v.name for v in variants]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate variant names in {path}")
    for variant in variants:
        unknown = set(variant.overrides) - set(Settings.model_fields)
        if unknown:
            raise ValueError(f"Variant '{variant.name}': unknown settings {sorted(unknown)}")
    return variants


def variant_settings(base: Settings, variant: Variant) -> Settings:
    """Base settings with the variant's overrides, validated."""
    return Settings.model_validate({**base.model_dump(), **variant.overrides})


def run_ablation(
    dataset_dir: Path,
    out_dir: Path,
    variants_path: Path,
    only: list[str] | None = None,
    retrieval_only: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run every selected variant that has no results yet, then rebuild the summary."""
    variants = load_variants(variants_path)
    base_settings = get_settings()
    shared: ForensicPipeline | None = None
    status: dict[str, str] = {}
    for variant in variants:
        if only and variant.name not in only:
            continue
        if retrieval_only and variant.requires_llm:
            status[variant.name] = "omitida: requiere LLM"
            continue
        run_dir = out_dir / variant.name
        if (run_dir / PREDICTIONS_FILE).exists() and not force:
            status[variant.name] = "ya calculada (use --force para repetir)"
            continue
        settings = variant_settings(base_settings, variant)
        if COMPONENT_SETTINGS & set(variant.overrides):
            pipeline = ForensicPipeline(settings=settings)
        else:
            # share the embedder, stores and LLM client: they do not depend on the overrides
            shared = shared or ForensicPipeline(settings=base_settings)
            pipeline = ForensicPipeline(
                settings=settings,
                llm_client=shared.llm_client,
                embedder=shared.embedder,
                vector_store=shared.vector_store,
                literature_store=shared.literature_store,
            )
        EvaluationHarness(pipeline).run(
            dataset_dir,
            run_dir,
            retrieval_only=retrieval_only,
            extra_config={"variante": variant.name, "overrides": variant.overrides},
        )
        status[variant.name] = "calculada"
    summary = summarize(out_dir, variants)
    summary["estado"] = status
    return summary


# --------------------------------------------------------------------------- summary


def _get(data: dict[str, Any], *path: str) -> Any:
    for key in path:
        if not isinstance(data, dict):
            return None
        data = data.get(key)  # type: ignore[assignment]
    return data


def metrics_row(metrics: dict[str, Any]) -> dict[str, Any]:
    """The numbers compared across variants."""
    glob = _get(metrics, "retriever", "global") or {}
    return {
        "context_precision": glob.get("context_precision"),
        "context_recall": glob.get("context_recall"),
        "mrr": glob.get("mrr"),
        "hit_at_1": glob.get("hit_at_1"),
        "cp_disfrazados": _get(metrics, "slices", "sutileza", "disfrazado", "context_precision"),
        "auroc_similitud": _get(metrics, "retriever", "linea_base_similitud", "auroc_vs_negativos"),
        "f1": _get(metrics, "generator", "deteccion", "f1"),
        "macro_f1": _get(metrics, "generator", "tipos", "macro_f1"),
        "fpr_hard_negatives": _get(
            metrics, "abilities", "noise_robustness_fpr_hard_negatives", "valor"
        ),
        "fpr_limpios": _get(metrics, "abilities", "negative_rejection_fpr_limpios", "valor"),
        "faithfulness": _get(metrics, "generator", "faithfulness", "media"),
        "costo_usd": _get(metrics, "cost", "costo_usd"),
        "latencia_mediana_s": _get(metrics, "cost", "latencia_s", "mediana"),
    }


RETRIEVER_COLUMNS = (
    "context_precision",
    "context_recall",
    "mrr",
    "hit_at_1",
    "cp_disfrazados",
    "auroc_similitud",
)
GENERATOR_COLUMNS = ("f1", "macro_f1", "fpr_hard_negatives", "fpr_limpios", "faithfulness")


def summarize(out_dir: Path, variants: list[Variant]) -> dict[str, Any]:
    """Comparison of every variant that has results, with differences against the base."""
    rows: list[dict[str, Any]] = []
    for variant in variants:
        summary_path = out_dir / variant.name / "eval_summary.json"
        if not summary_path.exists():
            continue
        metrics = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "variante": variant.name,
                "descripcion": variant.description,
                "pregunta": variant.question,
                "requiere_llm": variant.requires_llm,
                "solo_retrieval": _get(metrics, "config", "solo_retrieval"),
                "llm_is_mock": _get(metrics, "config", "llm_is_mock"),
                **metrics_row(metrics),
            }
        )
    base = next((r for r in rows if r["variante"] == BASE_VARIANT), None)
    for row in rows:
        row["delta"] = {
            key: round(row[key] - base[key], 4)
            for key in (*RETRIEVER_COLUMNS, *GENERATOR_COLUMNS)
            if base is not None
            and row is not base
            and isinstance(row.get(key), int | float)
            and isinstance(base.get(key), int | float)
        }
    summary = {"variantes": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / SUMMARY_MD).write_text(render_summary(rows), encoding="utf-8")
    return summary


def _cell(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return "—"
    delta = row.get("delta", {}).get(key)
    return f"{value:.3f}" + (f" ({delta:+.3f})" if delta else "")


def render_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Ablación",
        "",
        "Cada celda: valor (diferencia contra `base`).",
        "",
        "## Retriever",
        "",
        "| Variante | Context Precision | Context Recall | MRR | Hit@1 | CP disfrazados | AUROC similitud |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['variante']} | " + " | ".join(_cell(r, k) for k in RETRIEVER_COLUMNS) + " |"
        for r in rows
        if r.get("context_precision") is not None or r["variante"] == BASE_VARIANT
    ]
    with_generator = [r for r in rows if r.get("f1") is not None]
    if with_generator:
        lines += ["", "## Generador y habilidades", ""]
        if any(r.get("llm_is_mock") for r in with_generator):
            lines += [
                "> **Aviso:** corridas con el LLM mock: estas columnas no miden el sistema real.",
                "",
            ]
        lines += [
            "| Variante | F1 | Macro-F1 | FPR hard negatives | FPR limpios | Faithfulness | Costo USD |",
            "|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {r['variante']} | "
            + " | ".join(_cell(r, k) for k in GENERATOR_COLUMNS)
            + f" | {r.get('costo_usd')} |"
            for r in with_generator
        ]
    lines += ["", "## Variantes", "", "| Variante | Qué cambia | Pregunta |", "|---|---|---|"]
    lines += [f"| {r['variante']} | {r['descripcion']} | {r['pregunta'] or '—'} |" for r in rows]
    return "\n".join(lines) + "\n"
