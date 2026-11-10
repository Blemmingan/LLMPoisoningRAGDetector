"""Markdown rendering of an evaluation run (sections follow the class 11 scheme)."""

from typing import Any

from corpus_forensics.evaluation.metrics import ATTACK_TYPES, CLEAN_LABEL, EvaluationMetrics


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _rate(prop: dict[str, Any] | None) -> str:
    """'0.800 [0.490, 0.943] (n=10)'."""
    if not prop or prop.get("valor") is None:
        return "—"
    low, high = prop["ic95"]
    return f"{prop['valor']:.3f} [{low:.2f}, {high:.2f}] (n={prop['n']})"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += [
        "| " + " | ".join(_fmt(c) if not isinstance(c, str) else c for c in row) + " |"
        for row in rows
    ]
    return "\n".join(lines)


def render_markdown(metrics: EvaluationMetrics) -> str:
    cfg = metrics.config
    parts = ["# Evaluación del Corpus Poisoning Forensic Analyst", ""]
    if cfg.get("llm_is_mock") and metrics.generator is not None:
        parts += [
            "> **Aviso:** el LLM es un mock. Las métricas del generador, de habilidades y la "
            "matriz de accionables no miden el sistema real; las del retriever sí.",
            "",
        ]
    if cfg.get("embedder_is_mock"):
        parts += ["> **Aviso:** los embeddings son mock; el retriever no es representativo.", ""]
    parts += [
        f"Documentos evaluados: **{metrics.total_samples}**",
        "",
        "## Configuración (reproducibilidad)",
        "",
        _table(["Parámetro", "Valor"], [[k, _fmt(v)] for k, v in cfg.items()]),
        "",
    ]

    r = metrics.retriever
    g = r.get("global", {})
    parts += [
        "## 1. Retriever",
        "",
        f"Sobre el chunk que contiene el ataque en cada envenenado "
        f"({r.get('ataque_localizado_en_un_chunk')} de {r.get('envenenados')} localizados), "
        f"K = {r.get('k')}. Relevante = ejemplo del mismo tipo de ataque que la etiqueta.",
        "",
        _table(
            ["Métrica", "Valor"],
            [
                [f"Context Precision@{r.get('k')}", g.get("context_precision")],
                ["Context Recall", g.get("context_recall")],
                ["MRR", g.get("mrr")],
                [f"NDCG@{r.get('k')}", g.get("ndcg")],
                ["Hit@1", g.get("hit_at_1")],
                [f"Hit@{r.get('k')}", g.get("hit_at_k")],
                ["Similitud top-1 media", g.get("similitud_top1")],
                ["Literatura: Context Precision", g.get("literatura_context_precision")],
                ["Literatura: Hit@K", g.get("literatura_hit_at_k")],
            ],
        ),
        "",
        'Por tipo de ataque ("al azar" = lo que acertaría un retriever aleatorio según la '
        "proporción de cada tipo en la base):",
        "",
        _table(
            [
                "Tipo",
                "n",
                "Context Precision",
                "Context Recall",
                "MRR",
                "Hit@1",
                "Hit@1 al azar",
                f"Hit@{r.get('k')}",
                f"Hit@{r.get('k')} al azar",
            ],
            [
                [
                    t,
                    s["n"],
                    s["context_precision"],
                    s["context_recall"],
                    s["mrr"],
                    s["hit_at_1"],
                    s.get("azar_hit_at_1"),
                    s["hit_at_k"],
                    s.get("azar_hit_at_k"),
                ]
                for t, s in r.get("por_tipo", {}).items()
            ],
        ),
        "",
    ]
    base = r.get("linea_base_similitud", {})
    parts += [
        f"Línea base: {base.get('descripcion')}: AUROC vs negativos "
        f"{_fmt(base.get('auroc_vs_negativos'))}, vs hard negatives "
        f"{_fmt(base.get('auroc_vs_hard_negatives'))}.",
        "",
    ]

    gen = metrics.generator
    if gen is not None:
        d = gen["deteccion"]
        types = gen["tipos"]
        frag = gen["fragmento"]
        parts += [
            "## 2. Generador",
            "",
            "### Answer relevance (contra el ground truth)",
            "",
            _table(
                ["Métrica", "Valor"],
                [
                    ["Precision (malicioso)", d["precision"]],
                    ["Recall (malicioso)", d["recall"]],
                    ["F1 (malicioso)", d["f1"]],
                    ["Accuracy", d["accuracy"]],
                    ["Macro-F1 por tipo de ataque", types["macro_f1"]],
                    ["Tipo correcto en algún hallazgo", _rate(gen["tipo_en_algun_hallazgo"])],
                    [
                        "Ocultamiento identificado (disfrazados)",
                        _rate(gen["ocultamiento_identificado"]),
                    ],
                    [
                        f"Fragmento: cobertura ≥ {frag['umbral_cobertura']}",
                        _rate(frag["coincidencia"]),
                    ],
                    [
                        "Fragmento: cobertura media / precisión media",
                        f"{_fmt(frag['cobertura_media'])} / {_fmt(frag['precision_media'])}",
                    ],
                ],
            ),
            "",
            _table(
                ["Tipo", "n", "Precision", "Recall", "F1"],
                [
                    [t, m["n"], m["precision"], m["recall"], m["f1"]]
                    for t, m in types["por_tipo"].items()
                ],
            ),
            "",
            "Matriz de confusión (filas = etiqueta, columnas = predicción):",
            "",
        ]
        labels = [*ATTACK_TYPES, CLEAN_LABEL]
        conf = types["matriz_confusion"]
        # predicted labels outside the five types (e.g. "otro") get their own column
        extra_cols = sorted({p for row in conf.values() for p in row} - set(labels))
        columns = [*labels, *extra_cols]
        parts += [
            _table(
                ["etiqueta \\ predicción", *columns],
                [
                    [gt, *[conf.get(gt, {}).get(p, 0) for p in columns]]
                    for gt in labels
                    if gt in conf
                ],
            ),
            "",
            "### Faithfulness",
            "",
            f"Media {_fmt(gen['faithfulness']['media'])} sobre {gen['faithfulness']['hallazgos']} "
            "hallazgos (fragmento textual en el documento, ejemplo citado realmente recuperado, "
            "tipo coherente con el ejemplo citado, fuente de la mitigación realmente recuperada).",
            "",
        ]
    if metrics.abilities is not None:
        a = metrics.abilities
        parts += [
            "## 3. Habilidades",
            "",
            _table(
                ["Habilidad", "Tasa de falsos positivos"],
                [
                    [
                        "Noise robustness (hard negatives)",
                        _rate(a["noise_robustness_fpr_hard_negatives"]),
                    ],
                    ["Negative rejection (limpios)", _rate(a["negative_rejection_fpr_limpios"])],
                ],
            ),
            "",
        ]
    if metrics.quadrants:
        parts += [
            "## 4. Matriz de accionables (por tipo de ataque, umbral 0.5)",
            "",
            _table(
                ["Tipo", "Retriever (CP)", "Generador (F1)", "Acción"],
                [
                    [q["tipo"], q["retriever_cp"], q["generador_f1"], q.get("accion", "—")]
                    for q in metrics.quadrants
                ],
            ),
            "",
        ]
    sweep = metrics.threshold_sweep
    if sweep is not None:
        best = sweep.get("mejor_f1") or {}
        parts += [
            "## Umbral del veredicto",
            "",
            f"Umbral actual {sweep['umbral_actual']}; filtros: {sweep['filtros']}. "
            f"Mejor F1 {_fmt(best.get('f1'))} con umbral {_fmt(best.get('umbral'))}. "
            f"AUROC del score del documento (máxima confianza elegible): "
            f"{_fmt(sweep.get('auroc_score_documento'))}.",
            "",
            _table(
                ["Umbral", "Precision", "Recall", "F1", "FPR hard negatives", "FPR limpios"],
                [
                    [
                        row["umbral"],
                        row["precision"],
                        row["recall"],
                        row["f1"],
                        row["fpr_hard_negatives"],
                        row["fpr_limpios"],
                    ]
                    for row in sweep["filas"]
                    if round(row["umbral"] * 100) % 10 == 0 or row is best
                ],
            ),
            "",
        ]
    c = metrics.cost
    parts += [
        "## 5. Costo",
        "",
        _table(
            ["Métrica", "Valor"],
            [
                [
                    "Latencia mediana / p95 (s)",
                    f"{_fmt(c['latencia_s']['mediana'])} / {_fmt(c['latencia_s']['p95'])}",
                ],
                ["Tiempo total (s)", c["latencia_s"]["total"]],
                ["Chunks / fallidos", f"{c['chunks']} / {c['chunks_fallidos']}"],
                ["Llamadas al modelo (en caché)", f"{c['llamadas']} ({c['llamadas_en_cache']})"],
                ["Tokens prompt / completion", f"{c['tokens_prompt']} / {c['tokens_completion']}"],
                [
                    "Costo total / por documento (USD)",
                    f"{c['costo_usd']} / {c['costo_usd_por_documento']}",
                ],
            ],
        ),
        "",
        "## 6. Porciones",
        "",
        "Recall sobre envenenados y FPR sobre el resto, con intervalo de Wilson al 95%.",
        "",
    ]
    for dimension, table in metrics.slices.items():
        parts += [
            f"### {dimension}",
            "",
            _table(
                ["Valor", "n", "Context Precision", "Hit@1", "Recall", "FPR"],
                [
                    [
                        v,
                        row["n"],
                        row.get("context_precision"),
                        row.get("hit_at_1"),
                        _rate(row.get("recall")),
                        _rate(row.get("fpr")),
                    ]
                    for v, row in table.items()
                ],
            ),
            "",
        ]
    return "\n".join(parts)
