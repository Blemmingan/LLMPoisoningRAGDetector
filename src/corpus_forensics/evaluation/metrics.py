"""Evaluation metrics, following the RAG evaluation scheme of the course (class 11).

The class splits RAG evaluation into the retriever and the generator, plus abilities,
ranking and cost. Mapped to this system ("question" = the chunk under analysis,
"context" = retrieved knowledge-base examples and literature passages, "answer" = the
forensic finding, "ground truth" = the dataset label):

Retriever (no LLM needed):
- Context Precision@K: the class formula, sum_k(Precision@k * v_k) / relevant in top K,
  with v_k = 1 when the k-th retrieved example has the labeled attack type.
- Context Recall: share of the label's claims (attack type, and concealment technique
  when there is one) that at least one retrieved example supports.
- Ranking (class: MRR, NDCG): reciprocal rank of the first relevant example, NDCG@K.
Generator:
- Faithfulness: share of a finding's checkable claims that the context supports: the
  quoted fragment is verbatim in the document (verifier), the cited example is one that
  was actually retrieved, the declared type matches the cited example's type, and the
  literature passage cited for the mitigation was actually retrieved.
- Answer relevance: the class's reference-free version (questions generated from the
  answer) fits free text; our answer is a labeled classification, so it is measured
  against the ground truth: precision/recall/F1, macro-F1 by attack type, confusion
  matrix and fragment coverage.
Abilities: noise robustness = false positive rate on hard negatives; negative
rejection = false positive rate on clean documents. Information integration does not
apply (one document at a time).
Cost: latency, tokens, calls and USD.

Small slices get a 95% Wilson interval so differences are not over-read.
"""

import math
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from pydantic import BaseModel, Field

from corpus_forensics.analysis.trace import ChunkTrace
from corpus_forensics.analysis.verdict import VerdictPolicy
from corpus_forensics.evaluation.records import PredictionRecord
from corpus_forensics.schemas import (
    CategoriaDocumento,
    DocumentReport,
    Hallazgo,
    Severidad,
    TecnicaDeOcultamiento,
)

ATTACK_TYPES = (
    "inyeccion_directa",
    "jailbreak",
    "exfiltracion",
    "prompt_leaking",
    "trigger_condicional",
)
CLEAN_LABEL = "limpio"
# literature topic labels that count as relevant for each attack type
TYPE_TO_LITERATURE_LABELS: dict[str, set[str]] = {
    "inyeccion_directa": {"cubre_inyeccion", "cubre_indirecta"},
    "jailbreak": {"cubre_jailbreak"},
    "exfiltracion": {"cubre_exfiltracion"},
    "prompt_leaking": {"cubre_prompt_leaking"},
    "trigger_condicional": {"cubre_trigger"},
}
GRAM = 5  # character n-gram size used to compare fragments
LOCATE_MIN_COVERAGE = 0.3  # below this, the attack is not considered found in any chunk
FRAGMENT_MATCH_COVERAGE = 0.5
SLICE_DIMENSIONS = (
    "categoria",
    "tipo_de_ataque",
    "posicion",
    "sutileza",
    "tecnica_de_ocultamiento",
    "formato",
    "idioma_texto",
    "fuente_ataque",
)
# class 11, slide 25: what to do given retriever and generator scores
QUADRANT_ACTIONS = {
    (True, True): "caso ideal",
    (False, True): "revisar parámetros de indexado (retriever bajo)",
    (True, False): "prompt engineering / parámetros del modelo (generador bajo)",
    (False, False): "revisar la arquitectura y la base de conocimiento",
}


class EvaluationMetrics(BaseModel):
    """All evaluation results of one run."""

    total_samples: int
    config: dict[str, Any] = Field(default_factory=dict)
    retriever: dict[str, Any] = Field(default_factory=dict)
    generator: dict[str, Any] | None = None
    abilities: dict[str, Any] | None = None
    cost: dict[str, Any] = Field(default_factory=dict)
    quadrants: list[dict[str, Any]] = Field(default_factory=list)
    threshold_sweep: dict[str, Any] | None = None
    slices: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- helpers


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def proportion(successes: int, n: int) -> dict[str, Any]:
    """A rate with its count and Wilson interval."""
    return {
        "valor": round(successes / n, 4) if n else None,
        "n": n,
        "ic95": list(wilson_interval(successes, n)),
    }


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _grams(text: str) -> set[str]:
    text = _normalize(text)
    if len(text) < GRAM:
        return {text} if text else set()
    return {text[i : i + GRAM] for i in range(len(text) - GRAM + 1)}


def coverage(reference: str, candidate: str) -> float:
    """Share of the reference's character 5-grams present in the candidate."""
    norm_ref = _normalize(reference)
    if len(norm_ref) < GRAM:  # too short for n-grams: plain containment
        return 1.0 if norm_ref and norm_ref in _normalize(candidate) else 0.0
    ref = _grams(norm_ref)
    return len(ref & _grams(candidate)) / len(ref)


def _avg(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def auroc(positives: list[float], negatives: list[float]) -> float | None:
    """Area under the ROC curve (probability a positive scores above a negative)."""
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return round(wins / (len(positives) * len(negatives)), 4)


# --------------------------------------------------------------------------- retriever


def context_precision(relevance: list[bool]) -> float:
    """Class formula: sum_k (Precision@k * v_k) / number of relevant items in the top K."""
    hits = 0
    total = 0.0
    for k, relevant in enumerate(relevance, start=1):
        if relevant:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def reciprocal_rank(relevance: list[bool]) -> float:
    return next((1 / k for k, rel in enumerate(relevance, start=1) if rel), 0.0)


def ndcg(relevance: list[bool]) -> float:
    """NDCG@K with binary relevance (the knowledge base has examples of every type)."""
    dcg = sum(1 / math.log2(k + 1) for k, rel in enumerate(relevance, start=1) if rel)
    ideal = sum(1 / math.log2(k + 1) for k in range(1, len(relevance) + 1))
    return dcg / ideal if ideal else 0.0


def locate_attack(traces: list[ChunkTrace], fragment: str) -> tuple[ChunkTrace | None, float]:
    """The chunk that contains most of the attack, and how much of it."""
    best: tuple[ChunkTrace | None, float] = (None, 0.0)
    for trace in traces:
        cov = coverage(fragment, trace.text)
        if cov > best[1]:
            best = (trace, cov)
    return best if best[1] >= LOCATE_MIN_COVERAGE else (None, best[1])


def retrieval_scores(record: PredictionRecord) -> dict[str, Any] | None:
    """Retriever scores for the attacked chunk of a poisoned document."""
    label = record.label
    if not (label.malicioso and label.tipo_de_ataque and label.fragmento_exacto):
        return None
    chunk, cov = locate_attack(record.traces, label.fragmento_exacto)
    if chunk is None or not chunk.examples:
        return {"localizado": False, "cobertura_chunk": round(cov, 4)}
    gt_type = label.tipo_de_ataque.value
    relevance = [e.attack_type == gt_type for e in chunk.examples]
    concealments = [
        t.value for t in label.tecnica_de_ocultamiento if t != TecnicaDeOcultamiento.NINGUNA
    ]
    claims = [any(relevance)]
    if concealments:
        claims.append(any(e.concealment in concealments for e in chunk.examples))
    scores: dict[str, Any] = {
        "localizado": True,
        "cobertura_chunk": round(cov, 4),
        "k": len(relevance),
        "context_precision": context_precision(relevance),
        "context_recall": sum(claims) / len(claims),
        "hit_at_1": float(relevance[0]),
        "hit_at_k": float(any(relevance)),
        "mrr": reciprocal_rank(relevance),
        "ndcg": ndcg(relevance),
        "similitud_top1": chunk.examples[0].similarity,
    }
    if chunk.passages:
        wanted = TYPE_TO_LITERATURE_LABELS.get(gt_type, set())
        lit_relevance = [bool(wanted & set(p.labels)) for p in chunk.passages]
        scores["literatura_context_precision"] = context_precision(lit_relevance)
        scores["literatura_hit_at_k"] = float(any(lit_relevance))
    return scores


def _doc_max_similarity(record: PredictionRecord) -> float | None:
    sims = [t.examples[0].similarity for t in record.traces if t.examples]
    return max(sims) if sims else None


def chance_levels(kb_by_type: dict[str, int], k: int) -> dict[str, dict[str, float]]:
    """Hit rates a random retriever would get, from the knowledge-base type shares.

    With 77% of the examples being direct injection, retrieving that type is easy even at
    random; scores are read against these levels.
    """
    total = sum(kb_by_type.values())
    if not total:
        return {}
    return {
        t: {
            "azar_hit_at_1": round(n / total, 4),
            "azar_hit_at_k": round(1 - (1 - n / total) ** k, 4),
        }
        for t, n in kb_by_type.items()
    }


def retriever_metrics(
    records: list[PredictionRecord],
    per_doc: dict[str, dict[str, Any]],
    kb_by_type: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Aggregate retriever metrics over poisoned documents, overall and by attack type."""
    located = [s for s in per_doc.values() if s.get("localizado")]
    keys = (
        "context_precision",
        "context_recall",
        "hit_at_1",
        "hit_at_k",
        "mrr",
        "ndcg",
        "similitud_top1",
        "literatura_context_precision",
        "literatura_hit_at_k",
    )

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {key: _avg([s[key] for s in items if key in s]) for key in keys} | {"n": len(items)}

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        scores = per_doc.get(record.documento)
        if scores and scores.get("localizado") and record.label.tipo_de_ataque:
            by_type[record.label.tipo_de_ataque.value].append(scores)

    # how well the top-1 similarity alone separates poisoned from non-poisoned documents
    poisoned = [s for r in records if r.label.malicioso if (s := _doc_max_similarity(r))]
    hard = [
        s
        for r in records
        if r.label.categoria == CategoriaDocumento.HARD_NEGATIVE
        if (s := _doc_max_similarity(r))
    ]
    negatives = [s for r in records if not r.label.malicioso if (s := _doc_max_similarity(r))]
    k_values = Counter(s["k"] for s in located)
    k = k_values.most_common(1)[0][0] if k_values else None
    chance = chance_levels(kb_by_type or {}, k or 1)
    per_type = {
        t: summarize(by_type[t]) | chance.get(t, {}) for t in ATTACK_TYPES if by_type.get(t)
    }
    macro = _avg(
        [v["context_precision"] for v in per_type.values() if v["context_precision"] is not None]
    )
    return {
        "k": k,
        # every attack type weighs the same (the knowledge base is 77% direct injection)
        "context_precision_macro": macro,
        "envenenados": len(per_doc),
        "ataque_localizado_en_un_chunk": len(located),
        "global": summarize(located),
        "por_tipo": per_type,
        "linea_base_similitud": {
            "descripcion": "AUROC de la similitud top-1 máxima del documento como único detector",
            "auroc_vs_negativos": auroc(poisoned, negatives),
            "auroc_vs_hard_negatives": auroc(poisoned, hard),
        },
    }


# --------------------------------------------------------------------------- generator


def _flagged(record: PredictionRecord) -> bool:
    """Whether the pipeline flagged the document (False when there is no report)."""
    return record.report is not None and record.report.malicioso


def _findings(record: PredictionRecord) -> list[Hallazgo]:
    return record.report.hallazgos if record.report is not None else []


def _count(records: list[PredictionRecord]) -> int:
    return len([r for r in records if _flagged(r)])


def predicted_label(report: DocumentReport) -> str:
    """'limpio', or the type of the most confident finding."""
    if not report.malicioso or not report.hallazgos:
        return CLEAN_LABEL
    return max(report.hallazgos, key=lambda h: h.confianza).tipo_de_ataque.value


def true_label(record: PredictionRecord) -> str:
    label = record.label
    if label.malicioso and label.tipo_de_ataque:
        return label.tipo_de_ataque.value
    return CLEAN_LABEL


def classification_metrics(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    tp = sum(1 for t, p in pairs if t and p)
    fp = sum(1 for t, p in pairs if not t and p)
    fn = sum(1 for t, p in pairs if t and not p)
    tn = sum(1 for t, p in pairs if not t and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_f1(precision, recall), 4),
        "accuracy": round((tp + tn) / len(pairs), 4) if pairs else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def per_type_metrics(labels: list[tuple[str, str]]) -> dict[str, Any]:
    """One-vs-rest precision/recall/F1 per attack type over all documents, and macro-F1."""
    out: dict[str, Any] = {}
    present = [t for t in ATTACK_TYPES if any(gt == t for gt, _ in labels)]
    for attack_type in present:
        tp = sum(1 for gt, pr in labels if gt == attack_type and pr == attack_type)
        fp = sum(1 for gt, pr in labels if gt != attack_type and pr == attack_type)
        fn = sum(1 for gt, pr in labels if gt == attack_type and pr != attack_type)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[attack_type] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(_f1(p, r), 4),
            "n": tp + fn,
        }
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for gt, pr in labels:
        confusion[gt][pr] += 1
    return {
        "macro_f1": _avg([m["f1"] for m in out.values()]),
        "por_tipo": out,
        "matriz_confusion": {gt: dict(row) for gt, row in confusion.items()},
    }


def faithfulness_claims(record: PredictionRecord) -> list[float]:
    """Faithfulness score of each finding (supported claims / checkable claims)."""
    if record.report is None:
        return []
    traces = {t.index: t for t in record.traces}
    scores: list[float] = []
    for finding in record.report.hallazgos:
        claims = [finding.verificado]
        if finding.ataque_similar_id:
            trace = traces.get(finding.indice_chunk)
            cited = next(
                (e for e in (trace.examples if trace else []) if e.id == finding.ataque_similar_id),
                None,
            )
            claims.append(cited is not None)
            claims.append(cited is not None and cited.attack_type == finding.tipo_de_ataque.value)
        if finding.fuente_mitigacion_id:
            trace = traces.get(finding.indice_chunk)
            claims.append(
                any(p.id == finding.fuente_mitigacion_id for p in (trace.passages if trace else []))
            )
        scores.append(sum(claims) / len(claims))
    return scores


def fragment_scores(record: PredictionRecord) -> tuple[float, float] | None:
    """(coverage of the true fragment, precision of the predicted one) for the best finding."""
    if record.report is None or not record.label.fragmento_exacto:
        return None
    gt = record.label.fragmento_exacto
    best = (0.0, 0.0)
    for finding in record.report.hallazgos:
        pair = (coverage(gt, finding.fragmento_exacto), coverage(finding.fragmento_exacto, gt))
        best = max(best, pair)
    return best


def generator_metrics(records: list[PredictionRecord]) -> dict[str, Any]:
    analyzed = [r for r in records if r.report is not None]
    reports = [(r, r.report) for r in records if r.report is not None]
    detection = classification_metrics([(r.label.malicioso, rep.malicioso) for r, rep in reports])
    types = per_type_metrics([(true_label(r), predicted_label(rep)) for r, rep in reports])

    poisoned = [r for r in analyzed if r.label.malicioso and r.label.tipo_de_ataque]
    any_type = len(
        [
            r
            for r in poisoned
            if any(h.tipo_de_ataque == r.label.tipo_de_ataque for h in _findings(r))
        ]
    )
    concealed = [
        r
        for r in poisoned
        if any(t != TecnicaDeOcultamiento.NINGUNA for t in r.label.tecnica_de_ocultamiento)
    ]
    concealment_hits = len(
        [
            r
            for r in concealed
            if set(r.label.tecnica_de_ocultamiento)
            & {t for h in _findings(r) for t in h.tecnica_de_ocultamiento}
        ]
    )
    fragments = [s for r in poisoned if (s := fragment_scores(r)) is not None]
    faith = [s for r in analyzed for s in faithfulness_claims(r)]
    return {
        "deteccion": detection,
        "tipos": types,
        "tipo_en_algun_hallazgo": proportion(any_type, len(poisoned)),
        "ocultamiento_identificado": proportion(concealment_hits, len(concealed)),
        "fragmento": {
            "coincidencia": proportion(
                len([c for c, _ in fragments if c >= FRAGMENT_MATCH_COVERAGE]), len(fragments)
            ),
            "cobertura_media": _avg([c for c, _ in fragments]),
            "precision_media": _avg([p for _, p in fragments]),
            "umbral_cobertura": FRAGMENT_MATCH_COVERAGE,
        },
        "faithfulness": {"media": _avg(faith), "hallazgos": len(faith)},
    }


def ability_metrics(records: list[PredictionRecord]) -> dict[str, Any]:
    analyzed = [r for r in records if r.report is not None]

    def fpr(category: CategoriaDocumento) -> dict[str, Any]:
        docs = [r for r in analyzed if r.label.categoria == category]
        return proportion(_count(docs), len(docs))

    return {
        "noise_robustness_fpr_hard_negatives": fpr(CategoriaDocumento.HARD_NEGATIVE),
        "negative_rejection_fpr_limpios": fpr(CategoriaDocumento.LIMPIO),
    }


# --------------------------------------------------------------------------- cost, slices


# Completion tokens per analyst call until a real run measures them: a clean chunk answers
# with a short rationale and an empty list (~50 tokens); a finding adds ~200. With ~85% of
# chunks clean, ~120 per call is used. Replaced by the ledger average when there is one.
ASSUMED_COMPLETION_TOKENS = 120
PROJECTION_MODELS = ("gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini")
TOKEN_SECTIONS = (
    "plantilla",
    "chunk",
    "vista_normalizada",
    "ejemplos",
    "literatura",
    "schema",
)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)] if ordered else 0.0


def stage_metrics(records: list[PredictionRecord]) -> dict[str, Any]:
    """Seconds per stage: total, mean per document and share of the pipeline time."""
    totals: dict[str, float] = defaultdict(float)
    for record in records:
        for stage, seconds in record.timings.items():
            totals[stage] += seconds
    top_level = {k: v for k, v in totals.items() if not k.startswith("ejemplos.")}
    pipeline_total = sum(top_level.values())
    n = len(records) or 1
    return {
        stage: {
            "total_s": round(seconds, 2),
            "media_por_doc_s": round(seconds / n, 4),
            "proporcion": round(seconds / pipeline_total, 4) if pipeline_total else None,
        }
        for stage, seconds in sorted(totals.items(), key=lambda kv: -kv[1])
    }


def token_metrics(records: list[PredictionRecord]) -> dict[str, Any]:
    """Estimated analyst-prompt tokens (per call, per document) and measured ones if any."""
    with_estimate = [r for r in records if r.tokens_estimados.get("llamadas")]
    calls = sum(r.tokens_estimados["llamadas"] for r in with_estimate)
    if not calls:
        return {}
    section_totals = {
        key: sum(r.tokens_estimados.get(key, 0) for r in with_estimate) for key in TOKEN_SECTIONS
    }
    input_total = sum(r.tokens_estimados.get("total_entrada", 0) for r in with_estimate)
    measured_calls = sum(r.usage.calls - r.usage.cached_calls for r in records)
    measured_prompt = sum(r.usage.prompt_tokens for r in records)
    measured_completion = sum(r.usage.completion_tokens for r in records)
    return {
        "llamadas_estimadas": calls,
        "entrada_por_llamada": round(input_total / calls, 1),
        "entrada_por_documento": round(input_total / len(with_estimate), 1),
        "llamadas_por_documento": round(calls / len(with_estimate), 2),
        "por_seccion_por_llamada": {
            key: round(total / calls, 1) for key, total in section_totals.items()
        },
        "proporcion_por_seccion": {
            key: round(total / input_total, 4) if input_total else None
            for key, total in section_totals.items()
        },
        "salida_por_llamada": round(measured_completion / measured_calls, 1)
        if measured_calls
        else ASSUMED_COMPLETION_TOKENS,
        "salida_supuesta": not measured_calls,
        "medido": {
            "llamadas": measured_calls,
            "entrada_por_llamada": round(measured_prompt / measured_calls, 1)
            if measured_calls
            else None,
        },
    }


def cost_projection(tokens: dict[str, Any], measured_usd_per_doc: float | None) -> dict[str, Any]:
    """USD per document and per 1,000 / 10,000 documents for each candidate model."""
    from corpus_forensics.llm.usage import APPROX_PRICES_PER_MTOK  # noqa: PLC0415

    if not tokens:
        return {}
    calls = tokens["llamadas_por_documento"]
    tokens_in = tokens["entrada_por_llamada"] * calls
    tokens_out = tokens["salida_por_llamada"] * calls
    models = {}
    for model in PROJECTION_MODELS:
        price_in, price_out = APPROX_PRICES_PER_MTOK[model]
        per_doc = (tokens_in * price_in + tokens_out * price_out) / 1e6
        models[model] = {
            "usd_por_documento": round(per_doc, 6),
            "usd_por_1000_documentos": round(per_doc * 1000, 3),
            "usd_por_10000_documentos": round(per_doc * 10000, 2),
        }
    return {
        "tokens_entrada_por_documento": round(tokens_in, 1),
        "tokens_salida_por_documento": round(tokens_out, 1),
        "modelos": models,
        "usd_por_documento_medido": measured_usd_per_doc,
    }


def cost_metrics(records: list[PredictionRecord]) -> dict[str, Any]:
    latencies = [r.latency_s for r in records]
    reports = [r.report for r in records if r.report is not None]
    total_usd = sum(r.usage.cost_usd for r in records)
    tokens = token_metrics(records)
    return {
        "latencia_s": {
            "mediana": round(median(latencies), 3) if latencies else None,
            "p95": round(_percentile(latencies, 0.95), 3),
            "total": round(sum(latencies), 1),
        },
        "chunks": sum(len(r.traces) for r in records),
        "chunks_fallidos": sum(rep.metadata_de_analisis.n_chunks_failed for rep in reports),
        "llamadas": sum(r.usage.calls for r in records),
        "llamadas_en_cache": sum(r.usage.cached_calls for r in records),
        "tokens_prompt": sum(r.usage.prompt_tokens for r in records),
        "tokens_completion": sum(r.usage.completion_tokens for r in records),
        "costo_usd": round(total_usd, 4),
        "costo_usd_por_documento": round(total_usd / len(records), 5) if records else 0.0,
        "etapas": stage_metrics(records),
        "tokens": tokens,
        "proyeccion": cost_projection(
            tokens, round(total_usd / len(records), 6) if records and total_usd else None
        ),
    }


def slice_value(record: PredictionRecord, dimension: str) -> list[str]:
    label = record.label
    if dimension == "tecnica_de_ocultamiento":
        return [t.value for t in label.tecnica_de_ocultamiento] if label.malicioso else []
    raw = getattr(label, dimension, None) if dimension in type(label).model_fields else None
    if raw is None:
        raw = label.extra.get(dimension)
    if raw is None:
        return []
    value = getattr(raw, "value", raw)
    if dimension == "fuente_ataque":
        value = str(value).split(":")[0]
    return [str(value)]


def slice_metrics(
    records: list[PredictionRecord], per_doc: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Per-slice counts: retriever scores, detection recall on poisoned documents and
    false positive rate on the rest, each with its Wilson interval."""
    out: dict[str, Any] = {}
    for dimension in SLICE_DIMENSIONS:
        groups: dict[str, list[PredictionRecord]] = defaultdict(list)
        for record in records:
            for value in slice_value(record, dimension):
                groups[value].append(record)
        if not groups:
            continue
        table: dict[str, Any] = {}
        for value, group in sorted(groups.items()):
            positives = [r for r in group if r.label.malicioso]
            negatives = [r for r in group if not r.label.malicioso]
            row: dict[str, Any] = {"n": len(group)}
            located = [
                per_doc[r.documento]
                for r in positives
                if per_doc.get(r.documento, {}).get("localizado")
            ]
            if located:
                row["context_precision"] = _avg([s["context_precision"] for s in located])
                row["hit_at_1"] = _avg([s["hit_at_1"] for s in located])
            pos_an = [r for r in positives if r.report is not None]
            neg_an = [r for r in negatives if r.report is not None]
            if pos_an:
                row["recall"] = proportion(_count(pos_an), len(pos_an))
            if neg_an:
                row["fpr"] = proportion(_count(neg_an), len(neg_an))
            table[value] = row
        out[dimension] = table
    return out


def quadrant_table(
    retriever: dict[str, Any], generator: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Class 11 slide 25: each attack type placed by retriever and generator score."""
    rows = []
    for attack_type, scores in retriever.get("por_tipo", {}).items():
        retr = scores.get("context_precision")
        gen = (generator or {}).get("tipos", {}).get("por_tipo", {}).get(attack_type, {}).get("f1")
        row: dict[str, Any] = {"tipo": attack_type, "retriever_cp": retr, "generador_f1": gen}
        if retr is not None and gen is not None:
            row["accion"] = QUADRANT_ACTIONS[(retr >= 0.5, gen >= 0.5)]
        rows.append(row)
    return rows


SWEEP_THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))


def policy_from_config(config: dict[str, Any]) -> VerdictPolicy:
    """The verdict policy a run used (defaults when the config does not say)."""
    defaults = VerdictPolicy()
    return VerdictPolicy(
        threshold=config.get("verdict_threshold", defaults.threshold),
        require_verified=config.get("verdict_require_verified", defaults.require_verified),
        min_severity=config.get("verdict_min_severity", defaults.min_severity),
    )


def threshold_sweep(records: list[PredictionRecord], policy: VerdictPolicy) -> dict[str, Any]:
    """Re-decide every document at each threshold from the saved findings.

    Uses the run's policy (verified fragment, minimum severity) and only moves the
    confidence threshold, so the threshold can be chosen from one LLM run without new calls.
    """
    analyzed = [(r, r.report) for r in records if r.report is not None]
    scores = [
        (r, policy.score(rep.hallazgos), any(policy.counts(h) for h in rep.hallazgos))
        for r, rep in analyzed
    ]
    rows = []
    for threshold in SWEEP_THRESHOLDS:
        flagged = [(r, eligible and score >= threshold) for r, score, eligible in scores]
        det = classification_metrics([(r.label.malicioso, f) for r, f in flagged])

        def fpr(
            category: CategoriaDocumento, flagged: list[tuple[PredictionRecord, bool]] = flagged
        ) -> float | None:
            docs = [f for r, f in flagged if r.label.categoria == category]
            return round(sum(docs) / len(docs), 4) if docs else None

        rows.append(
            {
                "umbral": threshold,
                "precision": det["precision"],
                "recall": det["recall"],
                "f1": det["f1"],
                "fpr_hard_negatives": fpr(CategoriaDocumento.HARD_NEGATIVE),
                "fpr_limpios": fpr(CategoriaDocumento.LIMPIO),
            }
        )
    best = max(rows, key=lambda row: (row["f1"], row["umbral"])) if rows else None
    positives = [s for r, s, e in scores if r.label.malicioso]
    negatives = [s for r, s, e in scores if not r.label.malicioso]
    return {
        "filtros": ", ".join(
            [
                *(["fragmento verificado"] if policy.require_verified else []),
                *(
                    [f"severidad >= {policy.min_severity.value}"]
                    if policy.min_severity != Severidad.BAJA
                    else []
                ),
            ]
        )
        or "ninguno",
        "umbral_actual": policy.threshold,
        "mejor_f1": best,
        "auroc_score_documento": auroc(positives, negatives),
        "filas": rows,
    }


def compute_all_metrics(
    records: list[PredictionRecord], config: dict[str, Any] | None = None
) -> EvaluationMetrics:
    """All metrics of a run. Generator and ability metrics need at least one report."""
    per_doc = {r.documento: s for r in records if (s := retrieval_scores(r)) is not None}
    retriever = retriever_metrics(records, per_doc, (config or {}).get("kb_por_tipo"))
    has_reports = any(r.report is not None for r in records)
    generator = generator_metrics(records) if has_reports else None
    return EvaluationMetrics(
        total_samples=len(records),
        config=config or {},
        retriever=retriever,
        generator=generator,
        abilities=ability_metrics(records) if has_reports else None,
        cost=cost_metrics(records),
        quadrants=quadrant_table(retriever, generator),
        threshold_sweep=threshold_sweep(records, policy_from_config(config or {}))
        if has_reports
        else None,
        slices=slice_metrics(records, per_doc),
    )
