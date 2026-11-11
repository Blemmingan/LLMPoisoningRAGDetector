"""Computational cost and token metrics."""

import pytest

from corpus_forensics.analysis.timing import StageTimer
from corpus_forensics.evaluation.dataset import EvalLabel
from corpus_forensics.evaluation.metrics import (
    ASSUMED_COMPLETION_TOKENS,
    EvaluationMetrics,
    cost_metrics,
    cost_projection,
    stage_metrics,
    token_metrics,
)
from corpus_forensics.evaluation.records import PredictionRecord, UsageDelta
from corpus_forensics.evaluation.report import render_cost
from corpus_forensics.llm.tokens import count_tokens, encoding_for, schema_tokens
from corpus_forensics.schemas import CategoriaDocumento


def _record(
    name: str, timings: dict[str, float], tokens: dict[str, int], usage: UsageDelta | None = None
) -> PredictionRecord:
    return PredictionRecord(
        documento=name,
        label=EvalLabel(documento=name, categoria=CategoriaDocumento.LIMPIO, malicioso=False),
        timings=timings,
        tokens_estimados=tokens,
        usage=usage or UsageDelta(),
        latency_s=1.0,
    )


TOKENS = {
    "plantilla": 900,
    "chunk": 100,
    "vista_normalizada": 0,
    "ejemplos": 300,
    "literatura": 200,
    "schema": 500,
    "total_entrada": 2000,
    "llamadas": 2,
}


def test_stage_timer_accumulates() -> None:
    timer = StageTimer()
    with timer.stage("a"):
        pass
    with timer.stage("a"):
        pass
    assert set(timer.snapshot()) == {"a"}
    timer.reset()
    assert timer.snapshot() == {}


def test_stage_shares_exclude_sub_stages() -> None:
    records = [
        _record("a", {"carga": 1.0, "recuperacion_ejemplos": 3.0, "ejemplos.reranker": 2.5}, TOKENS)
    ]
    stages = stage_metrics(records)
    assert stages["recuperacion_ejemplos"]["proporcion"] == 0.75
    assert stages["ejemplos.reranker"]["proporcion"] == pytest.approx(0.625)


def test_token_metrics_and_projection() -> None:
    tokens = token_metrics([_record("a", {}, TOKENS), _record("b", {}, TOKENS)])
    assert tokens["entrada_por_llamada"] == 1000 and tokens["llamadas_por_documento"] == 2
    assert tokens["por_seccion_por_llamada"]["plantilla"] == 450
    assert tokens["salida_por_llamada"] == ASSUMED_COMPLETION_TOKENS and tokens["salida_supuesta"]
    projection = cost_projection(tokens, None)
    mini = projection["modelos"]["gpt-4o-mini"]
    expected = (2000 * 0.15 + 2 * ASSUMED_COMPLETION_TOKENS * 0.60) / 1e6
    assert mini["usd_por_documento"] == pytest.approx(expected, rel=1e-3)


def test_measured_usage_replaces_the_assumption() -> None:
    usage = UsageDelta(calls=2, prompt_tokens=2100, completion_tokens=300, cost_usd=0.001)
    tokens = token_metrics([_record("a", {}, TOKENS, usage)])
    assert tokens["salida_por_llamada"] == 150 and not tokens["salida_supuesta"]
    assert tokens["medido"]["entrada_por_llamada"] == 1050


def test_token_counting_uses_the_model_encoding() -> None:
    assert encoding_for("openai/gpt-4o-mini") == "o200k_base"
    assert encoding_for("text-davinci-003") == "cl100k_base"
    assert count_tokens("hola mundo", "gpt-4o-mini") > 0
    assert schema_tokens("gpt-4o-mini") > 100


def test_cost_report_renders() -> None:
    records = [_record("a", {"carga": 0.1, "recuperacion_ejemplos": 0.4}, TOKENS)]
    metrics = EvaluationMetrics(
        total_samples=1,
        cost=cost_metrics(records),
        config={"recursos": {"memoria_pico_mb": 1810.3}},
    )
    text = render_cost(metrics)
    assert "Tiempo por etapa" in text and "gpt-4o-mini" in text and "1810" in text
