"""Ablation variants, their settings and the comparison table."""

import json
from pathlib import Path

import pytest

from corpus_forensics.analysis.prompt_builder import PROMPTS_DIR, resolve_prompt_path
from corpus_forensics.analysis.verifier import FindingVerifier
from corpus_forensics.config import Settings
from corpus_forensics.evaluation.ablation import (
    Variant,
    load_variants,
    summarize,
    variant_settings,
)
from corpus_forensics.schemas import Document, Hallazgo, Severidad, TipoDeAtaque

REPO_VARIANTS = Path(__file__).resolve().parents[1] / "data" / "eval" / "ablations.yaml"


def test_repository_variants_are_valid() -> None:
    variants = load_variants(REPO_VARIANTS)
    names = [v.name for v in variants]
    assert names[0] == "base" and "sin_verificador" in names
    for variant in variants:
        variant_settings(Settings(), variant)  # overrides validate against Settings
    prompt = next(v for v in variants if v.name == "prompt_sin_hard_negatives")
    assert resolve_prompt_path(prompt.overrides["analyst_prompt"]).exists()


def test_unknown_override_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "v.yaml"
    path.write_text("variants:\n  - name: x\n    overrides: {not_a_setting: 1}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_setting"):
        load_variants(path)


def test_variant_settings_apply_overrides() -> None:
    settings = variant_settings(Settings(), Variant(name="v", overrides={"top_k": 0}))
    assert settings.top_k == 0


def test_prompt_without_hard_negatives_differs_only_there() -> None:
    base = (PROMPTS_DIR / "analyst_prompt.txt").read_text(encoding="utf-8")
    variant = resolve_prompt_path("analyst_prompt_sin_hard_negatives").read_text(encoding="utf-8")
    assert "HARD NEGATIVE" in base and "HARD NEGATIVE" not in variant
    assert "Untrusted Content" in variant and "{chunk_text}" in variant


def test_verifier_penalty_can_be_disabled() -> None:
    doc = Document(id="d", raw_content="clean text", visible_text="clean text")

    def finding() -> Hallazgo:
        return Hallazgo(
            tipo_de_ataque=TipoDeAtaque.INYECCION_DIRECTA,
            fragmento_exacto="not in the document",
            objetivo_del_atacante="x",
            severidad=Severidad.ALTA,
            confianza=0.9,
            mitigacion="x",
        )

    assert FindingVerifier().verify([finding()], doc)[0].confianza == pytest.approx(0.18)
    kept = FindingVerifier(penalize=False).verify([finding()], doc)[0]
    assert kept.confianza == 0.9 and not kept.verificado


def _write_run(out: Path, name: str, cp: float, f1: float | None) -> None:
    metrics = {
        "config": {"solo_retrieval": f1 is None, "llm_is_mock": True},
        "retriever": {
            "global": {"context_precision": cp, "context_recall": 0.5, "mrr": cp, "hit_at_1": cp}
        },
        "slices": {"sutileza": {"disfrazado": {"context_precision": cp}}},
        "generator": None
        if f1 is None
        else {"deteccion": {"f1": f1}, "tipos": {"macro_f1": 0.1}, "faithfulness": {"media": 0.5}},
        "cost": {"costo_usd": 0.0, "latencia_s": {"mediana": 0.1}},
    }
    (out / name).mkdir(parents=True)
    (out / name / "eval_summary.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_summary_has_deltas_against_base(tmp_path: Path) -> None:
    _write_run(tmp_path, "base", 0.6, 0.5)
    _write_run(tmp_path, "sin_normalizacion", 0.5, 0.4)
    variants = [Variant(name="base"), Variant(name="sin_normalizacion"), Variant(name="falta")]
    summary = summarize(tmp_path, variants)
    rows = {r["variante"]: r for r in summary["variantes"]}
    assert set(rows) == {"base", "sin_normalizacion"}
    assert rows["sin_normalizacion"]["delta"]["context_precision"] == pytest.approx(-0.1)
    assert rows["sin_normalizacion"]["delta"]["f1"] == pytest.approx(-0.1)
    md = (tmp_path / "ablation_summary.md").read_text(encoding="utf-8")
    assert "0.500 (-0.100)" in md and "Aviso" in md
