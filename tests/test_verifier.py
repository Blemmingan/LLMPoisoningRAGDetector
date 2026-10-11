"""Tests for anti-hallucination finding verifier."""

from corpus_forensics.analysis.verifier import FindingVerifier
from corpus_forensics.schemas import (
    Document,
    Hallazgo,
    Severidad,
    TecnicaDeOcultamiento,
    TipoDeAtaque,
)


def test_verifier_literal_match_success(sample_document: Document) -> None:
    """Verify that a legitimate verbatim fragment is verified with exact character offsets."""
    verifier = FindingVerifier()

    target_fragment = "Ignore previous instructions"
    finding = Hallazgo(
        tipo_de_ataque=TipoDeAtaque.INYECCION_DIRECTA,
        fragmento_exacto=target_fragment,
        tecnica_de_ocultamiento=[TecnicaDeOcultamiento.NINGUNA],
        objetivo_del_atacante="Prompt override",
        severidad=Severidad.ALTA,
        confianza=0.95,
        mitigacion="Block instruction",
        verificado=False,
    )

    results = verifier.verify([finding], sample_document)
    assert len(results) == 1
    v = results[0]

    assert v.verificado is True
    assert v.confianza == 0.95
    assert sample_document.visible_text[v.inicio_caracter : v.fin_caracter] == target_fragment
    assert "Literal substring verified" in (v.notas_verificacion or "")


def test_verifier_hallucination_penalty(sample_document: Document) -> None:
    """Verify that a hallucinated fragment (not in document) fails check and gets confidence penalized."""
    verifier = FindingVerifier()

    hallucinated_fragment = "This string does not appear anywhere in the document."
    finding = Hallazgo(
        tipo_de_ataque=TipoDeAtaque.INYECCION_DIRECTA,
        fragmento_exacto=hallucinated_fragment,
        tecnica_de_ocultamiento=[TecnicaDeOcultamiento.NINGUNA],
        objetivo_del_atacante="Fake attack",
        severidad=Severidad.ALTA,
        confianza=0.90,
        mitigacion="None",
        verificado=False,
    )

    results = verifier.verify([finding], sample_document)
    assert len(results) == 1
    v = results[0]

    assert v.verificado is False
    assert v.confianza < 0.90  # Heavily downgraded
    assert "FAILED" in (v.notas_verificacion or "")


def test_verifier_empty_fragment(sample_document: Document) -> None:
    """Verify handling of empty fragment."""
    verifier = FindingVerifier()
    finding = Hallazgo(
        tipo_de_ataque=TipoDeAtaque.OTRO,
        fragmento_exacto="",
        tecnica_de_ocultamiento=[TecnicaDeOcultamiento.NINGUNA],
        objetivo_del_atacante="None",
        severidad=Severidad.BAJA,
        confianza=0.5,
        mitigacion="None",
    )
    results = verifier.verify([finding], sample_document)
    assert results[0].verificado is False
    assert results[0].confianza == 0.0
