"""Hidden text detection in PDFs (benign markers only)."""

from pathlib import Path

import pytest

from corpus_forensics.ingestion.pdf_loader import PdfLoader, hidden_reason
from corpus_forensics.schemas import HIDDEN_SECTION_PDF_HIDDEN_TEXT
from tests.pdf_factory import text_segment, write_pdf

VISIBLE = "The lighthouse guides ships along the coast at night."


@pytest.mark.parametrize(
    ("segment", "reason"),
    [
        (text_segment("HIDDEN MARKER ALPHA", 72, 650, prefix="3 Tr"), "modo_invisible"),
        (text_segment("HIDDEN MARKER BRAVO", 72, 650, prefix="1 1 1 rg"), "blanco"),
        (text_segment("HIDDEN MARKER CHARLIE", 72, 650, size=0.4), "fuente_minima"),
        (text_segment("HIDDEN MARKER DELTA", 900, 650), "fuera_de_pagina"),
        (text_segment("HIDDEN MARKER ECHO", 72, 650, prefix="/GS0 gs"), "transparente"),
    ],
)
def test_hidden_text_goes_to_hidden_channel(tmp_path: Path, segment: str, reason: str) -> None:
    path = write_pdf(tmp_path / "doc.pdf", [text_segment(VISIBLE, 72, 700), segment])
    doc = PdfLoader().load(path)
    marker = segment.split("(")[1].split(")")[0]
    assert "lighthouse" in doc.visible_text
    assert marker not in doc.visible_text
    assert HIDDEN_SECTION_PDF_HIDDEN_TEXT in doc.hidden_content
    assert marker in doc.hidden_content and reason in doc.hidden_content
    assert doc.metadata["hidden_text_reasons"] == {reason: 1}


def test_clean_pdf_has_no_hidden_text(tmp_path: Path) -> None:
    path = write_pdf(
        tmp_path / "clean.pdf",
        [text_segment(VISIBLE, 72, 700), text_segment("Second visible line.", 72, 680)],
    )
    doc = PdfLoader().load(path)
    assert "Second visible line." in doc.visible_text
    assert HIDDEN_SECTION_PDF_HIDDEN_TEXT not in doc.hidden_content
    assert doc.metadata["extractor"] == "pdfium" and doc.metadata["hidden_text_runs"] == 0


def test_hidden_reason_rules() -> None:
    page = (612.0, 792.0)
    box = (72.0, 80.0, 700.0, 710.0)
    assert hidden_reason(0, (0, 0, 0, 255), 12, box, page) is None
    assert hidden_reason(3, (0, 0, 0, 255), 12, box, page) == "modo_invisible"
    assert hidden_reason(0, (250, 250, 250, 255), 12, box, page) == "blanco"
    assert hidden_reason(0, (0, 0, 0, 0), 12, box, page) == "transparente"
    assert hidden_reason(0, (0, 0, 0, 255), 0.5, box, page) == "fuente_minima"
    assert hidden_reason(0, (0, 0, 0, 255), 12, (-50, -40, 700, 710), page) == "fuera_de_pagina"


def test_verifier_tags_findings_in_pdf_hidden_text(tmp_path: Path) -> None:
    from corpus_forensics.analysis.verifier import FindingVerifier
    from corpus_forensics.schemas import (
        CanalDocumento,
        Hallazgo,
        Severidad,
        TecnicaDeOcultamiento,
        TipoDeAtaque,
    )

    path = write_pdf(
        tmp_path / "doc.pdf",
        [
            text_segment(VISIBLE, 72, 700),
            text_segment("HIDDEN MARKER FOXTROT", 72, 650, prefix="1 1 1 rg"),
        ],
    )
    doc = PdfLoader().load(path)
    finding = Hallazgo(
        tipo_de_ataque=TipoDeAtaque.INYECCION_DIRECTA,
        fragmento_exacto="HIDDEN MARKER FOXTROT",
        objetivo_del_atacante="x",
        severidad=Severidad.ALTA,
        confianza=0.9,
        mitigacion="x",
    )
    verified = FindingVerifier().verify([finding], doc)[0]
    assert verified.verificado and verified.canal == CanalDocumento.OCULTO
    assert TecnicaDeOcultamiento.TEXTO_BLANCO in verified.tecnica_de_ocultamiento
