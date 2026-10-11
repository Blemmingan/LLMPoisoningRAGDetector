"""End-to-end smoke tests for ForensicPipeline on fixture documents."""

import json
from pathlib import Path

from corpus_forensics.analysis.pipeline import ForensicPipeline
from corpus_forensics.config import Settings
from corpus_forensics.reporting.json_writer import serialize_report_to_json
from corpus_forensics.reporting.markdown_renderer import render_report_to_markdown


def test_pipeline_smoke_end_to_end(fixtures_dir: Path, tmp_path: Path) -> None:
    """Run full pipeline over clean, poisoned, and hard-negative fixtures."""
    settings = Settings(
        vectorstore_dir=tmp_path / "vectorstore",
        output_dir=tmp_path / "reports",
        chunk_size=200,
        chunk_overlap=50,
        chunk_unit="char",
        top_k=2,
    )
    pipeline = ForensicPipeline(settings=settings)

    clean_file = fixtures_dir / "clean_doc.txt"
    poisoned_file = fixtures_dir / "poisoned_doc.txt"
    hard_neg_file = fixtures_dir / "hard_negative_doc.txt"

    # 1. Clean document analysis
    clean_report = pipeline.analyze_document(clean_file)
    assert clean_report.malicioso is False
    assert len(clean_report.hallazgos) == 0
    assert clean_report.metadata_de_analisis.llm_is_mock is True
    assert clean_report.metadata_de_analisis.embedder_is_mock is True

    # 2. Poisoned document analysis
    poisoned_report = pipeline.analyze_document(poisoned_file)
    assert poisoned_report.malicioso is True
    assert len(poisoned_report.hallazgos) >= 1

    # Check verified literal offsets against file content
    poisoned_text = poisoned_file.read_text(encoding="utf-8")
    for finding in poisoned_report.hallazgos:
        if finding.verificado:
            assert (
                poisoned_text[finding.inicio_caracter : finding.fin_caracter]
                == finding.fragmento_exacto
            )

    # 3. Hard negative document analysis
    hard_neg_report = pipeline.analyze_document(hard_neg_file)
    assert hard_neg_report.malicioso is False

    # 4. JSON Serialization check (Spanish keys requirement)
    json_str = serialize_report_to_json(poisoned_report)
    parsed_json = json.loads(json_str)
    assert "metadata de análisis" in parsed_json
    assert "documento" in parsed_json
    assert "malicioso" in parsed_json
    assert "hallazgos" in parsed_json

    # 5. Markdown rendering check
    md_str = render_report_to_markdown(poisoned_report)
    assert "MALICIOSO" in md_str
    assert "Metadatos de Análisis" in md_str
    assert "Hallazgos Detectados" in md_str

    # 6. Non-parametric memory rule: Analyzed docs must NOT be persisted in vector store
    assert pipeline.vector_store.count() == 0
