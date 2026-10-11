"""Markdown report renderer producing human-readable forensic reports."""

from pathlib import Path

from corpus_forensics.schemas import DocumentReport


def render_report_to_markdown(report: DocumentReport) -> str:
    """Render a DocumentReport into a human-readable Markdown string.

    Args:
        report: DocumentReport model.

    Returns:
        Formatted Markdown text.
    """
    lines: list[str] = []

    verdict_badge = "🚨 **MALICIOSO**" if report.malicioso else "✅ **LIMPIO**"
    lines.append("# Reporte Forense de Inyección de Prompt / Envenenamiento de Corpus")
    lines.append("")
    lines.append(f"**Documento Analizado:** `{report.documento}`  ")
    lines.append(f"**Veredicto Final:** {verdict_badge}")
    lines.append("")

    # Metadata de análisis
    meta = report.metadata_de_analisis
    lines.append("## Metadatos de Análisis")
    lines.append(f"- **Total de Chunks:** {meta.n_chunks}")
    lines.append(f"- **Loader Utilizado:** `{meta.loader_used}`")
    lines.append(f"- **LLM Provider:** `{'MOCK' if meta.llm_is_mock else 'REAL'}`")
    lines.append(f"- **Embedder Provider:** `{'MOCK' if meta.embedder_is_mock else 'REAL'}`")
    lines.append(f"- **Fecha y Hora (UTC):** `{meta.timestamp}`")
    lines.append("")

    # Hallazgos
    lines.append(f"## Hallazgos Detectados ({len(report.hallazgos)})")
    lines.append("")

    if not report.hallazgos:
        lines.append(
            "> No se detectaron patrones sospechosos de inyección de instrucciones ni técnicas de ocultamiento."
        )
    else:
        for idx, h in enumerate(report.hallazgos, start=1):
            lines.append(f"### Hallazgo #{idx}: {h.tipo_de_ataque.value.upper()}")
            lines.append("")
            lines.append(f"- **Severidad:** `{h.severidad.value.upper()}`")
            lines.append(f"- **Confianza:** `{h.confianza:.2f}`")
            lines.append(
                f"- **Técnica de Ocultamiento:** {', '.join(t.value for t in h.tecnica_de_ocultamiento)}"
            )
            lines.append(f"- **Objetivo del Atacante:** {h.objetivo_del_atacante}")
            lines.append(
                f"- **Offsets en Documento:** `[{h.inicio_caracter}:{h.fin_caracter}]` (Chunk #{h.indice_chunk})"
            )
            lines.append(
                f"- **Verificado contra Texto Fuente:** `{'SÍ (Coincidencia Literal)' if h.verificado else 'NO (Alerta Anti-Alucinación)'}`"
            )
            if h.notas_verificacion:
                lines.append(f"- **Notas de Verificación:** _{h.notas_verificacion}_")
            if h.ataque_similar_id:
                lines.append(
                    f"- **Ataque Similar en KB:** `{h.ataque_similar_id}` (Variante nueva: `{h.posible_variante_nueva}`)"
                )
            lines.append("")
            lines.append("**Fragmento Exacto:**")
            lines.append("```text")
            lines.append(h.fragmento_exacto)
            lines.append("```")
            lines.append("")
            lines.append(f"**Mitigación Recomendada:** {h.mitigacion}")
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def write_markdown_report(report: DocumentReport, output_path: Path | str) -> Path:
    """Write DocumentReport to a Markdown file.

    Args:
        report: DocumentReport to write.
        output_path: File path where markdown will be saved.

    Returns:
        Path of written report.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_report_to_markdown(report)
    path.write_text(content, encoding="utf-8")
    return path
