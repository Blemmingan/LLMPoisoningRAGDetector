"""Anti-hallucination verification engine for forensic findings."""

from corpus_forensics.schemas import (
    HIDDEN_SECTION_HIDDEN_ELEMENTS,
    HIDDEN_SECTION_HTML_COMMENTS,
    HIDDEN_SECTION_PDF_METADATA,
    CanalDocumento,
    Document,
    Hallazgo,
    TecnicaDeOcultamiento,
)

_HIDDEN_SECTIONS = (
    HIDDEN_SECTION_HTML_COMMENTS,
    HIDDEN_SECTION_HIDDEN_ELEMENTS,
    HIDDEN_SECTION_PDF_METADATA,
)


def hidden_section_at(hidden_text: str, offset: int) -> str | None:
    """Return the hidden-channel section header that contains `offset`, if any."""
    best_header: str | None = None
    best_pos = -1
    for header in _HIDDEN_SECTIONS:
        pos = hidden_text.rfind(header, 0, offset + 1)
        if pos > best_pos:
            best_pos, best_header = pos, header
    return best_header


class FindingVerifier:
    """Performs anti-hallucination validation and sanity checks on forensic findings."""

    def verify(self, findings: list[Hallazgo], document: Document) -> list[Hallazgo]:
        """Verify each finding against the source document.

        Checks:
        1. Literal substring check: Verifies that `fragmento_exacto` literally
           exists in the document, first in the visible text and then in the hidden
           channel, and records the channel (`canal`) and channel-relative offsets.
        2. Confidence penalty: Downgrades confidence score if the fragment is not found.
        3. Diagnostic notes: Documents verification status or failure rationale.

        Args:
            findings: List of candidate findings from aggregator.
            document: Ingestion Document containing source text.

        Returns:
            List of verified Hallazgo instances with updated offsets, flags, and notes.
        """
        visible_text = document.visible_text or document.raw_content
        hidden_text = document.hidden_content

        verified_findings: list[Hallazgo] = []
        for finding in findings:
            fragment = finding.fragmento_exacto
            if not fragment:
                finding.verificado = False
                finding.confianza = 0.0
                finding.notas_verificacion = (
                    "Anti-hallucination check failed: Empty fragment string."
                )
                verified_findings.append(finding)
                continue

            # Literal substring search: visible text first, then the hidden channel
            canal = CanalDocumento.VISIBLE
            idx = visible_text.find(fragment)
            if idx == -1 and hidden_text:
                canal = CanalDocumento.OCULTO
                idx = hidden_text.find(fragment)

            if idx != -1:
                # Substring matched literally
                finding.canal = canal
                finding.inicio_caracter = idx
                finding.fin_caracter = idx + len(fragment)
                finding.verificado = True
                if canal == CanalDocumento.VISIBLE:
                    finding.notas_verificacion = (
                        "Literal substring verified against the visible document text."
                    )
                else:
                    finding.notas_verificacion = (
                        "Literal substring verified against the hidden channel "
                        "(HTML comments, hidden elements or PDF metadata)."
                    )
                    # Deterministic concealment tagging: the loader strips the <!-- -->
                    # markers, so the LLM cannot always tell the fragment was a comment.
                    if hidden_section_at(hidden_text, idx) == HIDDEN_SECTION_HTML_COMMENTS:
                        self._add_technique(finding, TecnicaDeOcultamiento.COMENTARIO_HTML)
            else:
                # Anti-hallucination failure: model quoted text that does not exist in the source document
                finding.verificado = False
                # Penalize confidence score drastically
                finding.confianza = round(max(0.0, finding.confianza * 0.2), 2)
                finding.notas_verificacion = (
                    "Anti-hallucination check FAILED: 'fragmento_exacto' does not exist verbatim "
                    "in the source document."
                )

            # TODO(team): Implement severity / confidence sanity checks:
            # 1. Flag finding if confidence > 0.9 but fragment length < 5 characters.
            # 2. Check for contradiction between attack type and concealment technique.
            # 3. Calibrate confidence score using vector similarity score of closest attack.

            verified_findings.append(finding)

        return verified_findings

    @staticmethod
    def _add_technique(finding: Hallazgo, technique: TecnicaDeOcultamiento) -> None:
        """Add a concealment technique, replacing the 'ninguna' placeholder if present."""
        techniques: list[TecnicaDeOcultamiento] = [
            t for t in finding.tecnica_de_ocultamiento if t != TecnicaDeOcultamiento.NINGUNA
        ]
        if technique not in techniques:
            techniques.append(technique)
        finding.tecnica_de_ocultamiento = techniques
