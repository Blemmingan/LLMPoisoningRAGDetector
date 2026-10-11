"""Anti-hallucination verification engine for forensic findings."""

from corpus_forensics.schemas import Document, Hallazgo


class FindingVerifier:
    """Performs anti-hallucination validation and sanity checks on forensic findings."""

    def verify(self, findings: list[Hallazgo], document: Document) -> list[Hallazgo]:
        """Verify each finding against the source document.

        Checks:
        1. Literal substring check: Verifies that `fragmento_exacto` literally
           exists in the source text and computes exact global character offsets.
        2. Confidence penalty: Downgrades confidence score if the fragment is not found.
        3. Diagnostic notes: Documents verification status or failure rationale.

        Args:
            findings: List of candidate findings from aggregator.
            document: Ingestion Document containing source text.

        Returns:
            List of verified Hallazgo instances with updated offsets, flags, and notes.
        """
        source_text = document.visible_text or document.raw_content

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

            # Literal substring search in source document
            idx = source_text.find(fragment)

            if idx != -1:
                # Substring matched literally
                finding.inicio_caracter = idx
                finding.fin_caracter = idx + len(fragment)
                finding.verificado = True
                finding.notas_verificacion = (
                    "Literal substring verified against source document text."
                )
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
