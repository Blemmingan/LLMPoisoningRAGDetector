"""Aggregates chunk-level findings into document-level forensic reports."""

from collections.abc import Sequence

from corpus_forensics.schemas import (
    HIDDEN_CHANNEL_SEPARATOR,
    Chunk,
    ChunkFinding,
    Hallazgo,
    Severidad,
    TecnicaDeOcultamiento,
)

# Fragments of the same attack type closer than this (whitespace only in between) are
# one payload that the chunk boundary cut in two.
MAX_STITCH_GAP = 3
_SEVERITY_RANK = {
    Severidad.BAJA: 0,
    Severidad.MEDIA: 1,
    Severidad.ALTA: 2,
    Severidad.CRITICA: 3,
}


def reconstruct_text(chunks: Sequence[Chunk]) -> str | None:
    """The analyzed text rebuilt from overlapping chunks (None if they leave a gap)."""
    text = ""
    for chunk in sorted(chunks, key=lambda c: c.start):
        if chunk.start > len(text):
            return None
        if chunk.end > len(text):
            text = text[: chunk.start] + chunk.text
    return text


def locate(finding: Hallazgo, chunks_by_index: dict[int, Chunk]) -> tuple[int, int] | None:
    """Absolute span of a finding's fragment inside its chunk, if it is quoted verbatim."""
    chunk = chunks_by_index.get(finding.indice_chunk)
    if chunk is None or not finding.fragmento_exacto:
        return None
    idx = chunk.text.find(finding.fragmento_exacto)
    if idx == -1:
        return None
    return chunk.start + idx, chunk.start + idx + len(finding.fragmento_exacto)


def merge_findings(first: Hallazgo, second: Hallazgo, fragment: str) -> Hallazgo:
    """One finding for a stitched payload: strongest assessment, all techniques."""
    lead = first if first.confianza >= second.confianza else second
    techniques = list(dict.fromkeys(t for h in (first, second) for t in h.tecnica_de_ocultamiento))
    if len(techniques) > 1:
        techniques = [t for t in techniques if t != TecnicaDeOcultamiento.NINGUNA]
    severity = max(first.severidad, second.severidad, key=lambda s: _SEVERITY_RANK[s])
    return lead.model_copy(
        update={
            "fragmento_exacto": fragment,
            "indice_chunk": min(first.indice_chunk, second.indice_chunk),
            "confianza": max(first.confianza, second.confianza),
            "severidad": severity,
            "tecnica_de_ocultamiento": techniques,
        }
    )


def stitch(findings: list[Hallazgo], chunks: Sequence[Chunk]) -> list[Hallazgo]:
    """Join same-type findings whose fragments overlap or abut across chunk boundaries.

    The joined fragment is cut from the document text, so it stays a verbatim quote
    (the verifier checks it afterwards). Findings whose fragment is not verbatim in
    their chunk, or that would join visible and hidden channels, are left as they are.
    """
    text = reconstruct_text(chunks)
    if text is None:
        return findings
    by_index = {c.index: c for c in chunks}
    located = [(span, h) for h in findings if (span := locate(h, by_index)) is not None]
    others = [h for h in findings if locate(h, by_index) is None]
    located.sort(key=lambda item: item[0])

    merged: list[tuple[tuple[int, int], Hallazgo]] = []
    for (start, end), finding in located:
        if merged:
            (m_start, m_end), current = merged[-1]
            gap = text[m_end:start] if start > m_end else ""
            joinable = (
                finding.tipo_de_ataque == current.tipo_de_ataque
                and start <= m_end + MAX_STITCH_GAP
                and not gap.strip()
            )
            fragment = text[m_start : max(end, m_end)]
            if joinable and HIDDEN_CHANNEL_SEPARATOR not in fragment:
                new_span = (m_start, max(end, m_end))
                merged[-1] = (new_span, merge_findings(current, finding, fragment))
                continue
        merged.append(((start, end), finding))
    return [h for _, h in merged] + others


class FindingAggregator:
    """Aggregates per-chunk forensic findings into a unified document-level findings list.

    Strategy Overview:
    1. Cross-Chunk Overlap Resolution:
       When an injection spans the boundary between Chunk N and Chunk N+1,
       the overlapping window ensures both chunks contain partial fragments.
       The aggregator identifies adjacent findings whose fragments overlap or abut
       and stitches them into a continuous payload.
    2. Deduplication:
       Duplicate detections of identical fragments in consecutive overlapping chunks
       are merged, retaining the highest confidence score and broadest character offset.
    3. Conflict Resolution:
       If overlapping chunks produce divergent severity or attack types for the same fragment,
       the highest severity level and consensus attack type is selected.
    """

    def aggregate(
        self,
        chunk_findings: Sequence[ChunkFinding],
        chunks: Sequence[Chunk],
    ) -> list[Hallazgo]:
        """Merge per-chunk findings into document-level Hallazgo items.

        Args:
            chunk_findings: List of structured findings emitted by the analyst per chunk.
            chunks: List of corresponding Chunk objects.

        Returns:
            Deduplicated, merged list of document-level Hallazgo findings.
        """
        all_hallazgos: list[Hallazgo] = []
        for cf in chunk_findings:
            all_hallazgos.extend(cf.hallazgos)

        if not all_hallazgos:
            return []

        # Deduplication by fragment exact text and attack type
        deduped: list[Hallazgo] = []
        seen_fragments: set[tuple[str, str]] = set()

        for h in all_hallazgos:
            # Key by (normalized fragment, attack_type)
            key = (h.fragmento_exacto.strip().lower(), h.tipo_de_ataque.value)
            if key in seen_fragments:
                # Update existing finding if higher confidence
                for existing in deduped:
                    if (
                        existing.fragmento_exacto.strip().lower(),
                        existing.tipo_de_ataque.value,
                    ) == key:
                        if h.confianza > existing.confianza:
                            existing.confianza = h.confianza
                        break
                continue

            seen_fragments.add(key)
            deduped.append(h)

        # Cross-chunk stitching: a payload cut by a chunk boundary becomes one finding.
        # (Payloads split across distant paragraphs are out of scope.)
        return stitch(deduped, chunks)
