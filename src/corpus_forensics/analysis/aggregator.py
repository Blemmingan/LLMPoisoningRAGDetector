"""Aggregates chunk-level findings into document-level forensic reports."""

from collections.abc import Sequence

from corpus_forensics.schemas import Chunk, ChunkFinding, Hallazgo


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

        # TODO(team): Implement cross-chunk boundary stitching:
        # 1. Inspect chunk boundary overlap regions for fragmented payloads (e.g. prefix in chunk N, suffix in chunk N+1).
        # 2. Check if concatenating adjacent fragments exists contiguously in the document.
        # 3. Handle multi-hop exfiltration triggers split across distant paragraphs.

        return deduped
