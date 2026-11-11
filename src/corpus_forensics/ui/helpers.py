"""Pure helpers for the Streamlit app (kept apart so they can be unit-tested)."""

import html
import json
from pathlib import Path
from typing import Any

from corpus_forensics.analysis.trace import ChunkTrace
from corpus_forensics.schemas import DocumentReport, Hallazgo

SEVERITY_COLORS = {"baja": "#8fa3b8", "media": "#d9a441", "alta": "#e0703c", "critica": "#c0392b"}
SUMMARY_FILES = ("eval_summary.md", "ablation_summary.md", "cost_report.md")


def highlight(text: str, fragments: list[str]) -> str:
    """Escaped HTML of `text` with every occurrence of the fragments wrapped in <mark>.

    Overlapping matches are merged; the text is never interpreted as HTML.
    """
    spans: list[tuple[int, int]] = []
    for fragment in {f for f in fragments if f}:
        start = text.find(fragment)
        while start != -1:
            spans.append((start, start + len(fragment)))
            start = text.find(fragment, start + 1)
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    out: list[str] = []
    cursor = 0
    for start, end in merged:
        out.append(html.escape(text[cursor:start]))
        out.append(f"<mark>{html.escape(text[start:end])}</mark>")
        cursor = end
    out.append(html.escape(text[cursor:]))
    return "".join(out).replace("\n", "<br>")


def finding_title(finding: Hallazgo) -> str:
    return (
        f"{finding.tipo_de_ataque.value} · severidad {finding.severidad.value} · "
        f"confianza {finding.confianza:.2f} · canal {finding.canal.value}"
    )


def retrieval_rows(traces: list[ChunkTrace]) -> list[dict[str, Any]]:
    """One row per (chunk, retrieved example) for a table."""
    rows: list[dict[str, Any]] = []
    for trace in traces:
        preview = " ".join(trace.text.split())[:90]
        for example in trace.examples:
            rows.append(
                {
                    "chunk": trace.index,
                    "texto del chunk": preview,
                    "rank": example.rank,
                    "ejemplo": example.id,
                    "tipo": example.attack_type,
                    "ocultamiento": example.concealment,
                    "similitud": example.similarity,
                }
            )
    return rows


def report_json(report: DocumentReport) -> str:
    return json.dumps(report.model_dump(mode="json", by_alias=True), indent=2, ensure_ascii=False)


def load_labels(dataset_dir: Path) -> list[dict[str, Any]]:
    """Dataset labels (empty if the dataset has not been built)."""
    path = dataset_dir / "labels.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_summaries(root: Path) -> dict[str, Path]:
    """Evaluation, ablation and cost summaries under a folder, by relative name."""
    if not root.exists():
        return {}
    found: dict[str, Path] = {}
    for name in SUMMARY_FILES:
        for path in sorted(root.rglob(name)):
            found[str(path.relative_to(root))] = path
    return found
