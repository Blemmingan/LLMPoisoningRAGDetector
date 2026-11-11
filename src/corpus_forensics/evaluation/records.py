"""Per-document evaluation records (`predictions.jsonl`).

Each record keeps the label, the forensic report (absent in retrieval-only runs), what
was retrieved for every chunk, latency and model usage. Metrics are computed from these
records, so they can be recomputed or extended without running the pipeline again
(`corpus-forensics eval metrics`), which matters once every run costs money.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from corpus_forensics.analysis.trace import ChunkTrace
from corpus_forensics.evaluation.dataset import EvalLabel
from corpus_forensics.schemas import DocumentReport


class UsageDelta(BaseModel):
    """Model calls made while analyzing one document (read from the usage ledger)."""

    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class PredictionRecord(BaseModel):
    """Everything the evaluation needs about one document."""

    documento: str
    label: EvalLabel
    report: DocumentReport | None = None
    traces: list[ChunkTrace] = Field(default_factory=list)
    latency_s: float = 0.0
    usage: UsageDelta = Field(default_factory=UsageDelta)
    timings: dict[str, float] = Field(
        default_factory=dict, description="Seconds per pipeline stage for this document"
    )
    tokens_estimados: dict[str, int] = Field(
        default_factory=dict,
        description="Analyst prompt tokens counted locally (by section), before any call",
    )


def write_records(records: list[PredictionRecord], path: Path) -> None:
    """Write records as JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")


def read_records(path: Path) -> list[PredictionRecord]:
    """Read records written by `write_records`."""
    with open(path, encoding="utf-8") as f:
        return [PredictionRecord.model_validate(json.loads(line)) for line in f if line.strip()]
