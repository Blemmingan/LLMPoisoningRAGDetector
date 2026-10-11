"""JSON report writer for forensic analysis results."""

import json
from pathlib import Path

from corpus_forensics.schemas import DocumentReport


def serialize_report_to_json(report: DocumentReport, indent: int = 2) -> str:
    """Serialize a DocumentReport into a JSON string with Spanish keys.

    Args:
        report: DocumentReport model.
        indent: JSON indentation level.

    Returns:
        JSON string representation.
    """
    # model_dump(by_alias=True) ensures 'metadata de análisis' is used as key
    data = report.model_dump(by_alias=True, mode="json")
    return json.dumps(data, indent=indent, ensure_ascii=False)


def write_json_report(report: DocumentReport, output_path: Path | str) -> Path:
    """Write DocumentReport to a JSON file.

    Args:
        report: DocumentReport to write.
        output_path: File path where report will be saved.

    Returns:
        Path of written report.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = serialize_report_to_json(report)
    path.write_text(content, encoding="utf-8")
    return path
