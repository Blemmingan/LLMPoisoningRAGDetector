"""Reporting module for JSON serialization and Markdown generation."""

from corpus_forensics.reporting.json_writer import serialize_report_to_json, write_json_report
from corpus_forensics.reporting.markdown_renderer import (
    render_report_to_markdown,
    write_markdown_report,
)

__all__ = [
    "serialize_report_to_json",
    "write_json_report",
    "render_report_to_markdown",
    "write_markdown_report",
]
