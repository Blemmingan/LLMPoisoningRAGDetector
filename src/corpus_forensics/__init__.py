"""Corpus Poisoning Forensic Analyst.

A forensic tool to detect indirect prompt injections and corpus poisoning
prior to indexing into RAG architectures.
"""

from corpus_forensics.analysis.pipeline import ForensicPipeline
from corpus_forensics.config import Settings, get_settings
from corpus_forensics.schemas import DocumentReport, Hallazgo

__version__ = "0.1.0"

__all__ = [
    "ForensicPipeline",
    "DocumentReport",
    "Hallazgo",
    "Settings",
    "get_settings",
]
