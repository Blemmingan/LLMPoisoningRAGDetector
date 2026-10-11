"""Ingestion module for document loading."""

from corpus_forensics.ingestion.base import DocumentLoader, get_loader_for_path, register_loader
from corpus_forensics.ingestion.html_loader import HtmlLoader
from corpus_forensics.ingestion.pdf_loader import PdfLoader
from corpus_forensics.ingestion.text_loader import TextLoader

__all__ = [
    "DocumentLoader",
    "get_loader_for_path",
    "register_loader",
    "TextLoader",
    "HtmlLoader",
    "PdfLoader",
]
