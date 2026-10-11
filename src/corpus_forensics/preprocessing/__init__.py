"""Preprocessing module for normalization, obfuscation detection, and chunking."""

from corpus_forensics.preprocessing.chunker import Chunker
from corpus_forensics.preprocessing.normalizer import (
    detect_base64_candidates,
    detect_invisible_chars,
    extract_html_comments,
    hint_language,
    normalize_and_annotate,
    normalize_unicode,
)

__all__ = [
    "Chunker",
    "normalize_and_annotate",
    "normalize_unicode",
    "detect_invisible_chars",
    "detect_base64_candidates",
    "extract_html_comments",
    "hint_language",
]
