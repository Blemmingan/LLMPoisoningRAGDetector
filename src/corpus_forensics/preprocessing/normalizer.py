"""Text normalizer and forensic obfuscation detection hooks.

Provides modular analysis hooks for invisible characters, base64 payloads,
Unicode anomalies, HTML comments, and language hints.
Each hook returns structured annotations rather than just stripping text.
"""

import base64
import re
import unicodedata
from typing import Any, Literal

# Known zero-width and invisible unicode codepoints
ZERO_WIDTH_CHARS = {
    "\u200b": "ZERO_WIDTH_SPACE",
    "\u200c": "ZERO_WIDTH_NON_JOINER",
    "\u200d": "ZERO_WIDTH_JOINER",
    "\ufeff": "ZERO_WIDTH_NO_BREAK_SPACE_BOM",
    "\u2060": "WORD_JOINER",
    "\u00ad": "SOFT_HYPHEN",
    "\u200e": "LEFT_TO_RIGHT_MARK",
    "\u200f": "RIGHT_TO_LEFT_MARK",
}

BASE64_REGEX = re.compile(r"(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")
HTML_COMMENT_REGEX = re.compile(r"<!--([\s\S]*?)-->")


def normalize_unicode(
    text: str, form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC"
) -> tuple[str, dict[str, Any]]:
    """Normalize unicode text and document alterations.

    Args:
        text: Input string.
        form: Normalization form (default 'NFKC').

    Returns:
        Tuple of (normalized_text, annotation_dict).
    """
    normalized = unicodedata.normalize(form, text)
    # TODO(team): Integrate homoglyph and confusable detection (e.g. Cyrillic 'а' replacing Latin 'a').
    changed = normalized != text
    annotations = {
        "form": form,
        "changed": changed,
        "original_len": len(text),
        "normalized_len": len(normalized),
    }
    return normalized, annotations


def detect_invisible_chars(text: str) -> list[dict[str, Any]]:
    """Scan for zero-width and invisible unicode characters.

    Args:
        text: Input string.

    Returns:
        List of detected occurrences with position, codepoint, and character name.
    """
    findings: list[dict[str, Any]] = []
    for idx, char in enumerate(text):
        if char in ZERO_WIDTH_CHARS:
            findings.append(
                {
                    "char": ZERO_WIDTH_CHARS[char],
                    "codepoint": f"U+{ord(char):04X}",
                    "offset": idx,
                }
            )
    # TODO(team): Add detection for Unicode Tag Characters (U+E0001..U+E007F) used in ASCII-smuggling.
    # TODO(team): Detect bidirectional override characters (LRO/RLO) used for visual spoofing.
    return findings


def detect_base64_candidates(text: str, min_len: int = 16) -> list[dict[str, Any]]:
    """Detect potential base64-encoded payload candidates.

    Args:
        text: Input string.
        min_len: Minimum string length to consider as candidate.

    Returns:
        List of candidate matches with offset, candidate snippet, and decoded preview.
    """
    candidates: list[dict[str, Any]] = []
    for match in BASE64_REGEX.finditer(text):
        candidate = match.group(0)
        if len(candidate) < min_len:
            continue
        decoded_preview = None
        try:
            raw_bytes = base64.b64decode(candidate, validate=True)
            # Check if it looks like UTF-8 text
            decoded_preview = raw_bytes.decode("utf-8")
        except Exception:
            decoded_preview = None

        candidates.append(
            {
                "offset_start": match.start(),
                "offset_end": match.end(),
                "snippet": candidate[:30] + ("..." if len(candidate) > 30 else ""),
                "decoded_preview": decoded_preview,
                "is_valid_utf8": decoded_preview is not None,
            }
        )
    # TODO(team): Integrate Shannon entropy checks to differentiate random tokens from actual payloads.
    return candidates


def extract_html_comments(text: str) -> list[dict[str, Any]]:
    """Detect and extract HTML comment blocks within text.

    Args:
        text: Input string.

    Returns:
        List of HTML comments with offsets and extracted content.
    """
    comments: list[dict[str, Any]] = []
    for match in HTML_COMMENT_REGEX.finditer(text):
        comments.append(
            {
                "offset_start": match.start(),
                "offset_end": match.end(),
                "content": match.group(1).strip(),
            }
        )
    # TODO(team): Analyze comments for imperative prompts (e.g. 'Instruction:', 'System:').
    return comments


def hint_language(text: str) -> dict[str, Any]:
    """Provide language and script distribution hints for the text.

    Args:
        text: Input string.

    Returns:
        Dictionary with detected script hints.
    """
    # MOCK / STUB: Basic script counting
    scripts: dict[str, int] = {}
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            script = name.split()[0] if name else "UNKNOWN"
            scripts[script] = scripts.get(script, 0) + 1

    primary_script = max(scripts, key=lambda k: scripts[k]) if scripts else "NONE"
    # TODO(team): Integrate full language identification library (e.g., fasttext, lingua).
    return {
        "primary_script": primary_script,
        "script_distribution": scripts,
        "is_mixed_script": len(scripts) > 1,
    }


def normalize_and_annotate(text: str) -> tuple[str, dict[str, Any]]:
    """Run all preprocessing hooks and produce text with forensic annotations.

    Args:
        text: Input text to analyze.

    Returns:
        Tuple of (clean_text, annotations_dict).
    """
    clean_text, unicode_info = normalize_unicode(text)
    invisible = detect_invisible_chars(text)
    b64_candidates = detect_base64_candidates(text)
    html_comments = extract_html_comments(text)
    lang_hints = hint_language(text)

    annotations: dict[str, Any] = {
        "unicode": unicode_info,
        "invisible_chars": invisible,
        "has_invisible_chars": len(invisible) > 0,
        "base64_candidates": b64_candidates,
        "has_base64": len(b64_candidates) > 0,
        "html_comments": html_comments,
        "has_html_comments": len(html_comments) > 0,
        "language_hints": lang_hints,
    }

    return clean_text, annotations
