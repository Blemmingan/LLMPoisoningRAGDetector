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

# Bidirectional controls (Trojan Source) and variation selectors change how text is read
# or tokenized without changing what is visible.
BIDI_CONTROLS = {chr(c) for c in (*range(0x202A, 0x202F), *range(0x2066, 0x206A))}
VARIATION_SELECTORS = {chr(c) for c in range(0xFE00, 0xFE10)} | {
    chr(c) for c in range(0xE0100, 0xE01F0)
}
TAG_RANGE = (0xE0000, 0xE007F)

# Common look-alike letters folded to their Latin "skeleton". A small, documented subset of
# the Unicode confusables table (UTS #39, confusables.txt) covering Cyrillic and Greek
# letters that are visually identical to Latin ones. Fullwidth, circled and mathematical
# letters are already folded by NFKC.
CONFUSABLES: dict[str, str] = {
    # Cyrillic
    "а": "a",
    "в": "b",
    "е": "e",
    "ё": "e",
    "һ": "h",
    "і": "i",
    "ї": "i",
    "ј": "j",
    "к": "k",
    "м": "m",
    "н": "h",
    "о": "o",
    "р": "p",
    "с": "c",
    "т": "t",
    "у": "y",
    "х": "x",
    "ѕ": "s",
    "ԁ": "d",
    "ԛ": "q",
    "ԝ": "w",
    "ɡ": "g",
    "А": "A",
    "В": "B",
    "Е": "E",
    "Ѕ": "S",
    "І": "I",
    "Ј": "J",
    "К": "K",
    "М": "M",
    "Н": "H",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Т": "T",
    "У": "Y",
    "Х": "X",
    # Greek
    "α": "a",
    "ε": "e",
    "ι": "i",
    "κ": "k",
    "ν": "v",
    "ο": "o",
    "ρ": "p",
    "τ": "t",
    "υ": "u",
    "χ": "x",
    "Α": "A",
    "Β": "B",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Ι": "I",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
    "Ο": "O",
    "Ρ": "P",
    "Τ": "T",
    "Υ": "Y",
    "Χ": "X",
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
    # Cyrillic/Greek look-alikes are handled by fold_confusables (UTS #39 subset).
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
    for idx, char in enumerate(text):
        code = ord(char)
        if TAG_RANGE[0] <= code <= TAG_RANGE[1]:
            findings.append({"char": "UNICODE_TAG", "codepoint": f"U+{code:04X}", "offset": idx})
        elif char in BIDI_CONTROLS:
            findings.append({"char": "BIDI_CONTROL", "codepoint": f"U+{code:04X}", "offset": idx})
        elif char in VARIATION_SELECTORS:
            findings.append(
                {"char": "VARIATION_SELECTOR", "codepoint": f"U+{code:04X}", "offset": idx}
            )
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


def reveal_unicode_tags(text: str) -> tuple[str, list[str]]:
    """Make Unicode Tag characters visible.

    Each run of tag characters (U+E0000..U+E007F, used for "ASCII smuggling") is replaced
    by a visible `[UNICODE_TAGS: ...]` marker holding the text it encodes.
    """
    revealed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        decoded = "".join(
            chr(ord(c) - 0xE0000) for c in match.group(0) if 0xE0020 <= ord(c) <= 0xE007E
        )
        revealed.append(decoded)
        return f"[UNICODE_TAGS: {decoded}]"

    return re.sub("[\U000e0000-\U000e007f]+", replace, text), revealed


def strip_invisible(text: str) -> tuple[str, int]:
    """Remove zero-width characters, bidi controls and variation selectors."""
    removable = set(ZERO_WIDTH_CHARS) | BIDI_CONTROLS | VARIATION_SELECTORS
    kept = [c for c in text if c not in removable]
    return "".join(kept), len(text) - len(kept)


def fold_confusables(text: str) -> tuple[str, int]:
    """Fold common Cyrillic/Greek look-alikes to Latin when they sit inside Latin words.

    Only words that already contain Latin letters are folded, so genuine Russian or Greek
    text is left untouched.
    """
    folded_count = 0

    def fold_word(match: re.Match[str]) -> str:
        nonlocal folded_count
        word = match.group(0)
        if not any("LATIN" in unicodedata.name(c, "") for c in word if c.isalpha()):
            return word
        out = []
        for c in word:
            if c in CONFUSABLES:
                folded_count += 1
                out.append(CONFUSABLES[c])
            else:
                out.append(c)
        return "".join(out)

    return re.sub(r"\w+", fold_word, text), folded_count


def canonicalize(text: str) -> tuple[str, dict[str, Any]]:
    """Defensive normalization: what a model would read, made visible for analysis.

    Steps: reveal Unicode Tags, strip invisible/bidi/variation characters, NFKC (folds
    fullwidth, circled and mathematical letters), fold Cyrillic/Greek look-alikes inside
    Latin words, and append decoded base64 blobs. Used for retrieval and shown to the
    analyst next to the original text; offsets and verification keep using the original.
    """
    revealed_text, tags = reveal_unicode_tags(text)
    stripped, n_invisible = strip_invisible(revealed_text)
    nfkc = unicodedata.normalize("NFKC", stripped)
    folded, n_confusables = fold_confusables(nfkc)
    decoded = [
        c["decoded_preview"]
        for c in detect_base64_candidates(text)
        if c["is_valid_utf8"] and c["decoded_preview"] and c["decoded_preview"].isprintable()
    ]
    canonical = folded
    if decoded:
        canonical += "\n" + "\n".join(f"[BASE64_DECODED: {d}]" for d in decoded)
    info = {
        "unicode_tags": tags,
        "removed_invisible": n_invisible,
        "nfkc_changed": nfkc != stripped,
        "confusables_folded": n_confusables,
        "base64_decoded": decoded,
        "changed": canonical != text,
    }
    return canonical, info


def normalize_and_annotate(text: str) -> tuple[str, dict[str, Any]]:
    """Run all preprocessing hooks.

    Args:
        text: Input text to analyze.

    Returns:
        Tuple of (canonical_text, annotations_dict). The canonical text is the defensive
        normalization from `canonicalize`.
    """
    canonical, canon_info = canonicalize(text)
    _, unicode_info = normalize_unicode(text)
    invisible = detect_invisible_chars(text)
    b64_candidates = detect_base64_candidates(text)
    html_comments = extract_html_comments(text)
    lang_hints = hint_language(text)

    annotations: dict[str, Any] = {
        "unicode": unicode_info,
        "invisible_chars": invisible,
        "has_invisible_chars": len(invisible) > 0,
        "has_unicode_tags": bool(canon_info["unicode_tags"]),
        "has_homoglyphs": canon_info["confusables_folded"] > 0 or canon_info["nfkc_changed"],
        "base64_candidates": b64_candidates,
        "has_base64": len(b64_candidates) > 0,
        "html_comments": html_comments,
        "has_html_comments": len(html_comments) > 0,
        "language_hints": lang_hints,
        "canonical": canon_info,
    }

    return canonical, annotations
