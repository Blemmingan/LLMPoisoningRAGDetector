"""Tests for the defensive normalization applied before retrieval and analysis."""

import base64

from corpus_forensics.preprocessing.normalizer import (
    canonicalize,
    detect_invisible_chars,
    fold_confusables,
    normalize_and_annotate,
    reveal_unicode_tags,
    strip_invisible,
)
from corpus_forensics.schemas import Chunk


def _tags(text: str) -> str:
    """Encode benign text as Unicode Tag characters (only to test that we reveal them)."""
    return "".join(chr(0xE0000 + ord(c)) for c in text)


def test_reveal_unicode_tags_makes_hidden_text_visible() -> None:
    revealed, runs = reveal_unicode_tags(f"Visible{_tags('hello')} text")
    assert revealed == "Visible[UNICODE_TAGS: hello] text"
    assert runs == ["hello"]


def test_strip_invisible_removes_zero_width_bidi_and_variation_selectors() -> None:
    text = "a​b‮c️d"
    cleaned, removed = strip_invisible(text)
    assert cleaned == "abcd" and removed == 3


def test_detect_invisible_reports_tags_bidi_and_selectors() -> None:
    kinds = {f["char"] for f in detect_invisible_chars(f"x{_tags('a')}‮︀")}
    assert {"UNICODE_TAG", "BIDI_CONTROL", "VARIATION_SELECTOR"} <= kinds


def test_fold_confusables_only_inside_latin_words() -> None:
    mixed, n = fold_confusables("hеllo wоrld")  # Cyrillic е and о inside Latin words
    assert mixed == "hello world" and n == 2
    russian, n = fold_confusables("привет мир")  # genuine Cyrillic text stays as is
    assert russian == "привет мир" and n == 0


def test_canonicalize_combines_all_steps() -> None:
    blob = base64.b64encode(b"secret note for testing").decode()
    text = f"Ｈｅｌｌｏ​ wоrld{_tags('hi')} {blob}"
    canonical, info = canonicalize(text)
    assert canonical.startswith("Hello world[UNICODE_TAGS: hi]")
    assert "[BASE64_DECODED: secret note for testing]" in canonical
    assert info["removed_invisible"] == 1 and info["confusables_folded"] == 1
    assert info["nfkc_changed"] and info["changed"]


def test_plain_text_is_unchanged() -> None:
    canonical, info = canonicalize("Plain text, nothing hidden.")
    assert canonical == "Plain text, nothing hidden." and not info["changed"]


def test_normalize_and_annotate_flags() -> None:
    _, ann = normalize_and_annotate(f"hеllo{_tags('x')}")
    assert ann["has_unicode_tags"] and ann["has_homoglyphs"] and ann["has_invisible_chars"]


def test_chunker_stores_normalized_text() -> None:
    from corpus_forensics.preprocessing.chunker import Chunker
    from corpus_forensics.schemas import Document

    doc = Document(id="d", raw_content="", visible_text="hеllo wоrld", source_type="text")
    chunk = Chunker(chunk_size=100, chunk_overlap=10).chunk_document(doc)[0]
    assert chunk.text == "hеllo wоrld" and chunk.normalized_text == "hello world"


def test_retriever_searches_normalized_text_unless_disabled() -> None:
    from corpus_forensics.analysis.retriever import AttackRetriever

    chunk = Chunk(
        id="c", document_id="d", text="hеllo", normalized_text="hello", start=0, end=5, index=0
    )
    on = AttackRetriever(vector_store=None, embedder=None)  # type: ignore[arg-type]
    off = AttackRetriever(vector_store=None, embedder=None, use_normalized_text=False)  # type: ignore[arg-type]
    assert on.query_text(chunk) == "hello" and off.query_text(chunk) == "hеllo"


def test_prompt_shows_normalized_view_only_when_it_differs() -> None:
    from corpus_forensics.analysis.prompt_builder import PromptBuilder

    same = Chunk(
        id="c", document_id="d", text="abc", normalized_text="abc", start=0, end=3, index=0
    )
    diff = Chunk(
        id="c", document_id="d", text="аbc", normalized_text="abc", start=0, end=3, index=0
    )
    assert "Identical to the chunk text" in PromptBuilder().build_prompt(same, [])
    prompt = PromptBuilder().build_prompt(diff, [])
    assert "NORMALIZED VIEW" in prompt and "never quote from here" in prompt
