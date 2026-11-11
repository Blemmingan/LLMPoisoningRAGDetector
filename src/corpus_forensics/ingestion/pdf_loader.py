"""PDF document loader with visible / hidden text separation.

Text is read character by character with pdfium (pypdfium2), which reports for each
character its render mode, fill color (with alpha), font size and box. A character is
treated as hidden when a reader cannot see it on the rendered page:

- render mode 3 (invisible text, the classic "text layer under an image" trick),
- fully transparent fill (alpha 0),
- white or near-white fill (pages are assumed to have a white background),
- font size or glyph box below 1 pt,
- box entirely outside the page.

Hidden runs go to the hidden channel under `[PDF_HIDDEN_TEXT]` with the reason, so the
analyst sees them and the verifier can tag findings found there; the visible channel
only keeps what a reader would see. Document metadata goes under `[PDF_METADATA]`.
Text drawn under an opaque shape is not detected (it would need full rendering).

If pdfium cannot read the file, pypdf extracts the text without the hidden-text check.
"""

import ctypes
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from corpus_forensics.ingestion.base import DocumentLoader, register_loader
from corpus_forensics.schemas import (
    HIDDEN_SECTION_PDF_HIDDEN_TEXT,
    HIDDEN_SECTION_PDF_METADATA,
    Document,
)

logger = logging.getLogger(__name__)

RENDER_MODE_INVISIBLE = 3
NEAR_WHITE = 245  # RGB components at or above this count as white on a white page
MIN_VISIBLE_SIZE_PT = 1.0


@dataclass
class HiddenRun:
    """Consecutive hidden characters on a page, with why they are hidden."""

    page: int
    reasons: set[str] = field(default_factory=set)
    chars: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join("".join(self.chars).split())


def hidden_reason(
    render_mode: int,
    rgba: tuple[int, int, int, int] | None,
    font_size: float,
    box: tuple[float, float, float, float] | None,
    page_size: tuple[float, float],
) -> str | None:
    """Why a character is invisible on the rendered page, or None if it is visible."""
    if render_mode == RENDER_MODE_INVISIBLE:
        return "modo_invisible"
    if rgba is not None:
        r, g, b, a = rgba
        if a == 0:
            return "transparente"
        if min(r, g, b) >= NEAR_WHITE:
            return "blanco"
    if box is not None:
        left, right, bottom, top = box
        width, height = page_size
        if right < 0 or left > width or top < 0 or bottom > height:
            return "fuera_de_pagina"
        if 0 < font_size < MIN_VISIBLE_SIZE_PT or 0 < (top - bottom) < MIN_VISIBLE_SIZE_PT * 0.5:
            return "fuente_minima"
    elif 0 < font_size < MIN_VISIBLE_SIZE_PT:
        return "fuente_minima"
    return None


def _char_info(
    raw: Any, textpage: Any, index: int
) -> tuple[int, tuple[int, int, int, int] | None, float, tuple[float, float, float, float] | None]:
    """Render mode, fill color, font size and box of one character."""
    text_obj = raw.FPDFText_GetTextObject(textpage, index)
    render_mode = int(raw.FPDFTextObj_GetTextRenderMode(text_obj)) if text_obj else 0
    r, g, b, a = (ctypes.c_uint() for _ in range(4))
    rgba = None
    if raw.FPDFText_GetFillColor(textpage, index, r, g, b, a):
        rgba = (r.value, g.value, b.value, a.value)
    font_size = float(raw.FPDFText_GetFontSize(textpage, index))
    left, right, bottom, top = (ctypes.c_double() for _ in range(4))
    box = None
    if raw.FPDFText_GetCharBox(textpage, index, left, right, bottom, top):
        box = (left.value, right.value, bottom.value, top.value)
    return render_mode, rgba, font_size, box


def extract_with_pdfium(path: Path) -> tuple[str, list[HiddenRun], int]:
    """(visible text, hidden runs, page count) separated character by character."""
    import pypdfium2 as pdfium  # noqa: PLC0415
    import pypdfium2.raw as raw  # noqa: PLC0415

    pdf = pdfium.PdfDocument(str(path))
    pages_text: list[str] = []
    runs: list[HiddenRun] = []
    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            textpage = page.get_textpage()
            page_size = page.get_size()
            visible: list[str] = []
            current: HiddenRun | None = None
            previous_hidden = False
            for i in range(raw.FPDFText_CountChars(textpage.raw)):
                char = chr(raw.FPDFText_GetUnicode(textpage.raw, i))
                if char.isspace() or raw.FPDFText_IsGenerated(textpage.raw, i) == 1:
                    # whitespace and generated breaks follow the previous character
                    reason = "continuacion" if previous_hidden else None
                else:
                    reason = hidden_reason(*_char_info(raw, textpage.raw, i), page_size)
                if reason is None:
                    visible.append(char)
                    current = None
                    previous_hidden = False
                    continue
                if current is None:
                    current = HiddenRun(page=page_index + 1)
                    runs.append(current)
                if reason != "continuacion":
                    current.reasons.add(reason)
                current.chars.append(char)
                previous_hidden = True
            pages_text.append("".join(visible).replace("\r\n", "\n").replace("\r", "\n"))
            textpage.close()
            page.close()
        page_count = len(pdf)
    finally:
        pdf.close()
    return (
        "\n\n".join(t.strip() for t in pages_text).strip(),
        [r for r in runs if r.text],
        page_count,
    )


class PdfLoader(DocumentLoader):
    """Loads PDF documents separating visible text, hidden text and metadata."""

    @property
    def name(self) -> str:
        return "pdf_loader"

    def load(self, path: Path | str) -> Document:
        """Load a PDF, keeping what a reader sees apart from what is hidden.

        Args:
            path: Path to the PDF file.

        Returns:
            Document with the visible text and a hidden channel holding hidden text runs
            (`[PDF_HIDDEN_TEXT]`) and document metadata (`[PDF_METADATA]`).
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        reader = PdfReader(str(file_path))
        hidden_runs: list[HiddenRun] = []
        extractor = "pdfium"
        try:
            visible_text, hidden_runs, page_count = extract_with_pdfium(file_path)
        except Exception as exc:  # malformed or unsupported file: plain extraction
            logger.warning("pdfium could not read %s (%s); using pypdf", file_path, exc)
            extractor = "pypdf"
            visible_text = "\n\n".join(p.extract_text() or "" for p in reader.pages).strip()
            page_count = len(reader.pages)

        hidden_parts: list[str] = []
        if hidden_runs:
            lines = [
                f"(página {run.page}; {', '.join(sorted(run.reasons))}) {run.text}"
                for run in hidden_runs
            ]
            hidden_parts.append(HIDDEN_SECTION_PDF_HIDDEN_TEXT + "\n" + "\n".join(lines))
        pdf_metadata: dict[str, Any] = dict(reader.metadata) if reader.metadata else {}
        meta_strings = [f"{k}: {v}" for k, v in pdf_metadata.items() if v]
        if meta_strings:
            hidden_parts.append(HIDDEN_SECTION_PDF_METADATA + "\n" + "\n".join(meta_strings))

        reasons: dict[str, int] = {}
        for run in hidden_runs:
            for reason in run.reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
        return Document(
            id=str(file_path),
            raw_content=visible_text,
            visible_text=visible_text,
            hidden_content="\n\n".join(hidden_parts),
            source_type="pdf",
            metadata={
                "file_name": file_path.name,
                "num_pages": page_count,
                "is_encrypted": reader.is_encrypted,
                "extractor": extractor,
                "hidden_text_runs": len(hidden_runs),
                "hidden_text_reasons": reasons,
            },
        )


register_loader([".pdf"], PdfLoader)
