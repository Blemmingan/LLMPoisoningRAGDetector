"""PDF document loader using pypdf."""

from pathlib import Path
from typing import Any

from pypdf import PdfReader

from corpus_forensics.ingestion.base import DocumentLoader, register_loader
from corpus_forensics.schemas import HIDDEN_SECTION_PDF_METADATA, Document


class PdfLoader(DocumentLoader):
    """Loads PDF documents extracting page text and cataloging metadata."""

    @property
    def name(self) -> str:
        return "pdf_loader"

    def load(self, path: Path | str) -> Document:
        """Load PDF document and extract text.

        Args:
            path: Path to the PDF file.

        Returns:
            Document instance containing extracted text and document metadata.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        reader = PdfReader(str(file_path))
        page_texts: list[str] = []

        for page in reader.pages:
            text = page.extract_text() or ""
            page_texts.append(text)

        visible_text = "\n\n".join(page_texts).strip()

        # TODO(team): Advanced PDF forensics:
        # 1. Detect invisible / white-rendered text layers (font rendering modes 3 / fill-color match).
        # 2. Extract embedded annotations, form fields, and JavaScript actions.
        # 3. Analyze raw object streams and unreferenced objects for malicious payloads.
        # 4. Integrate OCR fallback for scanned or image-flattened pages.

        hidden_channel = ""
        pdf_metadata: dict[str, Any] = dict(reader.metadata) if reader.metadata else {}
        meta_strings = [f"{k}: {v}" for k, v in pdf_metadata.items() if v]
        if meta_strings:
            hidden_channel = HIDDEN_SECTION_PDF_METADATA + "\n" + "\n".join(meta_strings)

        return Document(
            id=str(file_path),
            raw_content=visible_text,
            visible_text=visible_text,
            hidden_content=hidden_channel,
            source_type="pdf",
            metadata={
                "file_name": file_path.name,
                "num_pages": len(reader.pages),
                "is_encrypted": reader.is_encrypted,
            },
        )


register_loader([".pdf"], PdfLoader)
