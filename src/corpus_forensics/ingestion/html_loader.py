"""HTML document loader with dual-channel visible/hidden content extraction."""

import re
from pathlib import Path

from bs4 import BeautifulSoup, Comment

from corpus_forensics.ingestion.base import DocumentLoader, register_loader
from corpus_forensics.schemas import (
    HIDDEN_SECTION_HIDDEN_ELEMENTS,
    HIDDEN_SECTION_HTML_COMMENTS,
    Document,
)


class HtmlLoader(DocumentLoader):
    """HTML loader that separates visible rendered text from hidden content channels.

    Preserves comments, elements with display:none or visibility:hidden,
    and provides extension hooks for CSS-based steganography / white text.
    """

    @property
    def name(self) -> str:
        return "html_loader"

    def load(self, path: Path | str) -> Document:
        """Load HTML file and separate visible and hidden content.

        Args:
            path: Path to the HTML document.

        Returns:
            Document instance containing visible_text and hidden_content channels.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        raw_content = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw_content, "html.parser")

        # Channel 1: Extract comments
        comments = [
            c.strip() for c in soup.find_all(string=lambda t: isinstance(t, Comment)) if c.strip()
        ]

        # Channel 2: Extract visually hidden elements (display:none, visibility:hidden, hidden attr)
        hidden_elements: list[str] = []
        for tag in soup.find_all(True):
            style_str = str(tag.get("style") or "")
            is_hidden = False
            if tag.has_attr("hidden"):
                is_hidden = True
            elif style_str and re.search(
                r"(display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0)",
                style_str,
                re.IGNORECASE,
            ):
                is_hidden = True

            if is_hidden:
                hidden_text = tag.get_text(strip=True)
                if hidden_text:
                    hidden_elements.append(hidden_text)

        # TODO(team): Advanced hidden content extraction:
        # 1. Parse external/embedded stylesheets to detect classes with display:none / visibility:hidden.
        # 2. Detect white-on-white text (e.g. color: #fff; background-color: #fff;).
        # 3. Detect text positioned off-screen (e.g. left: -9999px; position: absolute;).
        # 4. Detect zero-font-size elements (e.g. font-size: 0px;).

        hidden_channel_parts: list[str] = []
        if comments:
            hidden_channel_parts.append(HIDDEN_SECTION_HTML_COMMENTS + "\n" + "\n".join(comments))
        if hidden_elements:
            hidden_channel_parts.append(
                HIDDEN_SECTION_HIDDEN_ELEMENTS + "\n" + "\n".join(hidden_elements)
            )

        hidden_content = "\n\n".join(hidden_channel_parts)

        # Extract visible text by removing script, style, and comments
        for tag in soup(["script", "style"]):
            tag.decompose()
        visible_text = soup.get_text(separator="\n", strip=True)

        return Document(
            id=str(file_path),
            raw_content=raw_content,
            visible_text=visible_text,
            hidden_content=hidden_content,
            source_type="html",
            metadata={
                "file_name": file_path.name,
                "comments_count": len(comments),
                "hidden_elements_count": len(hidden_elements),
            },
        )


register_loader([".html", ".htm"], HtmlLoader)
