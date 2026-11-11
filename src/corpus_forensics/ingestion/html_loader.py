"""HTML document loader with dual-channel visible/hidden content extraction.

Hidden channel:
- `[HTML_COMMENTS]`: comment contents.
- `[HIDDEN_ELEMENTS]`: text of elements a reader cannot see, with the reason. An element
  is hidden by the `hidden` attribute or by CSS, inline or in `<style>` rules (resolved
  with CSS selectors): display:none, visibility:hidden, opacity 0, font-size 0, white or
  transparent text without a non-white background, off-screen positioning (offsets or
  text-indent of -100px or more) and zero-size clipping.

Hidden elements are removed from the visible text, so a fragment found there is
reported in the hidden channel. External stylesheets and computed backgrounds of
ancestors are not resolved (it would need a browser engine).
"""

import re
from pathlib import Path

from bs4 import BeautifulSoup, Comment, Tag

from corpus_forensics.ingestion.base import DocumentLoader, register_loader
from corpus_forensics.schemas import (
    HIDDEN_SECTION_HIDDEN_ELEMENTS,
    HIDDEN_SECTION_HTML_COMMENTS,
    Document,
)

_ZERO = r"0(?:\.0+)?(?:px|pt|em|rem|%|vh|vw)?\s*(?:!important)?\s*(?:;|$)"
HIDING_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("display_none", re.compile(r"display\s*:\s*none", re.I)),
    ("visibility_hidden", re.compile(r"visibility\s*:\s*(?:hidden|collapse)", re.I)),
    ("opacidad_cero", re.compile(rf"(?:^|;)\s*opacity\s*:\s*{_ZERO}", re.I)),
    ("fuente_cero", re.compile(rf"font-size\s*:\s*{_ZERO}", re.I)),
    (
        "fuera_de_pantalla",
        re.compile(r"(?:left|top|right|bottom|text-indent|margin-left)\s*:\s*-\d{3,}", re.I),
    ),
    (
        "recorte",
        re.compile(
            r"clip\s*:\s*rect\(\s*0[\s,a-z]*0[\s,a-z]*0[\s,a-z]*0|clip-path\s*:\s*inset\(\s*50%",
            re.I,
        ),
    ),
)
WHITE_TEXT = re.compile(
    r"(?:^|;)\s*color\s*:\s*(?:white|transparent|#fff(?:fff)?\b|rgba?\(\s*255\s*,\s*255\s*,\s*255"
    r"|rgba\([^)]*,\s*0(?:\.0+)?\s*\))",
    re.I,
)
BACKGROUND = re.compile(r"background(?:-color)?\s*:\s*([^;]+)", re.I)
WHITE_BACKGROUND = re.compile(
    r"^\s*(?:white|#fff(?:fff)?\b|rgba?\(\s*255\s*,\s*255\s*,\s*255)", re.I
)
CSS_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")


def hiding_reason(declarations: str) -> str | None:
    """Why a block of CSS declarations makes text invisible, or None."""
    for reason, pattern in HIDING_RULES:
        if pattern.search(declarations):
            return reason
    if WHITE_TEXT.search(declarations):
        background = BACKGROUND.search(declarations)
        if background is None or WHITE_BACKGROUND.match(background.group(1)):
            return "texto_blanco"
    return None


def stylesheet_hidden(soup: BeautifulSoup) -> dict[int, tuple[Tag, str]]:
    """Elements hidden by rules of the document's <style> blocks, by element id()."""
    marked: dict[int, tuple[Tag, str]] = {}
    for style in soup.find_all("style"):
        css = re.sub(r"/\*.*?\*/", "", style.get_text(), flags=re.S)
        for selectors, declarations in CSS_RULE.findall(css):
            reason = hiding_reason(declarations)
            if reason is None:
                continue
            for selector in selectors.split(","):
                selector = selector.strip()
                if not selector or selector.startswith("@") or "::" in selector:
                    continue
                try:
                    elements = soup.select(selector)
                except Exception:  # selector soupsieve does not support
                    continue
                for element in elements:
                    marked.setdefault(id(element), (element, reason))
    return marked


def hidden_elements(soup: BeautifulSoup) -> list[tuple[Tag, str]]:
    """Outermost hidden elements with their reason, in document order."""
    marked = stylesheet_hidden(soup)
    for tag in soup.find_all(True):
        if tag.name in ("style", "script"):
            continue
        if tag.has_attr("hidden"):
            marked.setdefault(id(tag), (tag, "atributo_hidden"))
            continue
        reason = hiding_reason(str(tag.get("style") or ""))
        if reason:
            marked.setdefault(id(tag), (tag, reason))
    outermost = []
    for tag in soup.find_all(True):
        if id(tag) in marked and not any(id(parent) in marked for parent in tag.parents):
            outermost.append(marked[id(tag)])
    return outermost


class HtmlLoader(DocumentLoader):
    """HTML loader that separates visible rendered text from hidden content channels."""

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

        # Channel 1: comments
        comments = [
            c.strip() for c in soup.find_all(string=lambda t: isinstance(t, Comment)) if c.strip()
        ]

        # Channel 2: visually hidden elements (attribute, inline CSS, <style> rules)
        hidden: list[str] = []
        reasons: dict[str, int] = {}
        for tag, reason in hidden_elements(soup):
            text = tag.get_text(" ", strip=True)
            if text:
                hidden.append(f"({reason}) {text}")
                reasons[reason] = reasons.get(reason, 0) + 1
            tag.decompose()  # what a reader cannot see is not visible text

        hidden_channel_parts: list[str] = []
        if comments:
            hidden_channel_parts.append(HIDDEN_SECTION_HTML_COMMENTS + "\n" + "\n".join(comments))
        if hidden:
            hidden_channel_parts.append(HIDDEN_SECTION_HIDDEN_ELEMENTS + "\n" + "\n".join(hidden))

        for tag in soup(["script", "style"]):
            tag.decompose()
        visible_text = soup.get_text(separator="\n", strip=True)

        return Document(
            id=str(file_path),
            raw_content=raw_content,
            visible_text=visible_text,
            hidden_content="\n\n".join(hidden_channel_parts),
            source_type="html",
            metadata={
                "file_name": file_path.name,
                "comments_count": len(comments),
                "hidden_elements_count": len(hidden),
                "hidden_element_reasons": reasons,
            },
        )


register_loader([".html", ".htm"], HtmlLoader)
