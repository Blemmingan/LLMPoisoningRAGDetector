"""Loader for OWASP Top 10 for LLM Applications 2025 (CC BY-SA 4.0).

Uses the official pages of LLM01 (Prompt Injection) and LLM07 (System Prompt Leakage).
"""

from collections.abc import Iterable
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from corpus_forensics.knowledge_base.loaders.base import (
    LABEL_PENDING,
    LABEL_STATUS_KEY,
    BaseKBLoader,
    make_entry_id,
    scalar_metadata,
)
from corpus_forensics.knowledge_base.schemas import AttackEntry
from corpus_forensics.schemas import TipoDeAtaque

PAGES = {"llm01.html": "LLM01:2025", "llm07.html": "LLM07:2025"}
HEADINGS = ("h2", "h3", "h4")


def _section_text(heading: Tag) -> str:
    """Text of the paragraphs and list items after `heading`, up to the next heading.

    Walks the document in order instead of siblings: the OWASP site wraps each heading
    in its own <div>, so the paragraphs are not siblings of the heading.
    """
    parts: list[str] = []
    for element in heading.find_all_next(["p", "li", *HEADINGS]):
        if element.name in HEADINGS:
            break
        if element.name == "p" and element.find_parent("li") is not None:
            continue  # already included through its <li>
        text = element.get_text(" ", strip=True)
        if text:
            parts.append(text)
    return " ".join(" ".join(parts).split())


def _subsections(soup: BeautifulSoup, title_prefix: str) -> list[tuple[str, str]]:
    """(title, text) of every h4 under the h3 whose title starts with `title_prefix`."""
    start = next(
        (h for h in soup.find_all("h3") if h.get_text(strip=True).startswith(title_prefix)),
        None,
    )
    if start is None:
        return []
    items: list[tuple[str, str]] = []
    for heading in start.find_all_next(HEADINGS):
        if heading.name in ("h2", "h3"):
            break
        items.append((heading.get_text(" ", strip=True).rstrip("\\"), _section_text(heading)))
    return items


class OwaspLlmTop10Loader(BaseKBLoader):
    """Loads the 'Example Attack Scenarios' of LLM01 and LLM07 as entries.

    Each entry carries the page's 'Prevention and Mitigation Strategies' (titles only)
    in its mitigation field and the OWASP risk ID in its metadata.
    """

    @property
    def source_name(self) -> str:
        return "owasp_llm_top10"

    def load(self, source_path: Path | str | None = None) -> Iterable[AttackEntry]:
        """Load scenarios from the downloaded HTML pages."""
        if source_path is None:
            return
        root = Path(source_path)
        for filename, risk_id in PAGES.items():
            file = root / filename
            if not file.exists():
                continue
            soup = BeautifulSoup(file.read_text(encoding="utf-8"), "html.parser")
            mitigations = "; ".join(
                title for title, _ in _subsections(soup, "Prevention and Mitigation")
            )
            for title, text in _subsections(soup, "Example Attack Scenarios"):
                if not text:
                    continue
                yield AttackEntry(
                    id=make_entry_id(self.source_name, risk_id, title + text),
                    text=text,
                    source=self.source_name,
                    attack_type=TipoDeAtaque.OTRO,
                    attacker_objective=title,
                    mitigation=mitigations,
                    metadata=scalar_metadata(
                        {
                            "tipo_entrada": "escenario",
                            "riesgo_owasp": risk_id,
                            "titulo": title,
                            LABEL_STATUS_KEY: LABEL_PENDING,
                        }
                    ),
                )
