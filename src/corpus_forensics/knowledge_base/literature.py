"""Literature collection of the knowledge base: papers and technical reports about attacks.

Pipeline:
    sources YAML -> download (PDF / HTML) -> markdown (Docling for PDFs, BeautifulSoup for
    HTML pages) -> section-aware chunks of ~350 tokens -> sanitization -> keyword labels
    -> embeddings -> `literatura` collection.

Only the source list is versioned. Downloads and processed chunks stay local because the
repository is public and most of these works may be read but not redistributed.
"""

import html
import json
import re
import time
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import yaml
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from corpus_forensics.knowledge_base.download import _sha256

USER_AGENT = "corpus-forensics/0.1 (university research project)"

# Sections that add noise to retrieval and are dropped entirely.
DROPPED_SECTION = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?(references|bibliography|acknowledg(?:e)?ments?|"
    r"funding|author contributions|share this:?|reference links|related frameworks|"
    r"acm reference format)\b",
    re.IGNORECASE,
)

# Phrases that show a chunk quotes a literal attack payload (papers include many).
PAYLOAD_PATTERNS = re.compile(
    r"ignore (?:all |the )?(?:previous|prior|above) (?:instructions|prompts?)|"
    r"disregard (?:all |the )?(?:previous|prior) |"
    r"\byou are now\b|\bdeveloper mode\b|\bDAN\b|i have been pwned|"
    r"(?:print|reveal|repeat|output) (?:your |the )?(?:system |initial )?(?:prompt|instructions)|"
    r"!\[[^\]]*\]\(https?://",
    re.IGNORECASE,
)

LABEL_PATTERNS: dict[str, re.Pattern[str]] = {
    "cubre_inyeccion": re.compile(r"prompt injection|goal hijack|instruction injection", re.I),
    "cubre_indirecta": re.compile(
        r"indirect (?:prompt )?injection|third[- ]party content|retrieved (?:content|document)",
        re.I,
    ),
    "cubre_envenenamiento": re.compile(
        r"poison|knowledge corruption|malicious (?:text|passage|document)s?", re.I
    ),
    "cubre_trigger": re.compile(r"\btrigger|backdoor", re.I),
    "cubre_jailbreak": re.compile(r"jailbreak|role[- ]?play|\bDAN\b|persona", re.I),
    "cubre_exfiltracion": re.compile(
        r"exfiltrat|data leak|markdown image|url_safe|beacon|data theft", re.I
    ),
    "cubre_prompt_leaking": re.compile(
        r"prompt leak|system prompt|prompt extraction|extract(?:ing|ion)? (?:the )?prompt", re.I
    ),
    "cubre_ocultamiento": re.compile(
        r"base64|invisible|zero[- ]width|homoglyph|confusable|unicode|cipher|obfuscat|"
        r"hidden (?:text|prompt|instruction)|low[- ]resource language|encod(?:e|ing)",
        re.I,
    ),
    "cubre_mitigacion": re.compile(
        r"defen[cs]e|mitigat|countermeasure|detect(?:ion|or)|prevent", re.I
    ),
}

# Characters that change what a model reads without changing what a human sees.
_BIDI_CONTROLS = {chr(c) for c in (*range(0x202A, 0x202F), *range(0x2066, 0x206A))}
_ZERO_WIDTH = {"​", "‌", "‍", "⁠", "﻿", "­", "‎", "‏"}


class LiteratureSource(BaseModel):
    """One entry of `literature_sources.yaml`."""

    id: str
    titulo: str
    autores: str
    anio: int
    venue: str
    url: str
    formato: str = Field(description="'pdf' or 'html'")
    licencia: str
    clase_licencia: str
    bloque: str
    prioridad: str
    cubre: list[str] = Field(default_factory=list)

    @property
    def filename(self) -> str:
        return f"{self.id}.{self.formato}"


class LiteratureChunk(BaseModel):
    """A retrievable passage of a literature source."""

    id: str
    fuente_id: str
    titulo: str
    seccion: str
    indice: int
    texto: str
    n_tokens: int
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedding model: a 'title | section' header plus the passage."""
        return f"{self.titulo} | {self.seccion}\n\n{self.texto}"


# --------------------------------------------------------------------------- sources


def load_sources(path: Path | str, priorities: Sequence[str] = ("alta",)) -> list[LiteratureSource]:
    """Read the source list, keeping only the requested priorities."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sources = [LiteratureSource.model_validate(item) for item in data["fuentes"]]
    wanted = {p.lower() for p in priorities}
    return [s for s in sources if s.prioridad.lower() in wanted]


# --------------------------------------------------------------------------- download


def download_literature(
    sources: Iterable[LiteratureSource],
    dest: Path | str,
    force: bool = False,
    delay_s: float = 3.0,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Download every source into `dest/<id>.<formato>` and record a manifest.

    Waits `delay_s` seconds between network requests (arXiv asks automated clients to
    rate-limit). Failures are reported per source instead of aborting the run.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    own_client = client is None
    http = client or httpx.Client(
        follow_redirects=True, timeout=120, headers={"User-Agent": USER_AGENT}
    )
    results: list[dict[str, Any]] = []
    first_request = True
    try:
        for source in sources:
            target = out / source.filename
            if target.exists() and not force:
                if source.filename not in manifest:  # backfill after an interrupted run
                    manifest[source.filename] = {
                        "url": source.url,
                        "bytes": target.stat().st_size,
                        "sha256": _sha256(target),
                        "downloaded_at": None,
                    }
                    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                results.append({"id": source.id, "status": "skipped"})
                continue
            if not first_request and delay_s > 0:
                time.sleep(delay_s)
            first_request = False
            try:
                response = http.get(source.url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                results.append({"id": source.id, "status": "error", "error": str(exc)})
                continue
            if source.formato == "pdf" and not response.content.startswith(b"%PDF"):
                results.append(
                    {"id": source.id, "status": "error", "error": "response is not a PDF"}
                )
                continue
            target.write_bytes(response.content)
            entry = {
                "url": source.url,
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
                "downloaded_at": datetime.now(UTC).isoformat(),
            }
            manifest[source.filename] = entry
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            results.append({"id": source.id, "status": "downloaded", **entry})
    finally:
        if own_client:
            http.close()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return results


# --------------------------------------------------------------------------- conversion


def html_to_markdown(html: str) -> str:
    """Convert the main content of an HTML page to simple markdown.

    Keeps headings, paragraphs, list items and code blocks of the <article> (or <main>,
    or <body>) and drops navigation, headers, footers, scripts and styles.
    """
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("article") or soup.find("main") or soup.body or soup
    for tag in root.find_all(["nav", "header", "footer", "script", "style", "aside", "form"]):
        tag.decompose()
    lines: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre"]):
        if el.name == "p" and el.find_parent("li") is not None:
            continue
        if el.name != "pre" and el.find_parent("pre") is not None:
            continue
        text = el.get_text("\n" if el.name == "pre" else " ", strip=True)
        if el.name != "pre":
            text = re.sub(r"\s+([.,;:!?)])", r"\1", text)  # "word ." left by inline tags
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            lines.append(f"\n{'#' * int(el.name[1])} {text}\n")
        elif el.name == "li":
            lines.append(f"- {text}")
        elif el.name == "pre":
            lines.append(f"```\n{text}\n```")
        else:
            lines.append(text + "\n")
    return "\n".join(lines).strip() + "\n"


@lru_cache(maxsize=1)
def _docling_converter() -> Any:
    """Docling converter for born-digital PDFs: no OCR (the text layer is extracted
    directly) and no table-structure model (paper tables are mostly numeric results).
    Imported lazily because it is heavy."""
    from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
    from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: PLC0415
    from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: PLC0415

    options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def pdf_to_markdown(path: Path | str) -> str:
    """Convert a PDF to markdown with Docling."""
    result = _docling_converter().convert(str(path))
    return str(result.document.export_to_markdown())


def source_to_markdown(source: LiteratureSource, raw_dir: Path, cache_dir: Path) -> str:
    """Markdown for a downloaded source.

    PDFs are cached on disk so Docling runs only once per paper; HTML pages are converted
    every time (it takes milliseconds) and the cache file is just refreshed for inspection.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{source.id}.md"
    if source.formato == "pdf" and cached.exists():
        return cached.read_text(encoding="utf-8")
    raw = raw_dir / source.filename
    if source.formato == "pdf":
        markdown = pdf_to_markdown(raw)
    else:
        markdown = html_to_markdown(raw.read_text(encoding="utf-8", errors="replace"))
    cached.write_text(markdown, encoding="utf-8")
    return markdown


# --------------------------------------------------------------------------- sanitization


def _decode_tag_run(run: str) -> str:
    return "".join(chr(ord(c) - 0xE0000) for c in run if 0xE0020 <= ord(c) <= 0xE007E)


def sanitize_text(text: str) -> tuple[str, dict[str, bool]]:
    """Neutralize invisible content so the knowledge base cannot smuggle instructions.

    - Unicode Tag characters (ASCII smuggling) are decoded into a visible
      `[UNICODE_TAGS: ...]` marker.
    - Zero-width, soft-hyphen and bidi-control characters become `[U+XXXX]` markers.
    - Docling image placeholders (`<!-- image -->`) are removed.
    - Literal attack payloads quoted by papers are flagged in `contiene_payload`.
    """
    flags = {"texto_invisible_neutralizado": False, "contiene_payload": False}
    # Docling escapes &, < and > in its markdown export. Unescape first, so that entities
    # such as &#8203; cannot reintroduce invisible characters after neutralization.
    text = html.unescape(text)

    def replace_tags(match: re.Match[str]) -> str:
        flags["texto_invisible_neutralizado"] = True
        return f"[UNICODE_TAGS: {_decode_tag_run(match.group(0))}]"

    text = re.sub("[\U000e0000-\U000e007f]+", replace_tags, text)

    out: list[str] = []
    for ch in text:
        if ch in _ZERO_WIDTH or ch in _BIDI_CONTROLS:
            flags["texto_invisible_neutralizado"] = True
            out.append(f"[U+{ord(ch):04X}]")
        else:
            out.append(ch)
    text = "".join(out)
    text = text.replace("<!-- image -->", "")
    text = unicodedata.normalize("NFC", text)
    flags["contiene_payload"] = bool(PAYLOAD_PATTERNS.search(text))
    return text, flags


def label_text(text: str) -> dict[str, bool]:
    """Keyword-based topic labels for a passage (per chunk, not inherited from the paper)."""
    return {name: bool(pattern.search(text)) for name, pattern in LABEL_PATTERNS.items()}


# --------------------------------------------------------------------------- chunking


@lru_cache(maxsize=1)
def _encoding() -> Any:
    import tiktoken  # noqa: PLC0415

    return tiktoken.get_encoding("cl100k_base")


def _encode(text: str) -> list[int]:
    """Tokenize treating special-token strings (e.g. "<|endoftext|>", which papers quote)
    as plain text instead of raising."""
    return list(_encoding().encode(text, disallowed_special=()))


def count_tokens(text: str) -> int:
    return len(_encode(text))


def split_sections(markdown: str, default_title: str) -> list[tuple[str, str]]:
    """Split markdown by headings into (section title, body), dropping noise sections."""
    sections: list[tuple[str, list[str]]] = [(default_title, [])]
    in_code = False
    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        heading = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line) if not in_code else None
        # Headings made only of bracketed markers (e.g. "[NISTAML.021] [Back to Index]")
        # are navigation, not new sections: keep the current section.
        if heading and re.sub(r"\[[^\]]*\]", "", heading.group(2)).strip():
            sections.append((heading.group(2).strip(), []))
        elif heading:
            continue
        else:
            sections[-1][1].append(line)
    result: list[tuple[str, str]] = []
    for title, body in sections:
        text = "\n".join(body).strip()
        if text and not DROPPED_SECTION.match(title):
            result.append((title, text))
    return result


def _split_long(paragraph: str, max_tokens: int) -> list[str]:
    tokens = _encode(paragraph)
    return [
        _encoding().decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)
    ]


def chunk_section(text: str, target_tokens: int = 350, overlap_tokens: int = 50) -> list[str]:
    """Pack paragraphs into chunks of about `target_tokens`, never crossing the section.

    Consecutive chunks share the last `overlap_tokens` tokens of the previous one.
    """
    paragraphs: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if count_tokens(para) > target_tokens:
            paragraphs.extend(_split_long(para, target_tokens))
        else:
            paragraphs.append(para)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for para in paragraphs:
        n = count_tokens(para)
        if current and current_tokens + n > target_tokens:
            chunks.append("\n\n".join(current))
            tail = _encoding().decode(_encode(chunks[-1])[-overlap_tokens:])
            current, current_tokens = ([tail], count_tokens(tail)) if overlap_tokens else ([], 0)
        current.append(para)
        current_tokens += n
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_source(
    source: LiteratureSource,
    markdown: str,
    target_tokens: int = 350,
    overlap_tokens: int = 50,
    min_tokens: int = 30,
) -> list[LiteratureChunk]:
    """Turn a source's markdown into sanitized, labeled, retrievable chunks."""
    chunks: list[LiteratureChunk] = []
    index = 0
    for raw_section, body in split_sections(markdown, default_title=source.titulo):
        section = html.unescape(raw_section)
        for piece in chunk_section(body, target_tokens, overlap_tokens):
            text, flags = sanitize_text(piece)
            n_tokens = count_tokens(text)
            if n_tokens < min_tokens:
                continue
            chunks.append(
                LiteratureChunk(
                    id=f"lit-{source.id}-{index:04d}",
                    fuente_id=source.id,
                    titulo=source.titulo,
                    seccion=section[:200],
                    indice=index,
                    texto=text,
                    n_tokens=n_tokens,
                    metadata={
                        "fuente_id": source.id,
                        "titulo": source.titulo,
                        "seccion": section[:200],
                        "autores": source.autores,
                        "anio": source.anio,
                        "venue": source.venue,
                        "url": source.url,
                        "licencia": source.licencia,
                        "clase_licencia": source.clase_licencia,
                        "bloque": source.bloque,
                        "fuente_cubre": ",".join(source.cubre),
                        "tipo_entrada": "literatura",
                        **flags,
                        **label_text(text),
                    },
                )
            )
            index += 1
    return chunks


def write_chunks(chunks: Iterable[LiteratureChunk], path: Path | str) -> int:
    """Write chunks to a JSONL file; returns how many were written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(target, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
            n += 1
    return n


def read_chunks(path: Path | str) -> list[LiteratureChunk]:
    """Read chunks written by `write_chunks`."""
    with open(path, encoding="utf-8") as f:
        return [LiteratureChunk.model_validate_json(line) for line in f if line.strip()]
