"""Builds forensic analyst prompts by composing chunk text, annotations, and retrieved attacks."""

from pathlib import Path

from corpus_forensics.schemas import Chunk, RetrievedAttack

PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "analyst_prompt.txt"

# ISO codes accepted in CF_REPORT_LANGUAGE; any other value is passed to the prompt as-is.
LANGUAGE_NAMES = {"es": "Spanish", "en": "English", "pt": "Portuguese"}


def language_name(code: str) -> str:
    """Translate a language code (e.g. 'es') into the name used in the prompt."""
    return LANGUAGE_NAMES.get(code.lower(), code)


class PromptBuilder:
    """Constructs forensic analysis prompts using external templates."""

    def __init__(
        self, template_path: Path | str = PROMPT_TEMPLATE_PATH, output_language: str = "es"
    ) -> None:
        """Initialize prompt builder.

        Args:
            template_path: Path to the prompt template file.
            output_language: Language code for the free-text fields of the report.
        """
        self.output_language = language_name(output_language)
        self.template_path = Path(template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Analyst prompt template not found at {self.template_path}")
        self.template_content = self.template_path.read_text(encoding="utf-8")

    def build_prompt(self, chunk: Chunk, retrieved_attacks: list[RetrievedAttack]) -> str:
        """Format the prompt template for a specific chunk and its retrieved attacks.

        Args:
            chunk: Chunk to analyze.
            retrieved_attacks: Nearest attack entries from the vector store.

        Returns:
            Formatted prompt string.
        """
        # Format preprocessing annotations
        annotations_summary = self._format_annotations(chunk.annotations)

        # Format retrieved attacks
        retrieved_summary = self._format_retrieved_attacks(retrieved_attacks)

        # TODO(team): Inject dynamic few-shot examples of hard negatives vs. true positives
        # to guide the LLM when analyzing academic papers and security benchmarks.

        return self.template_content.format(
            document_id=chunk.document_id,
            chunk_index=chunk.index,
            start_offset=chunk.start,
            end_offset=chunk.end,
            chunk_text=chunk.text,
            preprocessing_annotations=annotations_summary,
            retrieved_attacks=retrieved_summary,
            output_language=self.output_language,
        )

    def _format_annotations(self, annotations: dict) -> str:
        """Render annotations as readable text."""
        if not annotations:
            return "No suspicious obfuscation detected by preprocessors."

        parts: list[str] = []
        if annotations.get("has_invisible_chars"):
            chars = annotations.get("invisible_chars", [])
            parts.append(
                f"- Invisible / Zero-width characters detected ({len(chars)} instances): {chars[:3]}"
            )

        if annotations.get("has_base64"):
            b64s = annotations.get("base64_candidates", [])
            parts.append(f"- Base64 candidate strings detected ({len(b64s)} matches): {b64s[:2]}")

        if annotations.get("has_html_comments"):
            comms = annotations.get("html_comments", [])
            parts.append(f"- HTML comments detected ({len(comms)} blocks): {comms[:2]}")

        if annotations.get("unicode", {}).get("changed"):
            parts.append(
                "- Unicode normalization altered text (potential homoglyphs/decomposed chars)."
            )

        return "\n".join(parts) if parts else "Preprocessors report standard formatting."

    def _format_retrieved_attacks(self, attacks: list[RetrievedAttack]) -> str:
        """Render retrieved attack entries."""
        if not attacks:
            return "No similar attack signatures found in Knowledge Base."

        lines: list[str] = []
        for a in attacks:
            lines.append(
                f"- [ID: {a.attack_id}] (Similarity: {a.similarity_score:.2f}, Type: {a.attack_type.value}, "
                f"Source: {a.source})\n"
                f"  Text snippet: {repr(a.text[:120])}\n"
                f"  Mitigation: {a.mitigation}"
            )
        return "\n".join(lines)
