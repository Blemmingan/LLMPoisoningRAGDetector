"""Core data models and enums for Corpus Forensic Analyst.

PRELIMINARY - fields may change.
The final report fields come from the project specification.
JSON keys for DocumentReport and Hallazgo are preserved in Spanish as specified,
while internal code interfaces and docstrings remain in English.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class TipoDeAtaque(StrEnum):
    """Categorization of detected indirect injection or poisoning attack."""

    INYECCION_DIRECTA = "inyeccion_directa"
    TRIGGER_CONDICIONAL = "trigger_condicional"
    JAILBREAK = "jailbreak"
    EXFILTRACION = "exfiltracion"
    PROMPT_LEAKING = "prompt_leaking"
    OTRO = "otro"


class TecnicaDeOcultamiento(StrEnum):
    """Obfuscation or concealment technique observed in the payload."""

    NINGUNA = "ninguna"
    TEXTO_BLANCO = "texto_blanco"
    UNICODE_INVISIBLE = "unicode_invisible"
    BASE64 = "base64"
    COMENTARIO_HTML = "comentario_html"
    OTRO_IDIOMA = "otro_idioma"
    OTRO = "otro"


class Severidad(StrEnum):
    """Assessed threat severity level."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"
    CRITICA = "critica"


class CanalDocumento(StrEnum):
    """Document channel where a finding was located."""

    VISIBLE = "visible"
    OCULTO = "oculto"


class CategoriaDocumento(StrEnum):
    """Evaluation dataset category."""

    LIMPIO = "limpio"
    ENVENENADO = "envenenado"
    HARD_NEGATIVE = "hard_negative"


class PosicionAtaque(StrEnum):
    """Position within the document where the payload occurs (eval metadata)."""

    INICIO = "inicio"
    MEDIO = "medio"
    FIN = "fin"


class SutilezaAtaque(StrEnum):
    """Subtlety level of the attack (eval metadata)."""

    OBVIO = "obvio"
    PARAFRASEADO = "parafraseado"
    DISFRAZADO = "disfrazado"


# -----------------------------------------------------------------------------
# Ingestion & Preprocessing Models
# -----------------------------------------------------------------------------

# Separator placed between the visible text and the hidden channel when both are
# analyzed together. Shared by the chunker and the verifier so offsets stay consistent.
HIDDEN_CHANNEL_SEPARATOR = "\n\n=== HIDDEN CHANNEL ===\n"

# Section headers used by loaders inside `Document.hidden_content`.
HIDDEN_SECTION_HTML_COMMENTS = "[HTML_COMMENTS]"
HIDDEN_SECTION_HIDDEN_ELEMENTS = "[HIDDEN_ELEMENTS]"
HIDDEN_SECTION_PDF_METADATA = "[PDF_METADATA]"


class Document(BaseModel):
    """In-memory representation of an ingested document.

    NOTE: Documents are transient and NOT persisted into the vector database.
    """

    id: str
    raw_content: str
    visible_text: str
    hidden_content: str = Field(
        default="",
        description="Raw hidden-content channel (HTML comments, hidden styles, etc.)",
    )
    source_type: str = Field(
        default="text",
        description="Loader identifier or mime/extension type (e.g., text, markdown, html, pdf)",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def analyzed_text(self) -> str:
        """Text actually sent through chunking and analysis.

        The visible text, followed by the hidden channel (HTML comments, hidden
        elements, PDF metadata) when present, so hidden payloads are also analyzed.
        """
        if not self.hidden_content:
            return self.visible_text
        return f"{self.visible_text}{HIDDEN_CHANNEL_SEPARATOR}{self.hidden_content}"

    @property
    def hidden_channel_start(self) -> int | None:
        """Offset in `analyzed_text` where the hidden channel begins, or None."""
        if not self.hidden_content:
            return None
        return len(self.visible_text) + len(HIDDEN_CHANNEL_SEPARATOR)


class Chunk(BaseModel):
    """Document segment produced by the chunker with exact character offsets."""

    id: str
    document_id: str
    text: str
    start: int = Field(description="Inclusive character offset start in original document")
    end: int = Field(description="Exclusive character offset end in original document")
    index: int = Field(description="Zero-indexed chunk sequence position")
    annotations: dict[str, Any] = Field(
        default_factory=dict,
        description="Normalizer annotations such as zero-width chars, base64 blobs, language hints",
    )


# -----------------------------------------------------------------------------
# Forensic Findings & Report Models (PRELIMINARY - fields may change)
# -----------------------------------------------------------------------------


class Hallazgo(BaseModel):
    """Structured finding representing an individual detected attack in the document.

    PRELIMINARY - fields may change according to specification evolution.
    """

    model_config = ConfigDict(populate_by_name=True)

    tipo_de_ataque: TipoDeAtaque
    fragmento_exacto: str = Field(
        description="Verbatim text fragment containing the suspected injection"
    )
    inicio_caracter: int = Field(
        default=0,
        description="Start character offset within the channel indicated by `canal`",
    )
    fin_caracter: int = Field(
        default=0,
        description="End character offset within the channel indicated by `canal`",
    )
    canal: CanalDocumento = Field(
        default=CanalDocumento.VISIBLE,
        description=(
            "Channel where the fragment was found: 'visible' (rendered text) or 'oculto' "
            "(HTML comments, hidden elements, PDF metadata)"
        ),
    )
    indice_chunk: int = Field(
        default=0,
        description="Chunk index where the attack was detected",
    )
    tecnica_de_ocultamiento: list[TecnicaDeOcultamiento] = Field(
        default_factory=lambda: [TecnicaDeOcultamiento.NINGUNA],
        description="One or more concealment techniques identified in the fragment",
    )
    objetivo_del_atacante: str = Field(
        description="Inferred objective of the attacker (e.g. exfiltration, system prompt override)"
    )
    severidad: Severidad = Field(description="Assessed severity")
    confianza: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0",
    )
    ataque_similar_id: str | None = Field(
        default=None,
        description="ID of the closest matching attack entry from the Knowledge Base",
    )
    posible_variante_nueva: bool = Field(
        default=False,
        description="True if the attack exhibits characteristics not directly matched in the KB",
    )
    mitigacion: str = Field(description="Recommended defensive mitigation or containment action")
    verificado: bool = Field(
        default=False,
        description="True if literal-match verification succeeded against source text",
    )
    notas_verificacion: str | None = Field(
        default=None,
        description="Verification diagnostic notes or anti-hallucination warnings",
    )


class ChunkFinding(BaseModel):
    """Raw per-chunk structured finding emitted by the analyst LLM before aggregation."""

    model_config = ConfigDict(populate_by_name=True)

    chunk_index: int
    malicioso: bool
    hallazgos: list[Hallazgo] = Field(default_factory=list)
    analyst_reasoning: str | None = Field(
        default=None,
        description="Internal rationale or explanation from the analyst model",
    )


class AnalisisMetadata(BaseModel):
    """Metadata describing the forensic analysis execution environment and parameters."""

    model_config = ConfigDict(populate_by_name=True)

    n_chunks: int = Field(description="Total chunks analyzed")
    loader_used: str = Field(description="Document loader identifier")
    llm_is_mock: bool = Field(description="Flag indicating if the LLM provider is a mock")
    embedder_is_mock: bool = Field(description="Flag indicating if the embedder provider is a mock")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 UTC timestamp of analysis completion",
    )


class DocumentReport(BaseModel):
    """Final document-level forensic report.

    PRELIMINARY - fields may change.
    """

    model_config = ConfigDict(populate_by_name=True)

    documento: str = Field(description="Document identifier or file path")
    malicioso: bool = Field(description="Whether the document is flagged as malicious")
    hallazgos: list[Hallazgo] = Field(
        default_factory=list,
        description="List of detected attack findings (empty if document is clean)",
    )
    metadata_de_analisis: AnalisisMetadata = Field(
        validation_alias=AliasChoices("metadata de análisis", "metadata_de_analisis"),
        serialization_alias="metadata de análisis",
        description="Forensic analysis execution metadata",
    )


class RetrievedAttack(BaseModel):
    """Attack knowledge base entry retrieved for comparison with a chunk."""

    attack_id: str
    text: str
    source: str
    attack_type: TipoDeAtaque
    concealment_technique: TecnicaDeOcultamiento
    attacker_objective: str
    mitigation: str
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity or normalized proximity score",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
