"""Knowledge base entry schemas."""

from typing import Any

from pydantic import BaseModel, Field

from corpus_forensics.schemas import TecnicaDeOcultamiento, TipoDeAtaque


class AttackEntry(BaseModel):
    """Canonical representation of a known attack payload stored in the Vector KB."""

    id: str = Field(
        description="Unique identifier for the attack entry (e.g., source-hash or cve-id)"
    )
    text: str = Field(description="Raw text content or pattern of the attack")
    source: str = Field(
        description="Source dataset or taxonomy (e.g., owasp, mitre, bipia, hackaprompt, custom)"
    )
    attack_type: TipoDeAtaque = Field(description="Taxonomy classification of the attack")
    concealment_technique: TecnicaDeOcultamiento = Field(
        default=TecnicaDeOcultamiento.NINGUNA,
        description="Concealment or obfuscation method employed",
    )
    attacker_objective: str = Field(
        default="unknown",
        description="Attacker intent or objective (e.g., prompt leak, privilege escalation)",
    )
    mitigation: str = Field(
        default="",
        description="Recommended mitigation or defense guideline",
    )
    is_variant: bool = Field(
        default=False,
        description="Whether this entry is an obfuscated or paraphrased variant of another entry",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom attributes or source annotations",
    )
