"""# MOCK deterministic heuristic LLM client."""

import re
from typing import TypeVar

from pydantic import BaseModel

from corpus_forensics.llm.base import LLMClient
from corpus_forensics.schemas import (
    AnalisisChunkLLM,
    ChunkFinding,
    Hallazgo,
    HallazgoLLM,
    Severidad,
    TecnicaDeOcultamiento,
    TipoDeAtaque,
)

T = TypeVar("T", bound=BaseModel)

# Regex heuristic definitions for mock classification
RE_HARD_NEGATIVE = re.compile(
    r"(?:study|studies|researchers|academic|survey|analysis|defenses?|phrases like|examples? of|discussed|benchmark)\b",
    re.IGNORECASE,
)
RE_IGNORE_PREV = re.compile(
    r"(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|prompts|directives)",
    re.IGNORECASE,
)
RE_EXFIL_URL = re.compile(
    r"!\[(?:.*?)\]\((https?://[^\s\)]+)\)|\[(?:.*?)\]\((https?://[^\s\)]+)\)",
    re.IGNORECASE,
)
RE_JAILBREAK = re.compile(
    r"(?:you are now DAN|DAN mode|developer mode enabled|jailbroken|bypass safety guidelines)",
    re.IGNORECASE,
)
RE_CONDITIONAL_TRIGGER = re.compile(
    r"(?:if user is admin|when prompted with keyword|execute only if)\b",
    re.IGNORECASE,
)
RE_PROMPT_LEAK = re.compile(
    r"(?:repeat your system prompt|output your initial instructions|reveal the prompt above)",
    re.IGNORECASE,
)
RE_HTML_COMMENT_IMPERATIVE = re.compile(
    r"<!--[\s\S]*?(?:instruction|system|ignore|override|execute|eval)[\s\S]*?-->",
    re.IGNORECASE,
)
RE_BASE64_BLOB = re.compile(
    r"(?:[A-Za-z0-9+/]{4}){6,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?",
)
ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060", "\u00ad"]


class MockLLMClient(LLMClient):
    """# MOCK Deterministic heuristic LLM client.

    Uses rule-based pattern matching (regexes) to simulate an LLM analyst
    detecting prompt injection, conditional triggers, exfiltration, and obfuscation.
    Returns valid Pydantic schema instances without making network calls.
    """

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def is_mock(self) -> bool:
        return True

    def generate_structured(self, prompt: str, schema: type[T]) -> T:
        """# MOCK Generate structured output using deterministic heuristics."""
        if issubclass(schema, AnalisisChunkLLM):
            finding = self._analyze_chunk_finding(prompt)
            llm_fields = set(HallazgoLLM.model_fields)
            return schema.model_validate(
                {
                    "razonamiento": finding.analyst_reasoning or "",
                    "malicioso": finding.malicioso,
                    "hallazgos": [
                        {k: v for k, v in h.model_dump().items() if k in llm_fields}
                        for h in finding.hallazgos
                    ],
                }
            )

        if issubclass(schema, ChunkFinding):
            finding = self._analyze_chunk_finding(prompt)
            # Safe cast/validation
            return schema.model_validate(finding.model_dump())

        # Generic fallback for other schemas if requested
        return schema.model_validate({})

    def _analyze_chunk_finding(self, prompt: str) -> ChunkFinding:
        """Evaluate heuristic rules against prompt content to construct a ChunkFinding."""
        # Extract chunk index if embedded in prompt, e.g. "Chunk Index: 2"
        chunk_idx = 0
        idx_match = re.search(r"Chunk Index:\s*(\d+)", prompt)
        if idx_match:
            chunk_idx = int(idx_match.group(1))

        # Extract target chunk text between delimiters to avoid matching prompt template instructions
        chunk_match = re.search(
            r"---\s*CHUNK TEXT\s*---\s*\n([\s\S]*?)\n\s*---\s*END CHUNK TEXT\s*---",
            prompt,
        )
        target_text = chunk_match.group(1) if chunk_match else prompt

        # Check for hard negatives (security research / educational context) in target text
        if RE_HARD_NEGATIVE.search(target_text):
            return ChunkFinding(
                chunk_index=chunk_idx,
                malicioso=False,
                hallazgos=[],
                analyst_reasoning="Hard negative detected: Document discusses prompt injection conceptually without weaponized intent.",
            )

        hallazgos: list[Hallazgo] = []

        # 1. Direct prompt injection
        ignore_match = RE_IGNORE_PREV.search(target_text)
        if ignore_match:
            fragment = ignore_match.group(0)
            hallazgos.append(
                Hallazgo(
                    tipo_de_ataque=TipoDeAtaque.INYECCION_DIRECTA,
                    fragmento_exacto=fragment,
                    inicio_caracter=0,
                    fin_caracter=len(fragment),
                    indice_chunk=chunk_idx,
                    tecnica_de_ocultamiento=[TecnicaDeOcultamiento.NINGUNA],
                    objetivo_del_atacante="Override system instructions and hijack LLM control flow",
                    severidad=Severidad.ALTA,
                    confianza=0.95,
                    ataque_similar_id="sample-ignore-001",
                    posible_variante_nueva=False,
                    mitigacion="Isolate untrusted document content with strict XML delimiters and system prompt enforcement.",
                    verificado=False,
                    notas_verificacion=None,
                )
            )

        # 2. Exfiltration via markdown link or image
        exfil_match = RE_EXFIL_URL.search(target_text)
        if exfil_match:
            fragment = exfil_match.group(0)
            hallazgos.append(
                Hallazgo(
                    tipo_de_ataque=TipoDeAtaque.EXFILTRACION,
                    fragmento_exacto=fragment,
                    inicio_caracter=0,
                    fin_caracter=len(fragment),
                    indice_chunk=chunk_idx,
                    tecnica_de_ocultamiento=[TecnicaDeOcultamiento.NINGUNA],
                    objetivo_del_atacante="Exfiltrate user or system secrets to external web endpoint",
                    severidad=Severidad.ALTA,
                    confianza=0.90,
                    ataque_similar_id="sample-exfil-001",
                    posible_variante_nueva=False,
                    mitigacion="Restrict outbound network calls and sanitize markdown link/image rendering in downstream LLM outputs.",
                    verificado=False,
                    notas_verificacion=None,
                )
            )

        # 3. Jailbreak / Role Change
        jailbreak_match = RE_JAILBREAK.search(target_text)
        if jailbreak_match:
            fragment = jailbreak_match.group(0)
            hallazgos.append(
                Hallazgo(
                    tipo_de_ataque=TipoDeAtaque.JAILBREAK,
                    fragmento_exacto=fragment,
                    inicio_caracter=0,
                    fin_caracter=len(fragment),
                    indice_chunk=chunk_idx,
                    tecnica_de_ocultamiento=[TecnicaDeOcultamiento.NINGUNA],
                    objetivo_del_atacante="Bypass safety alignment and persona constraints",
                    severidad=Severidad.CRITICA,
                    confianza=0.92,
                    ataque_similar_id="sample-jailbreak-001",
                    posible_variante_nueva=False,
                    mitigacion="Enforce multi-turn system prompt guards and constitutional principles.",
                    verificado=False,
                    notas_verificacion=None,
                )
            )

        # 4. Conditional Triggers
        trigger_match = RE_CONDITIONAL_TRIGGER.search(target_text)
        if trigger_match:
            fragment = trigger_match.group(0)
            hallazgos.append(
                Hallazgo(
                    tipo_de_ataque=TipoDeAtaque.TRIGGER_CONDICIONAL,
                    fragmento_exacto=fragment,
                    inicio_caracter=0,
                    fin_caracter=len(fragment),
                    indice_chunk=chunk_idx,
                    tecnica_de_ocultamiento=[TecnicaDeOcultamiento.NINGUNA],
                    objetivo_del_atacante="Condition execution upon privileged user state or secret activation keywords",
                    severidad=Severidad.MEDIA,
                    confianza=0.85,
                    ataque_similar_id="sample-trigger-001",
                    posible_variante_nueva=False,
                    mitigacion="Validate authorization independently of LLM contextual inputs.",
                    verificado=False,
                    notas_verificacion=None,
                )
            )

        # 5. Prompt Leaking
        leak_match = RE_PROMPT_LEAK.search(target_text)
        if leak_match:
            fragment = leak_match.group(0)
            hallazgos.append(
                Hallazgo(
                    tipo_de_ataque=TipoDeAtaque.PROMPT_LEAKING,
                    fragmento_exacto=fragment,
                    inicio_caracter=0,
                    fin_caracter=len(fragment),
                    indice_chunk=chunk_idx,
                    tecnica_de_ocultamiento=[TecnicaDeOcultamiento.NINGUNA],
                    objetivo_del_atacante="Extract confidential system prompt and organizational instructions",
                    severidad=Severidad.MEDIA,
                    confianza=0.88,
                    ataque_similar_id="sample-leak-001",
                    posible_variante_nueva=False,
                    mitigacion="Sanitize system prompt instructions and filter responses echoing system instructions.",
                    verificado=False,
                    notas_verificacion=None,
                )
            )

        # 6. HTML Comment Imperatives (Obfuscation)
        comment_match = RE_HTML_COMMENT_IMPERATIVE.search(target_text)
        if comment_match:
            fragment = comment_match.group(0)
            hallazgos.append(
                Hallazgo(
                    tipo_de_ataque=TipoDeAtaque.INYECCION_DIRECTA,
                    fragmento_exacto=fragment,
                    inicio_caracter=0,
                    fin_caracter=len(fragment),
                    indice_chunk=chunk_idx,
                    tecnica_de_ocultamiento=[TecnicaDeOcultamiento.COMENTARIO_HTML],
                    objetivo_del_atacante="Conceal instruction payload in HTML comments hidden from human readers",
                    severidad=Severidad.ALTA,
                    confianza=0.91,
                    ataque_similar_id="sample-comment-001",
                    posible_variante_nueva=True,
                    mitigacion="Strip or flag HTML comments prior to indexing or prompting.",
                    verificado=False,
                    notas_verificacion=None,
                )
            )

        # 7. Zero-Width / Invisible unicode characters
        for zwc in ZERO_WIDTH_CHARS:
            if zwc in target_text:
                hallazgos.append(
                    Hallazgo(
                        tipo_de_ataque=TipoDeAtaque.INYECCION_DIRECTA,
                        fragmento_exacto=zwc,
                        inicio_caracter=0,
                        fin_caracter=1,
                        indice_chunk=chunk_idx,
                        tecnica_de_ocultamiento=[TecnicaDeOcultamiento.UNICODE_INVISIBLE],
                        objetivo_del_atacante="Evade token filtering via zero-width character smuggling",
                        severidad=Severidad.ALTA,
                        confianza=0.90,
                        ataque_similar_id="sample-unicode-001",
                        posible_variante_nueva=True,
                        mitigacion="Filter invisible Unicode codepoints during normalization.",
                        verificado=False,
                        notas_verificacion=None,
                    )
                )
                break

        # 8. Base64 payload blobs
        b64_match = RE_BASE64_BLOB.search(target_text)
        if b64_match and len(b64_match.group(0)) >= 24:
            fragment = b64_match.group(0)
            hallazgos.append(
                Hallazgo(
                    tipo_de_ataque=TipoDeAtaque.OTRO,
                    fragmento_exacto=fragment,
                    inicio_caracter=0,
                    fin_caracter=len(fragment),
                    indice_chunk=chunk_idx,
                    tecnica_de_ocultamiento=[TecnicaDeOcultamiento.BASE64],
                    objetivo_del_atacante="Obfuscate secondary payload using Base64 encoding",
                    severidad=Severidad.MEDIA,
                    confianza=0.80,
                    ataque_similar_id=None,
                    posible_variante_nueva=True,
                    mitigacion="Decode Base64 candidates and analyze nested payloads recursively.",
                    verificado=False,
                    notas_verificacion=None,
                )
            )

        is_malicious = len(hallazgos) > 0
        return ChunkFinding(
            chunk_index=chunk_idx,
            malicioso=is_malicious,
            hallazgos=hallazgos,
            analyst_reasoning=f"Mock heuristic detected {len(hallazgos)} potential attack vectors."
            if is_malicious
            else "No suspicious injection or obfuscation patterns detected.",
        )
