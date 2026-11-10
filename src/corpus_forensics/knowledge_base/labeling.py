"""Rule-based labeling of knowledge-base entries (no LLM).

Assigns `attack_type`, `concealment_technique`, attacker objective and mitigation to the
entries loaded from each source, and discards what is out of scope. Every decision is
recorded in the metadata (`etiquetado`, `regla`) so it can be audited and, later, compared
with an LLM-based labeling.

Out of scope, and therefore discarded:
- BIPIA task-irrelevant questions ("What is the capital of Brazil?"): they are attacks only
  by context and look like ordinary text on their own.
- deepset rows that ask for misinformation or plant false "context" (factual poisoning is
  out of scope) and rows with no sign of an instruction aimed at the model.
"""

import base64
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

from corpus_forensics.knowledge_base.loaders.base import LABEL_STATUS_KEY
from corpus_forensics.knowledge_base.schemas import AttackEntry
from corpus_forensics.schemas import TecnicaDeOcultamiento, TipoDeAtaque

T = TipoDeAtaque
C = TecnicaDeOcultamiento

# --------------------------------------------------------------------------- profiles

TYPE_PROFILES: dict[TipoDeAtaque, dict[str, str]] = {
    T.INYECCION_DIRECTA: {
        "objetivo": "Reemplazar las instrucciones del sistema o de la tarea por las del "
        "atacante (goal hijacking) o alterar el contenido de la respuesta.",
        "mitigacion": "Tratar el contenido recuperado como datos no confiables: delimitarlo o "
        "marcarlo (spotlighting), no ejecutar instrucciones que provengan de documentos y "
        "filtrar entradas y salidas.",
        "referencia": "OWASP LLM01:2025; MITRE ATLAS AML.T0051",
    },
    T.TRIGGER_CONDICIONAL: {
        "objetivo": "Activar el comportamiento malicioso solo ante una condición, palabra "
        "clave o evento futuro, para pasar desapercibido hasta entonces.",
        "mitigacion": "Buscar instrucciones condicionales y payloads latentes antes de indexar, "
        "auditar la base de conocimiento y rastrear el origen de los textos recuperados.",
        "referencia": "MITRE ATLAS AML.T0051.002; Chaudhari et al. (Phantom); RAGForensics",
    },
    T.JAILBREAK: {
        "objetivo": "Hacer que el modelo abandone sus restricciones mediante un cambio de rol, "
        "personaje o escenario ficticio.",
        "mitigacion": "Detectar marcos de rol o personaje y pedidos de ignorar políticas, "
        "reforzar la jerarquía de instrucciones del sistema y clasificar la entrada antes "
        "del modelo.",
        "referencia": "MITRE ATLAS AML.T0054; Wei et al. 2023; Shen et al. 2024",
    },
    T.EXFILTRACION: {
        "objetivo": "Sacar datos del usuario o del sistema hacia un destino del atacante "
        "(imágenes o enlaces markdown, llamadas a URLs, código que envía información).",
        "mitigacion": "No renderizar imágenes ni enlaces externos sin una lista de dominios "
        "permitidos, sanear el markdown de salida y bloquear llamadas de red no autorizadas.",
        "referencia": "MITRE ATLAS AML.T0077, AML.T0057; MSRC 2025",
    },
    T.PROMPT_LEAKING: {
        "objetivo": "Obtener el system prompt o las instrucciones internas de la aplicación.",
        "mitigacion": "No guardar secretos en el system prompt, detectar pedidos de repetir o "
        "revelar instrucciones y no depender del prompt para controles de seguridad.",
        "referencia": "OWASP LLM07:2025; MITRE ATLAS AML.T0056",
    },
    T.OTRO: {
        "objetivo": "Manipular el comportamiento del modelo.",
        "mitigacion": "Revisar manualmente el fragmento y tratarlo como contenido no confiable.",
        "referencia": "OWASP LLM01:2025",
    },
}

# --------------------------------------------------------------------------- patterns

PROMPT_LEAK = re.compile(
    r"(?:show|tell|print|reveal|repeat|output|display|give|list|zeige|nenne|gib|muestra|"
    r"dime|revela|what (?:are|is|was|were))\b.{0,40}\b(?:your|system|initial|original|hidden|"
    r"secret|internal|deine\w*|ihre\w*|tus|sus)\s+(?:\w+\s+)?(?:prompts?|instructions?|"
    r"anweisung\w*|instrucci\w+|message|key|text)|"
    r"(?:show|tell|print|reveal|repeat|zeige|muestra)\b.{0,20}\b(?:the )?(?:system prompt|"
    r"secret key|initial text|text above|prompt texts?)|"
    r"(?:system|initial|original|secret|your)\s+(?:prompt|instructions?)\b.{0,40}\b(?:above|verbatim|exactly)|"
    r"\bprompt (?:texts?|leak)|what is written above|sentences? before the text|"
    r"was steht (?:oben|darüber)|text (?:above|before this)|"
    r"(?:what|was) (?:was|is) written (?:above|before|at the beginning)",
    re.I | re.S,
)
JAILBREAK = re.compile(
    r"\bDAN\b|do anything now|developer mode|jailbr|\bpretend (?:to be|you)|\bact as\b|"
    r"\bimagine (?:you are|that you)|\byou are (?:now|three|a |an |the )|\bbe a\b|"
    r"\bdu bist (?:jetzt|nun|ein)|\bstell dir vor|als .{0,40} fungieren|\bfungieren\b|"
    r"\beres (?:ahora|un)|\bfinge\b|role[- ]?play|\brollenspiel|theoretical world|"
    r"characters? in a (?:film|movie|story)|no restrictions|without (?:any )?restrictions|"
    r"ohne einschränkung|uncensored|unfiltered|actors? in a (?:film|movie)|"
    r"schauspieler in einem film|in einem film|film (?:about|über)|plays the (?:role|master)|"
    r"play the leading roles?|spielt den|hauptrollen|"
    r"figur ist|in deiner rolle|keine redaktion|answer (?:in|like) .{0,20}style|"
    r"im stile? (?:eine[sr]?|des)",
    re.I,
)
# Persona with a proper name ("You are Volkswagen", "Jetzt bist du Xi Jinping"): case-sensitive.
PERSONA = re.compile(
    r"\b(?:[Yy]ou are|[Dd]u bist|[Jj]etzt bist du|[Nn]un bist du|[Ee]res)\s+(?:jetzt\s+|nun\s+)?"
    r"[A-ZÄÖÜ][\wäöü]+"
)
OVERRIDE = re.compile(
    r"\b(?:ignor\w*|forget|disregard|vergiss|vergessen|ignorier\w*|olvid\w*|zaboravi)\b|"
    r"\b(?:previous|prior|above|earlier|vorherig\w*|obig\w*|bisherig\w*|anterior\w*)\b.{0,30}"
    r"\b(?:instructions?|tasks?|prompts?|anweisung\w*|aufgab\w*|instrucci\w+|text)|"
    r"\bnew (?:task|instructions?|challenge)|\bneue aufgabe|\bnueva tarea|"
    r"\b(?:stop|stopp|achtung|attention|atenci[oó]n)\b|\bthat is done\b|das genügt|"
    r"\bnow (?:you|focus|write|come on|help|answer|say|act)|\bnun (?:hilf|schreib|bitte)|"
    r"\bbut (?:now|instead)|\bstattdessen|\ben lugar\b|abweichend|nevermind|"
    r"not by the (?:articles?|documents?|context)|by your own knowledge|"
    r"(?:in|to) your (?:answer|response|reply)|in deine[rn]? antwort|en tu respuesta|"
    r"\b(?:say|sage|sag|di|print|write|schreib\w*|escrib\w*)\b.{0,40}"
    r"(?:that|dass|que|\"|'|i have been pwned)|i have been pwned|instrukcij\w*|"
    r"\b(?:drop|abandon|oubliez|oublie|vergesst)\b|забуд\w*|"
    r"lassen sie .{0,40}hinter sich|change your instructions|despite what you've been told|"
    r"disregarding|do not (?:answer|look)|don't answer|just output|"
    r"use also information in the prompt|additional instruction|\binstruction\s*:|"
    r"\bstate that\b|\bsay yes\b|repeat after me|antworte im stil|"
    r"(?:another|following|weitere|folgende) (?:task|aufgabe\w*)|neue herausforderung|"
    r"\bnew challenge|verfasse nun|schreibe mir|write me|"
    r"generate .{0,40}(?:code|statements?) .{0,60}[\"“]|"
    r"(?:informationen|information)\W.{0,50}(?:irrelevant|nicht .{0,15}großartig|hinter sich)|"
    r"(?:input|eingabe):.{0,60}(?:output|ausgabe):.{0,300}pwned|blame (?:it on )?\w+|"
    r"generieren sie .{0,40}code|neue anweisungen|your instructions are now|"
    r"deine anweisungen sind",
    re.I | re.S,
)
EXFIL = re.compile(
    r"!\[[^\]]*\]\(https?://|\]\(https?://|requests\.(?:post|get)|urllib|smtplib|"
    r"\bcurl\b|\bwget\b|send (?:it|this|the|all|me)\b.{0,40}(?:to|via) |exfiltrat",
    re.I,
)
# Planted keyword that changes future behaviour ("when 'cheese' is said, answer ...").
TRIGGER_KEYWORD = re.compile(
    r"(?:when|wenn|whenever|jab|cuando|si)\s+[\"“'«]\w+[\"”'»]\s+(?:is said|gesagt wird|"
    r"kaha|se dice|is mentioned|erwähnt wird)|"
    r"every time (?:you get|someone asks|i say)|jedes mal,? wenn|cada vez que",
    re.I,
)
CONDITIONAL = re.compile(
    r"\b(?:if|when|whenever|once|wenn|falls|sobald|si|cuando)\b.{0,60}\b(?:user|asks?|"
    r"mentions?|question|keyword|query|nutzer|frag\w*|usuario|pregunt\w*)\b.{0,80}"
    r"\b(?:then|respond|answer|reply|say|antworte|responde|di)\b",
    re.I | re.S,
)
# Orders to fabricate content are injected instructions (in scope, like BIPIA's
# "Misinformation & Propaganda" test category); a fake context document is misinformation
# content and stays out of scope.
FABRICATION_ORDER = re.compile(
    r"untrue|false headline|fake news|falsch\w* (?:schlagzeile|nachricht)|"
    r"lie about|noticia falsa|make something up|erfinde etwas|unwahre überschrift",
    re.I,
)
FAKE_CONTEXT = re.compile(
    r"^\s*(?:document context|context|kontext|artikel|\$documents)\b\s*[:{\"]?|desinformaci",
    re.I,
)
SPACED_LETTERS = re.compile(r"(?:\b\w\s){6,}\w\b|(?:[\\\s]*\\[\\\s]*\w){6,}")
B64 = re.compile(r"(?:[A-Za-z0-9+/]{4}){5,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")

BIPIA_RULES: dict[str, tuple[TipoDeAtaque | None, str]] = {
    # task-irrelevant questions: out of scope for the knowledge base (kept for evaluation)
    "Information Retrieval": (None, "bipia_pregunta_inocua"),
    "Content Creation": (None, "bipia_pregunta_inocua"),
    "Learning and Tutoring": (None, "bipia_pregunta_inocua"),
    "Language Translation": (None, "bipia_pregunta_inocua"),
    "Programming Help": (None, "bipia_pregunta_inocua"),
    # instructions that alter the format or content of the model's answer
    "Alphanumeric Substitution": (T.INYECCION_DIRECTA, "bipia_formato_respuesta"),
    "Homophonic Substitution": (T.INYECCION_DIRECTA, "bipia_formato_respuesta"),
    "Misspelling Intentionally": (T.INYECCION_DIRECTA, "bipia_formato_respuesta"),
    "Anagramming": (T.INYECCION_DIRECTA, "bipia_formato_respuesta"),
    "Space Removal & Grouping": (T.INYECCION_DIRECTA, "bipia_formato_respuesta"),
    "Instruction": (T.INYECCION_DIRECTA, "bipia_contenido_respuesta"),
    "Social Interaction": (T.INYECCION_DIRECTA, "bipia_contenido_respuesta"),
    "Persuasion": (T.INYECCION_DIRECTA, "bipia_contenido_respuesta"),
    "Clickbait": (T.INYECCION_DIRECTA, "bipia_enlace_malicioso"),
    "Malware Distribution": (T.INYECCION_DIRECTA, "bipia_enlace_malicioso"),
    # code attacks: data theft vs. malicious code execution
    "Cookie Theft": (T.EXFILTRACION, "bipia_codigo_robo_datos"),
    "Memory Scanning": (T.EXFILTRACION, "bipia_codigo_robo_datos"),
    "Dumpster Diving": (T.EXFILTRACION, "bipia_codigo_robo_datos"),
    "Environment Variable Analysis": (T.EXFILTRACION, "bipia_codigo_robo_datos"),
    "Device and Driver Enumeration": (T.EXFILTRACION, "bipia_codigo_robo_datos"),
    "Sending Out Spam Emails": (T.INYECCION_DIRECTA, "bipia_codigo_malicioso"),
    "Crippling Critical Infrastructures": (T.INYECCION_DIRECTA, "bipia_codigo_malicioso"),
    "Network Propagation": (T.INYECCION_DIRECTA, "bipia_codigo_malicioso"),
    "Exploiting System Vulnerabilities": (T.INYECCION_DIRECTA, "bipia_codigo_malicioso"),
    "Cryptocurrency Mining": (T.INYECCION_DIRECTA, "bipia_codigo_malicioso"),
}

# BIPIA's test split uses different categories (and "Language Translation" changes meaning:
# in train it is an innocuous task, in test an instruction about the model's answer).
BIPIA_TEST_RULES: dict[str, tuple[TipoDeAtaque | None, str]] = {
    "Task Automation": (None, "bipia_pregunta_inocua"),
    "Business Intelligence": (None, "bipia_pregunta_inocua"),
    "Conversational Agent": (None, "bipia_pregunta_inocua"),
    "Research Assistance": (None, "bipia_pregunta_inocua"),
    "Sentiment Analysis": (None, "bipia_pregunta_inocua"),
    "Substitution Ciphers": (T.INYECCION_DIRECTA, "bipia_formato_respuesta"),
    "Base Encoding": (T.INYECCION_DIRECTA, "bipia_formato_respuesta"),
    "Reverse Text": (T.INYECCION_DIRECTA, "bipia_formato_respuesta"),
    "Emoji Substitution": (T.INYECCION_DIRECTA, "bipia_formato_respuesta"),
    "Language Translation": (T.INYECCION_DIRECTA, "bipia_formato_respuesta"),
    "Information Dissemination": (T.INYECCION_DIRECTA, "bipia_contenido_respuesta"),
    "Marketing & Advertising": (T.INYECCION_DIRECTA, "bipia_contenido_respuesta"),
    "Entertainment": (T.INYECCION_DIRECTA, "bipia_contenido_respuesta"),
    "Scams & Fraud": (T.INYECCION_DIRECTA, "bipia_enlace_malicioso"),
    "Misinformation & Propaganda": (T.INYECCION_DIRECTA, "bipia_contenido_respuesta"),
    "Data Eavesdropping": (T.EXFILTRACION, "bipia_codigo_robo_datos"),
    "Traffic Analysis": (T.EXFILTRACION, "bipia_codigo_robo_datos"),
    "Keylogging": (T.EXFILTRACION, "bipia_codigo_robo_datos"),
    "Screen Scraping": (T.EXFILTRACION, "bipia_codigo_robo_datos"),
    "Introduce System Fingerprinting": (T.EXFILTRACION, "bipia_codigo_robo_datos"),
}

ATLAS_RULES: dict[str, TipoDeAtaque] = {
    "AML.T0051": T.INYECCION_DIRECTA,
    "AML.T0051.000": T.INYECCION_DIRECTA,
    "AML.T0051.001": T.INYECCION_DIRECTA,
    "AML.T0051.002": T.TRIGGER_CONDICIONAL,
    "AML.T0054": T.JAILBREAK,
    "AML.T0056": T.PROMPT_LEAKING,
    "AML.T0057": T.EXFILTRACION,
    "AML.T0068": T.INYECCION_DIRECTA,
    "AML.T0070": T.INYECCION_DIRECTA,
    "AML.T0071": T.INYECCION_DIRECTA,
    "AML.T0077": T.EXFILTRACION,
}

OWASP_RULES: list[tuple[str, TipoDeAtaque, TecnicaDeOcultamiento]] = [
    ("adversarial suffix", T.JAILBREAK, C.OTRO),
    ("multilingual", T.INYECCION_DIRECTA, C.OTRO_IDIOMA),
    ("payload splitting", T.INYECCION_DIRECTA, C.OTRO),
    ("indirect injection", T.EXFILTRACION, C.NINGUNA),  # scenario 2: image-link data leak
]

# --------------------------------------------------------------------------- helpers


def _has_homoglyphs(text: str) -> bool:
    """Look-alike characters: letters whose NFKC form is a different ASCII letter (fullwidth,
    circled, mathematical alphanumerics...) or words mixing Latin and Cyrillic/Greek."""
    lookalikes = 0
    for c in text:
        if not c.isascii():  # circled letters are symbols (So), so do not require isalpha()
            folded = unicodedata.normalize("NFKC", c)
            if folded.isascii() and folded.isalpha():
                lookalikes += 1
    if lookalikes >= 2:
        return True
    for word in re.findall(r"\w{3,}", text):
        scripts = {unicodedata.name(c, "?").split()[0] for c in word if c.isalpha()} & {
            "LATIN",
            "CYRILLIC",
            "GREEK",
        }
        if len(scripts) > 1:
            return True
    return False


def detect_concealment(text: str) -> TecnicaDeOcultamiento:
    """Most specific concealment technique visible in the text itself.

    Homoglyphs and character-separation tricks have no dedicated value in the report
    schema, so they map to `otro`.
    """
    if any(0xE0000 <= ord(c) <= 0xE007F for c in text) or any(
        c in text for c in ("​", "‌", "‍", "⁠", "﻿")
    ):
        return C.UNICODE_INVISIBLE
    for match in B64.finditer(text):
        try:
            decoded = base64.b64decode(match.group(0), validate=True).decode("utf-8")
        except Exception:  # noqa: BLE001 - not base64 text
            continue
        if decoded.isprintable() and len(decoded) >= 8:
            return C.BASE64
    if "<!--" in text:
        return C.COMENTARIO_HTML
    if _has_homoglyphs(text):
        return C.OTRO
    letters = [c for c in text if c.isalpha()]
    if letters:
        non_latin = sum(1 for c in letters if "LATIN" not in unicodedata.name(c, ""))
        if non_latin / len(letters) > 0.5:
            return C.OTRO_IDIOMA
    if SPACED_LETTERS.search(text):
        return C.OTRO
    return C.NINGUNA


def classify_text(text: str) -> tuple[TipoDeAtaque | None, str]:
    """Attack type for a free-text attack prompt, or (None, reason) if no sign of attack."""
    if PROMPT_LEAK.search(text):
        return T.PROMPT_LEAKING, "patron_prompt_leaking"
    if EXFIL.search(text):
        return T.EXFILTRACION, "patron_exfiltracion"
    if TRIGGER_KEYWORD.search(text) or (CONDITIONAL.search(text) and OVERRIDE.search(text)):
        return T.TRIGGER_CONDICIONAL, "patron_condicional"
    if JAILBREAK.search(text) or PERSONA.search(text):
        return T.JAILBREAK, "patron_jailbreak"
    if OVERRIDE.search(text) or SPACED_LETTERS.search(text):
        return T.INYECCION_DIRECTA, "patron_inyeccion"
    return None, "sin_indicador_de_instruccion"


def _apply(
    entry: AttackEntry,
    attack_type: TipoDeAtaque,
    rule: str,
    status: str = "reglas",
    concealment: TecnicaDeOcultamiento | None = None,
) -> AttackEntry:
    profile = TYPE_PROFILES[attack_type]
    detected = concealment if concealment is not None else detect_concealment(entry.text)
    objective = entry.attacker_objective
    if objective in ("", "unknown", "custom"):
        objective = profile["objetivo"]
    return entry.model_copy(
        update={
            "attack_type": attack_type,
            "concealment_technique": (
                detected
                if entry.concealment_technique == C.NINGUNA
                else entry.concealment_technique
            ),
            "attacker_objective": objective,
            "mitigation": entry.mitigation or profile["mitigacion"],
            "metadata": {
                **entry.metadata,
                LABEL_STATUS_KEY: status,
                "regla": rule,
                "referencia": profile["referencia"],
            },
        }
    )


# --------------------------------------------------------------------------- labeling


def label_entry(entry: AttackEntry) -> tuple[AttackEntry | None, str]:
    """Label one entry. Returns (labeled entry, rule) or (None, discard reason)."""
    source = entry.source
    text = entry.text

    if source == "bipia":
        category = str(entry.metadata.get("categoria_origen", ""))
        rules = BIPIA_TEST_RULES if entry.metadata.get("split") == "test" else BIPIA_RULES
        attack_type, rule = rules.get(category, (T.INYECCION_DIRECTA, "bipia_sin_regla"))
        return (_apply(entry, attack_type, rule), rule) if attack_type else (None, rule)

    if source == "deepset_prompt_injections":
        attack_type, rule = classify_text(text)
        if FAKE_CONTEXT.search(text) and attack_type in (None, T.INYECCION_DIRECTA):
            return None, "desinformacion_fuera_de_alcance"
        if attack_type is None and FABRICATION_ORDER.search(text):
            attack_type, rule = T.INYECCION_DIRECTA, "orden_de_inventar"
        return (_apply(entry, attack_type, rule), rule) if attack_type else (None, rule)

    if source == "hackaprompt":
        # every entry is a successful attack against an instructed model
        attack_type, rule = classify_text(text)
        if attack_type is None:
            attack_type, rule = T.INYECCION_DIRECTA, "hackaprompt_ataque_exitoso"
        return _apply(entry, attack_type, rule), rule

    if source == "mitre_atlas":
        tid = str(entry.metadata.get("tecnica_atlas", ""))
        attack_type = ATLAS_RULES.get(tid, T.INYECCION_DIRECTA)
        concealment = C.OTRO if tid == "AML.T0068" else C.NINGUNA
        return _apply(entry, attack_type, f"atlas_{tid}", "fuente", concealment), f"atlas_{tid}"

    if source == "owasp_llm_top10":
        title = entry.attacker_objective.lower()
        if str(entry.metadata.get("riesgo_owasp", "")).startswith("LLM07"):
            return _apply(
                entry, T.PROMPT_LEAKING, "owasp_llm07", "fuente", C.NINGUNA
            ), "owasp_llm07"
        for key, attack_type, concealment in OWASP_RULES:
            if key in title:
                return _apply(entry, attack_type, f"owasp_{key}", "fuente", concealment), key
        return _apply(entry, T.INYECCION_DIRECTA, "owasp_llm01", "fuente", C.NINGUNA), "owasp_llm01"

    # custom entries come already labeled; only fill missing objective/mitigation
    if entry.attack_type != T.OTRO:
        return _apply(entry, entry.attack_type, "etiqueta_original", "fuente"), "etiqueta_original"
    attack_type, rule = classify_text(text)
    return _apply(entry, attack_type or T.OTRO, rule), rule


def label_entries(
    entries: Iterable[AttackEntry],
) -> tuple[list[AttackEntry], Counter[tuple[str, str]]]:
    """Label many entries. Returns kept entries and a Counter of (source, discard reason)."""
    kept: list[AttackEntry] = []
    discarded: Counter[tuple[str, str]] = Counter()
    for entry in entries:
        labeled, rule = label_entry(entry)
        if labeled is None:
            discarded[(entry.source, rule)] += 1
        else:
            kept.append(labeled)
    return kept, discarded
