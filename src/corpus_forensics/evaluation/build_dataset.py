"""Builds the evaluation dataset from pinned public texts and held-out attacks.

Composition (defaults): 60 clean documents, 100 poisoned documents and 30 hard negatives,
in TXT, Markdown and HTML, in English and Spanish.

- Clean texts: Wikipedia articles at a fixed revision and READMEs at a fixed commit, listed
  in `data/eval/sources.yaml`.
- Poisoned documents: a clean text with one attack inserted verbatim as its own paragraph at
  the start, middle or end. Attacks come only from splits that are NOT in the knowledge
  base: deepset `test`, BIPIA `test` (text attacks; its categories differ from train) and
  HackAPrompt successful attacks outside the knowledge-base sample. Nothing is rewritten,
  paraphrased or obfuscated here: obfuscated cases are the ones that already come that way
  in the published data.
- Hard negatives: passages of Wikipedia articles that discuss prompt injection and
  adversarial attacks without being one.

Ground-truth attack type and concealment come from the rule-based labeling
(`knowledge_base/labeling.py`); `fragmento_exacto` is the inserted attack text.
The build is deterministic for a given seed and set of pinned sources.
"""

import html
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from bs4 import BeautifulSoup

from corpus_forensics.config import Settings
from corpus_forensics.knowledge_base.labeling import label_entry
from corpus_forensics.knowledge_base.loaders.base import normalize_text
from corpus_forensics.knowledge_base.loaders.bipia import BipiaLoader
from corpus_forensics.knowledge_base.loaders.deepset_prompt_injections import (
    DeepsetPromptInjectionsLoader,
)
from corpus_forensics.knowledge_base.loaders.hackaprompt import HackAPromptLoader
from corpus_forensics.knowledge_base.schemas import AttackEntry

# Wikimedia's robot policy asks for contact information in the User-Agent.
USER_AGENT = (
    "corpus-forensics/0.1 (https://github.com/Blemmingan/LLMPoisoningRAGDetector; "
    "university project)"
)
HARD_NEGATIVE_KEYWORDS = re.compile(
    r"prompt|injection|inyecci|jailbreak|adversari|antag[oó]nic|attack|ataque|malicious|"
    r"malicios|exfiltrat|poison|envenen",
    re.I,
)
POSITIONS = ("inicio", "medio", "fin")
ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
REQUEST_DELAY_S = 0.5  # be polite with Wikimedia and GitHub


@dataclass(frozen=True)
class SourceText:
    """A clean or hard-negative source split into paragraphs."""

    key: str
    title: str
    lang: str
    role: str
    kind: str  # "wikipedia" | "readme"
    paragraphs: list[tuple[str, str]]  # (section, paragraph)
    readme_ext: str = "md"


@dataclass(frozen=True)
class AttackSample:
    """A held-out attack with its rule-based labels."""

    entry: AttackEntry
    origin: str


# --------------------------------------------------------------------------- fetching


def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True)


def parse_wikipedia_html(page_html: str) -> list[tuple[str, str]]:
    """Paragraphs of a Wikipedia article with the section they belong to."""
    soup = BeautifulSoup(page_html, "html.parser")
    root = soup.select_one(".mw-parser-output") or soup
    for tag in root.select(
        "table, .infobox, .navbox, .reference, .reflist, .mw-references-wrap, "
        "sup, style, .hatnote, .thumb, figure, .mw-editsection"
    ):
        tag.decompose()
    section = "Introduction"
    out: list[tuple[str, str]] = []
    for el in root.find_all(["h2", "h3", "p"]):
        if el.name in ("h2", "h3"):
            section = el.get_text(" ", strip=True)
            continue
        # zero-width spaces sit where the reference markers were; leaving them would make the
        # invisible-character detector flag clean documents for an extraction artifact
        text = ZERO_WIDTH.sub("", el.get_text(" ", strip=True))
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\s+([.,;:!?)])", r"\1", text)
        if len(text) >= 80:
            out.append((section, text))
    return out


def fetch_wikipedia(
    lang: str, revid: int, cache_dir: Path, client: httpx.Client
) -> list[tuple[str, str]]:
    """Paragraphs of a pinned Wikipedia revision.

    The raw article HTML is cached on disk, so parser changes do not need a new download.
    """
    cache = cache_dir / f"wiki-{lang}-{revid}.html"
    if not cache.exists():
        time.sleep(REQUEST_DELAY_S)
        response = client.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "oldid": revid,
                "prop": "text",
                "format": "json",
                "formatversion": 2,
            },
        )
        response.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(response.json()["parse"]["text"], encoding="utf-8")
    return parse_wikipedia_html(cache.read_text(encoding="utf-8"))


def readme_paragraphs(text: str) -> list[tuple[str, str]]:
    """Prose paragraphs of a README: drops badges, images, HTML and code blocks."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    section = "README"
    out: list[tuple[str, str]] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        heading = re.match(r"^#{1,6}\s+(.*)", block)
        if heading:
            section = heading.group(1).strip()
            continue
        if not block or block.startswith(("[![", "![", "<", ".. ", "|", "=", "-")):
            continue
        if len(block) >= 80:
            out.append((section, " ".join(ZERO_WIDTH.sub("", block).split())))
    return out


def fetch_readme(repo: str, sha: str, path: str, cache_dir: Path, client: httpx.Client) -> str:
    """Raw README at a pinned commit (cached on disk)."""
    cache = cache_dir / f"readme-{repo.replace('/', '__')}-{sha[:12]}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    time.sleep(REQUEST_DELAY_S)
    response = client.get(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}")
    response.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(response.text, encoding="utf-8")
    return response.text


def load_source_texts(sources_path: Path, cache_dir: Path) -> list[SourceText]:
    """Fetch (or read from cache) every pinned source."""
    spec = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    texts: list[SourceText] = []
    with _client() as client:
        for page in spec.get("wikipedia", []):
            paragraphs = fetch_wikipedia(page["lang"], int(page["revid"]), cache_dir, client)
            texts.append(
                SourceText(
                    key=f"wikipedia:{page['lang']}:{page['title']}@{page['revid']}",
                    title=page["title"],
                    lang=page["lang"],
                    role=page["role"],
                    kind="wikipedia",
                    paragraphs=paragraphs,
                )
            )
        for readme in spec.get("readmes", []):
            raw = fetch_readme(readme["repo"], readme["sha"], readme["path"], cache_dir, client)
            texts.append(
                SourceText(
                    key=f"readme:{readme['repo']}@{readme['sha'][:12]}",
                    title=readme["repo"].split("/")[-1],
                    lang="en",
                    role="limpio",
                    kind="readme",
                    paragraphs=readme_paragraphs(raw),
                    readme_ext="md" if readme["path"].lower().endswith(".md") else "txt",
                )
            )
    return texts


# --------------------------------------------------------------------------- attacks


def held_out_attacks(settings: Settings) -> list[AttackSample]:
    """Every labeled attack that is not in the knowledge base.

    - deepset `test` split.
    - BIPIA `test` split: text attacks, plus the data-theft code attacks (their rules map
      them to exfiltration; destructive code categories have no rule and stay out).
    - HackAPrompt successful attacks of up to 600 characters outside the knowledge-base
      sample (compared after normalization).
    """
    raw = settings.kb_raw_dir
    samples: list[AttackSample] = []

    for entry in DeepsetPromptInjectionsLoader(splits=("test",)).load(
        raw / "deepset_prompt_injections"
    ):
        labeled, _ = label_entry(entry)
        if labeled is not None:
            samples.append(AttackSample(labeled, "deepset:test"))

    for entry in BipiaLoader(splits=("test",)).load(raw / "bipia"):
        labeled, rule = label_entry(entry)
        if labeled is None or rule == "bipia_sin_regla":
            continue
        modality = entry.metadata.get("modalidad")
        samples.append(
            AttackSample(labeled, f"bipia:test:{modality}:{entry.metadata.get('categoria_origen')}")
        )

    hack_dir = raw / "hackaprompt"
    in_kb = {normalize_text(e.text) for e in HackAPromptLoader().load(hack_dir)}
    for e in HackAPromptLoader(max_entries=100_000, max_chars=600, seed=1).load(hack_dir):
        if normalize_text(e.text) in in_kb:
            continue
        labeled, _ = label_entry(e)
        if labeled is not None:
            samples.append(AttackSample(labeled, "hackaprompt:held_out"))
    samples.sort(key=lambda s: s.entry.id)
    return samples


# Poisoned documents per attack type. The held-out data has almost no conditional
# triggers and no text exfiltration, so those types are capped by availability; direct
# injection takes the remainder.
PREFIX_DEDUP_CHARS = 40  # HackAPrompt has many resubmissions differing only at the end
TYPE_QUOTAS: dict[str, int] = {
    "exfiltracion": 15,
    "jailbreak": 20,
    "prompt_leaking": 12,
    "trigger_condicional": 10,
}


def select_attacks(samples: list[AttackSample], n: int, rng: random.Random) -> list[AttackSample]:
    """Pick `n` attacks stratified by type, round-robin over (source, concealment) buckets.

    The round-robin keeps rare buckets (e.g. obfuscated HackAPrompt attacks, deepset
    German/Spanish items) represented instead of letting the largest source dominate.
    """
    used_prefixes: set[str] = set()
    by_type: dict[str, list[AttackSample]] = {}
    for s in samples:
        by_type.setdefault(s.entry.attack_type.value, []).append(s)

    def round_robin(pool: list[AttackSample], k: int) -> list[AttackSample]:
        buckets: dict[tuple[str, str], list[AttackSample]] = {}
        for s in pool:
            key = (s.origin.split(":")[0], s.entry.concealment_technique.value)
            buckets.setdefault(key, []).append(s)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        order = sorted(buckets)
        chosen: list[AttackSample] = []
        while len(chosen) < k and any(buckets.values()):
            for key in order:
                while buckets[key] and len(chosen) < k:
                    candidate = buckets[key].pop()
                    prefix = normalize_text(candidate.entry.text)[:PREFIX_DEDUP_CHARS]
                    if prefix in used_prefixes:
                        continue  # near-duplicate of an attack already chosen
                    used_prefixes.add(prefix)
                    chosen.append(candidate)
                    break
        return chosen

    selected: list[AttackSample] = []
    for attack_type, quota in TYPE_QUOTAS.items():
        selected += round_robin(by_type.get(attack_type, []), quota)
    rest = [s for t, pool in by_type.items() if t not in TYPE_QUOTAS for s in pool]
    selected += round_robin(rest, n - len(selected))
    rng.shuffle(selected)
    return selected


# --------------------------------------------------------------------------- documents


def make_windows(
    paragraphs: list[tuple[str, str]],
    rng: random.Random,
    n: int,
    min_chars: int = 1500,
    max_chars: int = 3500,
) -> list[list[str]]:
    """Up to `n` non-overlapping runs of consecutive paragraphs of 1.5k-3.5k characters."""
    texts = [p for _, p in paragraphs]
    windows: list[list[str]] = []
    starts = list(range(len(texts)))
    rng.shuffle(starts)
    used: set[int] = set()
    for start in starts:
        if len(windows) >= n:
            break
        window: list[str] = []
        size = 0
        i = start
        while i < len(texts) and i not in used and size < min_chars:
            if size + len(texts[i]) > max_chars and window:
                break
            window.append(texts[i])
            size += len(texts[i])
            i += 1
        if size >= min_chars * 0.6:
            used.update(range(start, i))
            windows.append(window)
    return windows


def hard_negative_excerpts(source: SourceText, min_chars: int = 1200) -> list[list[str]]:
    """Groups of consecutive paragraphs that talk about attacks."""
    excerpts: list[list[str]] = []
    current: list[str] = []
    for _, text in source.paragraphs:
        if HARD_NEGATIVE_KEYWORDS.search(text):
            current.append(text)
            if sum(len(t) for t in current) >= min_chars:
                excerpts.append(current)
                current = []
        elif current and sum(len(t) for t in current) >= min_chars * 0.6:
            excerpts.append(current)
            current = []
        else:
            current = []
    if current and sum(len(t) for t in current) >= min_chars * 0.6:
        excerpts.append(current)
    return excerpts


def insert_attack(paragraphs: list[str], attack: str, position: str) -> list[str]:
    """Insert the attack verbatim as its own paragraph."""
    index = {"inicio": 0, "medio": len(paragraphs) // 2, "fin": len(paragraphs)}[position]
    return [*paragraphs[:index], attack, *paragraphs[index:]]


def render(title: str, paragraphs: list[str], fmt: str) -> str:
    """Serialize a document as TXT, Markdown or HTML (every paragraph visible)."""
    if fmt == "html":
        body = "\n".join(f"<p>{html.escape(p)}</p>" for p in paragraphs)
        return (
            f'<!DOCTYPE html>\n<html><head><meta charset="utf-8"><title>{html.escape(title)}'
            f"</title></head>\n<body>\n<h1>{html.escape(title)}</h1>\n{body}\n</body></html>\n"
        )
    if fmt == "md":
        return f"# {title}\n\n" + "\n\n".join(paragraphs) + "\n"
    return f"{title}\n\n" + "\n\n".join(paragraphs) + "\n"


def remove_previous_build(out_dir: Path) -> None:
    """Delete the files of an earlier build (only those its labels.jsonl lists)."""
    labels_path = out_dir / "labels.jsonl"
    if not labels_path.exists():
        return
    for label in read_labels(out_dir):
        (out_dir / label["documento"]).unlink(missing_ok=True)
    labels_path.unlink()
    (out_dir / "dataset_card.json").unlink(missing_ok=True)


def build_dataset(
    settings: Settings,
    out_dir: Path,
    sources_path: Path,
    cache_dir: Path,
    seed: int = 42,
    n_clean: int = 60,
    n_poisoned: int = 100,
    n_hard_negative: int = 30,
) -> dict[str, Any]:
    """Build the dataset in `out_dir` (documents + labels.jsonl + dataset_card.json)."""
    rng = random.Random(seed)
    texts = load_source_texts(sources_path, cache_dir)
    clean_sources = [t for t in texts if t.role == "limpio"]
    hn_sources = [t for t in texts if t.role == "hard_negative"]

    bases: list[tuple[SourceText, list[str], str]] = []
    for source in clean_sources:
        n_windows = 2 if source.kind == "wikipedia" else 1
        windows = make_windows(source.paragraphs, rng, n_windows)
        if not windows and source.kind == "readme" and source.paragraphs:
            windows = [[p for _, p in source.paragraphs]]  # short READMEs are used whole
        for window in windows:
            fmt = source.readme_ext if source.kind == "readme" else rng.choice(["txt", "html"])
            bases.append((source, window, fmt))
    rng.shuffle(bases)

    available = held_out_attacks(settings)
    attacks = select_attacks(available, n_poisoned, rng)
    # code snippets go into README-like documents; prose attacks into the rest, starting
    # with the bases not used by clean documents
    readme_bases = [b for b in bases if b[0].kind == "readme"]
    prose_bases = bases[n_clean:] + bases[:n_clean]

    remove_previous_build(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels: list[dict[str, Any]] = []

    def write(
        doc_id: str, title: str, paragraphs: list[str], fmt: str, label: dict[str, Any]
    ) -> None:
        name = f"{doc_id}.{fmt}"
        (out_dir / name).write_text(render(title, paragraphs, fmt), encoding="utf-8")
        labels.append({"documento": name, **label})

    for i, (source, window, fmt) in enumerate(bases[:n_clean]):
        write(
            f"limpio_{i:03d}",
            source.title,
            window,
            fmt,
            {
                "categoria": "limpio",
                "malicioso": False,
                "fuente_texto": source.key,
                "idioma_texto": source.lang,
                "formato": fmt,
            },
        )

    for i, attack in enumerate(attacks):
        is_code = ":code:" in attack.origin
        pool = readme_bases if is_code and readme_bases else prose_bases
        source, window, fmt = pool[i % len(pool)]
        position = POSITIONS[i % len(POSITIONS)]
        entry = attack.entry
        concealment = entry.concealment_technique.value
        write(
            f"envenenado_{i:03d}",
            source.title,
            insert_attack(window, entry.text, position),
            fmt,
            {
                "categoria": "envenenado",
                "malicioso": True,
                "tipo_de_ataque": entry.attack_type.value,
                "tecnica_de_ocultamiento": [concealment],
                "fragmento_exacto": entry.text,
                "posicion": position,
                "sutileza": "obvio" if concealment == "ninguna" else "disfrazado",
                "fuente_texto": source.key,
                "idioma_texto": source.lang,
                "formato": fmt,
                "fuente_ataque": attack.origin,
                "id_ataque": entry.id,
                "regla_etiqueta": entry.metadata.get("regla"),
            },
        )

    excerpts = [(s, ex) for s in hn_sources for ex in hard_negative_excerpts(s)]
    rng.shuffle(excerpts)
    for i, (source, excerpt) in enumerate(excerpts[:n_hard_negative]):
        fmt = rng.choice(["txt", "html"])
        write(
            f"hard_negative_{i:03d}",
            source.title,
            excerpt,
            fmt,
            {
                "categoria": "hard_negative",
                "malicioso": False,
                "fuente_texto": source.key,
                "idioma_texto": source.lang,
                "formato": fmt,
            },
        )

    with open(out_dir / "labels.jsonl", "w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")
    card = describe_labels(labels)
    card.update(
        {
            "seed": seed,
            "sources": str(sources_path),
            "available_attacks": len(available),
            "type_quotas": TYPE_QUOTAS,
        }
    )
    (out_dir / "dataset_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return card


def describe_labels(labels: list[dict[str, Any]]) -> dict[str, Any]:
    """Composition counts of a labels list."""
    poisoned = [lab for lab in labels if lab["categoria"] == "envenenado"]
    return {
        "total": len(labels),
        "categoria": dict(Counter(lab["categoria"] for lab in labels)),
        "formato": dict(Counter(lab.get("formato") for lab in labels)),
        "idioma_texto": dict(Counter(lab.get("idioma_texto") for lab in labels)),
        "tipo_de_ataque": dict(Counter(lab.get("tipo_de_ataque") for lab in poisoned)),
        "tecnica_de_ocultamiento": dict(
            Counter(t for lab in poisoned for t in lab.get("tecnica_de_ocultamiento", []))
        ),
        "posicion": dict(Counter(lab.get("posicion") for lab in poisoned)),
        "sutileza": dict(Counter(lab.get("sutileza") for lab in poisoned)),
        "fuente_ataque": dict(
            Counter(str(lab.get("fuente_ataque", "")).split(":")[0] for lab in poisoned)
        ),
    }


def read_labels(dataset_dir: Path) -> list[dict[str, Any]]:
    """Read labels.jsonl as plain dicts."""
    with open(dataset_dir / "labels.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
