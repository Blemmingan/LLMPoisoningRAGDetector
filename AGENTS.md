# Engineering Conventions & Agent Guidelines

This document outlines mandatory architecture rules, operational constraints, and engineering guidelines for any AI agent or human contributor working on the **Corpus Poisoning Forensic Analyst** codebase.

---

## 1. Core Architectural Invariants

### Rule 1: Always Depend on Abstract Interfaces
- Never import a specific LLM client, embedding provider, or storage driver directly into `analysis/`, `evaluation/`, or `cli.py`.
- Pipeline logic must communicate strictly through:
  - `LLMClient` (`src/corpus_forensics/llm/base.py`)
  - `Embedder` (`src/corpus_forensics/embeddings/base.py`)
  - `VectorStore` (`src/corpus_forensics/vectorstore/base.py`)
  - `DocumentLoader` (`src/corpus_forensics/ingestion/base.py`)
- Real providers (OpenAI, Anthropic, HuggingFace, etc.) must be instantiated exclusively through factories (`get_llm_client`, `get_embedder`) configured via runtime settings.

### Rule 2: Non-Parametric Memory & Document Transience
- **NEVER persist analyzed documents into the vector database or disk storage.**
- Ingested documents are transient in-memory models (`Document`) processed through the pipeline and discarded once the `DocumentReport` is emitted.
- The vector store (`VectorStore`) stores **ONLY** known attack signatures (`AttackEntry`) from curated knowledge bases.

### Rule 3: Out-of-Scope Rule (No Factual Verification)
- This tool is a forensic parser detecting **form, intent, and obfuscation** of instructions.
- Do NOT add factual verification, truth-checking, or search-based ground-truth fact checking to this system. Factual misinformation is strictly out of scope.

### Rule 4: Mock/Real Flagging Policy
- Any mock component must be tagged with `# MOCK` in its docstrings and code comments.
- Every generated `DocumentReport` MUST accurately propagate whether the LLM and Embedder used were mocks:
  - `metadata_de_analisis.llm_is_mock`
  - `metadata_de_analisis.embedder_is_mock`
- Under no circumstances should mock data be labeled as real.

### Rule 5: Preservation of Spanish JSON Keys
- The report schemas in `src/corpus_forensics/schemas.py` are part of a shared inter-service specification:
  - `DocumentReport`: `documento`, `malicioso`, `hallazgos`, `metadata de análisis`.
  - `Hallazgo`: `tipo_de_ataque`, `fragmento_exacto`, `inicio_caracter`, `fin_caracter`, `canal`, `indice_chunk`, `tecnica_de_ocultamiento`, `objetivo_del_atacante`, `severidad`, `confianza`, `ataque_similar_id`, `posible_variante_nueva`, `mitigacion`, `verificado`, `notas_verificacion`.
- **Do NOT alter or translate these Spanish keys into English.**
- All internal code symbols, functions, variable names, and documentation must remain in English, but the serialized report keys must strictly adhere to the Spanish specification. If a field change is approved, update `schemas.py`, `json_writer.py`, `markdown_renderer.py`, `evaluation/metrics.py`, and test fixtures together.

---

## 2. Engineering & Quality Standards

### Quality Checks
Before completing any task or opening a pull request, all agents must verify:

```bash
# 1. Run all unit and integration tests
uv run pytest

# 2. Enforce code formatting and lint rules
uv run ruff check

# 3. Verify static type soundness
uv run mypy src
```

### Stubs and TODO Policy
- Use `# TODO(team):` tags with descriptive guidance when adding extension points.
- Never raise `NotImplementedError` in code paths executed by the default pipeline or smoke tests. Where a full implementation is pending, return a safe, documented default or mock value.

### File Separation
- Keep prompt templates in dedicated external files (`src/corpus_forensics/analysis/prompts/`) rather than hardcoding long prompt strings inside Python modules.
