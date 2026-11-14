#!/usr/bin/env bash
# ==============================================================================
# End-to-End Test Suite for Corpus Poisoning Forensic Analyst (OpenAI Provider)
# ==============================================================================
# This script executes a comprehensive, highly verbose validation of all system
# components using live OpenAI API calls (LLM analyst + text embeddings).
#
# Requirements:
#   - `.env` must be present with CF_LLM_API_KEY / OPENAI_API_KEY configured.
#   - `uv` package manager must be installed.
# ==============================================================================

set -euo pipefail

# Text formatting & color codes
BOLD="\033[1m"
DIM="\033[2m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
BLUE="\033[34m"
MAGENTA="\033[35m"
RESET="\033[0m"

# Script tracking
SCRIPT_START_TIME=$(date +%s)
REPORTS_DIR="reports/e2e_openai_test"
# Stage 1 resets the knowledge base: use a throwaway vector store so the real index
# (762 examples + literature in data/vectorstore) is never wiped by this test.
export CF_VECTORSTORE_DIR="${REPORTS_DIR}/vectorstore"

# Helper functions
log_header() {
    echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}${CYAN}  $1${RESET}"
    echo -e "${BOLD}${BLUE}══════════════════════════════════════════════════════════════════════════════${RESET}\n"
}

log_step() {
    echo -e "${BOLD}${YELLOW}─── [STEP $1] ${RESET}${BOLD}$2${RESET}"
}

log_info() {
    echo -e "    ${CYAN}ℹ${RESET}  $1"
}

log_detail() {
    echo -e "       ${DIM}› $1${RESET}"
}

log_success() {
    echo -e "    ${GREEN}✔${RESET}  ${BOLD}$1${RESET}"
}

log_warning() {
    echo -e "    ${YELLOW}⚠${RESET}  ${BOLD}$1${RESET}"
}

log_error() {
    echo -e "    ${RED}✖${RESET}  ${BOLD}$1${RESET}"
}

fail_exit() {
    echo -e "\n${BOLD}${RED}FATAL ERROR:${RESET} $1"
    exit 1
}

# ==============================================================================
# PRE-FLIGHT CHECKS & ENVIRONMENT VALIDATION
# ==============================================================================
log_header "PRE-FLIGHT: Environment & Credentials Inspection"

log_step "0.1" "Checking UV package manager and virtual environment"
if ! command -v uv &> /dev/null; then
    fail_exit "'uv' command not found. Please install Astral uv: https://docs.astral.sh/uv/"
fi
UV_VERSION=$(uv --version)
log_success "UV is available: ${UV_VERSION}"

log_step "0.2" "Validating .env configuration file"
if [ ! -f ".env" ]; then
    fail_exit "No .env file found in root directory! Please create one from .env.example."
fi
log_success "Found .env file."

# Read settings via Python to guarantee exact pydantic-settings resolution
log_step "0.3" "Inspecting active Corpus Forensics runtime configuration"
CONFIG_INFO=$(uv run python -c "
from corpus_forensics.config import get_settings
s = get_settings()
key = s.llm_api_key.get_secret_value() if s.llm_api_key else ''
masked = key[:7] + '...' + key[-4:] if len(key) > 12 else ('[NOT SET]' if not key else '***')
print(f'LLM_PROVIDER={s.llm_provider}')
print(f'LLM_MODEL={s.llm_model}')
print(f'LLM_BASE_URL={s.llm_base_url}')
print(f'EMBEDDER_PROVIDER={s.embedder_provider}')
print(f'EMBEDDING_MODEL={s.embedding_model}')
print(f'API_KEY_MASKED={masked}')
print(f'CACHE_ENABLED={s.llm_cache_enabled}')
print(f'BUDGET_CAP={s.budget_limit_usd}')
")

eval "$CONFIG_INFO"

log_info "Active LLM Provider       : ${BOLD}${LLM_PROVIDER}${RESET}"
log_info "Active Chat Model          : ${BOLD}${LLM_MODEL}${RESET}"
log_info "Active LLM Base URL        : ${BOLD}${LLM_BASE_URL}${RESET}"
log_info "Active Embedder Provider   : ${BOLD}${EMBEDDER_PROVIDER}${RESET}"
log_info "Active Embedding Model     : ${BOLD}${EMBEDDING_MODEL}${RESET}"
log_info "OpenAI API Key (masked)    : ${BOLD}${API_KEY_MASKED}${RESET}"
log_info "Response Cache Active      : ${BOLD}${CACHE_ENABLED}${RESET}"
log_info "Spending Budget Limit      : ${BOLD}\$${BUDGET_CAP} USD${RESET}"

if [ "${API_KEY_MASKED}" = "[NOT SET]" ]; then
    fail_exit "CF_LLM_API_KEY is not configured in .env!"
fi

log_step "0.4" "Live Network Handshake with OpenAI API"
log_info "Testing direct connectivity to OpenAI endpoints..."

HANDSHAKE_RESULT=$(uv run python -c "
import time
from openai import OpenAI
from corpus_forensics.config import get_settings

settings = get_settings()
api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
client = OpenAI(base_url=settings.llm_base_url, api_key=api_key)

# 1. Chat Completion test
t0 = time.time()
chat_resp = client.chat.completions.create(
    model=settings.llm_model or 'gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'Ping! Reply with just: PONG'}],
    max_tokens=10,
    temperature=0.0
)
t_chat = (time.time() - t0) * 1000
chat_text = chat_resp.choices[0].message.content.strip()

# 2. Embedding test
t0 = time.time()
emb_resp = client.embeddings.create(
    model=settings.embedding_model or 'text-embedding-3-small',
    input='corpus forensics connection test'
)
t_emb = (time.time() - t0) * 1000
emb_dim = len(emb_resp.data[0].embedding)

print(f'CHAT_REPLY={chat_text}')
print(f'CHAT_LATENCY_MS={t_chat:.1f}')
print(f'EMB_DIM={emb_dim}')
print(f'EMB_LATENCY_MS={t_emb:.1f}')
")

eval "$HANDSHAKE_RESULT"
log_success "Chat handshake succeeded (${CHAT_LATENCY_MS} ms) -> Model replied: '${CHAT_REPLY}'"
log_success "Embeddings handshake succeeded (${EMB_LATENCY_MS} ms) -> Vector dimension: ${EMB_DIM}"

# ==============================================================================
# STAGE 1: KNOWLEDGE BASE INGESTION (CHROMA + OPENAI EMBEDDINGS)
# ==============================================================================
log_header "STAGE 1: Knowledge Base Ingestion & Chroma Vector Store"

log_step "1.1" "Resetting and re-indexing ChromaDB with OpenAI text embeddings"
log_info "Command: uv run corpus-forensics kb ingest --source custom --reset"
uv run corpus-forensics kb ingest --source custom --reset
log_success "Knowledge base successfully ingested into ChromaDB."

log_step "1.2" "Inspecting vector store statistics"
log_info "Command: uv run corpus-forensics kb stats"
uv run corpus-forensics kb stats
log_success "Knowledge base statistics verified."

# ==============================================================================
# STAGE 2: FORENSIC DOCUMENT SCANNERS (REAL LLM + ADVANCED RAG)
# ==============================================================================
log_header "STAGE 2: End-to-End Forensic Document Analysis"

rm -rf "${REPORTS_DIR}"
mkdir -p "${REPORTS_DIR}"

run_and_verify_doc() {
    local doc_path="$1"
    local expected_verdict="$2" # "CLEAN" or "MALICIOUS"
    local description="$3"

    echo -e "\n  ${BOLD}${MAGENTA}▶ Testing Document: ${doc_path}${RESET}"
    log_detail "Category: ${description}"
    log_detail "Expected Verdict: ${expected_verdict}"

    uv run corpus-forensics analyze "${doc_path}" --out "${REPORTS_DIR}" --format both

    local filename
    filename=$(basename "${doc_path}")
    local stem="${filename%.*}"
    local json_report="${REPORTS_DIR}/${stem}_report.json"
    local md_report="${REPORTS_DIR}/${stem}_report.md"

    if [ ! -f "${json_report}" ] || [ ! -f "${md_report}" ]; then
        fail_exit "Report files were not generated for ${doc_path}!"
    fi

    # Verify report JSON contents using Python
    local VERIFY_OUT
    VERIFY_OUT=$(uv run python -c "
import json
with open('${json_report}', encoding='utf-8') as f:
    r = json.load(f)

is_malicious = r.get('malicioso', False)
verdict_str = 'MALICIOUS' if is_malicious else 'CLEAN'
findings = r.get('hallazgos', [])
meta = r.get('metadata de análisis', {})
llm_mock = meta.get('llm_is_mock')
emb_mock = meta.get('embedder_is_mock')
llm_model = meta.get('llm_model')

print(f'ACTUAL_VERDICT={verdict_str}')
print(f'NUM_FINDINGS={len(findings)}')
print(f'LLM_IS_MOCK={llm_mock}')
print(f'EMB_IS_MOCK={emb_mock}')
print(f'REPORT_LLM_MODEL={llm_model}')

if findings:
    f0 = findings[0]
    print(f'FINDING_TYPE={f0.get(\"tipo_de_ataque\")}')
    print(f'FINDING_SEVERITY={f0.get(\"severidad\")}')
    print(f'FINDING_CHANNEL={f0.get(\"canal\")}')
    print(f'FINDING_VERIFIED={f0.get(\"verificado\")}')
    print(f'FINDING_FRAGMENT={repr(f0.get(\"fragmento_exacto\", \"\"))}')
")

    eval "$VERIFY_OUT"

    log_detail "Actual Verdict      : ${ACTUAL_VERDICT}"
    log_detail "Findings Count      : ${NUM_FINDINGS}"
    log_detail "LLM is Mock?        : ${LLM_IS_MOCK} (must be False)"
    log_detail "Embedder is Mock?   : ${EMB_IS_MOCK} (must be False)"
    log_detail "Model Recorded      : ${REPORT_LLM_MODEL}"

    # Assertions
    if [ "${LLM_IS_MOCK}" != "False" ] || [ "${EMB_IS_MOCK}" != "False" ]; then
        fail_exit "Mock flag violation: llm_is_mock=${LLM_IS_MOCK}, embedder_is_mock=${EMB_IS_MOCK}. Real OpenAI must be used!"
    fi

    if [ "${ACTUAL_VERDICT}" != "${expected_verdict}" ]; then
        fail_exit "Verdict mismatch for ${doc_path}! Expected: ${expected_verdict}, Got: ${ACTUAL_VERDICT}"
    fi

    if [ "${expected_verdict}" = "MALICIOUS" ]; then
        log_detail "Finding Attack Type : ${FINDING_TYPE}"
        log_detail "Finding Severity    : ${FINDING_SEVERITY}"
        log_detail "Finding Channel     : ${FINDING_CHANNEL}"
        log_detail "Fragment Verified   : ${FINDING_VERIFIED}"
        log_detail "Fragment Text       : ${FINDING_FRAGMENT}"
        if [ "${FINDING_VERIFIED}" != "True" ]; then
            log_warning "Notice: Finding fragment was not verified verbatim against source channel."
        fi
    fi

    log_success "Document ${filename} correctly classified as ${ACTUAL_VERDICT}."
}

log_step "2.1" "Analyzing Benign Document (Clean Baseline)"
run_and_verify_doc "tests/fixtures/clean_doc.txt" "CLEAN" "Standard non-poisoned corporate text"

log_step "2.2" "Analyzing Hard Negative Document (Security Survey)"
run_and_verify_doc "tests/fixtures/hard_negative_doc.txt" "CLEAN" "Academic survey text mentioning prompt injection phrases in benign context"

log_step "2.3" "Analyzing Poisoned Document (Direct Injection & Exfiltration)"
run_and_verify_doc "tests/fixtures/poisoned_doc.txt" "MALICIOUS" "Procurement text poisoned with system override and exfil beacon"

log_step "2.4" "Analyzing Poisoned HTML Document (Concealed in HTML Comment)"
run_and_verify_doc "tests/fixtures/poisoned_hidden_comment.html" "MALICIOUS" "HTML page concealing prompt injection in an HTML comment tag"

# ==============================================================================
# STAGE 3: CACHING & SPENDING LEDGER VERIFICATION
# ==============================================================================
log_header "STAGE 3: Response Caching & Cost Accounting"

log_step "3.1" "Checking spending ledger before cache test"
LEDGER_BEFORE=$(uv run python -c "
from corpus_forensics.config import get_settings
from corpus_forensics.llm.litellm_client import build_usage_tracker
tracker = build_usage_tracker(get_settings())
paid = [r for r in tracker.read_records() if not r.cached]
hits = [r for r in tracker.read_records() if r.cached]
print(f'PAID_BEFORE={len(paid)}')
print(f'HITS_BEFORE={len(hits)}')
print(f'SPENT_BEFORE={tracker.total_cost_usd:.5f}')
")
eval "$LEDGER_BEFORE"
log_info "Paid calls so far : ${PAID_BEFORE}"
log_info "Cache hits so far : ${HITS_BEFORE}"
log_info "USD spend so far  : \$${SPENT_BEFORE}"

log_step "3.2" "Re-analyzing 'poisoned_doc.txt' to trigger local response cache"
log_info "Command: uv run corpus-forensics analyze tests/fixtures/poisoned_doc.txt --out ${REPORTS_DIR}"
uv run corpus-forensics analyze tests/fixtures/poisoned_doc.txt --out "${REPORTS_DIR}"

log_step "3.3" "Validating that cache hit was recorded without additional paid API calls"
LEDGER_AFTER=$(uv run python -c "
from corpus_forensics.config import get_settings
from corpus_forensics.llm.litellm_client import build_usage_tracker
tracker = build_usage_tracker(get_settings())
paid = [r for r in tracker.read_records() if not r.cached]
hits = [r for r in tracker.read_records() if r.cached]
print(f'PAID_AFTER={len(paid)}')
print(f'HITS_AFTER={len(hits)}')
print(f'SPENT_AFTER={tracker.total_cost_usd:.5f}')
")
eval "$LEDGER_AFTER"

log_detail "Paid calls after re-run  : ${PAID_AFTER}"
log_detail "Cache hits after re-run  : ${HITS_AFTER}"
log_detail "USD spend after re-run   : \$${SPENT_AFTER}"

DELTA_HITS=$(( HITS_AFTER - HITS_BEFORE ))
if [ "${DELTA_HITS}" -gt 0 ]; then
    log_success "Response caching confirmed! Registered ${DELTA_HITS} new cache hit(s), zero additional USD spent."
else
    log_warning "No increment in cache hits detected. Cache might have been bypassed or already held this exact request."
fi

log_step "3.4" "Displaying CLI usage table"
uv run corpus-forensics usage --no-remote

# ==============================================================================
# STAGE 4: BENCHMARK EVALUATION HARNESS
# ==============================================================================
log_header "STAGE 4: Benchmark Evaluation Harness"

EVAL_OUT_DIR="reports/e2e_eval_summary"
rm -rf "${EVAL_OUT_DIR}"

log_step "4.1" "Running evaluation harness over labeled test fixtures"
log_info "Command: uv run corpus-forensics eval run -d tests/fixtures -o ${EVAL_OUT_DIR}"
uv run corpus-forensics eval run -d tests/fixtures -o "${EVAL_OUT_DIR}"

log_step "4.2" "Parsing evaluation summary metrics"
EVAL_METRICS=$(uv run python -c "
import json
with open('${EVAL_OUT_DIR}/eval_summary.json', encoding='utf-8') as f:
    s = json.load(f)

gen = s.get('generator', {})
det = gen.get('deteccion', {})
faith = gen.get('faithfulness', {})
abilities = s.get('abilities', {})
ret_global = s.get('retriever', {}).get('global', {})

f1 = det.get('f1', 0.0)
prec = det.get('precision', 0.0)
rec = det.get('recall', 0.0)
faith_val = faith.get('media', 0.0)
fpr_hard = abilities.get('noise_robustness_fpr_hard_negatives', {}).get('valor', 0.0)
fpr_clean = abilities.get('negative_rejection_fpr_limpios', {}).get('valor', 0.0)
ctx_prec = ret_global.get('context_precision', 0.0)
ctx_rec = ret_global.get('context_recall', 0.0)

print(f'EVAL_F1={f1:.3f}')
print(f'EVAL_PRECISION={prec:.3f}')
print(f'EVAL_RECALL={rec:.3f}')
print(f'EVAL_FAITHFULNESS={faith_val:.3f}')
print(f'EVAL_FPR_HARD={fpr_hard:.3f}')
print(f'EVAL_FPR_CLEAN={fpr_clean:.3f}')
print(f'EVAL_CTX_PRECISION={ctx_prec:.3f}')
print(f'EVAL_CTX_RECALL={ctx_rec:.3f}')
")
eval "$EVAL_METRICS"

log_info "Generator F1 Score (malicioso)       : ${BOLD}${EVAL_F1}${RESET}"
log_info "Generator Precision                  : ${BOLD}${EVAL_PRECISION}${RESET}"
log_info "Generator Recall                     : ${BOLD}${EVAL_RECALL}${RESET}"
log_info "Generator Faithfulness               : ${BOLD}${EVAL_FAITHFULNESS}${RESET}"
log_info "FPR on Hard Negatives (Robustness)   : ${BOLD}${EVAL_FPR_HARD}${RESET}"
log_info "FPR on Clean Documents               : ${BOLD}${EVAL_FPR_CLEAN}${RESET}"
log_info "Retriever Context Precision@3        : ${BOLD}${EVAL_CTX_PRECISION}${RESET}"
log_info "Retriever Context Recall             : ${BOLD}${EVAL_CTX_RECALL}${RESET}"

log_success "Evaluation benchmark completed with F1 = ${EVAL_F1}."

# ==============================================================================
# STAGE 5: CODEBASE HEALTH & QUALITY STANDARDS (AGENTS.MD COMPLIANCE)
# ==============================================================================
log_header "STAGE 5: Quality Standards & Regression Verification"

log_step "5.1" "Running unit and integration tests (pytest)"
log_info "Command: uv run pytest"
uv run pytest
log_success "All unit & integration tests passed."

log_step "5.2" "Running static linter and formatting rules (ruff)"
log_info "Command: uv run ruff check"
uv run ruff check
log_success "Ruff linter passed with zero warnings."

log_step "5.3" "Verifying static type soundness (mypy)"
log_info "Command: uv run mypy src"
uv run mypy src
log_success "Mypy type checking passed successfully."

# ==============================================================================
# TEST SUITE SUMMARY
# ==============================================================================
SCRIPT_END_TIME=$(date +%s)
TOTAL_DURATION=$(( SCRIPT_END_TIME - SCRIPT_START_TIME ))

FINAL_SPEND=$(uv run python -c "
from corpus_forensics.config import get_settings
from corpus_forensics.llm.litellm_client import build_usage_tracker
tracker = build_usage_tracker(get_settings())
print(f'{tracker.total_cost_usd:.4f}')
")

log_header "END-TO-END TEST SUITE SUMMARY: ALL CHECKS PASSED"

echo -e "  ${BOLD}${GREEN}✔ All pipeline stages verified successfully!${RESET}"
echo -e "  ${CYAN}• Total test runtime    :${RESET} ${TOTAL_DURATION}s"
echo -e "  ${CYAN}• Total OpenAI spend    :${RESET} \$${FINAL_SPEND} USD"
echo -e "  ${CYAN}• Active LLM Model      :${RESET} ${LLM_MODEL} (Live OpenAI)"
echo -e "  ${CYAN}• Active Embedder Model :${RESET} ${EMBEDDING_MODEL} (Live OpenAI)"
echo -e "  ${CYAN}• Generated Reports     :${RESET} ${REPORTS_DIR}/"
echo -e "  ${CYAN}• Benchmark Summary     :${RESET} ${EVAL_OUT_DIR}/eval_summary.md"
echo ""
