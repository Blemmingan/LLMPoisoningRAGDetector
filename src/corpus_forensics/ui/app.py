"""Corpus Forensics web interface.

Run with `corpus-forensics ui` (or `streamlit run src/corpus_forensics/ui/app.py`).
Analyzes one document with the configured pipeline and shows the verdict, the findings
highlighted on the text (visible and hidden channels), the retrieved examples and the
cited sources; a second tab browses evaluation, ablation and cost reports.
"""

import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

from corpus_forensics.analysis.pipeline import ForensicPipeline
from corpus_forensics.config import Settings, get_settings
from corpus_forensics.reporting.markdown_renderer import render_report_to_markdown
from corpus_forensics.schemas import DocumentReport
from corpus_forensics.ui.helpers import (
    SEVERITY_COLORS,
    find_summaries,
    finding_title,
    highlight,
    load_labels,
    report_json,
    retrieval_rows,
)

DATASET_DIR = Path("data/eval/v1")
REPORTS_DIR = Path("reports")
ALLOWED_TYPES = ["txt", "md", "html", "htm", "pdf"]

st.set_page_config(page_title="Corpus Forensics", page_icon="🔎", layout="wide")
st.markdown(
    """<style>
    mark { background: #ffd54f; color: #111; padding: 0 2px; border-radius: 3px; }
    .doc-view { max-height: 420px; overflow-y: auto; padding: 12px 14px; border-radius: 8px;
                margin-bottom: 16px;
                border: 1px solid rgba(128,128,128,.35); font-size: .9rem; line-height: 1.5; }
    .pill { display: inline-block; padding: 1px 8px; border-radius: 10px; color: #fff;
            font-size: .8rem; margin-right: 6px; }
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando modelos y bases vectoriales…")
def get_pipeline(overrides: tuple[tuple[str, Any], ...]) -> ForensicPipeline:
    base = get_settings()
    settings = Settings.model_validate({**base.model_dump(), **dict(overrides)})
    return ForensicPipeline(settings=settings)


def sidebar() -> tuple[tuple[str, Any], ...]:
    base = get_settings()
    st.sidebar.header("Configuración")
    st.sidebar.caption(
        f"LLM: **{base.llm_provider}**"
        + (f" ({base.llm_model})" if base.llm_provider != "mock" else "")
        + f" · Embeddings: **{base.embedder_provider}**"
    )
    if base.llm_provider == "mock":
        st.sidebar.warning("El LLM es un mock: los hallazgos salen de reglas, no de un modelo.")
    st.sidebar.subheader("Advanced RAG")
    rewrite = st.sidebar.toggle("Reescritura de la consulta", value=base.query_rewrite)
    hybrid = st.sidebar.toggle("Búsqueda híbrida (BM25 + RRF)", value=base.hybrid_search)
    rerank = st.sidebar.toggle("Reranker (cross-encoder)", value=base.rerank)
    st.sidebar.subheader("Veredicto")
    threshold = st.sidebar.slider("Umbral de confianza", 0.05, 0.95, base.verdict_threshold, 0.05)
    verified = st.sidebar.toggle("Exigir fragmento verificado", value=base.verdict_require_verified)
    return (
        ("query_rewrite", rewrite),
        ("hybrid_search", hybrid),
        ("rerank", rerank),
        ("verdict_threshold", threshold),
        ("verdict_require_verified", verified),
    )


def pick_document() -> tuple[Path | None, dict[str, Any] | None]:
    """The document to analyze (uploaded or from the dataset) and its label, if any."""
    source = st.radio("Documento", ["Subir un archivo", "Ejemplo del dataset"], horizontal=True)
    if source == "Subir un archivo":
        uploaded = st.file_uploader("TXT, Markdown, HTML o PDF", type=ALLOWED_TYPES)
        if uploaded is None:
            return None, None
        folder = Path(tempfile.mkdtemp(prefix="cf-ui-"))
        path = folder / uploaded.name
        path.write_bytes(uploaded.getvalue())
        return path, None
    labels = load_labels(DATASET_DIR)
    if not labels:
        st.info("No hay dataset de evaluación: generarlo con `corpus-forensics eval build`.")
        return None, None
    options = {f"{lab['documento']} ({lab['categoria']})": lab for lab in labels}
    choice = st.selectbox("Documento del dataset", list(options))
    label = options[choice]
    return DATASET_DIR / label["documento"], label


def show_report(report: DocumentReport, seconds: float, pipeline: ForensicPipeline) -> None:
    meta = report.metadata_de_analisis
    if report.malicioso:
        st.error(f"**MALICIOSO** · {len(report.hallazgos)} hallazgo(s)")
    else:
        st.success("**LIMPIO** · no hay hallazgos que superen la política del veredicto")
    cols = st.columns(4)
    cols[0].metric("Chunks", meta.n_chunks)
    cols[1].metric("Hallazgos", len(report.hallazgos))
    cols[2].metric("Tiempo", f"{seconds:.2f} s" if seconds < 1 else f"{seconds:.1f} s")
    tokens = pipeline.last_token_estimate.get("total_entrada")
    cols[3].metric("Tokens de entrada (estim.)", f"{tokens:,}".replace(",", ".") if tokens else "—")
    st.caption(f"Política: {meta.verdict_policy} · loader: {meta.loader_used}")

    st.subheader("Hallazgos")
    if not report.hallazgos:
        st.write("Sin hallazgos.")
    for finding in report.hallazgos:
        color = SEVERITY_COLORS.get(finding.severidad.value, "#888")
        with st.expander(finding_title(finding), expanded=report.malicioso):
            st.markdown(
                f'<span class="pill" style="background:{color}">{finding.severidad.value}</span>'
                f'<span class="pill" style="background:#555">'
                f"{', '.join(t.value for t in finding.tecnica_de_ocultamiento)}</span>"
                + ("✅ verificado" if finding.verificado else "⚠️ no verificado"),
                unsafe_allow_html=True,
            )
            st.code(finding.fragmento_exacto, language="text")
            st.markdown(f"**Objetivo del atacante:** {finding.objetivo_del_atacante}")
            st.markdown(f"**Mitigación:** {finding.mitigacion}")
            if finding.fuente_mitigacion:
                st.caption(f"Fuente: {finding.fuente_mitigacion}")
            if finding.ataque_similar_id:
                st.caption(
                    f"Ataque similar en la base: {finding.ataque_similar_id}"
                    + (" · posible variante nueva" if finding.posible_variante_nueva else "")
                )
            if finding.notas_verificacion:
                st.caption(finding.notas_verificacion)

    fragments = [h.fragmento_exacto for h in report.hallazgos]
    doc = pipeline.last_document
    if doc is not None:
        st.subheader("Documento")
        visible_tab, hidden_tab = st.tabs(["Texto visible", "Canal oculto"])
        with visible_tab:
            st.markdown(
                f'<div class="doc-view">{highlight(doc.visible_text, fragments)}</div>',
                unsafe_allow_html=True,
            )
        with hidden_tab:
            if doc.hidden_content:
                st.markdown(
                    f'<div class="doc-view">{highlight(doc.hidden_content, fragments)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.write("El documento no tiene contenido oculto.")

    with st.expander("Ejemplos recuperados de la base (por chunk)"):
        st.dataframe(retrieval_rows(pipeline.last_trace), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    stem = Path(report.documento).stem
    left.download_button(
        "Descargar reporte JSON", report_json(report), f"{stem}_report.json", "application/json"
    )
    right.download_button(
        "Descargar reporte Markdown",
        render_report_to_markdown(report),
        f"{stem}_report.md",
        "text/markdown",
    )


def analyze_tab(overrides: tuple[tuple[str, Any], ...]) -> None:
    path, label = pick_document()
    if path is None:
        return
    if label is not None and label.get("malicioso"):
        with st.expander("Etiqueta real del documento (ground truth)"):
            st.json(
                {
                    k: label[k]
                    for k in (
                        "categoria",
                        "tipo_de_ataque",
                        "tecnica_de_ocultamiento",
                        "posicion",
                        "fragmento_exacto",
                    )
                    if k in label
                }
            )
    if st.button("Analizar", type="primary"):
        pipeline = get_pipeline(overrides)
        with st.spinner("Analizando…"):
            start = time.perf_counter()
            report = pipeline.analyze_document(path)
            st.session_state["result"] = (report, time.perf_counter() - start, overrides)
    result = st.session_state.get("result")
    if result is not None and result[2] == overrides:
        show_report(result[0], result[1], get_pipeline(overrides))


def evaluation_tab() -> None:
    root = Path(st.text_input("Carpeta de resultados", str(REPORTS_DIR)))
    summaries = find_summaries(root)
    if not summaries:
        st.info(
            "No hay resultados en esa carpeta. Se generan con `corpus-forensics eval run`, "
            "`eval ablation` o `eval cost`."
        )
        return
    choice = st.selectbox("Reporte", list(summaries))
    st.markdown(summaries[choice].read_text(encoding="utf-8"))


def main() -> None:
    st.title("🔎 Corpus Poisoning Forensic Analyst")
    st.caption(
        "Inspecciona documentos antes de indexarlos en un RAG y reporta instrucciones "
        "inyectadas, triggers, jailbreaks, exfiltración y variantes ofuscadas."
    )
    overrides = sidebar()
    analyze, evaluation = st.tabs(["Analizar documento", "Evaluación"])
    with analyze:
        analyze_tab(overrides)
    with evaluation:
        evaluation_tab()


main()
