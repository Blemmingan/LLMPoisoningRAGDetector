"""Typer Command-Line Interface for Corpus Forensic Analyst."""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich import print as rprint
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from corpus_forensics.analysis.pipeline import ForensicPipeline
from corpus_forensics.config import get_settings
from corpus_forensics.embeddings.factory import get_embedder
from corpus_forensics.evaluation.harness import EvaluationHarness
from corpus_forensics.knowledge_base.download import download_source, list_downloadable_sources
from corpus_forensics.knowledge_base.ingest import ingest_knowledge_base
from corpus_forensics.llm.base import BudgetExceededError
from corpus_forensics.llm.litellm_client import build_usage_tracker
from corpus_forensics.reporting.json_writer import write_json_report
from corpus_forensics.reporting.markdown_renderer import write_markdown_report
from corpus_forensics.vectorstore.chroma_store import ChromaVectorStore

app = typer.Typer(
    name="corpus-forensics",
    help="Forensic Analyst detecting indirect prompt injection and corpus poisoning.",
    no_args_is_help=True,
)
kb_app = typer.Typer(help="Manage attack knowledge base and persistent vector store.")
eval_app = typer.Typer(help="Run benchmark evaluation over labeled datasets.")

app.add_typer(kb_app, name="kb")
app.add_typer(eval_app, name="eval")

console = Console()


def stop_on_budget(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn spending-cap / no-credit errors into a clear message and a non-zero exit."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except BudgetExceededError as exc:
            rprint(f"[bold red]Stopped:[/bold red] {exc}")
            raise typer.Exit(code=2) from exc

    return wrapper


# -----------------------------------------------------------------------------
# analyze command
# -----------------------------------------------------------------------------


@app.command(name="analyze")
def analyze(
    paths: Annotated[
        list[Path],
        typer.Argument(help="One or more document paths or directories to analyze."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Directory where reports will be written."),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Report format: 'json', 'md', or 'both'.",
        ),
    ] = "both",
) -> None:
    """Analyze one or more files/dirs for prompt injection and output forensic reports."""
    settings = get_settings()
    out_dir = out or settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    supported_extensions = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}
    files_to_analyze: list[Path] = []

    for p in paths:
        if p.is_file():
            files_to_analyze.append(p)
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in supported_extensions:
                    files_to_analyze.append(child)
        else:
            rprint(f"[yellow]Warning:[/yellow] Path does not exist: {p}")

    if not files_to_analyze:
        rprint("[red]Error:[/red] No valid documents found to analyze.")
        raise typer.Exit(code=1)

    rprint("[bold cyan]Forensic Analysis Pipeline[/bold cyan]")
    rprint(f"Target documents: [bold]{len(files_to_analyze)}[/bold]")
    rprint(f"LLM Provider: [dim]{settings.llm_provider}[/dim] (mock: {settings.llm_is_mock})")
    rprint(f"Embedder: [dim]{settings.embedder_provider}[/dim] (mock: {settings.embedder_is_mock})")
    rprint(f"Reports destination: [dim]{out_dir}[/dim]\n")

    pipeline = ForensicPipeline(settings=settings)

    results_table = Table(title="Forensic Analysis Summary")
    results_table.add_column("Document", style="cyan")
    results_table.add_column("Verdict", justify="center")
    results_table.add_column("Findings", justify="center")
    results_table.add_column("Report Files", style="dim")

    for file_path in files_to_analyze:
        try:
            report = pipeline.analyze_document(file_path)
            stem = file_path.stem
            report_links: list[str] = []

            # Save reports according to format
            fmt = format.lower()
            if fmt in ("json", "both"):
                json_path = out_dir / f"{stem}_report.json"
                write_json_report(report, json_path)
                report_links.append(json_path.name)

            if fmt in ("md", "both"):
                md_path = out_dir / f"{stem}_report.md"
                write_markdown_report(report, md_path)
                report_links.append(md_path.name)

            verdict_badge = (
                "[bold red]MALICIOUS[/bold red]"
                if report.malicioso
                else "[bold green]CLEAN[/bold green]"
            )
            results_table.add_row(
                file_path.name,
                verdict_badge,
                str(len(report.hallazgos)),
                ", ".join(report_links),
            )
        except BudgetExceededError as e:
            console.print(results_table)
            rprint(f"[bold red]Stopped:[/bold red] {e}")
            raise typer.Exit(code=2) from e
        except Exception as e:
            results_table.add_row(file_path.name, "[red]ERROR[/red]", "-", str(e))

    console.print(results_table)


# -----------------------------------------------------------------------------
# kb commands
# -----------------------------------------------------------------------------


@kb_app.command(name="download")
def kb_download(
    source: Annotated[
        str,
        typer.Option("--source", "-s", help="Source name or 'all'."),
    ] = "all",
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Download again even if files already exist."),
    ] = False,
) -> None:
    """Download the raw files of the knowledge-base sources (no scraping, no LLM calls)."""
    settings = get_settings()
    sources = list_downloadable_sources() if source.lower() == "all" else [source.lower()]
    hf_token = settings.hf_token.get_secret_value() if settings.hf_token else None

    table = Table(title="Knowledge base download")
    table.add_column("Source", style="cyan")
    table.add_column("File")
    table.add_column("Status")
    table.add_column("Size", justify="right")
    for name in sources:
        try:
            result = download_source(name, settings.kb_raw_dir, hf_token=hf_token, force=force)
        except Exception as exc:  # report and continue with the other sources
            table.add_row(name, "-", f"[red]ERROR[/red] {exc}", "-")
            continue
        for f in result["files"]:
            size = "-"
            if "bytes" in f:
                kb = f["bytes"] / 1000
                size = f"{kb / 1000:.1f} MB" if kb >= 1000 else f"{kb:.0f} kB"
            table.add_row(name, f["file"], f["status"], size)
    console.print(table)
    rprint(f"[dim]Raw files in {settings.kb_raw_dir}[/dim]")


@kb_app.command(name="ingest")
@stop_on_budget
def kb_ingest(
    source: Annotated[
        str,
        typer.Option(
            "--source", "-s", help="Source name ('custom', 'owasp_llm_top10', etc.) or 'all'."
        ),
    ] = "all",
    reset: Annotated[
        bool,
        typer.Option("--reset", "-r", help="Reset vector store before ingesting."),
    ] = False,
    path: Annotated[
        Path | None,
        typer.Option("--path", "-p", help="Custom path for data files."),
    ] = None,
) -> None:
    """Load known attack sources into the persistent vector database."""
    settings = get_settings()
    vector_store = ChromaVectorStore(persist_dir=settings.vectorstore_dir)
    embedder = get_embedder(
        provider=settings.embedder_provider,
        dimension=settings.embedding_dimension,
    )

    rprint("[bold cyan]Knowledge Base Ingestion[/bold cyan]")
    rprint(f"Target source: [bold]{source}[/bold]")
    rprint(f"Reset database: [bold]{reset}[/bold]")
    rprint(f"Persistence path: [dim]{settings.vectorstore_dir}[/dim]\n")

    result = ingest_knowledge_base(
        vector_store=vector_store,
        embedder=embedder,
        source=source,
        source_path=path,
        reset=reset,
    )

    rprint("[green]✓ Ingestion complete![/green]")
    rprint(f"Total entries added in run: [bold]{result['total_ingested']}[/bold]")
    rprint(f"Current entries in store: [bold]{result['store_count']}[/bold]")

    table = Table(title="Breakdown by Source")
    table.add_column("Source", style="cyan")
    table.add_column("Entries Ingested", justify="right")
    for src, count in result.get("by_source", {}).items():
        table.add_row(src, str(count))
    console.print(table)

    if result.get("discarded"):
        disc_table = Table(title="Discarded as out of scope (rule-based labeling)")
        disc_table.add_column("Source: reason", style="yellow")
        disc_table.add_column("Entries", justify="right")
        for reason, count in sorted(result["discarded"].items()):
            disc_table.add_row(reason, str(count))
        console.print(disc_table)


@kb_app.command(name="stats")
def kb_stats() -> None:
    """Show attack entry counts by attack type and source in vector DB."""
    settings = get_settings()
    vector_store = ChromaVectorStore(persist_dir=settings.vectorstore_dir)
    stats = vector_store.get_stats()

    total = stats.get("total_entries", 0)
    rprint("[bold cyan]Knowledge Base Statistics[/bold cyan]")
    rprint(f"Total known attack signatures: [bold]{total}[/bold]\n")

    if total == 0:
        rprint("[yellow]Vector database is currently empty. Run 'kb ingest' first.[/yellow]")
        return

    # Source table
    source_table = Table(title="Entries by Source")
    source_table.add_column("Source", style="cyan")
    source_table.add_column("Count", justify="right")
    for src, cnt in stats.get("by_source", {}).items():
        source_table.add_row(src, str(cnt))
    console.print(source_table)
    print()

    # Attack type table
    type_table = Table(title="Entries by Attack Type")
    type_table.add_column("Attack Type", style="magenta")
    type_table.add_column("Count", justify="right")
    for at, cnt in stats.get("by_attack_type", {}).items():
        type_table.add_row(at, str(cnt))
    console.print(type_table)
    print()

    conceal_table = Table(title="Entries by Concealment Technique")
    conceal_table.add_column("Technique", style="green")
    conceal_table.add_column("Count", justify="right")
    for tech, cnt in stats.get("by_concealment", {}).items():
        conceal_table.add_row(tech, str(cnt))
    console.print(conceal_table)
    rprint(f"[dim]Label status: {stats.get('by_label_status', {})}[/dim]")


# -----------------------------------------------------------------------------
# eval commands
# -----------------------------------------------------------------------------


@eval_app.command(name="build")
def eval_build(
    out: Annotated[Path, typer.Option("--out", "-o", help="Output dataset directory.")] = Path(
        "data/eval/v1"
    ),
    sources: Annotated[Path, typer.Option("--sources", help="Pinned clean sources (YAML).")] = Path(
        "data/eval/sources.yaml"
    ),
    cache: Annotated[
        Path, typer.Option("--cache", help="Download cache for the pinned sources.")
    ] = Path("data/eval/cache"),
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    n_clean: Annotated[int, typer.Option("--n-clean")] = 60,
    n_poisoned: Annotated[int, typer.Option("--n-poisoned")] = 100,
    n_hard_negative: Annotated[int, typer.Option("--n-hard-negative")] = 30,
) -> None:
    """Build the evaluation dataset from pinned texts and held-out attacks (no LLM calls)."""
    from corpus_forensics.evaluation.build_dataset import build_dataset

    card = build_dataset(
        get_settings(),
        out_dir=out,
        sources_path=sources,
        cache_dir=cache,
        seed=seed,
        n_clean=n_clean,
        n_poisoned=n_poisoned,
        n_hard_negative=n_hard_negative,
    )
    rprint(f"[green]✓ Dataset written to {out} ({card['total']} documents)[/green]")
    _print_dataset_card(card)


@eval_app.command(name="describe")
def eval_describe(
    dataset: Annotated[Path, typer.Option("--dataset", "-d", help="Dataset directory.")] = Path(
        "data/eval/v1"
    ),
) -> None:
    """Show the composition of an evaluation dataset."""
    from corpus_forensics.evaluation.build_dataset import describe_labels, read_labels

    _print_dataset_card(describe_labels(read_labels(dataset)))


def _print_dataset_card(card: dict[str, Any]) -> None:
    table = Table(title="Evaluation dataset composition")
    table.add_column("Dimension", style="cyan")
    table.add_column("Counts")
    for key, value in card.items():
        if isinstance(value, dict):
            table.add_row(
                key,
                escape(
                    ", ".join(f"{k}: {v}" for k, v in sorted(value.items(), key=lambda kv: -kv[1]))
                ),
            )
    console.print(table)


@eval_app.command(name="run")
@stop_on_budget
def eval_run(
    dataset: Annotated[
        Path,
        typer.Option(
            "--dataset", "-d", help="Directory containing test documents and labels.jsonl."
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Output directory for evaluation results."),
    ] = None,
) -> None:
    """Run benchmark evaluation over labeled dataset and output metrics summary."""
    rprint("[bold cyan]Running Forensic Benchmark Evaluation[/bold cyan]")
    rprint(f"Dataset path: [bold]{dataset}[/bold]\n")

    harness = EvaluationHarness()
    results = harness.run(dataset_dir=dataset, output_dir=out)

    metrics = results["metrics"]
    rprint(f"[green]✓ Evaluation completed over {results['total_evaluated']} samples![/green]\n")

    # Metrics table
    metrics_table = Table(title="Benchmark Performance Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", justify="right", style="bold")

    metrics_table.add_row("Precision (Malicioso)", f"{metrics.precision:.4f}")
    metrics_table.add_row("Recall (Malicioso)", f"{metrics.recall:.4f}")
    metrics_table.add_row("F1-Score", f"{metrics.f1:.4f}")
    metrics_table.add_row("Accuracy", f"{metrics.accuracy:.4f}")
    metrics_table.add_row("FPR on Hard Negatives", f"{metrics.fpr_hard_negatives:.4f}")
    metrics_table.add_row("Attack Type Accuracy", f"{metrics.attack_type_accuracy:.4f}")
    metrics_table.add_row(
        "Concealment Tech Accuracy", f"{metrics.concealment_technique_accuracy:.4f}"
    )
    metrics_table.add_row("Fragment Match Rate", f"{metrics.fragment_match_rate:.4f}")

    console.print(metrics_table)
    rprint(f"\n[dim]Summary JSON written to: {results['json_summary']}[/dim]")
    rprint(f"[dim]Summary Markdown written to: {results['markdown_summary']}[/dim]")


# -----------------------------------------------------------------------------
# usage command
# -----------------------------------------------------------------------------


@app.command(name="usage")
def usage(
    remote: Annotated[
        bool,
        typer.Option(
            "--remote/--no-remote",
            help="Also query the proxy for the team spend (free, no tokens used).",
        ),
    ] = True,
) -> None:
    """Show model usage and cost from the local ledger and, optionally, the team budget."""
    settings = get_settings()
    tracker = build_usage_tracker(settings)
    records = tracker.read_records()

    paid = [r for r in records if not r.cached]
    table = Table(title="Local usage ledger")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold")
    table.add_row("Paid calls", str(len(paid)))
    table.add_row("Cache hits", str(len(records) - len(paid)))
    table.add_row("Prompt tokens", str(sum(r.prompt_tokens for r in paid)))
    table.add_row("Completion tokens", str(sum(r.completion_tokens for r in paid)))
    table.add_row("Cost (USD)", f"{tracker.total_cost_usd:.4f}")
    table.add_row("Local cap (USD)", f"{settings.budget_limit_usd:.2f}")
    console.print(table)

    if not remote:
        return
    if not settings.llm_base_url or settings.llm_api_key is None:
        rprint("[yellow]Proxy not configured; skipping team spend.[/yellow]")
        return

    import httpx

    headers = {"x-litellm-api-key": settings.llm_api_key.get_secret_value()}
    base = settings.llm_base_url.rstrip("/")
    try:
        key_info = httpx.get(f"{base}/key/info", headers=headers, timeout=30).json()
        team_id = key_info.get("info", {}).get("team_id")
        team = httpx.get(
            f"{base}/team/info", params={"team_id": team_id}, headers=headers, timeout=30
        ).json()
        info = team.get("team_info", team)
        spend = float(info.get("spend") or 0.0)
        budget = info.get("max_budget")
        rprint(
            f"Team [bold]{info.get('team_alias')}[/bold]: spent [bold]{spend:.4f}[/bold] USD"
            + (f" of {float(budget):.2f} USD" if budget is not None else "")
        )
    except (httpx.HTTPError, ValueError) as exc:
        rprint(f"[yellow]Could not read team spend from proxy: {exc}[/yellow]")


# -----------------------------------------------------------------------------
# lit commands (literature collection: papers and technical reports)
# -----------------------------------------------------------------------------

lit_app = typer.Typer(help="Literature collection: papers and reports about attack types.")
app.add_typer(lit_app, name="lit")


def _lit_sources() -> list[Any]:
    from corpus_forensics.knowledge_base.literature import load_sources

    settings = get_settings()
    priorities = [p.strip() for p in settings.lit_priorities.split(",") if p.strip()]
    return load_sources(settings.literature_sources_path, priorities=priorities)


@lit_app.command(name="download")
def lit_download(
    force: Annotated[bool, typer.Option("--force", "-f", help="Download again.")] = False,
) -> None:
    """Download the literature sources (PDF/HTML) listed in literature_sources.yaml."""
    from corpus_forensics.knowledge_base.literature import download_literature

    settings = get_settings()
    sources = _lit_sources()
    rprint(f"[bold cyan]Downloading {len(sources)} literature sources[/bold cyan]")
    results = download_literature(sources, settings.kb_raw_dir / "literatura", force=force)
    table = Table(title="Literature download")
    table.add_column("Source", style="cyan")
    table.add_column("Status")
    table.add_column("Size", justify="right")
    for r in results:
        size = f"{r['bytes'] / 1000:.0f} kB" if "bytes" in r else "-"
        status = r["status"] if r["status"] != "error" else f"[red]error[/red] {r['error']}"
        table.add_row(r["id"], status, size)
    console.print(table)


@lit_app.command(name="build")
def lit_build() -> None:
    """Convert downloaded sources to markdown (Docling for PDFs) and split into chunks."""
    from corpus_forensics.knowledge_base.literature import (
        chunk_source,
        source_to_markdown,
        write_chunks,
    )

    settings = get_settings()
    raw_dir = settings.kb_raw_dir / "literatura"
    md_dir = settings.kb_processed_dir / "markdown"
    all_chunks = []
    table = Table(title="Literature chunks")
    table.add_column("Source", style="cyan")
    table.add_column("Chunks", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("With payload", justify="right")
    for source in _lit_sources():
        if not (raw_dir / source.filename).exists():
            table.add_row(source.id, "[yellow]not downloaded[/yellow]", "-", "-")
            continue
        try:
            markdown = source_to_markdown(source, raw_dir, md_dir)
        except Exception as exc:  # keep going with the other sources
            table.add_row(source.id, f"[red]error[/red] {exc}", "-", "-")
            continue
        chunks = chunk_source(
            source,
            markdown,
            target_tokens=settings.lit_chunk_tokens,
            overlap_tokens=settings.lit_chunk_overlap,
        )
        all_chunks.extend(chunks)
        table.add_row(
            source.id,
            str(len(chunks)),
            str(sum(c.n_tokens for c in chunks)),
            str(sum(1 for c in chunks if c.metadata.get("contiene_payload"))),
        )
    out = settings.kb_processed_dir / "literatura_chunks.jsonl"
    n = write_chunks(all_chunks, out)
    console.print(table)
    rprint(f"[green]{n} chunks[/green], {sum(c.n_tokens for c in all_chunks)} tokens -> {out}")


@lit_app.command(name="ingest")
@stop_on_budget
def lit_ingest(
    reset: Annotated[
        bool,
        typer.Option(
            "--reset", "-r", help="Recreate the collection (needed after changing embedder)."
        ),
    ] = False,
) -> None:
    """Embed the built chunks and load them into the `literatura` collection.

    Without --reset it resumes, embedding only passages not loaded yet.
    """
    from corpus_forensics.knowledge_base.literature import read_chunks
    from corpus_forensics.vectorstore.literature_store import LiteratureStore

    settings = get_settings()
    chunks = read_chunks(settings.kb_processed_dir / "literatura_chunks.jsonl")
    embedder = get_embedder(
        provider=settings.embedder_provider, dimension=settings.embedding_dimension
    )
    store = LiteratureStore(settings.vectorstore_dir)
    if reset:
        store.reset()
    else:  # resume: skip passages already loaded
        done = store.existing_ids()
        chunks = [c for c in chunks if c.id not in done]
        if done:
            rprint(f"[dim]Resuming: {len(done)} passages already loaded.[/dim]")
    batch = 100
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        vectors = embedder.embed_documents([c.embedding_text for c in part])
        store.add(
            ids=[c.id for c in part],
            texts=[c.texto for c in part],
            embeddings=vectors,
            metadatas=[c.metadata for c in part],
        )
    rprint(
        f"[green]Loaded {len(chunks)} passages[/green] into 'literatura' "
        f"(embedder: {settings.embedder_provider}, mock: {embedder.is_mock}). "
        f"Total in collection: {store.count()}"
    )


@lit_app.command(name="stats")
def lit_stats() -> None:
    """Show passage counts per source in the `literatura` collection."""
    from corpus_forensics.vectorstore.literature_store import LiteratureStore

    stats = LiteratureStore(get_settings().vectorstore_dir).stats()
    table = Table(title=f"Literatura: {stats['total']} passages")
    table.add_column("Source", style="cyan")
    table.add_column("Passages", justify="right")
    for src, n in stats["by_source"].items():
        table.add_row(src, str(n))
    console.print(table)


@lit_app.command(name="search")
@stop_on_budget
def lit_search(
    query: Annotated[str, typer.Argument(help="Text to search for.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 3,
) -> None:
    """Search the literature collection (costs one embedding call)."""
    from corpus_forensics.vectorstore.literature_store import LiteratureStore

    settings = get_settings()
    embedder = get_embedder(
        provider=settings.embedder_provider, dimension=settings.embedding_dimension
    )
    passages = LiteratureStore(settings.vectorstore_dir).query(
        embedder.embed_query(query), top_k=top_k
    )
    for p in passages:
        # escape(): passages contain brackets (e.g. markdown images) that Rich would
        # otherwise swallow as style markup
        rprint(
            f"[bold]{p.similarity_score:.3f}[/bold] [cyan]{escape(p.titulo)}[/cyan] | "
            f"{escape(p.seccion)}"
        )
        rprint(f"  {escape(p.texto[:300])}...\n")


if __name__ == "__main__":
    app()
