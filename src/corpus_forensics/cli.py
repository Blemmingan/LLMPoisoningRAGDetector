"""Typer Command-Line Interface for Corpus Forensic Analyst."""

from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from corpus_forensics.analysis.pipeline import ForensicPipeline
from corpus_forensics.config import get_settings
from corpus_forensics.embeddings.factory import get_embedder
from corpus_forensics.evaluation.harness import EvaluationHarness
from corpus_forensics.knowledge_base.ingest import ingest_knowledge_base
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
        except Exception as e:
            results_table.add_row(file_path.name, "[red]ERROR[/red]", "-", str(e))

    console.print(results_table)


# -----------------------------------------------------------------------------
# kb commands
# -----------------------------------------------------------------------------


@kb_app.command(name="ingest")
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


# -----------------------------------------------------------------------------
# eval commands
# -----------------------------------------------------------------------------


@eval_app.command(name="run")
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


if __name__ == "__main__":
    app()
