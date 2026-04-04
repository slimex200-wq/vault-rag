"""vault-rag CLI — Click entry point for all pipeline commands."""
from __future__ import annotations

from pathlib import Path

import click

from vault_rag.config import VaultConfig


# ---------------------------------------------------------------------------
# Module-level helper (patchable in tests)
# ---------------------------------------------------------------------------


def _get_config() -> VaultConfig:
    return VaultConfig()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="vault-rag")
def cli() -> None:
    """vault-rag: Karpathy-style RAG for Obsidian vaults."""


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


@cli.command("scan")
def scan_cmd() -> None:
    """Scan vault and show note count + first 10 notes with tags."""
    from vault_rag.ingest.scanner import VaultScanner

    config = _get_config()
    scanner = VaultScanner(config)
    notes = scanner.scan()

    click.echo(f"Vault: {config.vault_path}")
    click.echo(f"Total notes: {len(notes)}")
    click.echo("")

    for note in notes[:10]:
        tags_str = ", ".join(note.tags) if note.tags else "(no tags)"
        click.echo(f"  {note.relative_path}")
        click.echo(f"    title : {note.title}")
        click.echo(f"    tags  : {tags_str}")


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


@cli.command("health")
def health_cmd() -> None:
    """Run health checks and show a summary report."""
    from vault_rag.engine.health import HealthChecker
    from vault_rag.ingest.scanner import VaultScanner

    config = _get_config()
    scanner = VaultScanner(config)
    notes = scanner.scan()
    checker = HealthChecker(notes)
    report = checker.report()

    click.echo(f"Health report for: {config.vault_path}")
    click.echo(f"  Total notes   : {report['total_notes']}")
    click.echo(f"  Broken links  : {report['broken_link_count']}")
    click.echo(f"  Orphan notes  : {report['orphan_count']}")
    click.echo(f"  Untagged notes: {report['untagged_count']}")

    if report["broken_link_count"]:
        click.echo("\nBroken links:")
        for bl in report["broken_links"]:
            click.echo(f"  {bl['source']} -> [[{bl['target']}]]")

    if report["orphan_notes"]:
        click.echo("\nOrphan notes:")
        for path in report["orphan_notes"]:
            click.echo(f"  {path}")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@cli.command("search")
@click.argument("query")
@click.option("-n", "--num", default=5, show_default=True, help="Number of results")
def search_cmd(query: str, num: int) -> None:
    """Semantic search via VectorStore + embeddings."""
    from vault_rag.engine.indexer import create_openai_embed_fn
    from vault_rag.engine.qa import QAEngine
    from vault_rag.store.vector_store import VectorStore

    config = _get_config()
    store = VectorStore(persist_dir=config.chroma_path)

    if store.count() == 0:
        click.echo("Vector store is empty. Run `vault-rag index` first.")
        return

    embed_fn = create_openai_embed_fn(
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
    )
    engine = QAEngine(
        vector_store=store,
        embed_fn=embed_fn,
        client=None,  # not needed for search-only
        model=config.qa_model,
    )
    results = engine.search(query, n_results=num)

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"Search results for: {query!r}\n")
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        title = meta.get("title", r["id"])
        path = meta.get("relative_path", r["id"])
        tags = meta.get("tags", "")
        distance = r["distance"]
        click.echo(f"[{i}] {title}  (distance: {distance:.4f})")
        click.echo(f"     path: {path}")
        if tags:
            click.echo(f"     tags: {tags}")
        click.echo("")


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


@cli.command("index")
@click.option("--full", is_flag=True, default=False, help="Full reindex (clear + rebuild)")
def index_cmd(full: bool) -> None:
    """Build or update the embedding index."""
    from vault_rag.engine.indexer import Indexer, create_openai_embed_fn
    from vault_rag.ingest.scanner import VaultScanner
    from vault_rag.store.vector_store import VectorStore

    config = _get_config()
    scanner = VaultScanner(config)
    store = VectorStore(persist_dir=config.chroma_path)

    if full:
        click.echo("Full reindex: clearing existing index...")

    notes = scanner.scan()

    embed_fn = create_openai_embed_fn(
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
    )
    indexer = Indexer(config=config, vector_store=store, embed_fn=embed_fn)

    if full:
        count = indexer.reindex(notes)
    else:
        count = indexer.index(notes)

    click.echo(f"Indexed {count} note(s). Total in store: {store.count()}")


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


@cli.command("ask")
@click.argument("question")
def ask_cmd(question: str) -> None:
    """RAG Q&A: retrieve context, generate answer, show sources."""
    from anthropic import Anthropic

    from vault_rag.engine.indexer import create_openai_embed_fn
    from vault_rag.engine.qa import QAEngine
    from vault_rag.store.vector_store import VectorStore

    config = _get_config()
    store = VectorStore(persist_dir=config.chroma_path)

    if store.count() == 0:
        click.echo("Vector store is empty. Run `vault-rag index` first.")
        return

    embed_fn = create_openai_embed_fn(
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
    )
    client = Anthropic()
    engine = QAEngine(
        vector_store=store,
        embed_fn=embed_fn,
        client=client,
        model=config.qa_model,
    )
    result = engine.answer(question)

    click.echo("Answer:\n")
    click.echo(result.answer)

    if result.sources:
        click.echo("\nSources:")
        for src in result.sources:
            meta = src["metadata"]
            title = meta.get("title", src["id"])
            path = meta.get("relative_path", src["id"])
            click.echo(f"  - {title} ({path})")


# ---------------------------------------------------------------------------
# clip
# ---------------------------------------------------------------------------


@cli.command("clip")
@click.argument("url")
@click.option("--folder", default="Reference", show_default=True, help="Target folder in vault")
def clip_cmd(url: str, folder: str) -> None:
    """Clip a web page into the vault as a markdown note."""
    from vault_rag.ingest.web_clipper import WebClipper

    config = _get_config()
    clipper = WebClipper(vault_path=config.vault_path)
    note_path = clipper.clip(url=url, folder=folder)
    click.echo(f"Clipped: {note_path}")


# ---------------------------------------------------------------------------
# ingest-pdf
# ---------------------------------------------------------------------------


@cli.command("ingest-pdf")
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option("--folder", default="Research", show_default=True, help="Target folder in vault")
def ingest_pdf_cmd(pdf_path: Path, folder: str) -> None:
    """Ingest a PDF file into the vault as a markdown note."""
    from vault_rag.ingest.pdf_reader import PDFReader

    config = _get_config()
    reader = PDFReader()
    note_path = reader.extract_to_note(
        pdf_path=pdf_path,
        vault_path=config.vault_path,
        folder=folder,
    )
    click.echo(f"Ingested: {note_path}")


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------


@cli.command("compile")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def compile_cmd(path: Path) -> None:
    """Compile a single note: auto-summarize, tag, and suggest links."""
    from anthropic import Anthropic

    from vault_rag.engine.compiler import Compiler
    from vault_rag.ingest.scanner import VaultScanner

    config = _get_config()
    scanner = VaultScanner(config)

    # Scan all notes then find the matching one
    notes = scanner.scan()
    target_path = Path(path).resolve()
    note = next(
        (n for n in notes if n.path.resolve() == target_path),
        None,
    )

    if note is None:
        click.echo(f"Note not found in vault: {path}", err=True)
        raise SystemExit(1)

    client = Anthropic()
    compiler = Compiler(client=client, model=config.compile_model)
    result = compiler.compile(note)

    click.echo(f"Summary:\n{result.summary}\n")
    if result.tags:
        click.echo(f"Tags: {', '.join(result.tags)}")
    if result.suggested_links:
        click.echo(f"Suggested links: {', '.join(result.suggested_links)}")
