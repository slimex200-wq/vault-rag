"""vault-rag CLI - Click entry point for all pipeline commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from vault_rag.config import VaultConfig

# ---------------------------------------------------------------------------
# Module-level helper (patchable in tests)
# ---------------------------------------------------------------------------


def _get_config() -> VaultConfig:
    return VaultConfig()


# Cap on per-issue listings. Orphan/broken-link counts run into the thousands
# on a mature vault, and dumping every row buries the summary.
_LIST_CAP = 20


def _echo_truncated(total: int) -> None:
    if total > _LIST_CAP:
        click.echo(f"  ... and {total - _LIST_CAP} more")


def _lint_cursor_path(config: VaultConfig) -> Path:
    return Path(config.chroma_path).parent / "lint-cursor.json"


def _read_lint_cursor(config: VaultConfig) -> int:
    path = _lint_cursor_path(config)
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("offset", 0))
    except (ValueError, OSError, json.JSONDecodeError):
        return 0


def _write_lint_cursor(config: VaultConfig, offset: int) -> None:
    path = _lint_cursor_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"offset": offset}), encoding="utf-8")


# Metrics the --strict gate refuses to let grow. A one-off cleanup does not
# stick unless something fails when the vault slides back.
_GATED_METRICS = (
    "broken_link_count",
    "orphan_count",
    "isolated_count",
    "untagged_count",
    "vocabulary_violation_count",
    "singleton_tag_count",
)


def _health_baseline_path(config: VaultConfig) -> Path:
    return Path(config.chroma_path).parent / "health-baseline.json"


def _read_health_baseline(config: VaultConfig) -> dict[str, int] | None:
    path = _health_baseline_path(config)
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {k: int(v) for k, v in loaded.items() if k in _GATED_METRICS}


def _write_health_baseline(config: VaultConfig, metrics: dict[str, int]) -> Path:
    path = _health_baseline_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="vault-rag")
def cli() -> None:
    """vault-rag: Karpathy-style LLM Wiki for Obsidian vaults."""
    # Vault content is UTF-8 (Korean prose, em-dashes, arrows) but a Windows
    # console defaults to cp949, which raises UnicodeEncodeError mid-report and
    # kills the command. Reporting must never die on a character.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Internal helper: compile + index a single note by absolute path
# ---------------------------------------------------------------------------


def _auto_compile_and_index(note_path: Path, config: VaultConfig) -> None:
    """Compile and embed a single note. Prints results to stdout."""
    from anthropic import Anthropic

    from vault_rag.engine.compiler import Compiler
    from vault_rag.engine.indexer import Indexer, create_openai_embed_fn
    from vault_rag.engine.qa import QAEngine
    from vault_rag.ingest.scanner import VaultScanner
    from vault_rag.store.vector_store import VectorStore

    scanner = VaultScanner(config)
    notes = scanner.scan()
    target = next(
        (n for n in notes if n.path.resolve() == note_path.resolve()),
        None,
    )

    if target is None:
        click.echo("  Warning: note not found in vault scan - skipping compile.", err=True)
        return

    # Dedup check before compiling
    vs = VectorStore(persist_dir=config.chroma_path)
    embed_fn = create_openai_embed_fn(config.embedding_model, config.embedding_dimensions)
    if vs.count() > 0:
        qa = QAEngine(vector_store=vs, embed_fn=embed_fn, client=None, model="")
        note_content_text = target.title + " " + target.content
        dupes = qa.find_duplicates(note_content_text, threshold=0.3)
        if dupes:
            click.echo(f"  Warning: Similar notes found ({len(dupes)}):")
            for d in dupes:
                click.echo(f"    [{d['distance']:.3f}] {d['metadata'].get('title', d['id'])}")

    known = {n.title.lower() for n in notes} | {Path(n.relative_path).stem.lower() for n in notes}
    client = Anthropic()
    compiler = Compiler(client=client, model=config.compile_model)
    result = compiler.compile_and_write(target, known_titles=known)
    click.echo(f"  Summary: {result.summary}")
    if result.tags:
        click.echo(f"  Tags: {', '.join(result.tags)}")
    if result.suggested_links:
        click.echo(f"  Links: {', '.join(result.suggested_links)}")
    click.echo("  Written to note.")

    indexer = Indexer(config=config, vector_store=vs, embed_fn=embed_fn)
    indexer.index([target])
    click.echo("  Indexed.")


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
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Exit non-zero if any metric is worse than the saved baseline",
)
@click.option(
    "--save-baseline",
    is_flag=True,
    default=False,
    help="Record current metrics as the baseline --strict compares against",
)
def health_cmd(strict: bool, save_baseline: bool) -> None:
    """Run health checks and show a summary report."""
    from vault_rag.engine.health import HealthChecker
    from vault_rag.ingest.scanner import VaultScanner

    config = _get_config()
    scanner = VaultScanner(config)
    notes = scanner.scan()
    ignored_dirs = {".git", ".obsidian", "_trash"}
    existing_paths = {
        path.relative_to(config.vault_path).as_posix()
        for path in config.vault_path.rglob("*")
        if path.is_file()
        and not any(
            part in ignored_dirs
            or any(part.startswith(prefix) for prefix in config.excluded_dir_prefixes)
            for part in path.relative_to(config.vault_path).parts
        )
    }
    checker = HealthChecker(notes, existing_paths=existing_paths)
    report = checker.report()

    click.echo(f"Health report for: {config.vault_path}")
    click.echo(f"  Total notes   : {report['total_notes']}")
    click.echo(f"  Broken links  : {report['broken_link_count']}")
    click.echo(f"  Orphan notes  : {report['orphan_count']} (no inbound link)")
    click.echo(f"  Isolated notes: {report['isolated_count']} (no links at all)")
    click.echo(f"  Untagged notes: {report['untagged_count']}")
    click.echo(f"  Vocab breaches: {report['vocabulary_violation_count']}")
    click.echo(f"  Singleton tags: {report['singleton_tag_count']}")

    if report["vocabulary_violation_count"]:
        click.echo("\nVocabulary violations:")
        for line in report["vocabulary_violations"][:_LIST_CAP]:
            click.echo(f"  {line}")
        _echo_truncated(report["vocabulary_violation_count"])

    if report["broken_link_count"]:
        click.echo("\nBroken links:")
        for bl in report["broken_links"][:_LIST_CAP]:
            click.echo(f"  {bl['source']} -> [[{bl['target']}]]")
        _echo_truncated(report["broken_link_count"])

    if report["orphan_notes"]:
        click.echo("\nOrphan notes:")
        for path in report["orphan_notes"][:_LIST_CAP]:
            click.echo(f"  {path}")
        _echo_truncated(report["orphan_count"])

    current = {name: int(report[name]) for name in _GATED_METRICS}

    if save_baseline:
        saved_to = _write_health_baseline(config, current)
        click.echo(f"\nBaseline saved: {saved_to}")

    if strict:
        baseline = _read_health_baseline(config)
        if baseline is None:
            click.echo("\nNo baseline recorded. Run `vault-rag health --save-baseline` first.")
            raise SystemExit(2)
        regressions = [
            (name, baseline[name], current[name])
            for name in _GATED_METRICS
            if name in baseline and current[name] > baseline[name]
        ]
        if regressions:
            click.echo("\nREGRESSION vs baseline:")
            for name, was, now in regressions:
                click.echo(f"  {name}: {was} -> {now}  (+{now - was})")
            raise SystemExit(1)
        click.echo("\nNo regression against baseline.")


# ---------------------------------------------------------------------------
# link-orphans
# ---------------------------------------------------------------------------


@cli.command("link-orphans")
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Write the links (default: dry run)",
)
def link_orphans_cmd(apply_changes: bool) -> None:
    """Give every orphan an inbound link from its nearest INDEX."""
    from vault_rag.engine import orphan_linker
    from vault_rag.engine.health import HealthChecker
    from vault_rag.ingest.scanner import VaultScanner

    config = _get_config()
    notes = VaultScanner(config).scan()
    by_path = {note.relative_path: note for note in notes}

    orphan_paths = HealthChecker(notes).orphan_notes()
    orphans = [(path, by_path[path].title) for path in orphan_paths]
    index_paths = {p for p in by_path if p.rsplit("/", 1)[-1] == "INDEX.md"}

    links, unplaceable = orphan_linker.plan_links(orphans, index_paths)

    click.echo(f"Orphans: {len(orphans)}")
    click.echo(f"  linkable from an INDEX : {len(links)}")
    click.echo(f"  no INDEX above them    : {len(unplaceable)}")

    per_index: dict[str, int] = {}
    for link in links:
        per_index[link.index_path] = per_index.get(link.index_path, 0) + 1
    click.echo(f"\nINDEX files to update: {len(per_index)}")
    for index_path, count in sorted(per_index.items(), key=lambda kv: (-kv[1], kv[0]))[:_LIST_CAP]:
        click.echo(f"  {count:>3}  {index_path}")
    _echo_truncated(len(per_index))

    if unplaceable:
        click.echo("\nNeeds a human decision (no ancestor INDEX):")
        for path in unplaceable[:_LIST_CAP]:
            click.echo(f"  {path}")
        _echo_truncated(len(unplaceable))

    if not apply_changes:
        click.echo("\nDry run. Re-run with --apply to write.")
        return

    written = orphan_linker.apply_links(config.vault_path, links)
    click.echo(f"\nUpdated {len(written)} INDEX files.")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@cli.command("search")
@click.argument("query")
@click.option("-n", "--num", default=5, show_default=True, help="Number of results")
@click.option("--hybrid", is_flag=True, default=False, help="Hybrid search (vector + keyword)")
@click.option("--filter-tag", multiple=True, help="Filter by tag(s)")
@click.option(
    "--extract", "do_extract", is_flag=True, default=False, help="Extract key content (LLM)"
)
def search_cmd(
    query: str, num: int, hybrid: bool, filter_tag: tuple[str, ...], do_extract: bool
) -> None:
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

    client = None
    if do_extract:
        from anthropic import Anthropic

        client = Anthropic()

    engine = QAEngine(
        vector_store=store,
        embed_fn=embed_fn,
        client=client,
        model=config.qa_model,
    )

    if hybrid or filter_tag:
        results = engine.hybrid_search(
            query,
            n_results=num,
            filter_tags=list(filter_tag) if filter_tag else None,
            keyword=query if hybrid else None,
        )
    else:
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

    if do_extract and results:
        click.echo("--- Extracted Summary ---\n")
        extracted = engine.extract(results, query)
        click.echo(extracted)


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
@click.option("--follow-up", is_flag=True, default=False, help="Include follow-up questions")
@click.option("--save", is_flag=True, default=False, help="Save answer as a wiki page")
def ask_cmd(question: str, follow_up: bool, save: bool) -> None:
    """Query the wiki, synthesize an answer, optionally save as a page."""
    from anthropic import Anthropic

    from vault_rag.engine.indexer import create_openai_embed_fn
    from vault_rag.engine.qa import QAEngine
    from vault_rag.engine.wiki_log import WikiLog
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
    result = engine.answer(question, follow_up=follow_up)

    click.echo("Answer:\n")
    click.echo(result.answer)

    if result.sources:
        click.echo("\nSources:")
        for src in result.sources:
            meta = src["metadata"]
            title = meta.get("title", src["id"])
            path = meta.get("relative_path", src["id"])
            click.echo(f"  - {title} ({path})")

    log = WikiLog(config.vault_path)

    if save:
        from vault_rag.store.note_store import NoteStore

        note_store = NoteStore(config)
        source_titles = [s["metadata"].get("title", s["id"]) for s in result.sources]
        sources_md = "\n".join(f"- [[{t}]]" for t in source_titles)
        content = f"{result.answer}\n\n## Sources\n\n{sources_md}\n"
        tags = ["query", "synthesized"]
        note_path = note_store.create(
            title=question[:80], content=content, folder="Knowledge", tags=tags
        )
        click.echo(f"\nSaved as wiki page: {note_path.relative_to(config.vault_path)}")
        log.log_query(question, saved_as=question[:80])
    else:
        log.log_query(question)


# ---------------------------------------------------------------------------
# clip
# ---------------------------------------------------------------------------


@cli.command("clip")
@click.argument("url")
@click.option("--folder", default="Reference", show_default=True, help="Target folder in vault")
@click.option("--no-compile", is_flag=True, default=False, help="Skip auto-compile and index")
def clip_cmd(url: str, folder: str, no_compile: bool) -> None:
    """Clip a web page into the vault as a markdown note."""
    from vault_rag.ingest.web_clipper import WebClipper

    config = _get_config()
    clipper = WebClipper(vault_path=config.vault_path)
    note_path = clipper.clip(url=url, folder=folder)
    click.echo(f"Clipped: {note_path.relative_to(config.vault_path)}")

    if not no_compile:
        _auto_compile_and_index(note_path, config)


# ---------------------------------------------------------------------------
# ingest-pdf
# ---------------------------------------------------------------------------


@cli.command("ingest-pdf")
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option("--folder", default="Research", show_default=True, help="Target folder in vault")
@click.option("--no-compile", is_flag=True, default=False, help="Skip auto-compile and index")
def ingest_pdf_cmd(pdf_path: Path, folder: str, no_compile: bool) -> None:
    """Ingest a PDF file into the vault as a markdown note."""
    from vault_rag.ingest.pdf_reader import PDFReader

    config = _get_config()
    reader = PDFReader()
    note_path = reader.extract_to_note(
        pdf_path=pdf_path,
        vault_path=config.vault_path,
        folder=folder,
    )
    click.echo(f"Ingested: {note_path.relative_to(config.vault_path)}")

    if not no_compile:
        _auto_compile_and_index(note_path, config)


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------


@cli.command("compile")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--dry-run", is_flag=True, default=False, help="Show results without writing to file")
def compile_cmd(path: Path, dry_run: bool) -> None:
    """Compile a single note: auto-summarize, tag, and suggest links."""
    from anthropic import Anthropic

    from vault_rag.engine.compiler import Compiler
    from vault_rag.ingest.scanner import VaultScanner

    config = _get_config()
    scanner = VaultScanner(config)

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

    if dry_run:
        result = compiler.compile(note)
    else:
        known = {n.title.lower() for n in notes} | {
            Path(n.relative_path).stem.lower() for n in notes
        }
        result = compiler.compile_and_write(note, known_titles=known)
        click.echo("  Written to note.")

    click.echo(f"Summary:\n{result.summary}\n")
    if result.tags:
        click.echo(f"Tags: {', '.join(result.tags)}")
    if result.suggested_links:
        click.echo(f"Suggested links: {', '.join(result.suggested_links)}")
    if result.further_questions:
        click.echo("\nFurther Questions:")
        for i, q in enumerate(result.further_questions, 1):
            click.echo(f"  {i}. {q}")


# ---------------------------------------------------------------------------
# compile-new
# ---------------------------------------------------------------------------


@cli.command("compile-new")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Show what would be compiled without doing it"
)
def compile_new_cmd(dry_run: bool) -> None:
    """Auto-compile all untagged notes."""
    from vault_rag.ingest.scanner import VaultScanner

    config = _get_config()
    scanner = VaultScanner(config)
    notes = scanner.scan()
    untagged = [n for n in notes if not n.tags]
    click.echo(f"Found {len(untagged)} untagged notes")

    if dry_run:
        for n in untagged[:20]:
            click.echo(f"  {n.relative_path}")
        return

    from anthropic import Anthropic

    from vault_rag.engine.compiler import Compiler

    client = Anthropic()
    compiler = Compiler(client=client, model=config.compile_model)
    known = {n.title.lower() for n in notes} | {Path(n.relative_path).stem.lower() for n in notes}

    for i, note in enumerate(untagged):
        result, atoms_created = compiler.compile_and_extract(
            note, known_titles=known, vault_path=config.vault_path
        )
        click.echo(f"  [{i + 1}/{len(untagged)}] {note.relative_path}")
        if result.tags:
            click.echo(f"    Tags: {', '.join(result.tags)}")
        if result.quality_score:
            click.echo(f"    Quality: {result.quality_score} ({result.quality_tier})")
        if atoms_created:
            click.echo(f"    Atoms created: {len(atoms_created)}")


# ---------------------------------------------------------------------------
# relink
# ---------------------------------------------------------------------------


@cli.command("relink")
@click.option("-k", "--top-k", default=5, show_default=True, help="Number of similar notes to link")
@click.option(
    "--threshold",
    default=0.5,
    show_default=True,
    help="Max cosine distance for a link (0=identical, 2=opposite)",
)
@click.option("--dry-run", is_flag=True, default=False, help="Preview without writing")
@click.option("--limit", default=0, show_default=True, help="Only process first N notes (0 = all)")
@click.option(
    "--orphans-only", is_flag=True, default=False, help="Only process health-check orphan notes"
)
def relink_cmd(top_k: int, threshold: float, dry_run: bool, limit: int, orphans_only: bool) -> None:
    """Add `## Related` section via semantic similarity (no LLM)."""
    from vault_rag.engine.health import HealthChecker
    from vault_rag.engine.indexer import create_openai_embed_fn
    from vault_rag.ingest.scanner import VaultScanner
    from vault_rag.store.vector_store import VectorStore

    config = _get_config()
    scanner = VaultScanner(config)
    notes = scanner.scan()
    if orphans_only:
        # isolated_notes(), not orphan_notes(): this flag targets notes with no
        # links at all, which is what a `## Related` section can actually fix.
        # Adding outbound links to an inbound-orphan does not make it reachable.
        orphan_paths = set(HealthChecker(notes).isolated_notes())
        notes = [note for note in notes if note.relative_path in orphan_paths]
    if limit > 0:
        notes = notes[:limit]

    store = VectorStore(persist_dir=config.chroma_path)
    if store.count() == 0:
        click.echo("Vector store is empty. Run `vault-rag index --full` first.")
        return

    embed_fn = create_openai_embed_fn(
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
    )

    # Batch-embed note texts (same format as Indexer._note_to_text).
    def note_to_text(n):
        tags_str = " ".join(f"#{t}" for t in n.tags) if n.tags else ""
        parts = [n.title]
        if tags_str:
            parts.append(tags_str)
        parts.append(n.content[:2000])
        return "\n".join(parts)

    total = len(notes)
    written = 0
    skipped_existing = 0
    skipped_no_match = 0
    batch_size = 50

    for start in range(0, total, batch_size):
        batch = notes[start : start + batch_size]
        texts = [note_to_text(n) for n in batch]
        embeddings = embed_fn(texts)

        for note, emb in zip(batch, embeddings):
            try:
                raw = note.path.read_bytes()
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                skipped_no_match += 1
                continue
            if "## Related" in content:
                skipped_existing += 1
                continue

            result = store.query(query_embeddings=[emb], n_results=top_k + 1)
            ids = result.get("ids", [[]])[0]
            distances = result.get("distances", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]

            titles: list[str] = []
            for rid, dist, meta in zip(ids, distances, metadatas):
                if rid == note.relative_path:
                    continue
                if dist > threshold:
                    continue
                title = (meta or {}).get("title") or Path(rid).stem
                if title and title != note.title:
                    titles.append(title)
                if len(titles) >= top_k:
                    break

            if not titles:
                skipped_no_match += 1
                continue

            if dry_run:
                click.echo(f"  [{start + batch.index(note) + 1}/{total}] {note.relative_path}")
                for t in titles:
                    click.echo(f"    -> {t}")
                continue

            links_md = "\n".join(f"- [[{t}]]" for t in titles)
            suffix = f"\n\n## Related\n\n{links_md}\n".encode()
            note.path.write_bytes(raw.rstrip() + suffix)
            written += 1

    click.echo("")
    click.echo(f"Total notes         : {total}")
    click.echo(f"Written (new links) : {written}")
    click.echo(f"Skipped (had ##Related): {skipped_existing}")
    click.echo(f"Skipped (no match)  : {skipped_no_match}")


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


@cli.command("audit")
def audit_cmd() -> None:
    """Show notes by quality tier."""
    import re

    from vault_rag.ingest.scanner import VaultScanner

    cfg = _get_config()
    scanner = VaultScanner(cfg)
    notes = scanner.scan()

    high: list[str] = []
    medium: list[str] = []
    low: list[str] = []
    unscored: list[str] = []

    _score_re = re.compile(r"^quality_score:\s*(\d+)", re.MULTILINE)

    for note in notes:
        try:
            raw = note.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            unscored.append(note.relative_path)
            continue

        match = _score_re.search(raw)
        if not match:
            unscored.append(note.relative_path)
            continue

        score = int(match.group(1))
        if score >= 80:
            high.append(note.relative_path)
        elif score >= 40:
            medium.append(note.relative_path)
        else:
            low.append(note.relative_path)

    click.echo(f"HIGH   (80-100): {len(high)}")
    click.echo(f"MEDIUM (40-79):  {len(medium)}")
    click.echo(f"LOW    (0-39):   {len(low)}")
    click.echo(f"Unscored:        {len(unscored)}")

    if low:
        click.echo("\nLOW quality (deletion candidates):")
        for n in low[:10]:
            click.echo(f"  {n}")


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------


@cli.command("watch")
def watch_cmd() -> None:
    """Watch vault for .md changes and auto-compile + index."""
    from vault_rag.engine.watcher import VaultWatcher

    config = _get_config()
    click.echo(f"Watching {config.vault_path} ... (Ctrl+C to stop)")

    def on_change(path: Path) -> None:
        click.echo(f"\nChange detected: {path.name}")
        try:
            _auto_compile_and_index(path, config)
        except Exception as exc:  # noqa: BLE001
            click.echo(f"  Error: {exc}", err=True)

    watcher = VaultWatcher(config=config, on_change=on_change)
    watcher.start()


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@cli.group("generate")
def generate_group() -> None:
    """Generate outputs (slides, report, summary) from vault knowledge."""


@generate_group.command("slides")
@click.argument("topic")
@click.option("-n", "--num", default=10, show_default=True, help="Number of context notes")
def generate_slides_cmd(topic: str, num: int) -> None:
    """Generate a markdown slide deck on a topic."""
    from anthropic import Anthropic

    from vault_rag.engine.generator import Generator
    from vault_rag.engine.indexer import create_openai_embed_fn
    from vault_rag.store.vector_store import VectorStore

    config = _get_config()
    store = VectorStore(persist_dir=config.chroma_path)

    if store.count() == 0:
        click.echo("Vector store is empty. Run `vault-rag index` first.")
        return

    embed_fn = create_openai_embed_fn(config.embedding_model, config.embedding_dimensions)
    client = Anthropic()
    gen = Generator(client=client, model=config.qa_model, embed_fn=embed_fn, vector_store=store)
    result = gen.generate_slides(topic, n_context=num)
    path = gen.save(result, config.output_path)
    click.echo(f"Slides saved: {path}")
    click.echo(f"\n{result.content}")


@generate_group.command("report")
@click.argument("topic")
@click.option("-n", "--num", default=10, show_default=True, help="Number of context notes")
def generate_report_cmd(topic: str, num: int) -> None:
    """Generate a comprehensive report on a topic."""
    from anthropic import Anthropic

    from vault_rag.engine.generator import Generator
    from vault_rag.engine.indexer import create_openai_embed_fn
    from vault_rag.store.vector_store import VectorStore

    config = _get_config()
    store = VectorStore(persist_dir=config.chroma_path)

    if store.count() == 0:
        click.echo("Vector store is empty. Run `vault-rag index` first.")
        return

    embed_fn = create_openai_embed_fn(config.embedding_model, config.embedding_dimensions)
    client = Anthropic()
    gen = Generator(client=client, model=config.qa_model, embed_fn=embed_fn, vector_store=store)
    result = gen.generate_report(topic, n_context=num)
    path = gen.save(result, config.output_path)
    click.echo(f"Report saved: {path}")
    click.echo(f"\n{result.content}")


@generate_group.command("summary")
@click.argument("topic")
@click.option("-n", "--num", default=5, show_default=True, help="Number of context notes")
def generate_summary_cmd(topic: str, num: int) -> None:
    """Generate a concise executive briefing on a topic."""
    from anthropic import Anthropic

    from vault_rag.engine.generator import Generator
    from vault_rag.engine.indexer import create_openai_embed_fn
    from vault_rag.store.vector_store import VectorStore

    config = _get_config()
    store = VectorStore(persist_dir=config.chroma_path)

    if store.count() == 0:
        click.echo("Vector store is empty. Run `vault-rag index` first.")
        return

    embed_fn = create_openai_embed_fn(config.embedding_model, config.embedding_dimensions)
    client = Anthropic()
    gen = Generator(client=client, model=config.qa_model, embed_fn=embed_fn, vector_store=store)
    result = gen.generate_summary(topic, n_context=num)
    path = gen.save(result, config.output_path)
    click.echo(f"Summary saved: {path}")
    click.echo(f"\n{result.content}")


@generate_group.command("chart")
@click.argument("topic")
@click.option("-n", "--num", default=10, show_default=True, help="Number of context notes")
def generate_chart_cmd(topic: str, num: int) -> None:
    """Generate a Mermaid chart/diagram on a topic."""
    from anthropic import Anthropic

    from vault_rag.engine.generator import Generator
    from vault_rag.engine.indexer import create_openai_embed_fn
    from vault_rag.store.vector_store import VectorStore

    config = _get_config()
    store = VectorStore(persist_dir=config.chroma_path)

    if store.count() == 0:
        click.echo("Vector store is empty. Run `vault-rag index` first.")
        return

    embed_fn = create_openai_embed_fn(config.embedding_model, config.embedding_dimensions)
    client = Anthropic()
    gen = Generator(client=client, model=config.qa_model, embed_fn=embed_fn, vector_store=store)
    result = gen.generate_chart(topic, n_context=num)
    path = gen.save(result, config.output_path)
    click.echo(f"Chart saved: {path}")
    click.echo(f"\n{result.content}")


# ---------------------------------------------------------------------------
# ingest-image
# ---------------------------------------------------------------------------


@cli.command("ingest-image")
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option("--folder", default="Reference/design-ref", show_default=True, help="Target folder")
@click.option("--no-index", is_flag=True, default=False, help="Skip auto-index after saving")
def ingest_image_cmd(image_path: Path, folder: str, no_index: bool) -> None:
    """Analyze an image with Claude Vision and save as a descriptive note."""
    from anthropic import Anthropic

    from vault_rag.ingest.image_analyzer import ImageAnalyzer

    config = _get_config()
    client = Anthropic()
    analyzer = ImageAnalyzer(
        client=client, model=config.compile_model, vault_path=config.vault_path
    )

    click.echo(f"Analyzing: {image_path.name}")
    result, note_path = analyzer.analyze_and_save(image_path, folder=folder)

    click.echo(f"  Title: {result.title}")
    click.echo(f"  Style: {result.style}")
    click.echo(f"  Tags: {', '.join(result.tags)}")
    click.echo(f"  Note: {note_path.relative_to(config.vault_path)}")

    if not no_index:
        from vault_rag.engine.indexer import Indexer, create_openai_embed_fn
        from vault_rag.ingest.scanner import VaultScanner
        from vault_rag.store.vector_store import VectorStore

        vs = VectorStore(persist_dir=config.chroma_path)
        embed_fn = create_openai_embed_fn(config.embedding_model, config.embedding_dimensions)
        scanner = VaultScanner(config)
        notes = scanner.scan()
        target = next((n for n in notes if n.path.resolve() == note_path.resolve()), None)
        if target:
            indexer = Indexer(config=config, vector_store=vs, embed_fn=embed_fn)
            indexer.index([target])
            click.echo("  Indexed.")


# ---------------------------------------------------------------------------
# reindex (generate index.md)
# ---------------------------------------------------------------------------


@cli.command("reindex")
def reindex_cmd() -> None:
    """Regenerate index.md - categorized catalog of all wiki pages."""
    from vault_rag.engine.wiki_index import WikiIndex
    from vault_rag.ingest.scanner import VaultScanner

    config = _get_config()
    scanner = VaultScanner(config)
    notes = scanner.scan()

    wiki_index = WikiIndex(config.vault_path)
    path = wiki_index.generate(notes)
    click.echo(f"Index generated: {path.name} ({len(notes)} pages)")


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------


@cli.command("lint")
def lint_cmd() -> None:
    """LLM-powered wiki health check - contradictions, stale claims, data gaps."""
    from anthropic import Anthropic

    from vault_rag.engine.health import HealthChecker
    from vault_rag.engine.wiki_lint import WikiLinter
    from vault_rag.engine.wiki_log import WikiLog
    from vault_rag.ingest.scanner import VaultScanner

    config = _get_config()
    scanner = VaultScanner(config)
    notes = scanner.scan()

    # Structural checks first
    checker = HealthChecker(notes)
    health = checker.report()

    click.echo(f"Structural checks ({len(notes)} notes):")
    click.echo(f"  Broken links : {health['broken_link_count']}")
    click.echo(f"  Orphan notes : {health['orphan_count']} (no inbound link)")
    click.echo(f"  Isolated     : {health['isolated_count']} (no links at all)")
    click.echo(f"  Untagged     : {health['untagged_count']}")

    # LLM semantic checks over a rotating window, so repeated runs sweep the
    # whole vault instead of re-auditing the same head slice every time.
    offset = _read_lint_cursor(config)
    click.echo("\nRunning LLM analysis...")
    client = Anthropic()
    linter = WikiLinter(client=client, model=config.compile_model)
    result = linter.lint(notes, health_report=health, offset=offset)
    _write_lint_cursor(config, result.next_offset)
    click.echo(f"  Examined {result.pages_examined} pages from offset {offset}")

    if result.contradictions:
        click.echo(f"\nContradictions ({len(result.contradictions)}):")
        for c in result.contradictions:
            click.echo(f"  {c.get('page_a', '?')} vs {c.get('page_b', '?')}: {c.get('issue', '')}")

    if result.stale_claims:
        click.echo(f"\nStale claims ({len(result.stale_claims)}):")
        for s in result.stale_claims:
            click.echo(f"  [{s.get('page', '?')}] {s.get('claim', '')}")

    if result.missing_pages:
        click.echo(f"\nMissing pages ({len(result.missing_pages)}):")
        for m in result.missing_pages:
            click.echo(f"  - {m}")

    if result.data_gaps:
        click.echo(f"\nData gaps ({len(result.data_gaps)}):")
        for d in result.data_gaps:
            click.echo(f"  - {d}")

    click.echo(f"\nTotal issues: {result.total_issues}")

    log = WikiLog(config.vault_path)
    log.log_lint(issues_found=result.total_issues)
