"""CLI integration tests using click.testing.CliRunner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from vault_rag.cli import cli
from vault_rag.config import VaultConfig


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def tmp_config(tmp_path: Path) -> VaultConfig:
    """VaultConfig pointed at a temporary directory."""
    return VaultConfig(vault_path=tmp_path)


# ---------------------------------------------------------------------------
# 1. Help
# ---------------------------------------------------------------------------


def test_cli_has_help(runner: CliRunner) -> None:
    """--help exits cleanly and shows usage info."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


# ---------------------------------------------------------------------------
# 2. scan command
# ---------------------------------------------------------------------------


def test_scan_command(runner: CliRunner, tmp_path: Path) -> None:
    """scan counts .md files and lists them."""
    config = VaultConfig(vault_path=tmp_path)
    (tmp_path / "note1.md").write_text("# Hello\n\nContent.", encoding="utf-8")

    with patch("vault_rag.cli._get_config", return_value=config):
        result = runner.invoke(cli, ["scan"])

    assert result.exit_code == 0, result.output
    assert "1" in result.output


# ---------------------------------------------------------------------------
# 3. health command
# ---------------------------------------------------------------------------


def test_health_command(runner: CliRunner, tmp_path: Path) -> None:
    """health prints a report including orphan count."""
    config = VaultConfig(vault_path=tmp_path)
    (tmp_path / "orphan.md").write_text("# Orphan\n\nNo links, no tags.", encoding="utf-8")

    with patch("vault_rag.cli._get_config", return_value=config):
        result = runner.invoke(cli, ["health"])

    assert result.exit_code == 0, result.output
    # Report should contain total/orphan/broken counts
    output_lower = result.output.lower()
    assert any(word in output_lower for word in ["total", "orphan", "broken", "health"])


# ---------------------------------------------------------------------------
# 4. compile-new --dry-run
# ---------------------------------------------------------------------------


def test_compile_new_dry_run(runner: CliRunner, tmp_path: Path) -> None:
    """compile-new --dry-run lists untagged notes without calling the API."""
    config = VaultConfig(vault_path=tmp_path)
    (tmp_path / "untagged.md").write_text("# No Tags\n\nContent here.\n", encoding="utf-8")

    with patch("vault_rag.cli._get_config", return_value=config):
        result = runner.invoke(cli, ["compile-new", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "1" in result.output


# ---------------------------------------------------------------------------
# 5. compile --dry-run
# ---------------------------------------------------------------------------


def test_compile_dry_run(runner: CliRunner, tmp_path: Path) -> None:
    """compile --dry-run calls Anthropic but does not write to file."""
    config = VaultConfig(vault_path=tmp_path)
    note_file = tmp_path / "test-note.md"
    note_file.write_text("# Test\n\nSome content.\n", encoding="utf-8")

    mock_msg = MagicMock()
    mock_msg.content = [
        MagicMock(text='{"summary": "test summary", "tags": ["t"], "suggested_links": []}')
    ]

    with patch("vault_rag.cli._get_config", return_value=config):
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.return_value = mock_msg
            result = runner.invoke(cli, ["compile", "--dry-run", str(note_file)])

    assert result.exit_code == 0, result.output
    assert "test summary" in result.output


# ---------------------------------------------------------------------------
# 6. search command — empty store
# ---------------------------------------------------------------------------


def test_search_empty_store(runner: CliRunner, tmp_path: Path) -> None:
    """search on empty store prints helpful message."""
    config = VaultConfig(vault_path=tmp_path)

    with patch("vault_rag.cli._get_config", return_value=config):
        result = runner.invoke(cli, ["search", "some query"])

    assert result.exit_code == 0, result.output
    assert "empty" in result.output.lower() or "index" in result.output.lower()


# ---------------------------------------------------------------------------
# 7. ask command — empty store
# ---------------------------------------------------------------------------


def test_ask_empty_store(runner: CliRunner, tmp_path: Path) -> None:
    """ask on empty store prints helpful message."""
    config = VaultConfig(vault_path=tmp_path)

    with patch("vault_rag.cli._get_config", return_value=config):
        result = runner.invoke(cli, ["ask", "what is this?"])

    assert result.exit_code == 0, result.output
    assert "empty" in result.output.lower() or "index" in result.output.lower()


# ---------------------------------------------------------------------------
# 8. compile-new — no untagged notes
# ---------------------------------------------------------------------------


def test_compile_new_dry_run_no_untagged(runner: CliRunner, tmp_path: Path) -> None:
    """compile-new --dry-run on vault with all tagged notes shows 0."""
    config = VaultConfig(vault_path=tmp_path)
    (tmp_path / "tagged.md").write_text(
        "---\ntags: [existing]\n---\n# Tagged\n\nContent.\n", encoding="utf-8"
    )

    with patch("vault_rag.cli._get_config", return_value=config):
        result = runner.invoke(cli, ["compile-new", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "0" in result.output


# ---------------------------------------------------------------------------
# 9. compile — note not found in vault
# ---------------------------------------------------------------------------


def test_compile_note_not_found(runner: CliRunner, tmp_path: Path) -> None:
    """compile with a file outside vault scan yields exit code 1."""
    config = VaultConfig(vault_path=tmp_path)
    # Create a note in a subdirectory that is excluded
    excluded_dir = tmp_path / ".obsidian"
    excluded_dir.mkdir()
    note_file = excluded_dir / "hidden.md"
    note_file.write_text("# Hidden\n\nContent.\n", encoding="utf-8")

    with patch("vault_rag.cli._get_config", return_value=config):
        result = runner.invoke(cli, ["compile", "--dry-run", str(note_file)])

    assert result.exit_code != 0 or "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# 10. health command — with broken links
# ---------------------------------------------------------------------------


def test_health_with_broken_links(runner: CliRunner, tmp_path: Path) -> None:
    """health reports broken links when a note links to nonexistent target."""
    config = VaultConfig(vault_path=tmp_path)
    (tmp_path / "linker.md").write_text(
        "# Linker\n\nSee [[nonexistent-target]].\n", encoding="utf-8"
    )

    with patch("vault_rag.cli._get_config", return_value=config):
        result = runner.invoke(cli, ["health"])

    assert result.exit_code == 0, result.output
    assert "1" in result.output  # broken link count


# ---------------------------------------------------------------------------
# 11. search command — with results
# ---------------------------------------------------------------------------


def test_search_with_results(runner: CliRunner, tmp_path: Path) -> None:
    """search returns results when store has data."""
    from vault_rag.store.vector_store import VectorStore

    config = VaultConfig(vault_path=tmp_path)
    store = VectorStore(persist_dir=config.chroma_path)
    store.upsert(
        ids=["note-1"],
        embeddings=[[0.1] * 1536],
        documents=["Test document content"],
        metadatas=[{"title": "Test Note", "relative_path": "test.md", "tags": "test"}],
    )

    mock_embed_fn = MagicMock(return_value=[[0.1] * 1536])

    with patch("vault_rag.cli._get_config", return_value=config):
        with patch("vault_rag.engine.indexer.create_openai_embed_fn", return_value=mock_embed_fn):
            result = runner.invoke(cli, ["search", "test query"])

    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# 12. index command
# ---------------------------------------------------------------------------


def test_index_command(runner: CliRunner, tmp_path: Path) -> None:
    """index command embeds notes into the vector store."""
    config = VaultConfig(vault_path=tmp_path)
    (tmp_path / "note.md").write_text("# Note\n\nContent.\n", encoding="utf-8")

    mock_embed_fn = MagicMock(return_value=[[0.1] * 1536])

    with patch("vault_rag.cli._get_config", return_value=config):
        with patch("vault_rag.engine.indexer.create_openai_embed_fn", return_value=mock_embed_fn):
            result = runner.invoke(cli, ["index"])

    assert result.exit_code == 0, result.output
    assert "1" in result.output


# ---------------------------------------------------------------------------
# 13. watch command
# ---------------------------------------------------------------------------


def test_watch_command_help(runner: CliRunner) -> None:
    """watch --help shows usage info."""
    result = runner.invoke(cli, ["watch", "--help"])
    assert result.exit_code == 0
    assert "Watch" in result.output


# ---------------------------------------------------------------------------
# 14. generate commands
# ---------------------------------------------------------------------------


def test_generate_group_help(runner: CliRunner) -> None:
    """generate --help shows subcommands."""
    result = runner.invoke(cli, ["generate", "--help"])
    assert result.exit_code == 0
    assert "slides" in result.output
    assert "report" in result.output
    assert "summary" in result.output


def test_generate_slides_empty_store(runner: CliRunner, tmp_path: Path) -> None:
    """generate slides on empty store shows message."""
    config = VaultConfig(vault_path=tmp_path / "vault")

    with patch("vault_rag.cli._get_config", return_value=config):
        result = runner.invoke(cli, ["generate", "slides", "test topic"])

    assert result.exit_code == 0
    assert "empty" in result.output.lower() or "index" in result.output.lower()


def test_generate_report_empty_store(runner: CliRunner, tmp_path: Path) -> None:
    """generate report on empty store shows message."""
    config = VaultConfig(vault_path=tmp_path / "vault")

    with patch("vault_rag.cli._get_config", return_value=config):
        result = runner.invoke(cli, ["generate", "report", "test topic"])

    assert result.exit_code == 0
    assert "empty" in result.output.lower() or "index" in result.output.lower()


# ---------------------------------------------------------------------------
# 15. search with options
# ---------------------------------------------------------------------------


def test_search_hybrid_flag_help(runner: CliRunner) -> None:
    """search --help shows --hybrid and --filter-tag options."""
    result = runner.invoke(cli, ["search", "--help"])
    assert result.exit_code == 0
    assert "--hybrid" in result.output
    assert "--filter-tag" in result.output
    assert "--extract" in result.output


# ---------------------------------------------------------------------------
# 16. ask with --follow-up
# ---------------------------------------------------------------------------


def test_ask_follow_up_flag_help(runner: CliRunner) -> None:
    """ask --help shows --follow-up option."""
    result = runner.invoke(cli, ["ask", "--help"])
    assert result.exit_code == 0
    assert "--follow-up" in result.output


# ---------------------------------------------------------------------------
# 17. generate with data
# ---------------------------------------------------------------------------


def test_generate_slides_with_data(runner: CliRunner, tmp_path: Path) -> None:
    """generate slides with populated store produces output."""
    from vault_rag.store.vector_store import VectorStore

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    config = VaultConfig(vault_path=vault_dir)
    store = VectorStore(persist_dir=config.chroma_path)
    store.upsert(
        ids=["n1"],
        embeddings=[[0.1] * 512],
        documents=["AI agents content"],
        metadatas=[{"title": "AI", "tags": "ai", "relative_path": "n1.md"}],
    )

    mock_embed = MagicMock(return_value=[[0.1] * 512])
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text="## Slide 1\n\n- Point")]

    with patch("vault_rag.cli._get_config", return_value=config):
        with patch("vault_rag.engine.indexer.create_openai_embed_fn", return_value=mock_embed):
            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_anthropic.return_value.messages.create.return_value = mock_msg
                result = runner.invoke(cli, ["generate", "slides", "AI"])

    assert result.exit_code == 0, result.output
    assert "Slide" in result.output


def test_generate_report_with_data(runner: CliRunner, tmp_path: Path) -> None:
    """generate report with populated store produces output."""
    from vault_rag.store.vector_store import VectorStore

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    config = VaultConfig(vault_path=vault_dir)
    store = VectorStore(persist_dir=config.chroma_path)
    store.upsert(
        ids=["n1"],
        embeddings=[[0.1] * 512],
        documents=["Report content"],
        metadatas=[{"title": "Note", "tags": "t", "relative_path": "n1.md"}],
    )

    mock_embed = MagicMock(return_value=[[0.1] * 512])
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text="## Executive Summary\n\nReport body.")]

    with patch("vault_rag.cli._get_config", return_value=config):
        with patch("vault_rag.engine.indexer.create_openai_embed_fn", return_value=mock_embed):
            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_anthropic.return_value.messages.create.return_value = mock_msg
                result = runner.invoke(cli, ["generate", "report", "topic"])

    assert result.exit_code == 0, result.output
    assert "Executive Summary" in result.output


def test_generate_summary_with_data(runner: CliRunner, tmp_path: Path) -> None:
    """generate summary with populated store produces output."""
    from vault_rag.store.vector_store import VectorStore

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    config = VaultConfig(vault_path=vault_dir)
    store = VectorStore(persist_dir=config.chroma_path)
    store.upsert(
        ids=["n1"],
        embeddings=[[0.1] * 512],
        documents=["Summary content"],
        metadatas=[{"title": "Note", "tags": "t", "relative_path": "n1.md"}],
    )

    mock_embed = MagicMock(return_value=[[0.1] * 512])
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text="- Key point 1\n- Key point 2")]

    with patch("vault_rag.cli._get_config", return_value=config):
        with patch("vault_rag.engine.indexer.create_openai_embed_fn", return_value=mock_embed):
            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_anthropic.return_value.messages.create.return_value = mock_msg
                result = runner.invoke(cli, ["generate", "summary", "topic"])

    assert result.exit_code == 0, result.output
    assert "Key point" in result.output


def test_generate_summary_empty_store(runner: CliRunner, tmp_path: Path) -> None:
    """generate summary on empty store shows message."""
    config = VaultConfig(vault_path=tmp_path / "vault")

    with patch("vault_rag.cli._get_config", return_value=config):
        result = runner.invoke(cli, ["generate", "summary", "test topic"])

    assert result.exit_code == 0
    assert "empty" in result.output.lower() or "index" in result.output.lower()


# ---------------------------------------------------------------------------
# 18. audit command
# ---------------------------------------------------------------------------


def test_audit_command(runner: CliRunner, tmp_path: Path) -> None:
    """audit shows quality tier counts."""
    config = VaultConfig(vault_path=tmp_path)
    (tmp_path / "high.md").write_text(
        "---\nquality_score: 90\n---\n# High\n\nContent.\n", encoding="utf-8"
    )
    (tmp_path / "low.md").write_text(
        "---\nquality_score: 20\n---\n# Low\n\nContent.\n", encoding="utf-8"
    )
    (tmp_path / "none.md").write_text("# No Score\n\nContent.\n", encoding="utf-8")

    with patch("vault_rag.cli._get_config", return_value=config):
        result = runner.invoke(cli, ["audit"])

    assert result.exit_code == 0, result.output
    assert "HIGH" in result.output
    assert "LOW" in result.output


# ---------------------------------------------------------------------------
# 19. compile with further_questions output
# ---------------------------------------------------------------------------


def test_compile_dry_run_shows_further_questions(runner: CliRunner, tmp_path: Path) -> None:
    """compile --dry-run shows further_questions."""
    config = VaultConfig(vault_path=tmp_path)
    note_file = tmp_path / "fq.md"
    note_file.write_text("# FQ Test\n\nSome content.\n", encoding="utf-8")

    mock_msg = MagicMock()
    mock_msg.content = [
        MagicMock(
            text='{"summary": "FQ summary", "tags": ["t"], "suggested_links": [], '
            '"further_questions": ["Q1?", "Q2?"]}'
        )
    ]

    with patch("vault_rag.cli._get_config", return_value=config):
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.return_value = mock_msg
            result = runner.invoke(cli, ["compile", "--dry-run", str(note_file)])

    assert result.exit_code == 0, result.output
    assert "Further Questions" in result.output
    assert "Q1?" in result.output


# ---------------------------------------------------------------------------
# 20. generate chart
# ---------------------------------------------------------------------------


def test_generate_chart_help(runner: CliRunner) -> None:
    """generate --help shows chart subcommand."""
    result = runner.invoke(cli, ["generate", "--help"])
    assert result.exit_code == 0
    assert "chart" in result.output


def test_generate_chart_empty_store(runner: CliRunner, tmp_path: Path) -> None:
    """generate chart on empty store shows message."""
    config = VaultConfig(vault_path=tmp_path / "vault")

    with patch("vault_rag.cli._get_config", return_value=config):
        result = runner.invoke(cli, ["generate", "chart", "test topic"])

    assert result.exit_code == 0
    assert "empty" in result.output.lower() or "index" in result.output.lower()


# ---------------------------------------------------------------------------
# 21. ingest-image
# ---------------------------------------------------------------------------


def test_ingest_image_help(runner: CliRunner) -> None:
    """ingest-image --help shows usage."""
    result = runner.invoke(cli, ["ingest-image", "--help"])
    assert result.exit_code == 0
    assert "Analyze" in result.output


# ---------------------------------------------------------------------------
# 22. reindex
# ---------------------------------------------------------------------------


def test_reindex_command(runner: CliRunner, tmp_path: Path) -> None:
    """reindex generates index.md."""
    config = VaultConfig(vault_path=tmp_path)
    (tmp_path / "note.md").write_text("# Test\n\nContent.\n", encoding="utf-8")

    with patch("vault_rag.cli._get_config", return_value=config):
        result = runner.invoke(cli, ["reindex"])

    assert result.exit_code == 0, result.output
    assert "Index generated" in result.output
    assert (tmp_path / "index.md").exists()


# ---------------------------------------------------------------------------
# 23. lint
# ---------------------------------------------------------------------------


def test_lint_command_help(runner: CliRunner) -> None:
    """lint --help shows usage."""
    result = runner.invoke(cli, ["lint", "--help"])
    assert result.exit_code == 0
    assert "health check" in result.output.lower() or "lint" in result.output.lower()


# ---------------------------------------------------------------------------
# 24. ask --save
# ---------------------------------------------------------------------------


def test_ask_save_flag_help(runner: CliRunner) -> None:
    """ask --help shows --save option."""
    result = runner.invoke(cli, ["ask", "--help"])
    assert result.exit_code == 0
    assert "--save" in result.output
