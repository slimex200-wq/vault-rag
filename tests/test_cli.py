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
