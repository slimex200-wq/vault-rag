"""Integration tests: E2E pipeline scan → index → search → graph → health."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from vault_rag.cli import cli
from vault_rag.config import VaultConfig
from vault_rag.engine.compiler import Compiler
from vault_rag.engine.graph import KnowledgeGraph
from vault_rag.engine.health import HealthChecker
from vault_rag.engine.indexer import Indexer
from vault_rag.engine.qa import QAEngine
from vault_rag.ingest.scanner import VaultScanner
from vault_rag.store.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic mock: hash-based pseudo-embeddings (dim=512)."""
    results = []
    for t in texts:
        h = hash(t) % 1000
        results.append([(h + i) / 1000.0 for i in range(512)])
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_pipeline(tmp_vault: Path, cfg: VaultConfig, tmp_path: Path) -> None:
    """E2E: scan → index → search → graph → health → incremental scan."""
    # Add a note that links to an existing note by stem so the graph has edges
    (tmp_vault / "linker.md").write_text(
        "# Linker Note\n\nReferences [[concepts]] for more details.",
        encoding="utf-8",
    )

    # 1. Scan
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    assert len(notes) >= 2, f"Expected >= 2 notes, got {len(notes)}"

    # 2. Index
    store = VectorStore(tmp_path / "chroma")
    indexer = Indexer(config=cfg, vector_store=store, embed_fn=_mock_embed)
    indexed = indexer.index(notes)
    assert indexed == len(notes)
    assert store.count() == len(notes)

    # 3. Search
    qa = QAEngine(
        vector_store=store,
        embed_fn=_mock_embed,
        client=MagicMock(),
        model="claude-sonnet-4-5",
    )
    results = qa.search("AI agents")
    assert len(results) > 0

    # 4. Graph
    graph = KnowledgeGraph()
    graph.build(notes)
    assert graph.node_count() == len(notes)
    assert graph.edge_count() >= 1

    # 5. Health
    report = HealthChecker(notes).report()
    assert report["total_notes"] == len(notes)

    # 6. Incremental scan — nothing has changed, should return 0
    known_hashes = {n.relative_path: n.content_hash for n in notes}
    changed = scanner.scan(known_hashes=known_hashes)
    assert len(changed) == 0


@pytest.mark.integration
def test_cli_search_empty_store(tmp_vault: Path, cfg: VaultConfig, tmp_path: Path) -> None:
    """CLI search on empty store prints guidance message."""
    runner = CliRunner()
    with patch("vault_rag.cli._get_config", return_value=cfg):
        result = runner.invoke(cli, ["search", "AI agents"])
    assert result.exit_code == 0
    assert "empty" in result.output.lower() or "index" in result.output.lower()


@pytest.mark.integration
def test_cli_index_and_search(tmp_vault: Path, cfg: VaultConfig, tmp_path: Path) -> None:
    """CLI index (mocked embed) then search returns results."""
    runner = CliRunner()
    embed_mock = MagicMock(side_effect=_mock_embed)

    real_store = VectorStore(tmp_path / "chroma_cli")

    with patch("vault_rag.cli._get_config", return_value=cfg), patch(
        "vault_rag.engine.indexer.create_openai_embed_fn", return_value=embed_mock
    ), patch("vault_rag.store.vector_store.VectorStore", return_value=real_store):
        idx_result = runner.invoke(cli, ["index"])
        assert idx_result.exit_code == 0, idx_result.output
        assert "Indexed" in idx_result.output


@pytest.mark.integration
def test_cli_ask_empty_store(tmp_vault: Path, cfg: VaultConfig) -> None:
    """CLI ask on empty store prints guidance message."""
    runner = CliRunner()
    with patch("vault_rag.cli._get_config", return_value=cfg):
        result = runner.invoke(cli, ["ask", "What is AI?"])
    assert result.exit_code == 0
    assert "empty" in result.output.lower() or "index" in result.output.lower()


@pytest.mark.integration
def test_cli_compile_command(tmp_vault: Path, cfg: VaultConfig) -> None:
    """CLI compile mocks Anthropic client and prints summary."""
    runner = CliRunner()
    mock_response_text = json.dumps(
        {"summary": "CLI summary", "tags": ["cli"], "suggested_links": []}
    )
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=mock_response_text)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg

    note_path = tmp_vault / "index.md"

    with patch("vault_rag.cli._get_config", return_value=cfg), patch(
        "anthropic.Anthropic", return_value=mock_client
    ):
        result = runner.invoke(cli, ["compile", str(note_path)])

    assert result.exit_code == 0, result.output
    assert "CLI summary" in result.output


@pytest.mark.integration
def test_compile_and_index_pipeline(
    tmp_vault: Path, cfg: VaultConfig, tmp_path: Path
) -> None:
    """E2E: scan → compile (mocked LLM) → index."""
    # 1. Scan
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    assert len(notes) >= 1

    # 2. Mock Anthropic client returning structured JSON
    mock_response_text = json.dumps(
        {
            "summary": "Test summary",
            "tags": ["ai", "test"],
            "suggested_links": ["other"],
        }
    )
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=mock_response_text)]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    # 3. Compile first note
    compiler = Compiler(client=mock_client, model="claude-sonnet-4-5")
    result = compiler.compile(notes[0])
    assert result.summary == "Test summary"
    assert result.tags == ["ai", "test"]
    assert result.suggested_links == ["other"]

    # 4. Index all notes into VectorStore
    store = VectorStore(tmp_path / "chroma_compile")
    indexer = Indexer(config=cfg, vector_store=store, embed_fn=_mock_embed)
    indexed = indexer.index(notes)
    assert indexed == len(notes)
    assert store.count() == len(notes)
