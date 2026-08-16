"""Tests for incremental indexing: only changed notes get re-embedded."""

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
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    (v / "a.md").write_text("# A\n\nAlpha.\n", encoding="utf-8")
    (v / "b.md").write_text("# B\n\nBeta.\n", encoding="utf-8")
    return v


def _embed(texts: list[str]) -> list[list[float]]:
    return [[0.1, 0.2, 0.3] for _ in texts]


def _run(runner: CliRunner, vault: Path, store: MagicMock, args: list[str]):
    config = VaultConfig(vault_path=vault)
    with (
        patch("vault_rag.cli._get_config", return_value=config),
        patch("vault_rag.store.vector_store.VectorStore", return_value=store),
        patch("vault_rag.engine.indexer.create_embed_fn", return_value=_embed),
    ):
        return runner.invoke(cli, args)


def test_unchanged_vault_embeds_nothing(runner: CliRunner, vault: Path) -> None:
    """The whole point: a second run must not re-embed an unchanged vault."""
    store = MagicMock()
    store.content_hashes.return_value = {}
    first = _run(runner, vault, store, ["index"])
    assert first.exit_code == 0, first.output
    indexed_paths = {i for c in store.upsert.call_args_list for i in c.kwargs["ids"]}
    assert indexed_paths == {"a.md", "b.md"}

    # Feed back the hashes the first run would have stored.
    from vault_rag.ingest.scanner import VaultScanner

    hashes = {
        n.relative_path: n.content_hash for n in VaultScanner(VaultConfig(vault_path=vault)).scan()
    }
    store.reset_mock()
    store.content_hashes.return_value = hashes

    second = _run(runner, vault, store, ["index"])

    assert second.exit_code == 0, second.output
    assert "already current" in second.output
    store.upsert.assert_not_called()


def test_only_the_changed_note_is_reembedded(runner: CliRunner, vault: Path) -> None:
    from vault_rag.ingest.scanner import VaultScanner

    hashes = {
        n.relative_path: n.content_hash for n in VaultScanner(VaultConfig(vault_path=vault)).scan()
    }
    (vault / "b.md").write_text("# B\n\nBeta rewritten.\n", encoding="utf-8")

    store = MagicMock()
    store.content_hashes.return_value = hashes
    result = _run(runner, vault, store, ["index"])

    assert result.exit_code == 0, result.output
    embedded = [i for call in store.upsert.call_args_list for i in call.kwargs["ids"]]
    assert embedded == ["b.md"]


def test_deleted_notes_are_dropped_from_the_index(runner: CliRunner, vault: Path) -> None:
    from vault_rag.ingest.scanner import VaultScanner

    hashes = {
        n.relative_path: n.content_hash for n in VaultScanner(VaultConfig(vault_path=vault)).scan()
    }
    hashes["gone.md"] = "deadbeef"

    store = MagicMock()
    store.content_hashes.return_value = hashes
    result = _run(runner, vault, store, ["index"])

    assert result.exit_code == 0, result.output
    store.delete.assert_called_once_with(["gone.md"])


def test_full_reindex_still_rebuilds_everything(runner: CliRunner, vault: Path) -> None:
    from vault_rag.ingest.scanner import VaultScanner

    hashes = {
        n.relative_path: n.content_hash for n in VaultScanner(VaultConfig(vault_path=vault)).scan()
    }
    store = MagicMock()
    store.content_hashes.return_value = hashes

    result = _run(runner, vault, store, ["index", "--full"])

    assert result.exit_code == 0, result.output
    store.reset.assert_called_once()
    embedded = [i for call in store.upsert.call_args_list for i in call.kwargs["ids"]]
    assert sorted(embedded) == ["a.md", "b.md"]
