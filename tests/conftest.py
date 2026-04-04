"""Shared pytest fixtures for vault-rag tests."""
from pathlib import Path

import pytest

from vault_rag.config import VaultConfig


@pytest.fixture()
def tmp_vault(tmp_path: Path) -> Path:
    """Create a minimal Obsidian vault structure for testing."""
    vault = tmp_path / "TestVault"
    vault.mkdir()

    # Standard Obsidian directories
    (vault / ".obsidian").mkdir()
    (vault / "_trash").mkdir()
    (vault / "Projects").mkdir()
    (vault / "Knowledge").mkdir()
    (vault / "Templates").mkdir()

    # Sample notes
    (vault / "Projects" / "project-alpha.md").write_text(
        "# Project Alpha\n\nThis is a sample project note.\n\n[[Knowledge/concepts]]",
        encoding="utf-8",
    )
    (vault / "Knowledge" / "concepts.md").write_text(
        "# Core Concepts\n\nFoundational ideas for the vault.\n\ntags: #knowledge #concepts",
        encoding="utf-8",
    )
    (vault / "index.md").write_text(
        "# Vault Index\n\nTop-level index note.",
        encoding="utf-8",
    )

    return vault


@pytest.fixture()
def cfg(tmp_vault: Path) -> VaultConfig:
    """VaultConfig pointed at the temporary test vault."""
    return VaultConfig(vault_path=tmp_vault)
