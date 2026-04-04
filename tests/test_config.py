"""Tests for VaultConfig dataclass."""
from pathlib import Path

import pytest

from vault_rag.config import VaultConfig


def test_default_config_has_vault_path():
    cfg = VaultConfig()
    assert cfg.vault_path.exists(), f"vault_path does not exist: {cfg.vault_path}"


def test_config_with_custom_path(tmp_path):
    cfg = VaultConfig(vault_path=tmp_path)
    assert cfg.vault_path == tmp_path


def test_config_chroma_path():
    cfg = VaultConfig()
    assert "chroma" in str(cfg.chroma_path)


def test_config_embedding_model():
    cfg = VaultConfig()
    assert cfg.embedding_model == "text-embedding-3-small"


def test_config_excluded_dirs():
    cfg = VaultConfig()
    assert ".obsidian" in cfg.excluded_dirs
    assert "_trash" in cfg.excluded_dirs
