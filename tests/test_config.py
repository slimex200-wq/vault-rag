"""Tests for VaultConfig dataclass."""

from pathlib import Path

from vault_rag.config import VaultConfig


def test_default_config_has_vault_path():
    cfg = VaultConfig()
    # 기본 vault 경로가 Path 타입으로 설정돼 있는지만 검사한다.
    # 실제 디스크 존재 여부는 머신마다 달라(CI엔 없음) 코드 동작 테스트가 아니므로 제외.
    assert isinstance(cfg.vault_path, Path)
    assert cfg.vault_path.name == "Obsidian Vault"


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
