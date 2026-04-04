"""CLI integration tests using click.testing.CliRunner."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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
