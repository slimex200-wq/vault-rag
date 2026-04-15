"""Tests for WikiLog."""

from __future__ import annotations

from pathlib import Path

from vault_rag.engine.wiki_log import WikiLog


def test_log_creates_file(tmp_path: Path) -> None:
    log = WikiLog(tmp_path)
    log.log_ingest("Test Note", source="https://example.com")

    log_path = tmp_path / "log.md"
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "ingest | Test Note" in content
    assert "https://example.com" in content


def test_log_appends(tmp_path: Path) -> None:
    log = WikiLog(tmp_path)
    log.log_ingest("First")
    log.log_query("What is X?")
    log.log_compile("Note A", quality_score=85)

    content = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "ingest | First" in content
    assert "query | What is X?" in content
    assert "compile | Note A" in content
    assert "Quality: 85" in content


def test_log_query_with_save(tmp_path: Path) -> None:
    log = WikiLog(tmp_path)
    log.log_query("What is RAG?", saved_as="RAG 설명")

    content = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "Saved as: [[RAG 설명]]" in content


def test_log_lint(tmp_path: Path) -> None:
    log = WikiLog(tmp_path)
    log.log_lint(issues_found=5)

    content = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "lint | health check" in content
    assert "Issues found: 5" in content


def test_log_generate(tmp_path: Path) -> None:
    log = WikiLog(tmp_path)
    log.log_generate("slides", "AI agents", output_path="output/slides_20260405.md")

    content = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "generate:slides | AI agents" in content
    assert "output/slides_20260405.md" in content
