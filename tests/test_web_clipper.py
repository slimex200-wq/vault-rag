"""Tests for WebClipper (trafilatura-based web-to-markdown)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vault_rag.ingest.web_clipper import WebClipper


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    """Minimal vault directory."""
    v = tmp_path / "vault"
    v.mkdir()
    return v


class TestWebClipper:
    def test_clip_creates_markdown_note(self, vault: Path) -> None:
        """clip() creates a .md file containing the extracted title and body text."""
        mock_meta = MagicMock()
        mock_meta.title = "Test Article"
        mock_meta.date = "2026-04-04"

        with (
            patch("vault_rag.ingest.web_clipper.trafilatura") as mock_traf,
        ):
            mock_traf.fetch_url.return_value = "<html>...</html>"
            mock_traf.extract.return_value = "Body of the article."
            mock_traf.extract_metadata.return_value = mock_meta

            clipper = WebClipper(vault_path=vault)
            result = clipper.clip(url="https://example.com/article")

        assert result.exists(), "clip() must return a path to an existing file"
        assert result.suffix == ".md"
        text = result.read_text(encoding="utf-8")
        assert "Test Article" in text
        assert "Body of the article." in text

    def test_clip_sanitizes_filename(self, vault: Path) -> None:
        """Titles containing /:\\?* are stripped from the resulting filename stem."""
        mock_meta = MagicMock()
        mock_meta.title = "What/Is:This?"
        mock_meta.date = "2026-04-04"

        with patch("vault_rag.ingest.web_clipper.trafilatura") as mock_traf:
            mock_traf.fetch_url.return_value = "<html>...</html>"
            mock_traf.extract.return_value = "Some content."
            mock_traf.extract_metadata.return_value = mock_meta

            clipper = WebClipper(vault_path=vault)
            result = clipper.clip(url="https://example.com/test")

        stem = result.stem
        for char in r'/\:*?"<>|':
            assert char not in stem, f"Invalid char {char!r} found in filename stem {stem!r}"

    def test_clip_adds_frontmatter(self, vault: Path) -> None:
        """Clipped note must start with YAML frontmatter containing 'source:'."""
        mock_meta = MagicMock()
        mock_meta.title = "Frontmatter Test"
        mock_meta.date = "2026-04-04"

        with patch("vault_rag.ingest.web_clipper.trafilatura") as mock_traf:
            mock_traf.fetch_url.return_value = "<html>...</html>"
            mock_traf.extract.return_value = "Article text here."
            mock_traf.extract_metadata.return_value = mock_meta

            clipper = WebClipper(vault_path=vault)
            result = clipper.clip(url="https://example.com/fm-test")

        text = result.read_text(encoding="utf-8")
        assert text.startswith("---"), "Note must open with YAML frontmatter block"
        assert "source:" in text
