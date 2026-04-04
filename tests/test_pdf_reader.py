"""Tests for PDFReader (PyMuPDF-based PDF-to-markdown)."""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
import pytest

from vault_rag.ingest.pdf_reader import PDFReader, PDFResult


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal single-page PDF with known text content."""
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Hello from PDF")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    """Minimal vault directory."""
    v = tmp_path / "vault"
    v.mkdir()
    return v


class TestPDFReader:
    def test_extract_text_from_pdf(self, sample_pdf: Path) -> None:
        """extract() returns PDFResult with text and correct page_count."""
        reader = PDFReader()
        result = reader.extract(sample_pdf)

        assert isinstance(result, PDFResult)
        assert "Hello from PDF" in result.text
        assert result.page_count == 1

    def test_extract_creates_note(self, sample_pdf: Path, vault: Path) -> None:
        """extract_to_note() writes a .md file containing the extracted PDF text."""
        reader = PDFReader()
        note_path = reader.extract_to_note(sample_pdf, vault_path=vault)

        assert note_path.exists(), "extract_to_note() must create a file"
        assert note_path.suffix == ".md"
        text = note_path.read_text(encoding="utf-8")
        assert "Hello from PDF" in text
