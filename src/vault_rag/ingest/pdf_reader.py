"""PDFReader: extract text from PDF files and save as markdown notes."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PDFResult:
    """Result of extracting text from a PDF file."""

    text: str
    page_count: int
    metadata: dict = field(default_factory=dict)


class PDFReader:
    """Extract text from PDF files using PyMuPDF."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, pdf_path: Path) -> PDFResult:
        """Extract all text from *pdf_path* and return a :class:`PDFResult`.

        Parameters
        ----------
        pdf_path:
            Absolute (or resolvable) path to the PDF file.

        Returns
        -------
        PDFResult
            Extracted text, page count, and raw PDF metadata dict.
        """
        doc = fitz.open(str(pdf_path))
        try:
            pages = [page.get_text() for page in doc]
            text = "\n".join(pages)
            metadata: dict = dict(doc.metadata)
            page_count = len(doc)
        finally:
            doc.close()

        return PDFResult(text=text, page_count=page_count, metadata=metadata)

    def extract_to_note(
        self,
        pdf_path: Path,
        vault_path: Path,
        folder: str = "Research",
    ) -> Path:
        """Extract a PDF and save the content as a markdown note.

        Parameters
        ----------
        pdf_path:
            Source PDF file.
        vault_path:
            Root vault directory.
        folder:
            Sub-directory inside *vault_path* to write the note into.

        Returns
        -------
        Path
            Absolute path of the created ``.md`` file.
        """
        result = self.extract(pdf_path)
        title = result.metadata.get("title") or pdf_path.stem

        target_dir = vault_path / folder
        target_dir.mkdir(parents=True, exist_ok=True)

        note_path = target_dir / f"{pdf_path.stem}.md"
        note_path.write_text(
            self._render(title=title, source=str(pdf_path), result=result),
            encoding="utf-8",
        )
        return note_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render(title: str, source: str, result: PDFResult) -> str:
        """Return the full markdown note string with YAML frontmatter."""
        frontmatter = (
            "---\n"
            f"title: {title}\n"
            f"source: {source}\n"
            f"pages: {result.page_count}\n"
            "tags: [pdf]\n"
            "---\n"
        )
        return frontmatter + f"\n# {title}\n\n{result.text}\n"
