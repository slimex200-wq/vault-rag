"""WikiLog: append-only chronological log for wiki operations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class WikiLog:
    """Append entries to log.md in the vault root."""

    def __init__(self, vault_path: Path) -> None:
        self._path = vault_path / "log.md"

    def _ensure_exists(self) -> None:
        if not self._path.exists():
            self._path.write_text(
                "---\ntags: [system, log]\n---\n# Wiki Log\n\n"
                "Append-only chronological record of wiki operations.\n\n",
                encoding="utf-8",
            )

    def _append(self, operation: str, title: str, detail: str = "") -> None:
        self._ensure_exists()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"## [{ts}] {operation} | {title}\n"
        if detail:
            entry += f"{detail}\n"
        entry += "\n"
        with self._path.open("a", encoding="utf-8") as f:
            f.write(entry)

    def log_ingest(self, title: str, source: str = "", pages_updated: int = 0) -> None:
        """Log an ingest operation."""
        detail = ""
        if source:
            detail += f"- Source: {source}\n"
        if pages_updated:
            detail += f"- Pages updated: {pages_updated}\n"
        self._append("ingest", title, detail)

    def log_query(self, question: str, saved_as: str = "") -> None:
        """Log a query operation."""
        detail = ""
        if saved_as:
            detail += f"- Saved as: [[{saved_as}]]\n"
        self._append("query", question, detail)

    def log_compile(self, title: str, quality_score: int = 0) -> None:
        """Log a compile operation."""
        detail = ""
        if quality_score:
            detail += f"- Quality: {quality_score}\n"
        self._append("compile", title, detail)

    def log_lint(self, issues_found: int = 0) -> None:
        """Log a lint operation."""
        detail = f"- Issues found: {issues_found}\n" if issues_found else ""
        self._append("lint", "health check", detail)

    def log_generate(self, format: str, topic: str, output_path: str = "") -> None:
        """Log a generate operation."""
        detail = ""
        if output_path:
            detail += f"- Output: {output_path}\n"
        self._append(f"generate:{format}", topic, detail)
