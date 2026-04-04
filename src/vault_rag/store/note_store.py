"""NoteStore: filesystem CRUD for Obsidian markdown notes."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from vault_rag.config import VaultConfig

_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
_HEADING_RE = re.compile(r"^#\s+.+\n", re.MULTILINE)


def _slugify(title: str) -> str:
    """Convert a title to a lowercase-hyphenated filename slug."""
    slug = title.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug


def _build_note(title: str, content: str, tags: list[str] | None) -> str:
    """Assemble a complete markdown note with YAML frontmatter."""
    today = date.today().isoformat()
    tag_list = tags or []
    tag_lines = "\n".join(f"  - {t}" for t in tag_list)
    frontmatter = f"---\ncreated: {today}\ntags:\n{tag_lines}\n---\n"
    return f"{frontmatter}# {title}\n\n{content}\n"


class NoteStore:
    """Create, read, list, and update Obsidian markdown notes on disk."""

    def __init__(self, config: VaultConfig) -> None:
        self.vault = config.vault_path

    def create(
        self,
        title: str,
        content: str,
        folder: str = "Knowledge",
        tags: list[str] | None = None,
    ) -> Path:
        """Create a note with YAML frontmatter. Returns the absolute path."""
        folder_path = self.vault / folder
        folder_path.mkdir(parents=True, exist_ok=True)

        filename = _slugify(title) + ".md"
        path = folder_path / filename

        path.write_text(_build_note(title, content, tags), encoding="utf-8")
        return path

    def read(self, relative_path: str) -> str | None:
        """Read note content by vault-relative path. Returns None if missing."""
        path = self.vault / relative_path
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def list_folder(self, folder: str) -> list[Path]:
        """Return all .md files in *folder*, sorted by path."""
        folder_path = self.vault / folder
        if not folder_path.exists():
            return []
        return sorted(folder_path.glob("*.md"))

    def update(self, path: Path, content: str) -> None:
        """Replace the note body while preserving frontmatter and the H1 heading."""
        original = path.read_text(encoding="utf-8")

        # Extract frontmatter block (including trailing newline)
        fm_match = _FRONTMATTER_RE.match(original)
        frontmatter = fm_match.group(0) if fm_match else ""

        # Extract heading from the original text (after frontmatter)
        after_fm = original[len(frontmatter):]
        h1_match = _HEADING_RE.match(after_fm)
        heading = h1_match.group(0) if h1_match else ""

        path.write_text(f"{frontmatter}{heading}\n{content}\n", encoding="utf-8")
