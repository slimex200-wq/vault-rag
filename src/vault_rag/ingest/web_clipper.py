"""WebClipper: fetch a URL and save as an Obsidian-style markdown note."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import trafilatura


# Characters forbidden in Windows/POSIX filenames
_INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*]')


class WebClipper:
    """Clip a web page (via trafilatura) and save it as a markdown note in *vault_path*."""

    def __init__(self, vault_path: Path, compile_fn=None) -> None:
        self.vault_path = vault_path
        self.compile_fn = compile_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clip(
        self,
        url: str,
        html: str | None = None,
        folder: str = "Reference",
    ) -> Path:
        """Convert URL/HTML to a markdown note inside the vault.

        Parameters
        ----------
        url:
            Source URL (always recorded in frontmatter).
        html:
            Raw HTML content. When *None* the page is fetched via
            ``trafilatura.fetch_url(url)``.
        folder:
            Sub-directory inside *vault_path* to write the note into.

        Returns
        -------
        Path
            Absolute path of the newly created ``.md`` file.
        """
        if html is None:
            html = trafilatura.fetch_url(url)

        text: str = trafilatura.extract(html) or ""
        meta = trafilatura.extract_metadata(html)

        title: str = (meta.title if meta and meta.title else url)
        clipped_date: str = (
            meta.date if meta and meta.date else date.today().isoformat()
        )

        filename = self._sanitize(title) + ".md"
        target_dir = self.vault_path / folder
        target_dir.mkdir(parents=True, exist_ok=True)

        note_path = target_dir / filename
        note_path.write_text(
            self._render(title=title, url=url, clipped_date=clipped_date, body=text),
            encoding="utf-8",
        )
        if self.compile_fn is not None:
            self.compile_fn(note_path)
        return note_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sanitize(self, name: str) -> str:
        """Remove filesystem-invalid characters from *name*."""
        sanitized = _INVALID_CHARS_RE.sub("", name).strip()
        return sanitized or "untitled"

    @staticmethod
    def _render(title: str, url: str, clipped_date: str, body: str) -> str:
        """Return the full markdown note string with YAML frontmatter."""
        frontmatter = (
            "---\n"
            f"title: {title}\n"
            f"source: {url}\n"
            f"clipped: {clipped_date}\n"
            "tags: [clipped]\n"
            "---\n"
        )
        return frontmatter + f"\n# {title}\n\n{body}\n"
