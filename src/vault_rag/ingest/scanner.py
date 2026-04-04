"""VaultScanner: traverse an Obsidian vault and parse markdown notes."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from vault_rag.config import VaultConfig

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_RE_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_RE_YAML_TAGS = re.compile(r"tags:\s*\[([^\]]*)\]")
_RE_INLINE_TAGS = re.compile(r"(?:^|\s)#([a-zA-Z가-힣][\w가-힣-]*)", re.MULTILINE)
_RE_WIKILINKS = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_RE_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_RE_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScannedNote:
    """Parsed representation of a single Obsidian markdown note."""

    path: Path
    relative_path: str
    title: str
    content: str          # body without frontmatter
    tags: list[str]       # deduplicated, order-preserving
    links: list[str]      # wikilink targets
    modified: float       # mtime
    content_hash: str     # md5 hex digest of raw file content


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class VaultScanner:
    """Scans an Obsidian vault directory and parses .md files into ScannedNote objects."""

    def __init__(self, config: VaultConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self, known_hashes: dict[str, str] | None = None
    ) -> list[ScannedNote]:
        """Return ScannedNote list for all non-excluded .md files.

        When *known_hashes* is provided (relative_path → md5), only files
        whose content has changed are returned (incremental mode).
        """
        notes: list[ScannedNote] = []
        vault = self._config.vault_path

        for md_file in vault.rglob("*.md"):
            if self._is_excluded(md_file):
                continue

            relative = md_file.relative_to(vault).as_posix()
            raw = md_file.read_text(encoding="utf-8")
            content_hash = hashlib.md5(raw.encode("utf-8")).hexdigest()

            if known_hashes is not None:
                if known_hashes.get(relative) == content_hash:
                    continue  # unchanged — skip

            note = self._parse(
                path=md_file,
                relative=relative,
                content=raw,
                content_hash=content_hash,
            )
            notes.append(note)

        return notes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_excluded(self, path: Path) -> bool:
        """Return True if any component of *path* is in excluded_dirs."""
        excluded = set(self._config.excluded_dirs)
        return any(part in excluded for part in path.parts)

    def _parse(
        self,
        path: Path,
        relative: str,
        content: str,
        content_hash: str,
    ) -> ScannedNote:
        """Parse raw markdown content into a ScannedNote."""
        # --- frontmatter ---
        fm_match = _RE_FRONTMATTER.match(content)
        frontmatter = fm_match.group(1) if fm_match else ""
        body = content[fm_match.end():] if fm_match else content

        # --- tags from YAML frontmatter ---
        fm_tags: list[str] = []
        if frontmatter:
            yt_match = _RE_YAML_TAGS.search(frontmatter)
            if yt_match:
                fm_tags = [t.strip() for t in yt_match.group(1).split(",") if t.strip()]

        # --- inline tags from body ---
        inline_tags = _RE_INLINE_TAGS.findall(body)

        # --- combined, deduplicated, order-preserving ---
        all_tags = list(dict.fromkeys(fm_tags + inline_tags))

        # --- wikilinks (strip code blocks first, then extract and clean) ---
        content_no_code = _RE_CODE_BLOCK.sub("", content)
        raw_links = _RE_WIKILINKS.findall(content_no_code)
        links = [link.strip().rstrip("\\") for link in raw_links]

        # --- title: first H1 in body, fallback to stem ---
        h1 = _RE_H1.search(body)
        title = h1.group(1).strip() if h1 else path.stem

        return ScannedNote(
            path=path,
            relative_path=relative,
            title=title,
            content=body,
            tags=all_tags,
            links=links,
            modified=path.stat().st_mtime,
            content_hash=content_hash,
        )
