"""Give orphan notes an inbound link from their nearest INDEX.

An orphan is unreachable by navigation, so the repair has to create an *inbound*
edge. Adding a `## Related` section to the orphan itself only adds outbound
links and leaves it just as unreachable -- that is why earlier cleanups looked
successful and then "regressed".

Each INDEX owns one managed block delimited by HTML comments. Re-running
rewrites that block in place, so hand-written sections above it are never
touched.

The block is *merged*, never replaced wholesale: an entry already in it is
the only inbound edge its note has, so dropping the entry turns that note
back into an orphan and the next run re-adds it -- the linker would
oscillate instead of converging. Entries whose note no longer exists are
pruned, since those would be broken links.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MARKER_START = "<!-- vault-rag:unlinked:start -->"
MARKER_END = "<!-- vault-rag:unlinked:end -->"
HEADING = "## 미연결 노트"

_RE_BLOCK = re.compile(
    re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END),
    re.DOTALL,
)

_RE_ENTRY = re.compile(
    r"^- \[\[(?P<path>[^\]|]+)\|(?P<title>[^\]]*)\]\]\s*$",
    re.MULTILINE,
)

_INDEX_NAME = "INDEX.md"


@dataclass(frozen=True)
class OrphanLink:
    """One orphan and the INDEX that will link to it."""

    index_path: str
    note_path: str
    title: str


def _ancestor_dirs(note_path: str, *, skip_own_dir: bool) -> list[str]:
    """Directories to search for an INDEX, nearest first."""
    parts = note_path.split("/")[:-1]
    if skip_own_dir and parts:
        parts = parts[:-1]
    return ["/".join(parts[:i]) for i in range(len(parts), -1, -1)]


def plan_links(
    orphans: list[tuple[str, str]],
    index_paths: set[str],
) -> tuple[list[OrphanLink], list[str]]:
    """Map each orphan onto the nearest ancestor INDEX.

    Args:
        orphans: (relative_path, title) pairs.
        index_paths: relative paths of every INDEX.md in the vault.

    Returns:
        (links, unplaceable) -- unplaceable orphans have no INDEX above them
        and need a human decision rather than a generated link.
    """
    links: list[OrphanLink] = []
    unplaceable: list[str] = []

    for note_path, title in orphans:
        is_index = note_path.rsplit("/", 1)[-1] == _INDEX_NAME
        target: str | None = None
        for directory in _ancestor_dirs(note_path, skip_own_dir=is_index):
            candidate = f"{directory}/{_INDEX_NAME}" if directory else _INDEX_NAME
            if candidate == note_path:
                continue
            if candidate in index_paths:
                target = candidate
                break
        if target is None:
            unplaceable.append(note_path)
        else:
            links.append(OrphanLink(index_path=target, note_path=note_path, title=title))

    return links, unplaceable


def render_block(links: list[OrphanLink]) -> str:
    """Render the managed block for one INDEX."""
    lines = [MARKER_START, HEADING, ""]
    for link in sorted(links, key=lambda item: item.note_path):
        target = link.note_path.removesuffix(".md")
        lines.append(f"- [[{target}|{link.title}]]")
    lines.append(MARKER_END)
    return "\n".join(lines)


def _surviving_entries(original: str, vault_path: Path) -> list[tuple[str, str]]:
    """(note_path, title) already in the managed block, minus deleted notes."""
    match = _RE_BLOCK.search(original)
    if not match:
        return []

    kept: list[tuple[str, str]] = []
    for entry in _RE_ENTRY.finditer(match.group(0)):
        note_path = entry.group("path") + ".md"
        if (vault_path / note_path).exists():
            kept.append((note_path, entry.group("title")))
    return kept


def apply_links(vault_path: Path, links: list[OrphanLink]) -> list[str]:
    """Write the managed block into every affected INDEX.

    Returns the relative paths of the INDEX files that changed on disk.
    """
    by_index: dict[str, list[OrphanLink]] = {}
    for link in links:
        by_index.setdefault(link.index_path, []).append(link)

    written: list[str] = []
    for index_path, index_links in sorted(by_index.items()):
        path = vault_path / index_path
        original = path.read_text(encoding="utf-8")

        merged: dict[str, str] = dict(_surviving_entries(original, vault_path))
        for link in index_links:
            merged[link.note_path] = link.title
        block = render_block(
            [
                OrphanLink(index_path=index_path, note_path=note_path, title=title)
                for note_path, title in merged.items()
            ]
        )

        if _RE_BLOCK.search(original):
            updated = _RE_BLOCK.sub(lambda _m: block, original, count=1)
        else:
            updated = original.rstrip() + "\n\n" + block + "\n"

        if updated != original:
            path.write_text(updated, encoding="utf-8")
            written.append(index_path)

    return written
