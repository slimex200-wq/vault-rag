"""HealthChecker: vault sanity checks for broken links, orphans, untagged, and empty notes."""

from __future__ import annotations

import re

from vault_rag.ingest.scanner import ScannedNote

# Normalize spaces/hyphens and lowercase for fuzzy title matching
_RE_NORMALIZE = re.compile(r"[\s\-]+")


def _normalize(text: str) -> str:
    return _RE_NORMALIZE.sub(" ", text).lower().strip()


class HealthChecker:
    """Run health checks against a list of ScannedNote objects."""

    def __init__(self, notes: list[ScannedNote], existing_paths: set[str] | None = None) -> None:
        self._notes = notes

        # Pre-compute lookup sets for O(1) resolution
        self._titles: set[str] = {_normalize(n.title) for n in notes}
        self._stems: set[str] = {_normalize(n.path.stem) for n in notes}
        self._paths: set[str] = {n.relative_path for n in notes}

        # Path-based lookup: relative paths normalized to forward slashes, lowercased
        # e.g. "Projects/flatsnap/INDEX.md" and "projects/flatsnap/index.md"
        self._all_paths_lower: set[str] = {
            n.relative_path.replace("\\", "/").lower() for n in notes
        }
        self._existing_paths_lower: set[str] = {
            p.replace("\\", "/").lower() for p in (existing_paths or set())
        }
        self._existing_names_lower: set[str] = {
            p.replace("\\", "/").lower().rsplit("/", 1)[-1] for p in (existing_paths or set())
        }
        # Paths without .md extension for bare path links like [[Projects/flatsnap/INDEX]]
        self._paths_no_ext: set[str] = {
            p[:-3] if p.endswith(".md") else p for p in self._all_paths_lower
        }

        # Union of all normalized targets that can be resolved
        self._all_targets: set[str] = self._titles | self._stems

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def broken_links(self) -> list[dict]:
        """Return links pointing to non-existent notes.

        Returns:
            [{"source": relative_path, "target": link_text}, ...]
        """
        result: list[dict] = []
        for note in self._notes:
            for link in note.links:
                if not self._link_exists(link):
                    result.append({"source": note.relative_path, "target": link})
        return result

    def _link_exists(self, link: str) -> bool:
        """Return True if *link* resolves to a known note."""
        # Strip trailing backslash and whitespace (handles [[path\]] artifacts)
        cleaned = link.strip().rstrip("\\")
        if self._resolves(cleaned):
            return True
        # Retry with Obsidian subpath dropped ([[note#heading]], [[note#^block]]).
        # Full-string match runs first so note names containing '#' still win.
        if "#" in cleaned:
            base = cleaned.split("#", 1)[0].strip()
            if not base:
                return True  # pure in-file anchor like [[#heading]]
            return self._resolves(base)
        return False

    def _resolves(self, cleaned: str) -> bool:
        """Return True if *cleaned* matches a known title, stem, or path."""
        key = _normalize(cleaned)

        # Title / stem match (existing behaviour)
        if key in self._all_targets:
            return True

        # Path-based match: [[Projects/flatsnap/INDEX]] → Projects/flatsnap/INDEX.md
        normalized = cleaned.replace("\\", "/").lower()
        if normalized in self._all_paths_lower or normalized in self._existing_paths_lower:
            return True
        if "/" not in normalized and normalized in self._existing_names_lower:
            return True
        if normalized in self._paths_no_ext:
            return True
        if (normalized + ".md") in self._all_paths_lower:
            return True
        if (normalized + ".md") in self._existing_paths_lower:
            return True

        return False

    def orphan_notes(self) -> list[str]:
        """Return notes with no inbound *and* no outbound links.

        Returns:
            [relative_path, ...]
        """
        # Collect all inbound targets (normalized titles/stems + paths)
        inbound_targets: set[str] = set()
        inbound_paths: set[str] = set()
        for note in self._notes:
            for link in note.links:
                full = link.strip().rstrip("\\")
                # Register both the full text (note names may contain '#')
                # and the subpath-stripped base ([[note#heading]]).
                variants = {full}
                if "#" in full:
                    variants.add(full.split("#", 1)[0].strip())
                for variant in variants:
                    if not variant:
                        continue
                    inbound_targets.add(_normalize(variant))
                    path = variant.replace("\\", "/").lower()
                    if not path.endswith(".md"):
                        path += ".md"
                    inbound_paths.add(path)

        orphans: list[str] = []
        for note in self._notes:
            has_outbound = bool(note.links)
            is_linked = (
                _normalize(note.title) in inbound_targets
                or _normalize(note.path.stem) in inbound_targets
                or note.relative_path.replace("\\", "/").lower() in inbound_paths
            )
            if not has_outbound and not is_linked:
                orphans.append(note.relative_path)
        return orphans

    def untagged_notes(self) -> list[str]:
        """Return notes with an empty tags list.

        Returns:
            [relative_path, ...]
        """
        return [n.relative_path for n in self._notes if not n.tags]

    def empty_notes(self) -> list[str]:
        """Return notes with empty or whitespace-only content.

        Returns:
            [relative_path, ...]
        """
        return [n.relative_path for n in self._notes if not n.content.strip()]

    def report(self) -> dict:
        """Return a full health report with counts.

        Keys:
            total_notes, broken_links, orphan_notes, untagged_notes,
            empty_notes, broken_link_count, orphan_count, untagged_count
        """
        broken = self.broken_links()
        orphans = self.orphan_notes()
        untagged = self.untagged_notes()
        empty = self.empty_notes()

        return {
            "total_notes": len(self._notes),
            "broken_links": broken,
            "broken_link_count": len(broken),
            "orphan_notes": orphans,
            "orphan_count": len(orphans),
            "untagged_notes": untagged,
            "untagged_count": len(untagged),
            "empty_notes": empty,
        }
