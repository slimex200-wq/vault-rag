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

    def __init__(self, notes: list[ScannedNote]) -> None:
        self._notes = notes

        # Pre-compute lookup sets for O(1) resolution
        self._titles: set[str] = {_normalize(n.title) for n in notes}
        self._stems: set[str] = {_normalize(n.path.stem) for n in notes}
        self._paths: set[str] = {n.relative_path for n in notes}

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
                if _normalize(link) not in self._all_targets:
                    result.append({"source": note.relative_path, "target": link})
        return result

    def orphan_notes(self) -> list[str]:
        """Return notes with no inbound *and* no outbound links.

        Returns:
            [relative_path, ...]
        """
        # Collect all inbound targets (normalized)
        inbound_targets: set[str] = set()
        for note in self._notes:
            for link in note.links:
                inbound_targets.add(_normalize(link))

        orphans: list[str] = []
        for note in self._notes:
            has_outbound = bool(note.links)
            is_linked = (
                _normalize(note.title) in inbound_targets
                or _normalize(note.path.stem) in inbound_targets
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
