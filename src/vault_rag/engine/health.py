"""HealthChecker: vault sanity checks for broken links, orphans, untagged, and empty notes."""

from __future__ import annotations

import re

from vault_rag.engine import vocabulary
from vault_rag.engine.vocabulary import VocabularyViolation
from vault_rag.ingest.scanner import ScannedNote

# Normalize spaces/hyphens and lowercase for fuzzy title matching
_RE_NORMALIZE = re.compile(r"[\s\-]+")


def _normalize(text: str) -> str:
    return _RE_NORMALIZE.sub(" ", text).lower().strip()


def _strip_fragment(link: str) -> str:
    """Drop the ``#heading`` / ``^block`` part of a wikilink.

    ``[[note#some heading]]`` points at *note*; the fragment only picks a spot
    inside it. Resolving the whole string verbatim reports a live note as a
    broken link and hides a real inbound edge.
    """
    for separator in ("#", "^"):
        head = link.split(separator, 1)[0]
        if head != link:
            link = head
    return link.strip()


class HealthChecker:
    """Run health checks against a list of ScannedNote objects."""

    def __init__(self, notes: list[ScannedNote], existing_paths: set[str] | None = None) -> None:
        self._notes = notes

        # Pre-compute lookup sets for O(1) resolution
        self._titles: set[str] = {_normalize(n.title) for n in notes}
        self._stems: set[str] = {_normalize(n.path.stem) for n in notes}
        self._paths: set[str] = {n.relative_path for n in notes}

        # Link key -> note, so a resolved link can be attributed to its target.
        # Mirrors every form _link_exists() accepts, so inbound counting and
        # broken-link detection never disagree.
        self._by_key: dict[str, ScannedNote] = {}
        for n in notes:
            rel_lower = n.relative_path.replace("\\", "/").lower()
            for key in (
                _normalize(n.title),
                _normalize(n.path.stem),
                rel_lower,
                rel_lower[:-3] if rel_lower.endswith(".md") else rel_lower,
            ):
                self._by_key.setdefault(key, n)

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

    def _link_forms(self, link: str) -> list[str]:
        """Resolution candidates for *link*, most literal first.

        A `#` inside a wikilink is ambiguous: it starts a heading fragment in
        `[[note#heading]]`, but it is an ordinary character in a title such as
        `[[Batch 2 작업 #3a 완료]]`. Trying the verbatim string before the
        stripped one keeps both working; stripping first breaks the titles.
        """
        cleaned = link.strip().rstrip("\\")
        forms = [cleaned]
        stripped = _strip_fragment(cleaned)
        if stripped and stripped != cleaned:
            forms.append(stripped)
        return forms

    def _link_exists(self, link: str) -> bool:
        """Return True if *link* resolves to a known note."""
        if not _strip_fragment(link.strip().rstrip("\\")):
            return True  # pure in-page anchor like [[#heading]]
        return any(self._form_exists(form) for form in self._link_forms(link))

    def _form_exists(self, cleaned: str) -> bool:
        """Return True if one concrete candidate string resolves."""
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

    def _resolve(self, link: str) -> ScannedNote | None:
        """Return the note *link* points to, or None if it resolves to nothing."""
        for cleaned in self._link_forms(link):
            normalized = cleaned.replace("\\", "/").lower()
            for key in (_normalize(cleaned), normalized, normalized + ".md"):
                hit = self._by_key.get(key)
                if hit is not None:
                    return hit
        return None

    def _inbound_paths(self) -> set[str]:
        """Relative paths of notes that at least one *other* note links to."""
        linked: set[str] = set()
        for note in self._notes:
            for link in note.links:
                target = self._resolve(link)
                if target is not None and target.relative_path != note.relative_path:
                    linked.add(target.relative_path)
        return linked

    def orphan_notes(self) -> list[str]:
        """Return notes with no inbound links.

        An orphan is a page nothing links to, so it cannot be reached by
        following the graph. Outbound links do not rescue it -- a page that
        links out to ten others is still unreachable if nobody points at it.
        Resolution goes through _resolve(), so path-form links such as
        [[Projects/flatsnap/INDEX]] count as inbound for the .md target.

        Returns:
            [relative_path, ...]
        """
        linked = self._inbound_paths()
        return [n.relative_path for n in self._notes if n.relative_path not in linked]

    def isolated_notes(self) -> list[str]:
        """Return notes with neither inbound nor outbound links.

        Strict subset of orphan_notes() -- these are fully disconnected and
        are the highest-priority repair targets.

        Returns:
            [relative_path, ...]
        """
        linked = self._inbound_paths()
        return [
            n.relative_path for n in self._notes if not n.links and n.relative_path not in linked
        ]

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

    def vocabulary_violations(self) -> list[VocabularyViolation]:
        """Return frontmatter values outside their field's controlled vocabulary."""
        return vocabulary.check_vocabulary(self._notes)

    def singleton_tags(self) -> list[str]:
        """Return tags used exactly once -- the tag-sprawl signal."""
        return vocabulary.singleton_tags(self._notes)

    def report(self) -> dict:
        """Return a full health report with counts.

        Keys:
            total_notes, broken_links, orphan_notes, isolated_notes,
            untagged_notes, empty_notes, vocabulary_violations, singleton_tags,
            broken_link_count, orphan_count, isolated_count, untagged_count,
            vocabulary_violation_count, singleton_tag_count
        """
        broken = self.broken_links()
        orphans = self.orphan_notes()
        isolated = self.isolated_notes()
        untagged = self.untagged_notes()
        empty = self.empty_notes()
        vocab = self.vocabulary_violations()
        singles = self.singleton_tags()

        return {
            "total_notes": len(self._notes),
            "broken_links": broken,
            "broken_link_count": len(broken),
            "orphan_notes": orphans,
            "orphan_count": len(orphans),
            "isolated_notes": isolated,
            "isolated_count": len(isolated),
            "untagged_notes": untagged,
            "untagged_count": len(untagged),
            "empty_notes": empty,
            "vocabulary_violations": [v.describe() for v in vocab],
            "vocabulary_violation_count": len(vocab),
            "singleton_tags": singles,
            "singleton_tag_count": len(singles),
        }
