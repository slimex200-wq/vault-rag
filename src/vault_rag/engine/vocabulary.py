"""Controlled vocabulary for wiki classification fields.

A classification axis is only useful if its terms form a closed set. Once a
field accepts free text it stops being filterable -- `status` drifted into
17 distinct values on a live vault, mixing languages (`done` vs `완료`) and
absorbing whole sentences, which silently emptied every dashboard that
grouped by it.

This module owns the allowed terms and reports what falls outside them. It
deliberately does *not* rewrite values: mapping `approved` onto `done` is a
semantic judgement, and guessing it would destroy the distinction between a
decision that was approved and a project that shipped.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from vault_rag.ingest.scanner import ScannedNote

# Project / decision lifecycle. Lowercase; comparison is case-insensitive.
# `planning` is distinct from `active`: scoping and design work is under way but
# nothing is being built yet, and this vault uses that stage often enough that
# collapsing it into `active` would lose a real distinction.
STATUS_VOCABULARY: tuple[str, ...] = (
    "proposed",
    "planning",
    "active",
    "paused",
    "done",
    "archived",
)

# Confidence tier attached by the compiler.
QUALITY_TIER_VOCABULARY: tuple[str, ...] = ("HIGH", "MEDIUM", "LOW")

FIELD_VOCABULARIES: dict[str, tuple[str, ...]] = {
    "status": STATUS_VOCABULARY,
    "quality_tier": QUALITY_TIER_VOCABULARY,
}


@dataclass(frozen=True)
class VocabularyViolation:
    """A frontmatter value outside its field's controlled vocabulary."""

    path: str
    field_name: str
    value: str
    allowed: tuple[str, ...]

    def describe(self) -> str:
        shown = self.value if len(self.value) <= 40 else self.value[:37] + "..."
        return f"{self.path}: {self.field_name}={shown!r} not in {list(self.allowed)}"


def check_vocabulary(notes: list[ScannedNote]) -> list[VocabularyViolation]:
    """Return every frontmatter value outside its field's vocabulary.

    Notes that omit a governed field are not violations -- this checks the
    terms in use, not field coverage.
    """
    violations: list[VocabularyViolation] = []
    for note in notes:
        for field_name, allowed in FIELD_VOCABULARIES.items():
            raw = note.frontmatter.get(field_name)
            if raw is None:
                continue
            if not any(raw.strip().lower() == term.lower() for term in allowed):
                violations.append(
                    VocabularyViolation(
                        path=note.relative_path,
                        field_name=field_name,
                        value=raw,
                        allowed=allowed,
                    )
                )
    return violations


def singleton_tags(notes: list[ScannedNote]) -> list[str]:
    """Return tags used exactly once across the vault.

    A tag applied to a single note cannot group anything, so a high singleton
    ratio means the tag axis has degenerated into free-text keywords.
    """
    counts = Counter(tag for note in notes for tag in note.tags)
    return sorted(tag for tag, count in counts.items() if count == 1)
