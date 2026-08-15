"""Tests for the controlled-vocabulary checks."""

from __future__ import annotations

from pathlib import Path

from vault_rag.engine.vocabulary import (
    STATUS_VOCABULARY,
    check_vocabulary,
    singleton_tags,
)
from vault_rag.ingest.scanner import ScannedNote


def _note(
    name: str,
    frontmatter: dict[str, str] | None = None,
    tags: list[str] | None = None,
) -> ScannedNote:
    return ScannedNote(
        path=Path(f"/vault/{name}.md"),
        relative_path=f"{name}.md",
        title=name,
        content="body",
        tags=tags or [],
        links=[],
        modified=0.0,
        content_hash="x",
        frontmatter=frontmatter or {},
    )


def test_status_inside_vocabulary_is_clean() -> None:
    notes = [_note(v, {"status": v}) for v in STATUS_VOCABULARY]
    assert check_vocabulary(notes) == []


def test_status_match_is_case_insensitive() -> None:
    assert check_vocabulary([_note("a", {"status": "ACTIVE"})]) == []


def test_status_outside_vocabulary_is_flagged() -> None:
    violations = check_vocabulary([_note("a", {"status": "완료"})])
    assert len(violations) == 1
    assert violations[0].field_name == "status"
    assert violations[0].value == "완료"


def test_free_text_status_is_flagged() -> None:
    """The real vault had a whole sentence sitting in `status`."""
    sentence = "완전 해결 + smoke test 완료 + gap fix Batch 1 완료 (10/13)"
    violations = check_vocabulary([_note("a", {"status": sentence})])
    assert len(violations) == 1
    assert "..." in violations[0].describe()  # long values are truncated


def test_missing_field_is_not_a_violation() -> None:
    assert check_vocabulary([_note("a", {"title": "no status here"})]) == []


def test_quality_tier_vocabulary_enforced() -> None:
    assert check_vocabulary([_note("a", {"quality_tier": "HIGH"})]) == []
    assert len(check_vocabulary([_note("b", {"quality_tier": "excellent"})])) == 1


def test_singleton_tags_are_the_ones_used_once() -> None:
    notes = [
        _note("a", tags=["shared", "only-here"]),
        _note("b", tags=["shared"]),
        _note("c", tags=["another-single"]),
    ]
    assert singleton_tags(notes) == ["another-single", "only-here"]


def test_singleton_tags_empty_when_all_tags_repeat() -> None:
    notes = [_note("a", tags=["x"]), _note("b", tags=["x"])]
    assert singleton_tags(notes) == []
