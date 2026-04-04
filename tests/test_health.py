"""Tests for HealthChecker (Task 8: Health Checks)."""
from __future__ import annotations

from pathlib import Path

import pytest

from vault_rag.engine.health import HealthChecker
from vault_rag.ingest.scanner import ScannedNote


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _note(
    title: str,
    path: str,
    tags: list[str] | None = None,
    links: list[str] | None = None,
    content: str = "Some content",
) -> ScannedNote:
    return ScannedNote(
        path=Path(path),
        relative_path=path,
        title=title,
        content=content,
        tags=tags if tags is not None else [],
        links=links if links is not None else [],
        modified=0.0,
        content_hash="abc123",
    )


# ---------------------------------------------------------------------------
# 1. test_detect_broken_links
# ---------------------------------------------------------------------------

def test_detect_broken_links() -> None:
    a = _note("A", "a.md", links=["NonExistent"])
    b = _note("B", "b.md")
    checker = HealthChecker([a, b])
    broken = checker.broken_links()
    assert len(broken) == 1
    assert broken[0]["source"] == "a.md"
    assert broken[0]["target"] == "NonExistent"


# ---------------------------------------------------------------------------
# 2. test_detect_orphan_notes
# ---------------------------------------------------------------------------

def test_detect_orphan_notes() -> None:
    a = _note("A", "a.md", links=["B"])
    b = _note("B", "b.md")
    orphan = _note("Orphan", "orphan.md")
    checker = HealthChecker([a, b, orphan])
    orphans = checker.orphan_notes()
    assert "orphan.md" in orphans
    assert "a.md" not in orphans
    assert "b.md" not in orphans


# ---------------------------------------------------------------------------
# 3. test_detect_untagged_notes
# ---------------------------------------------------------------------------

def test_detect_untagged_notes() -> None:
    tagged = _note("Tagged", "tagged.md", tags=["ai"])
    untagged = _note("Untagged", "untagged.md", tags=[])
    checker = HealthChecker([tagged, untagged])
    untagged_list = checker.untagged_notes()
    assert "untagged.md" in untagged_list
    assert "tagged.md" not in untagged_list


# ---------------------------------------------------------------------------
# 4. test_detect_empty_notes
# ---------------------------------------------------------------------------

def test_detect_empty_notes() -> None:
    normal = _note("Normal", "normal.md", content="Has content")
    empty = _note("Empty", "empty.md", content="")
    whitespace = _note("Whitespace", "whitespace.md", content="   \n  ")
    checker = HealthChecker([normal, empty, whitespace])
    empty_list = checker.empty_notes()
    assert "empty.md" in empty_list
    assert "whitespace.md" in empty_list
    assert "normal.md" not in empty_list


# ---------------------------------------------------------------------------
# 5. test_full_report
# ---------------------------------------------------------------------------

def test_full_report() -> None:
    a = _note("A", "a.md", links=["B", "Missing"])
    b = _note("B", "b.md")
    orphan = _note("Orphan", "orphan.md")
    checker = HealthChecker([a, b, orphan])
    report = checker.report()

    assert report["total_notes"] == 3
    assert isinstance(report["broken_links"], list)
    assert isinstance(report["orphan_notes"], list)
    assert report["broken_link_count"] == len(report["broken_links"])
    assert report["orphan_count"] == len(report["orphan_notes"])
    assert "untagged_notes" in report
    assert "untagged_count" in report
    assert "empty_notes" in report

    broken_targets = [item["target"] for item in report["broken_links"]]
    assert "Missing" in broken_targets
    assert "B" not in broken_targets

    assert "orphan.md" in report["orphan_notes"]
