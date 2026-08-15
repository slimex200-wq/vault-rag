"""Tests for HealthChecker (Task 8: Health Checks)."""

from __future__ import annotations

from pathlib import Path

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
    frontmatter: dict[str, str] | None = None,
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
        frontmatter=frontmatter or {},
    )


def test_report_carries_vocabulary_and_tag_sprawl() -> None:
    notes = [
        _note("Good", "good.md", tags=["shared"], frontmatter={"status": "active"}),
        _note("Bad", "bad.md", tags=["shared", "once"], frontmatter={"status": "완료"}),
    ]
    report = HealthChecker(notes).report()

    assert report["vocabulary_violation_count"] == 1
    assert "bad.md" in report["vocabulary_violations"][0]
    assert report["singleton_tag_count"] == 1
    assert report["singleton_tags"] == ["once"]


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
    """Orphan = no inbound link. Outbound links do not rescue a page."""
    a = _note("A", "a.md", links=["B"])
    b = _note("B", "b.md")
    orphan = _note("Orphan", "orphan.md")
    checker = HealthChecker([a, b, orphan])
    orphans = checker.orphan_notes()
    assert "orphan.md" in orphans
    assert "a.md" in orphans  # links out to B, but nothing points at it
    assert "b.md" not in orphans  # a.md links to it


def test_isolated_notes_are_stricter_than_orphans() -> None:
    """isolated = no inbound AND no outbound; a strict subset of orphans."""
    a = _note("A", "a.md", links=["B"])
    b = _note("B", "b.md")
    orphan = _note("Orphan", "orphan.md")
    checker = HealthChecker([a, b, orphan])
    assert checker.isolated_notes() == ["orphan.md"]
    assert set(checker.isolated_notes()) <= set(checker.orphan_notes())


def test_path_form_link_counts_as_inbound() -> None:
    """[[Projects/flatsnap/INDEX]] must mark the .md target as linked."""
    hub = _note("Hub", "hub.md", links=["Projects/flatsnap/INDEX"])
    idx = _note("FlatSnap INDEX", "Projects/flatsnap/INDEX.md")
    checker = HealthChecker([hub, idx])
    assert "Projects/flatsnap/INDEX.md" not in checker.orphan_notes()


def test_self_link_does_not_rescue_orphan() -> None:
    """A note linking to itself is still unreachable from anywhere else."""
    solo = _note("Solo", "solo.md", links=["Solo"])
    checker = HealthChecker([solo])
    assert "solo.md" in checker.orphan_notes()


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


# ---------------------------------------------------------------------------
# 6. path-based wikilinks should not be broken
# ---------------------------------------------------------------------------


def test_path_based_links_not_broken() -> None:
    """[[Projects/flatsnap/INDEX]] should resolve to Projects/flatsnap/INDEX.md"""
    notes = [
        _note("Hub", "hub.md", links=["Projects/flatsnap/INDEX"]),
        _note("FlatSnap INDEX", "Projects/flatsnap/INDEX.md"),
    ]
    hc = HealthChecker(notes)
    broken = hc.broken_links()
    assert len(broken) == 0


# ---------------------------------------------------------------------------
# 7. explicit .md path wikilinks should not be broken
# ---------------------------------------------------------------------------


def test_explicit_md_path_links_not_broken() -> None:
    """[[Daily/2026-04-19.md]] should resolve directly."""
    notes = [
        _note("Hub", "hub.md", links=["Daily/2026-04-19.md"]),
        _note("2026-04-19", "Daily/2026-04-19.md"),
    ]
    hc = HealthChecker(notes)
    broken = hc.broken_links()
    assert len(broken) == 0


# ---------------------------------------------------------------------------
# 8. existing non-indexed files should not be broken
# ---------------------------------------------------------------------------


def test_existing_non_indexed_files_not_broken() -> None:
    """Existing template/image targets can be valid vault links even when not scanned as notes."""
    notes = [
        _note(
            "Hub", "hub.md", links=["Templates/INDEX", "Reference/design-ref/card.png", "card.png"]
        )
    ]
    hc = HealthChecker(
        notes,
        existing_paths={"Templates/INDEX.md", "Reference/design-ref/card.png"},
    )
    broken = hc.broken_links()
    assert len(broken) == 0


# ---------------------------------------------------------------------------
# 9. backslash-suffixed path links should not be broken
# ---------------------------------------------------------------------------


def test_heading_fragment_links_resolve() -> None:
    """[[note#heading]] targets the note; the fragment must not break it."""
    notes = [
        _note("Hub", "hub.md", links=["handoff-2026-05-07#남은 작업 — 우선순위"]),
        _note("Handoff", "handoff-2026-05-07.md"),
    ]
    hc = HealthChecker(notes)
    assert hc.broken_links() == []
    assert "handoff-2026-05-07.md" not in hc.orphan_notes()  # counts as inbound


def test_block_reference_links_resolve() -> None:
    notes = [_note("Hub", "hub.md", links=["target^block-id"]), _note("Target", "target.md")]
    assert HealthChecker(notes).broken_links() == []


def test_hash_inside_a_title_still_resolves() -> None:
    """`#` is an ordinary character in a title; fragment stripping must not win."""
    notes = [
        _note("Hub", "hub.md", links=["Batch 2 작업 #3a 완료 - DEV 덤프 버튼 구현"]),
        _note("Batch 2 작업 #3a 완료 - DEV 덤프 버튼 구현", "batch2.md"),
    ]
    hc = HealthChecker(notes)
    assert hc.broken_links() == []
    assert "batch2.md" not in hc.orphan_notes()


def test_pure_anchor_link_is_not_broken() -> None:
    """[[#heading]] is an in-page anchor, not a reference to another note."""
    assert HealthChecker([_note("Solo", "solo.md", links=["#section"])]).broken_links() == []


def test_backslash_links_not_broken() -> None:
    """[[Projects/flatsnap/INDEX\\]] should still resolve."""
    notes = [
        _note("Hub", "hub.md", links=["Projects/flatsnap/INDEX\\"]),
        _note("FlatSnap INDEX", "Projects/flatsnap/INDEX.md"),
    ]
    hc = HealthChecker(notes)
    broken = hc.broken_links()
    assert len(broken) == 0
