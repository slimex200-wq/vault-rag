"""Tests for orphan_linker: inbound-link repair driven from INDEX files."""

from __future__ import annotations

from pathlib import Path

from vault_rag.engine.orphan_linker import (
    MARKER_END,
    MARKER_START,
    OrphanLink,
    apply_links,
    plan_links,
    render_block,
)

INDEXES = {
    "INDEX.md",
    "Knowledge/INDEX.md",
    "Knowledge/patterns/INDEX.md",
    "Dev/INDEX.md",
}


def test_orphan_is_attached_to_nearest_index() -> None:
    links, unplaceable = plan_links([("Knowledge/patterns/a.md", "A")], INDEXES)
    assert unplaceable == []
    assert links[0].index_path == "Knowledge/patterns/INDEX.md"


def test_falls_back_to_a_higher_index() -> None:
    """No INDEX in the note's own folder -> walk up."""
    links, _ = plan_links([("Dev/mobile/deep/note.md", "N")], INDEXES)
    assert links[0].index_path == "Dev/INDEX.md"


def test_orphan_index_is_linked_from_its_parent_not_itself() -> None:
    links, _ = plan_links([("Knowledge/patterns/INDEX.md", "Patterns")], INDEXES)
    assert links[0].index_path == "Knowledge/INDEX.md"


def test_unplaceable_when_no_index_exists_above() -> None:
    links, unplaceable = plan_links([("Solo/note.md", "N")], set())
    assert links == []
    assert unplaceable == ["Solo/note.md"]


def test_render_block_is_sorted_and_uses_path_links() -> None:
    block = render_block(
        [
            OrphanLink("Dev/INDEX.md", "Dev/b.md", "B"),
            OrphanLink("Dev/INDEX.md", "Dev/a.md", "A"),
        ]
    )
    assert block.startswith(MARKER_START)
    assert block.endswith(MARKER_END)
    assert block.index("[[Dev/a|A]]") < block.index("[[Dev/b|B]]")
    assert ".md]]" not in block


def test_apply_appends_then_rewrites_in_place(tmp_path: Path) -> None:
    """Second run must replace the managed block, not stack another copy."""
    index = tmp_path / "Dev" / "INDEX.md"
    index.parent.mkdir(parents=True)
    index.write_text("# Dev\n\nHand written intro.\n", encoding="utf-8")

    written = apply_links(tmp_path, [OrphanLink("Dev/INDEX.md", "Dev/a.md", "A")])
    assert written == ["Dev/INDEX.md"]
    first = index.read_text(encoding="utf-8")
    assert "Hand written intro." in first
    assert first.count(MARKER_START) == 1

    apply_links(tmp_path, [OrphanLink("Dev/INDEX.md", "Dev/b.md", "B")])
    second = index.read_text(encoding="utf-8")
    assert second.count(MARKER_START) == 1  # replaced, not appended
    assert "[[Dev/b|B]]" in second
    assert "[[Dev/a|A]]" not in second
    assert "Hand written intro." in second


def test_apply_is_idempotent(tmp_path: Path) -> None:
    index = tmp_path / "Dev" / "INDEX.md"
    index.parent.mkdir(parents=True)
    index.write_text("# Dev\n", encoding="utf-8")
    links = [OrphanLink("Dev/INDEX.md", "Dev/a.md", "A")]

    apply_links(tmp_path, links)
    snapshot = index.read_text(encoding="utf-8")
    second_run = apply_links(tmp_path, links)

    assert second_run == []  # nothing changed on disk
    assert index.read_text(encoding="utf-8") == snapshot
