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
    (tmp_path / "Dev" / "a.md").write_text("A", encoding="utf-8")
    (tmp_path / "Dev" / "b.md").write_text("B", encoding="utf-8")

    written = apply_links(tmp_path, [OrphanLink("Dev/INDEX.md", "Dev/a.md", "A")])
    assert written == ["Dev/INDEX.md"]
    first = index.read_text(encoding="utf-8")
    assert "Hand written intro." in first
    assert first.count(MARKER_START) == 1

    apply_links(tmp_path, [OrphanLink("Dev/INDEX.md", "Dev/b.md", "B")])
    second = index.read_text(encoding="utf-8")
    assert second.count(MARKER_START) == 1  # one block, rewritten
    assert "[[Dev/b|B]]" in second
    assert "[[Dev/a|A]]" in second  # kept: the block is its only inbound edge
    assert "Hand written intro." in second


def test_apply_is_idempotent(tmp_path: Path) -> None:
    index = tmp_path / "Dev" / "INDEX.md"
    index.parent.mkdir(parents=True)
    index.write_text("# Dev\n", encoding="utf-8")
    (tmp_path / "Dev" / "a.md").write_text("A", encoding="utf-8")
    links = [OrphanLink("Dev/INDEX.md", "Dev/a.md", "A")]

    apply_links(tmp_path, links)
    snapshot = index.read_text(encoding="utf-8")
    second_run = apply_links(tmp_path, links)

    assert second_run == []  # nothing changed on disk
    assert index.read_text(encoding="utf-8") == snapshot


def test_prior_entries_survive_so_repaired_orphans_do_not_regress(tmp_path: Path) -> None:
    """The managed block *is* the inbound edge.

    Dropping last run's entries un-links those notes, so the next scan reports
    them as orphans again and the linker oscillates instead of converging.
    Real regression: Daily/2026-07-25..08-03 went back to orphan the moment
    08-28..30 were written into the block.
    """
    index = tmp_path / "Dev" / "INDEX.md"
    index.parent.mkdir(parents=True)
    index.write_text("# Dev\n", encoding="utf-8")
    (tmp_path / "Dev" / "a.md").write_text("A", encoding="utf-8")
    (tmp_path / "Dev" / "b.md").write_text("B", encoding="utf-8")

    apply_links(tmp_path, [OrphanLink("Dev/INDEX.md", "Dev/a.md", "A")])
    apply_links(tmp_path, [OrphanLink("Dev/INDEX.md", "Dev/b.md", "B")])

    body = index.read_text(encoding="utf-8")
    assert "[[Dev/a|A]]" in body
    assert "[[Dev/b|B]]" in body
    assert body.count(MARKER_START) == 1


def test_entry_for_a_deleted_note_is_pruned(tmp_path: Path) -> None:
    """Carrying a link to a note that no longer exists would be a broken link."""
    index = tmp_path / "Dev" / "INDEX.md"
    index.parent.mkdir(parents=True)
    index.write_text("# Dev\n", encoding="utf-8")
    (tmp_path / "Dev" / "gone.md").write_text("G", encoding="utf-8")
    (tmp_path / "Dev" / "b.md").write_text("B", encoding="utf-8")

    apply_links(tmp_path, [OrphanLink("Dev/INDEX.md", "Dev/gone.md", "G")])
    (tmp_path / "Dev" / "gone.md").unlink()
    apply_links(tmp_path, [OrphanLink("Dev/INDEX.md", "Dev/b.md", "B")])

    body = index.read_text(encoding="utf-8")
    assert "[[Dev/gone|G]]" not in body
    assert "[[Dev/b|B]]" in body
