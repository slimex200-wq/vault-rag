"""Tests for WikiIndex."""

from __future__ import annotations

from pathlib import Path

from vault_rag.engine.wiki_index import WikiIndex
from vault_rag.ingest.scanner import ScannedNote


def _make_note(title: str, rel_path: str, tags: list[str] | None = None) -> ScannedNote:
    return ScannedNote(
        path=Path(f"/vault/{rel_path}"),
        relative_path=rel_path,
        title=title,
        content=f"# {title}\n\nContent.",
        tags=tags or [],
        links=[],
        modified=0.0,
        content_hash="x",
    )


def test_generate_creates_index(tmp_path: Path) -> None:
    notes = [
        _make_note("AI Agents", "Knowledge/ai-agents.md", ["ai"]),
        _make_note("Project A", "Projects/proj-a/INDEX.md", ["project"]),
        _make_note("Daily Log", "Daily/2026-04-05.md", ["daily"]),
    ]

    idx = WikiIndex(tmp_path)
    path = idx.generate(notes)

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Wiki Index" in content
    assert "Total pages: 3" in content
    assert "[[AI Agents]]" in content
    assert "[[Project A]]" in content
    assert "Knowledge" in content
    assert "Projects" in content


def test_generate_empty_notes(tmp_path: Path) -> None:
    idx = WikiIndex(tmp_path)
    path = idx.generate([])

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Total pages: 0" in content


def test_generate_groups_by_category(tmp_path: Path) -> None:
    notes = [
        _make_note("Note A", "Knowledge/a.md"),
        _make_note("Note B", "Knowledge/patterns/b.md"),
        _make_note("Note C", "Research/tools/c.md"),
    ]

    idx = WikiIndex(tmp_path)
    path = idx.generate(notes)

    content = path.read_text(encoding="utf-8")
    assert "## Knowledge (2)" in content
    assert "## Research (1)" in content
    assert "### patterns" in content
    assert "### tools" in content
