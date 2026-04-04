"""Tests for the LLM Compiler (compile pass)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

from vault_rag.engine.compiler import Compiler, CompileResult
from vault_rag.ingest.scanner import ScannedNote

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_note(title: str = "Test Note", content: str = "Some content.") -> ScannedNote:
    return ScannedNote(
        path=Path("/vault/test.md"),
        relative_path="test.md",
        title=title,
        content=content,
        tags=["test"],
        links=[],
        modified=0.0,
        content_hash="abc123",
    )


def _mock_anthropic_response(text: str) -> MagicMock:
    """Return a mock Anthropic client whose messages.create returns text."""
    mock_content = MagicMock()
    mock_content.text = text

    mock_message = MagicMock()
    mock_message.content = [mock_content]

    mock_messages = MagicMock()
    mock_messages.create.return_value = mock_message

    mock_client = MagicMock()
    mock_client.messages = mock_messages

    return mock_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compile_returns_summary() -> None:
    response_json = json.dumps(
        {
            "summary": "AI agents overview",
            "tags": [],
            "suggested_links": [],
        }
    )
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="claude-test")

    result = compiler.compile(_make_note())

    assert result.summary == "AI agents overview"


def test_compile_returns_tags() -> None:
    response_json = json.dumps(
        {
            "summary": "Some summary.",
            "tags": ["ai", "agents"],
            "suggested_links": [],
        }
    )
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="claude-test")

    result = compiler.compile(_make_note())

    assert result.tags == ["ai", "agents"]


def test_compile_returns_suggested_links() -> None:
    response_json = json.dumps(
        {
            "summary": "Some summary.",
            "tags": [],
            "suggested_links": ["Large Language Models", "ReAct Framework"],
        }
    )
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="claude-test")

    result = compiler.compile(_make_note())

    assert result.suggested_links == ["Large Language Models", "ReAct Framework"]


def test_compile_handles_malformed_json() -> None:
    client = _mock_anthropic_response("not json")
    compiler = Compiler(client=client, model="claude-test")

    result = compiler.compile(_make_note())

    assert result == CompileResult()


def test_compile_batch() -> None:
    response_json = json.dumps(
        {
            "summary": "A note.",
            "tags": ["x"],
            "suggested_links": [],
        }
    )
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="claude-test")

    notes = [_make_note(title=f"Note {i}") for i in range(3)]
    results = compiler.compile_batch(notes)

    assert len(results) == 3
    assert client.messages.create.call_count == 3


def test_compile_and_write_updates_frontmatter(tmp_path: Path) -> None:
    """compile_and_write should update the note file with summary and tags."""
    note_path = tmp_path / "test.md"
    note_path.write_text(
        "---\ntags: [existing]\n---\n\n# Test\n\nContent here.\n", encoding="utf-8"
    )

    response_json = (
        '{"summary": "Test summary", "tags": ["ai", "new"], "suggested_links": ["Other Note"]}'
    )
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="test")

    note = ScannedNote(
        path=note_path,
        relative_path="test.md",
        title="Test",
        content="Content here.",
        tags=["existing"],
        links=[],
        modified=time.time(),
        content_hash="x",
    )
    compiler.compile_and_write(note)

    updated = note_path.read_text(encoding="utf-8")
    assert "Test summary" in updated
    assert "ai" in updated
    assert "existing" in updated  # preserved
    assert "## Related" in updated
    assert "[[Other Note]]" in updated


def test_compile_and_write_no_duplicate_related(tmp_path: Path) -> None:
    """Should not add ## Related twice."""
    note_path = tmp_path / "test.md"
    note_path.write_text("# Test\n\nContent.\n\n## Related\n\n- [[Existing]]\n", encoding="utf-8")

    response_json = '{"summary": "s", "tags": [], "suggested_links": ["New Link"]}'
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="test")

    note = ScannedNote(
        path=note_path,
        relative_path="test.md",
        title="Test",
        content="Content.",
        tags=[],
        links=[],
        modified=time.time(),
        content_hash="x",
    )
    compiler.compile_and_write(note)

    updated = note_path.read_text(encoding="utf-8")
    assert updated.count("## Related") == 1  # Not duplicated


def test_compile_and_write_filters_unknown_links(tmp_path: Path) -> None:
    """Only inject links that match known titles."""
    note_path = tmp_path / "test.md"
    note_path.write_text("# Test\n\nContent.\n", encoding="utf-8")

    response_json = (
        '{"summary": "s", "tags": [], "suggested_links": ["Known Note", "Unknown Note"]}'
    )
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="test")

    note = ScannedNote(
        path=note_path,
        relative_path="test.md",
        title="Test",
        content="Content.",
        tags=[],
        links=[],
        modified=time.time(),
        content_hash="x",
    )
    known = {"known note", "other"}
    compiler.compile_and_write(note, known_titles=known)

    updated = note_path.read_text(encoding="utf-8")
    assert "[[Known Note]]" in updated
    assert "[[Unknown Note]]" not in updated


def test_compile_and_write_creates_frontmatter_when_missing(tmp_path: Path) -> None:
    """Creates frontmatter if the note has none."""
    note_path = tmp_path / "no_fm.md"
    note_path.write_text("# No Frontmatter\n\nJust content.\n", encoding="utf-8")

    response_json = '{"summary": "Created FM", "tags": ["auto"], "suggested_links": []}'
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="test")

    note = ScannedNote(
        path=note_path,
        relative_path="no_fm.md",
        title="No Frontmatter",
        content="Just content.",
        tags=[],
        links=[],
        modified=time.time(),
        content_hash="y",
    )
    compiler.compile_and_write(note)

    updated = note_path.read_text(encoding="utf-8")
    assert updated.startswith("---")
    assert "Created FM" in updated
    assert "auto" in updated


# ---------------------------------------------------------------------------
# Quality scoring tests
# ---------------------------------------------------------------------------


def _note() -> ScannedNote:
    return _make_note()


def test_compile_returns_quality_score() -> None:
    response_json = '{"summary": "test", "tags": ["ai"], "suggested_links": [], "quality_score": 85, "atoms": []}'
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="test")
    result = compiler.compile(_note())
    assert result.quality_score == 85
    assert result.quality_tier == "HIGH"


def test_compile_returns_atoms() -> None:
    response_json = '{"summary": "test", "tags": [], "suggested_links": [], "quality_score": 70, "atoms": [{"title": "Concept A", "content": "Description A", "tags": ["a"]}]}'
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="test")
    result = compiler.compile(_note())
    assert len(result.atoms) == 1
    assert result.atoms[0]["title"] == "Concept A"


def test_quality_tier_calculation() -> None:
    response_json = (
        '{"summary": "test", "tags": [], "suggested_links": [], "quality_score": 30, "atoms": []}'
    )
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="test")
    result = compiler.compile(_note())
    assert result.quality_tier == "LOW"


def test_quality_tier_medium() -> None:
    response_json = (
        '{"summary": "test", "tags": [], "suggested_links": [], "quality_score": 60, "atoms": []}'
    )
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="test")
    result = compiler.compile(_note())
    assert result.quality_tier == "MEDIUM"


def test_compile_and_write_includes_quality_in_frontmatter(tmp_path: Path) -> None:
    """compile_and_write should write quality_score and quality_tier to frontmatter."""
    note_path = tmp_path / "q.md"
    note_path.write_text("# Quality Note\n\nContent.\n", encoding="utf-8")

    response_json = '{"summary": "Quality note", "tags": ["qa"], "suggested_links": [], "quality_score": 90, "atoms": []}'
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="test")

    note = ScannedNote(
        path=note_path,
        relative_path="q.md",
        title="Quality Note",
        content="Content.",
        tags=[],
        links=[],
        modified=time.time(),
        content_hash="q",
    )
    compiler.compile_and_write(note)

    updated = note_path.read_text(encoding="utf-8")
    assert "quality_score: 90" in updated
    assert "quality_tier: HIGH" in updated


# ---------------------------------------------------------------------------
# Atom extraction tests
# ---------------------------------------------------------------------------


def test_compile_and_extract_creates_atom_files(tmp_path: Path) -> None:
    """compile_and_extract should create sub-note files for each atom."""
    vault = tmp_path / "vault"
    vault.mkdir()
    note_path = vault / "multi.md"
    note_path.write_text("# Multi Concept\n\nContent about A and B.\n", encoding="utf-8")

    atoms = [
        {"title": "Concept Alpha", "content": "Alpha explanation.", "tags": ["alpha"]},
        {"title": "Concept Beta", "content": "Beta explanation.", "tags": ["beta"]},
    ]
    response_json = json.dumps(
        {
            "summary": "Multi-concept note",
            "tags": ["multi"],
            "suggested_links": [],
            "quality_score": 75,
            "atoms": atoms,
        }
    )
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="test")

    note = ScannedNote(
        path=note_path,
        relative_path="multi.md",
        title="Multi Concept",
        content="Content about A and B.",
        tags=[],
        links=[],
        modified=time.time(),
        content_hash="m",
    )
    result, created = compiler.compile_and_extract(note, vault_path=vault)

    assert len(created) == 2
    for p in created:
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert "Extracted from: [[Multi Concept]]" in content


def test_compile_and_extract_no_atoms(tmp_path: Path) -> None:
    """compile_and_extract returns empty list when note is already atomic."""
    vault = tmp_path / "vault"
    vault.mkdir()
    note_path = vault / "atomic.md"
    note_path.write_text("# Atomic Note\n\nSingle concept.\n", encoding="utf-8")

    response_json = '{"summary": "Single.", "tags": [], "suggested_links": [], "quality_score": 50, "atoms": []}'
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="test")

    note = ScannedNote(
        path=note_path,
        relative_path="atomic.md",
        title="Atomic Note",
        content="Single concept.",
        tags=[],
        links=[],
        modified=time.time(),
        content_hash="a",
    )
    result, created = compiler.compile_and_extract(note, vault_path=vault)

    assert created == []
