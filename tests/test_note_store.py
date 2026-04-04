"""Tests for NoteStore CRUD operations."""

from pathlib import Path

import pytest

from vault_rag.config import VaultConfig
from vault_rag.store.note_store import NoteStore


@pytest.fixture()
def store(cfg: VaultConfig) -> NoteStore:
    return NoteStore(cfg)


def test_create_note(store: NoteStore, tmp_vault: Path) -> None:
    path = store.create(
        title="My Test Note",
        content="This is the body.",
        folder="Knowledge",
        tags=["test", "demo"],
    )

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "# My Test Note" in text
    assert "This is the body." in text
    assert "tags:" in text
    assert "test" in text
    assert "demo" in text
    assert "created:" in text


def test_read_note(store: NoteStore) -> None:
    path = store.create(
        title="Readable Note",
        content="Hello world.",
        folder="Knowledge",
    )

    result = store.read(str(path.relative_to(store.vault)))
    assert result is not None
    assert "Hello world." in result


def test_read_note_missing_returns_none(store: NoteStore) -> None:
    result = store.read("nonexistent/file.md")
    assert result is None


def test_list_notes(store: NoteStore, tmp_vault: Path) -> None:
    store.create(title="Note A", content="AAA", folder="Projects")
    store.create(title="Note B", content="BBB", folder="Projects")

    notes = store.list_folder("Projects")

    # tmp_vault already has project-alpha.md — expect at least 2 new ones
    created = [p for p in notes if p.stem in ("note-a", "note-b")]
    assert len(created) == 2
    # list is sorted
    assert notes == sorted(notes)


def test_update_note(store: NoteStore) -> None:
    path = store.create(
        title="Updatable Note",
        content="Original content.",
        folder="Knowledge",
    )

    store.update(path, "Updated content.")

    text = path.read_text(encoding="utf-8")
    assert "Updated content." in text
    assert "Original content." not in text
    # frontmatter and heading are preserved
    assert "# Updatable Note" in text
    assert "created:" in text
