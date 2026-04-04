"""Tests for Indexer: embedding + batch upsert to VectorStore."""

from pathlib import Path
from unittest.mock import MagicMock

from vault_rag.config import VaultConfig
from vault_rag.engine.indexer import Indexer
from vault_rag.ingest.scanner import ScannedNote
from vault_rag.store.vector_store import VectorStore

EMBEDDING_DIM = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_note(
    title: str = "Test Note",
    content: str = "Some content",
    tags: list[str] | None = None,
    links: list[str] | None = None,
    path: str = "notes/test.md",
) -> ScannedNote:
    """ScannedNote 생성 헬퍼. 기본값으로 최소 필드를 채운다."""
    p = Path(path)
    return ScannedNote(
        path=p,
        relative_path=path,
        title=title,
        content=content,
        tags=tags or [],
        links=links or [],
        modified=0.0,
        content_hash="abc123",
    )


def _make_embed_fn(dim: int = EMBEDDING_DIM) -> MagicMock:
    """texts: list[str] → list[list[float]] 형태의 mock embed_fn."""
    mock = MagicMock()
    mock.side_effect = lambda texts: [[0.1] * dim for _ in texts]
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_index_notes_calls_embed_and_upsert(tmp_path: Path) -> None:
    """1개 노트 index 시 embed_fn 1회 호출, vs.count()==1."""
    cfg = VaultConfig(vault_path=tmp_path / "vault")
    store = VectorStore(persist_dir=tmp_path / "chroma")
    embed_fn = _make_embed_fn()

    indexer = Indexer(config=cfg, vector_store=store, embed_fn=embed_fn)
    note = _make_note(title="Hello", content="World")

    count = indexer.index([note])

    assert count == 1
    assert store.count() == 1
    embed_fn.assert_called_once()


def test_index_batches_large_sets(tmp_path: Path) -> None:
    """5개 노트를 batch_size=2로 index 시 embed_fn 3회 호출, vs.count()==5."""
    cfg = VaultConfig(vault_path=tmp_path / "vault")
    store = VectorStore(persist_dir=tmp_path / "chroma")
    embed_fn = _make_embed_fn()

    indexer = Indexer(config=cfg, vector_store=store, embed_fn=embed_fn, batch_size=2)
    notes = [_make_note(title=f"Note {i}", path=f"notes/note{i}.md") for i in range(5)]

    count = indexer.index(notes)

    assert count == 5
    assert store.count() == 5
    assert embed_fn.call_count == 3  # batches: [0,1], [2,3], [4]


def test_index_stores_metadata(tmp_path: Path) -> None:
    """index 결과에 title, tags 메타데이터가 포함되어야 한다."""
    cfg = VaultConfig(vault_path=tmp_path / "vault")
    store = VectorStore(persist_dir=tmp_path / "chroma")
    embed_fn = _make_embed_fn()

    indexer = Indexer(config=cfg, vector_store=store, embed_fn=embed_fn)
    note = _make_note(
        title="My Note",
        tags=["python", "rag"],
        path="notes/my_note.md",
    )

    indexer.index([note])

    result = store.get(ids=[note.relative_path])
    meta = result["metadatas"][0]
    assert meta["title"] == "My Note"
    assert "python" in meta["tags"]


def test_full_reindex_clears_and_rebuilds(tmp_path: Path) -> None:
    """reindex 시 기존 문서가 제거되고 새 노트만 남아야 한다."""
    cfg = VaultConfig(vault_path=tmp_path / "vault")
    store = VectorStore(persist_dir=tmp_path / "chroma")
    embed_fn = _make_embed_fn()

    indexer = Indexer(config=cfg, vector_store=store, embed_fn=embed_fn)

    old_note = _make_note(title="Old", path="notes/old.md")
    indexer.index([old_note])
    assert store.count() == 1

    new_note = _make_note(title="New", path="notes/new.md")
    count = indexer.reindex([new_note])

    assert count == 1
    assert store.count() == 1

    result = store.get(ids=[new_note.relative_path])
    assert result["ids"] == [new_note.relative_path]

    old_result = store.get(ids=[old_note.relative_path])
    assert old_result["ids"] == []
