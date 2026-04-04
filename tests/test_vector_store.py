"""Tests for VectorStore (ChromaDB wrapper)."""

from pathlib import Path

import pytest

from vault_rag.store.vector_store import VectorStore

EMBEDDING_DIM = 512
MOCK_EMBEDDING = [0.1] * EMBEDDING_DIM


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(persist_dir=tmp_path / "chroma")


def test_upsert_and_query(store: VectorStore) -> None:
    """upsert 1개 문서 후 query 시 해당 문서가 반환되어야 한다."""
    store.upsert(
        ids=["doc1"],
        documents=["Hello world"],
        metadatas=[{"source": "test.md"}],
        embeddings=[MOCK_EMBEDDING],
    )

    results = store.query(query_embeddings=[MOCK_EMBEDDING], n_results=1)

    assert results["ids"][0] == ["doc1"]
    assert results["documents"][0] == ["Hello world"]


def test_upsert_updates_existing(store: VectorStore) -> None:
    """동일 id로 upsert 하면 기존 문서가 업데이트되어야 한다."""
    store.upsert(
        ids=["doc1"],
        documents=["Original content"],
        metadatas=[{"source": "test.md"}],
        embeddings=[MOCK_EMBEDDING],
    )
    store.upsert(
        ids=["doc1"],
        documents=["Updated content"],
        metadatas=[{"source": "test.md"}],
        embeddings=[MOCK_EMBEDDING],
    )

    result = store.get(ids=["doc1"])
    assert result["documents"] == ["Updated content"]
    assert store.count() == 1


def test_delete(store: VectorStore) -> None:
    """upsert 2개 후 1개 삭제 시 count=1이어야 한다."""
    store.upsert(
        ids=["doc1", "doc2"],
        documents=["First", "Second"],
        metadatas=[{"source": "a.md"}, {"source": "b.md"}],
        embeddings=[MOCK_EMBEDDING, MOCK_EMBEDDING],
    )
    assert store.count() == 2

    store.delete(ids=["doc1"])

    assert store.count() == 1
    result = store.get(ids=["doc2"])
    assert result["documents"] == ["Second"]


def test_count(store: VectorStore) -> None:
    """초기 count=0, upsert 2개 후 count=2."""
    assert store.count() == 0

    store.upsert(
        ids=["doc1", "doc2"],
        documents=["Alpha", "Beta"],
        metadatas=[{"source": "a.md"}, {"source": "b.md"}],
        embeddings=[MOCK_EMBEDDING, MOCK_EMBEDDING],
    )

    assert store.count() == 2


def test_get_by_ids(store: VectorStore) -> None:
    """특정 id로 문서를 정확히 가져올 수 있어야 한다."""
    store.upsert(
        ids=["doc1", "doc2", "doc3"],
        documents=["Alpha", "Beta", "Gamma"],
        metadatas=[{"source": "a.md"}, {"source": "b.md"}, {"source": "c.md"}],
        embeddings=[MOCK_EMBEDDING, MOCK_EMBEDDING, MOCK_EMBEDDING],
    )

    result = store.get(ids=["doc2"])

    assert result["ids"] == ["doc2"]
    assert result["documents"] == ["Beta"]
