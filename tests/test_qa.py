"""Tests for the RAG Q&A engine."""
from __future__ import annotations

from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pytest

from vault_rag.engine.qa import QAEngine, QAResult
from vault_rag.store.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_embed(texts: list[str]) -> list[list[float]]:
    return [[0.1] * 512] * len(texts)


def _mock_anthropic(answer_text: str) -> MagicMock:
    """Return a mock Anthropic client whose messages.create returns answer_text."""
    mock_content = MagicMock()
    mock_content.text = answer_text

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


def test_answer_retrieves_and_generates(tmp_path: Path) -> None:
    """Upsert 1 doc, answer returns text and sources[0]['id'] == 'note1.md'."""
    store = VectorStore(persist_dir=tmp_path / "chroma")
    store.upsert(
        ids=["note1.md"],
        documents=["Python is a programming language."],
        metadatas=[{"title": "Python Basics"}],
        embeddings=[[0.1] * 512],
    )

    client = _mock_anthropic("Python is a high-level language.")
    engine = QAEngine(
        vector_store=store,
        embed_fn=_mock_embed,
        client=client,
        model="claude-test",
    )

    result = engine.answer("What is Python?")

    assert isinstance(result, QAResult)
    assert result.answer == "Python is a high-level language."
    assert len(result.sources) == 1
    assert result.sources[0]["id"] == "note1.md"


def test_answer_with_no_results(tmp_path: Path) -> None:
    """Empty store: still calls LLM and returns answer, sources is empty."""
    store = VectorStore(persist_dir=tmp_path / "chroma")

    client = _mock_anthropic("I don't have enough context to answer.")
    engine = QAEngine(
        vector_store=store,
        embed_fn=_mock_embed,
        client=client,
        model="claude-test",
    )

    result = engine.answer("What is the meaning of life?")

    assert result.answer == "I don't have enough context to answer."
    assert result.sources == []
    client.messages.create.assert_called_once()


def test_answer_passes_context_to_llm(tmp_path: Path) -> None:
    """Verify the user message sent to LLM contains the document content."""
    store = VectorStore(persist_dir=tmp_path / "chroma")
    store.upsert(
        ids=["note2.md"],
        documents=["Embeddings are dense vector representations."],
        metadatas=[{"title": "Embeddings Explained"}],
        embeddings=[[0.1] * 512],
    )

    client = _mock_anthropic("Embeddings represent text numerically.")
    engine = QAEngine(
        vector_store=store,
        embed_fn=_mock_embed,
        client=client,
        model="claude-test",
    )

    engine.answer("What are embeddings?")

    call_kwargs = client.messages.create.call_args
    messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][2]
    user_message_content = next(
        m["content"] for m in messages if m["role"] == "user"
    )

    assert "Embeddings are dense vector representations." in user_message_content


def test_search_only_returns_ranked_results(tmp_path: Path) -> None:
    """Upsert 2 docs, search returns them as a list of dicts with required keys."""
    store = VectorStore(persist_dir=tmp_path / "chroma")
    store.upsert(
        ids=["a.md", "b.md"],
        documents=["Document about cats.", "Document about dogs."],
        metadatas=[{"title": "Cats"}, {"title": "Dogs"}],
        embeddings=[[0.1] * 512, [0.1] * 512],
    )

    engine = QAEngine(
        vector_store=store,
        embed_fn=_mock_embed,
        client=MagicMock(),
        model="claude-test",
    )

    results = engine.search("pets", n_results=2)

    assert isinstance(results, list)
    assert len(results) == 2
    for item in results:
        assert "id" in item
        assert "document" in item
        assert "metadata" in item
        assert "distance" in item
