"""Tests for the Output Generator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from vault_rag.engine.generator import GenerateResult, Generator
from vault_rag.store.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_embed(texts: list[str]) -> list[list[float]]:
    return [[0.1] * 512] * len(texts)


def _mock_anthropic(text: str) -> MagicMock:
    mock_content = MagicMock()
    mock_content.text = text

    mock_message = MagicMock()
    mock_message.content = [mock_content]

    mock_messages = MagicMock()
    mock_messages.create.return_value = mock_message

    mock_client = MagicMock()
    mock_client.messages = mock_messages
    return mock_client


def _setup_store(tmp_path: Path) -> VectorStore:
    store = VectorStore(persist_dir=tmp_path / "chroma")
    store.upsert(
        ids=["note1.md", "note2.md"],
        documents=["AI agents are autonomous programs.", "RAG combines retrieval with generation."],
        metadatas=[
            {"title": "AI Agents", "tags": "ai", "relative_path": "note1.md"},
            {"title": "RAG Overview", "tags": "rag", "relative_path": "note2.md"},
        ],
        embeddings=[[0.1] * 512, [0.2] * 512],
    )
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_slides(tmp_path: Path) -> None:
    store = _setup_store(tmp_path)
    client = _mock_anthropic("## Title Slide\n\n---\n\n## Slide 1\n\n- Point 1\n- Point 2")
    gen = Generator(client=client, model="test", embed_fn=_mock_embed, vector_store=store)

    result = gen.generate_slides("AI agents")

    assert isinstance(result, GenerateResult)
    assert result.format == "slides"
    assert "Slide" in result.content
    assert len(result.sources) > 0
    client.messages.create.assert_called_once()


def test_generate_report(tmp_path: Path) -> None:
    store = _setup_store(tmp_path)
    client = _mock_anthropic("## Executive Summary\n\nKey findings about AI.")
    gen = Generator(client=client, model="test", embed_fn=_mock_embed, vector_store=store)

    result = gen.generate_report("AI agents")

    assert result.format == "report"
    assert "Executive Summary" in result.content


def test_generate_summary(tmp_path: Path) -> None:
    store = _setup_store(tmp_path)
    client = _mock_anthropic("- Key point 1\n- Key point 2\n\nSynthesis paragraph.")
    gen = Generator(client=client, model="test", embed_fn=_mock_embed, vector_store=store)

    result = gen.generate_summary("AI agents")

    assert result.format == "summary"
    assert "Key point" in result.content


def test_generate_with_empty_store(tmp_path: Path) -> None:
    store = VectorStore(persist_dir=tmp_path / "chroma_empty")
    client = _mock_anthropic("No context available.")
    gen = Generator(client=client, model="test", embed_fn=_mock_embed, vector_store=store)

    result = gen.generate_slides("anything")

    assert result.sources == []
    assert result.content == "No context available."


def test_save_creates_file(tmp_path: Path) -> None:
    store = _setup_store(tmp_path)
    client = _mock_anthropic("## Report Content")
    gen = Generator(client=client, model="test", embed_fn=_mock_embed, vector_store=store)

    result = gen.generate_report("test topic")
    output_dir = tmp_path / "output"
    path = gen.save(result, output_dir)

    assert path.exists()
    assert path.read_text(encoding="utf-8") == "## Report Content"
    assert "report_" in path.name


def test_generate_slides_uses_correct_max_tokens(tmp_path: Path) -> None:
    store = _setup_store(tmp_path)
    client = _mock_anthropic("slides")
    gen = Generator(client=client, model="test", embed_fn=_mock_embed, vector_store=store)

    gen.generate_slides("topic")

    call_kwargs = client.messages.create.call_args[1]
    assert call_kwargs["max_tokens"] == 2000


def test_generate_report_uses_correct_max_tokens(tmp_path: Path) -> None:
    store = _setup_store(tmp_path)
    client = _mock_anthropic("report")
    gen = Generator(client=client, model="test", embed_fn=_mock_embed, vector_store=store)

    gen.generate_report("topic")

    call_kwargs = client.messages.create.call_args[1]
    assert call_kwargs["max_tokens"] == 4000


def test_generate_chart(tmp_path: Path) -> None:
    store = _setup_store(tmp_path)
    client = _mock_anthropic("```mermaid\ngraph TD\n  A-->B\n```\n\nExplanation.")
    gen = Generator(client=client, model="test", embed_fn=_mock_embed, vector_store=store)

    result = gen.generate_chart("AI agents")

    assert result.format == "chart"
    assert "mermaid" in result.content
    assert len(result.sources) > 0


def test_generate_chart_uses_correct_max_tokens(tmp_path: Path) -> None:
    store = _setup_store(tmp_path)
    client = _mock_anthropic("chart")
    gen = Generator(client=client, model="test", embed_fn=_mock_embed, vector_store=store)

    gen.generate_chart("topic")

    call_kwargs = client.messages.create.call_args[1]
    assert call_kwargs["max_tokens"] == 2000
