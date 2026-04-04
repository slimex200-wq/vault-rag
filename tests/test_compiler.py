"""Tests for the LLM Compiler (compile pass)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vault_rag.engine.compiler import CompileResult, Compiler
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
    response_json = json.dumps({
        "summary": "AI agents overview",
        "tags": [],
        "suggested_links": [],
    })
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="claude-test")

    result = compiler.compile(_make_note())

    assert result.summary == "AI agents overview"


def test_compile_returns_tags() -> None:
    response_json = json.dumps({
        "summary": "Some summary.",
        "tags": ["ai", "agents"],
        "suggested_links": [],
    })
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="claude-test")

    result = compiler.compile(_make_note())

    assert result.tags == ["ai", "agents"]


def test_compile_returns_suggested_links() -> None:
    response_json = json.dumps({
        "summary": "Some summary.",
        "tags": [],
        "suggested_links": ["Large Language Models", "ReAct Framework"],
    })
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
    response_json = json.dumps({
        "summary": "A note.",
        "tags": ["x"],
        "suggested_links": [],
    })
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="claude-test")

    notes = [_make_note(title=f"Note {i}") for i in range(3)]
    results = compiler.compile_batch(notes)

    assert len(results) == 3
    assert client.messages.create.call_count == 3
