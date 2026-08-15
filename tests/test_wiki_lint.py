"""Tests for WikiLinter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

from vault_rag.engine.wiki_lint import LintResult, WikiLinter
from vault_rag.ingest.scanner import ScannedNote


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


def _make_note(title: str, content: str = "") -> ScannedNote:
    return ScannedNote(
        path=Path(f"/vault/{title}.md"),
        relative_path=f"{title}.md",
        title=title,
        content=content or f"---\ncompiled: 2026-04-05\n---\n# {title}\n\nContent.",
        tags=["test"],
        links=[],
        modified=0.0,
        content_hash="x",
    )


def test_lint_finds_contradictions() -> None:
    response = json.dumps(
        {
            "contradictions": [
                {"page_a": "Note A", "page_b": "Note B", "issue": "conflicting dates"}
            ],
            "stale_claims": [],
            "missing_pages": [],
            "data_gaps": [],
        }
    )
    client = _mock_anthropic(response)
    linter = WikiLinter(client=client, model="test")

    notes = [_make_note("Note A"), _make_note("Note B")]
    result = linter.lint(notes)

    assert len(result.contradictions) == 1
    assert result.contradictions[0]["issue"] == "conflicting dates"


def test_lint_finds_missing_pages() -> None:
    response = json.dumps(
        {
            "contradictions": [],
            "stale_claims": [],
            "missing_pages": ["RAG Architecture", "Vector DB Comparison"],
            "data_gaps": ["What embedding model performs best?"],
        }
    )
    client = _mock_anthropic(response)
    linter = WikiLinter(client=client, model="test")

    result = linter.lint([_make_note("Test")])

    assert len(result.missing_pages) == 2
    assert len(result.data_gaps) == 1


def test_lint_combines_structural_and_semantic() -> None:
    response = json.dumps(
        {
            "contradictions": [{"page_a": "A", "page_b": "B", "issue": "x"}],
            "stale_claims": [],
            "missing_pages": [],
            "data_gaps": [],
        }
    )
    client = _mock_anthropic(response)
    linter = WikiLinter(client=client, model="test")

    health = {"orphan_count": 5, "broken_link_count": 3, "untagged_notes": ["a.md"]}
    result = linter.lint([_make_note("Test")], health_report=health)

    assert result.orphan_count == 5
    assert result.broken_link_count == 3
    assert result.untagged_count == 1
    assert result.total_issues == 10  # 1 contradiction + 5 orphan + 3 broken + 1 untagged


def test_lint_handles_malformed_json() -> None:
    client = _mock_anthropic("not valid json")
    linter = WikiLinter(client=client, model="test")

    result = linter.lint([_make_note("Test")])

    assert isinstance(result, LintResult)
    assert result.contradictions == []


def test_lint_empty_notes() -> None:
    client = _mock_anthropic("{}")
    linter = WikiLinter(client=client, model="test")

    result = linter.lint([])

    assert result.total_issues == 0


# ---------------------------------------------------------------------------
# Rotating audit window
# ---------------------------------------------------------------------------


def _titles_in_last_prompt(client: MagicMock) -> list[str]:
    """Titles the linter actually sent to the model on its most recent call."""
    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    return re.findall(r"^\[(.+?)\]$", prompt, re.M)


def test_lint_window_rotates_and_wraps() -> None:
    client = _mock_anthropic("{}")
    linter = WikiLinter(client=client, model="test")
    notes = [_make_note(f"N{i}") for i in range(5)]

    first = linter.lint(notes, max_pages=2, offset=0)
    assert _titles_in_last_prompt(client) == ["N0", "N1"]
    assert first.pages_examined == 2
    assert first.next_offset == 2

    second = linter.lint(notes, max_pages=2, offset=first.next_offset)
    assert _titles_in_last_prompt(client) == ["N2", "N3"]
    assert second.next_offset == 4

    third = linter.lint(notes, max_pages=2, offset=second.next_offset)
    assert _titles_in_last_prompt(client) == ["N4", "N0"]  # wraps past the end
    assert third.next_offset == 1


def test_lint_rotation_sweeps_every_page() -> None:
    """Repeated runs must cover the whole vault, not re-read the head slice."""
    client = _mock_anthropic("{}")
    linter = WikiLinter(client=client, model="test")
    notes = [_make_note(f"N{i}") for i in range(7)]

    seen: set[str] = set()
    offset = 0
    for _ in range(4):  # ceil(7 / 2) passes at max_pages=2
        result = linter.lint(notes, max_pages=2, offset=offset)
        seen.update(_titles_in_last_prompt(client))
        offset = result.next_offset

    assert seen == {f"N{i}" for i in range(7)}


def test_lint_window_never_exceeds_pool() -> None:
    client = _mock_anthropic("{}")
    linter = WikiLinter(client=client, model="test")

    result = linter.lint([_make_note("Only")], max_pages=50, offset=0)

    assert result.pages_examined == 1
    assert result.next_offset == 0
