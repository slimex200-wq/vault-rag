"""LLM Compiler: distill ScannedNotes into summaries, tags, and suggested links."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from vault_rag.ingest.scanner import ScannedNote

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_COMPILE_PROMPT = (
    "You are a knowledge management assistant. Analyze this note and return JSON:\n"
    '{{"summary": "2-3 sentence summary in the note\'s language", '
    '"tags": ["lowercase", "relevant", "tags"], '
    '"suggested_links": ["titles of related concepts"]}}\n'
    "Note title: {title}\n"
    "Note tags: {tags}\n"
    "Note content: {content}\n"
    "Return ONLY valid JSON, no markdown fences."
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompileResult:
    """LLM-generated metadata for a single note."""

    summary: str = ""
    tags: list[str] = field(default_factory=list)
    suggested_links: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

class Compiler:
    """Send notes to the Anthropic API and parse structured metadata back."""

    def __init__(self, client: object, model: str) -> None:
        self._client = client
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, note: ScannedNote) -> CompileResult:
        """Send *note* to the LLM and return summary/tags/suggested_links."""
        prompt = _COMPILE_PROMPT.format(
            title=note.title,
            tags=", ".join(note.tags),
            content=note.content,
        )
        message = self._client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_response(message.content[0].text)

    def compile_batch(self, notes: list[ScannedNote]) -> list[CompileResult]:
        """Compile multiple notes sequentially."""
        return [self.compile(note) for note in notes]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(self, text: str) -> CompileResult:
        """Parse JSON response; return empty CompileResult on any failure."""
        try:
            data = json.loads(text)
            return CompileResult(
                summary=data.get("summary", ""),
                tags=data.get("tags", []),
                suggested_links=data.get("suggested_links", []),
            )
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            logger.warning("Failed to parse LLM response: %s", exc)
            return CompileResult()
