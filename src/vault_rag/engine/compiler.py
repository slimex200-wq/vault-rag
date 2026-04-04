"""LLM Compiler: distill ScannedNotes into summaries, tags, and suggested links."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

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

    def compile_and_write(
        self,
        note: ScannedNote,
        known_titles: set[str] | None = None,
    ) -> CompileResult:
        """Compile note and write results back to the file."""
        result = self.compile(note)
        if not result.summary and not result.tags and not result.suggested_links:
            return result

        content = note.path.read_text(encoding="utf-8")
        content = self._update_frontmatter(content, result)
        content = self._add_related_links(content, result.suggested_links, known_titles)
        note.path.write_text(content, encoding="utf-8")
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_frontmatter(self, content: str, result: CompileResult) -> str:
        """Update or create frontmatter with compiled summary and tags."""
        fm_re = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
        match = fm_re.match(content)

        if match:
            fm = match.group(1)
            body = content[match.end():]

            if result.summary:
                if "summary:" in fm:
                    fm = re.sub(r"summary:.*", f'summary: "{result.summary}"', fm)
                else:
                    fm += f'\nsummary: "{result.summary}"'

            if result.tags:
                # Handle both inline [a, b] and YAML list format
                existing: list[str] = []
                inline_match = re.search(r"tags:\s*\[([^\]]*)\]", fm)
                yaml_list_match = re.search(r"tags:\s*\n((?:\s+-\s+.+\n?)+)", fm)

                if inline_match:
                    existing = [
                        t.strip().strip("'\"")
                        for t in inline_match.group(1).split(",")
                        if t.strip()
                    ]
                    merged = list(dict.fromkeys(existing + result.tags))
                    fm = re.sub(
                        r"tags:\s*\[([^\]]*)\]",
                        f"tags: [{', '.join(merged)}]",
                        fm,
                    )
                elif yaml_list_match:
                    existing = [
                        line.strip().lstrip("- ").strip("'\"")
                        for line in yaml_list_match.group(1).splitlines()
                        if line.strip()
                    ]
                    merged = list(dict.fromkeys(existing + result.tags))
                    fm = re.sub(
                        r"tags:\s*\n(?:\s+-\s+.+\n?)+",
                        f"tags: [{', '.join(merged)}]\n",
                        fm,
                    )
                else:
                    fm += f"\ntags: [{', '.join(result.tags)}]"

            if "compiled:" not in fm:
                fm += f"\ncompiled: {datetime.now().strftime('%Y-%m-%d')}"

            return f"---\n{fm}\n---\n{body}"

        # No frontmatter — create one
        tags_str = ", ".join(result.tags) if result.tags else ""
        fm = (
            f'---\nsummary: "{result.summary}"\n'
            f"tags: [{tags_str}]\n"
            f"compiled: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"---\n\n"
        )
        return fm + content

    def _add_related_links(
        self,
        content: str,
        links: list[str],
        known_titles: set[str] | None = None,
    ) -> str:
        """Add ## Related section. Only include links that match known notes if provided."""
        if not links:
            return content
        if "## Related" in content:
            return content

        if known_titles:
            valid_links = [
                lnk
                for lnk in links
                if lnk.lower() in known_titles
                or lnk.lower().replace(" ", "-") in known_titles
            ]
        else:
            valid_links = links

        if not valid_links:
            return content

        links_md = "\n".join(f"- [[{lnk}]]" for lnk in valid_links)
        return content.rstrip() + f"\n\n## Related\n\n{links_md}\n"

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
