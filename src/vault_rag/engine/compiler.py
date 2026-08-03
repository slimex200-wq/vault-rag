"""LLM Compiler: distill ScannedNotes into summaries, tags, and suggested links."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from vault_rag.ingest.scanner import ScannedNote

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_COMPILE_PROMPT = (
    "You are a knowledge management assistant. Analyze this note and return JSON:\n"
    '{{"summary": "2-3 sentence summary in the note\'s language", '
    '"tags": ["lowercase", "relevant", "tags"], '
    '"suggested_links": ["EXACT titles from the existing notes list below"], '
    '"quality_score": 0-100 integer rating the knowledge value (novelty, actionability, cross-reference potential), '
    '"atoms": [{{"title": "atomic concept title", "content": "2-3 sentence explanation", "tags": ["tag1"]}}] '
    "— extract if the note contains 2+ distinct concepts that could be separate notes. "
    "Empty array if the note is already atomic. "
    '"further_questions": ["question 1", "question 2", "question 3"] '
    "— 3 follow-up research questions based on the note's content, in the note's language."
    "}}\n\n"
    "IMPORTANT: suggested_links must ONLY contain titles from this list of existing notes:\n"
    "{known_titles_list}\n\n"
    "Note title: {title}\n"
    "Note tags: {tags}\n"
    "Note content: {content}\n\n"
    "Return ONLY valid JSON, no markdown fences."
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


def _compute_tier(score: int) -> str:
    """Return quality tier string for a 0-100 score."""
    if score >= 80:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


@dataclass(frozen=True)
class CompileResult:
    """LLM-generated metadata for a single note."""

    summary: str = ""
    tags: list[str] = field(default_factory=list)
    suggested_links: list[str] = field(default_factory=list)
    quality_score: int = 0
    quality_tier: str = ""  # HIGH, MEDIUM, LOW
    atoms: list[dict] = field(default_factory=list)
    further_questions: list[str] = field(default_factory=list)


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

    def compile(self, note: ScannedNote, known_titles: set[str] | None = None) -> CompileResult:
        """Send *note* to the LLM and return summary/tags/suggested_links."""
        titles_list = (
            ", ".join(sorted(known_titles)[:100]) if known_titles else "(no list available)"
        )
        prompt = _COMPILE_PROMPT.format(
            title=note.title,
            tags=", ".join(note.tags),
            content=note.content[:3000],
            known_titles_list=titles_list,
        )
        message = self._client.messages.create(
            model=self.model,
            max_tokens=2000,
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
        result = self.compile(note, known_titles=known_titles)
        if not result.summary and not result.tags and not result.suggested_links:
            return result

        raw = note.path.read_bytes()
        original = raw.decode("utf-8")
        content = self._update_frontmatter(original, result)
        content = self._add_related_links(content, result.suggested_links, known_titles)
        content = self._add_further_questions(content, result.further_questions)
        if content == original:
            # No effective change - skip the write so watchers don't re-fire.
            return result
        note.path.write_bytes(content.encode("utf-8"))
        return result

    def compile_and_extract(
        self,
        note: ScannedNote,
        known_titles: set[str] | None = None,
        vault_path: Path | None = None,
    ) -> tuple[CompileResult, list[Path]]:
        """Compile note, write back, and create atomic sub-notes if needed."""
        result = self.compile_and_write(note, known_titles=known_titles)

        created_paths: list[Path] = []
        if result.atoms and vault_path:
            from vault_rag.config import VaultConfig
            from vault_rag.store.note_store import NoteStore

            cfg = VaultConfig(vault_path=vault_path)
            store = NoteStore(cfg)

            folder = str(Path(note.relative_path).parent)
            if folder == ".":
                folder = "Knowledge"

            for atom in result.atoms:
                title = atom.get("title", "").strip()
                atom_content = atom.get("content", "").strip()
                tags = atom.get("tags", [])
                if title and atom_content:
                    atom_content += f"\n\n> Extracted from: [[{note.title}]]"
                    path = store.create(title=title, content=atom_content, folder=folder, tags=tags)
                    created_paths.append(path)

        return result, created_paths

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_frontmatter(self, content: str, result: CompileResult) -> str:
        """Update or create frontmatter with compiled summary and tags."""
        fm_re = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
        match = fm_re.match(content)

        if match:
            fm = match.group(1)
            body = content[match.end() :]

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

            if result.quality_score:
                if "quality_score:" in fm:
                    fm = re.sub(r"quality_score:.*", f"quality_score: {result.quality_score}", fm)
                else:
                    fm += f"\nquality_score: {result.quality_score}"
                if "quality_tier:" in fm:
                    fm = re.sub(r"quality_tier:.*", f"quality_tier: {result.quality_tier}", fm)
                else:
                    fm += f"\nquality_tier: {result.quality_tier}"

            return f"---\n{fm}\n---\n{body}"

        # No frontmatter — create one
        tags_str = ", ".join(result.tags) if result.tags else ""
        quality_lines = ""
        if result.quality_score:
            quality_lines = (
                f"quality_score: {result.quality_score}\nquality_tier: {result.quality_tier}\n"
            )
        fm = (
            f'---\nsummary: "{result.summary}"\n'
            f"tags: [{tags_str}]\n"
            f"compiled: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"{quality_lines}"
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
                if lnk.lower() in known_titles or lnk.lower().replace(" ", "-") in known_titles
            ]
        else:
            valid_links = links

        if not valid_links:
            return content

        links_md = "\n".join(f"- [[{lnk}]]" for lnk in valid_links)
        return content.rstrip() + f"\n\n## Related\n\n{links_md}\n"

    def _add_further_questions(self, content: str, questions: list[str]) -> str:
        """Append ## Further Questions section if not already present."""
        if not questions:
            return content
        if "## Further Questions" in content:
            return content

        lines = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
        return content.rstrip() + f"\n\n## Further Questions\n\n{lines}\n"

    def _parse_response(self, text: str) -> CompileResult:
        """Parse JSON response; return empty CompileResult on any failure."""
        try:
            cleaned = (text or "").strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            data = json.loads(cleaned)
            score = int(data.get("quality_score", 0))
            score = max(0, min(100, score))
            return CompileResult(
                summary=data.get("summary", ""),
                tags=data.get("tags", []),
                suggested_links=data.get("suggested_links", []),
                quality_score=score,
                quality_tier=_compute_tier(score) if score else "",
                atoms=data.get("atoms", []),
                further_questions=data.get("further_questions", []),
            )
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            logger.warning("Failed to parse LLM response: %s", exc)
            return CompileResult()
