"""WikiIndex: auto-generate index.md catalog of all wiki pages."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from vault_rag.ingest.scanner import ScannedNote


class WikiIndex:
    """Generate and update index.md — a categorized catalog of all wiki pages."""

    def __init__(self, vault_path: Path) -> None:
        self._vault_path = vault_path
        self._path = vault_path / "index.md"

    def generate(self, notes: list[ScannedNote]) -> Path:
        """Generate index.md from all scanned notes."""
        categories: dict[str, list[ScannedNote]] = defaultdict(list)

        for note in notes:
            parts = Path(note.relative_path).parts
            if len(parts) >= 2:
                category = parts[0]
            else:
                category = "Root"
            categories[category].append(note)

        lines = [
            "---",
            "tags: [system, index]",
            f"updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "---",
            "# Wiki Index",
            "",
            f"Total pages: {len(notes)}",
            "",
        ]

        # Sort categories, put special ones first
        priority = ["Projects", "Knowledge", "Research", "Reference", "Dev", "Daily", "Decisions"]
        sorted_cats = sorted(
            categories.keys(),
            key=lambda c: (priority.index(c) if c in priority else 100, c),
        )

        for cat in sorted_cats:
            cat_notes = categories[cat]
            lines.append(f"## {cat} ({len(cat_notes)})")
            lines.append("")

            # Group by subfolder within category
            subgroups: dict[str, list[ScannedNote]] = defaultdict(list)
            for n in cat_notes:
                parts = Path(n.relative_path).parts
                if len(parts) >= 3:
                    sub = parts[1]
                else:
                    sub = ""
                subgroups[sub].append(n)

            for sub in sorted(subgroups.keys()):
                sub_notes = subgroups[sub]
                if sub:
                    lines.append(f"### {sub}")
                    lines.append("")

                for n in sorted(sub_notes, key=lambda x: x.title):
                    summary = ""
                    # Try to extract summary from frontmatter
                    if "summary:" in n.content[:500]:
                        for line in n.content[:500].splitlines():
                            if line.strip().startswith("summary:"):
                                summary = line.split(":", 1)[1].strip().strip('"').strip("'")
                                break
                    tags_str = f" `{', '.join(n.tags[:3])}`" if n.tags else ""
                    summary_str = f" — {summary[:80]}" if summary else ""
                    lines.append(f"- [[{n.title}]]{tags_str}{summary_str}")

                lines.append("")

        content = "\n".join(lines)
        self._path.write_text(content, encoding="utf-8")
        return self._path
