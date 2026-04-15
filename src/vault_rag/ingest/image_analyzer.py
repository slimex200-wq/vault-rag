"""ImageAnalyzer: analyze images via Claude Vision and create descriptive notes."""

from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from shutil import copy2

_INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*]')

_ANALYZE_PROMPT = (
    "Analyze this image and return a structured description for a design reference library.\n"
    "Write in Korean. Return in this exact format:\n\n"
    "Title: (concise descriptive title)\n"
    "Style: (design style keywords, e.g. dark mode, glassmorphism, brutalist)\n"
    "Colors: (primary colors and accents)\n"
    "Components: (UI components visible)\n"
    "Layout: (layout pattern)\n"
    "Tags: (comma-separated lowercase tags for search)\n"
    "Description: (2-3 sentence description)"
)


@dataclass(frozen=True)
class ImageAnalysisResult:
    """Result of an image analysis."""

    title: str
    style: str
    colors: str
    components: str
    layout: str
    tags: list[str]
    description: str
    image_path: Path


class ImageAnalyzer:
    """Analyze images with Claude Vision and create Obsidian notes."""

    def __init__(self, client: object, model: str, vault_path: Path) -> None:
        self._client = client
        self.model = model
        self.vault_path = vault_path

    def analyze(self, image_path: Path) -> ImageAnalysisResult:
        """Send image to Claude Vision and parse the structured response."""
        image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"

        message = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": _ANALYZE_PROMPT},
                    ],
                }
            ],
        )

        return self._parse_response(message.content[0].text, image_path)

    def analyze_and_save(
        self,
        image_path: Path,
        folder: str = "Reference/design-ref",
    ) -> tuple[ImageAnalysisResult, Path]:
        """Analyze image, copy to vault, create note with description."""
        result = self.analyze(image_path)

        target_dir = self.vault_path / folder
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy image to vault
        image_dest = target_dir / image_path.name
        if not image_dest.exists():
            copy2(image_path, image_dest)

        # Create note
        sanitized = _INVALID_CHARS_RE.sub("", result.title).strip() or "untitled"
        note_path = target_dir / f"{sanitized}.md"

        tags_str = ", ".join(result.tags)
        note_content = (
            f"---\n"
            f'title: "{result.title}"\n'
            f"tags: [{tags_str}]\n"
            f"created: {date.today().isoformat()}\n"
            f'summary: "{result.description[:100]}"\n'
            f"---\n\n"
            f"# {result.title}\n\n"
            f"![[{image_path.name}]]\n\n"
            f"- **Style**: {result.style}\n"
            f"- **Colors**: {result.colors}\n"
            f"- **Components**: {result.components}\n"
            f"- **Layout**: {result.layout}\n\n"
            f"{result.description}\n"
        )
        note_path.write_text(note_content, encoding="utf-8")

        return result, note_path

    def _parse_response(self, text: str, image_path: Path) -> ImageAnalysisResult:
        """Parse structured text response into ImageAnalysisResult."""
        fields: dict[str, str] = {}
        for line in text.splitlines():
            for key in ("Title", "Style", "Colors", "Components", "Layout", "Tags", "Description"):
                if line.startswith(f"{key}:"):
                    fields[key.lower()] = line.split(":", 1)[1].strip()
                    break

        tags_raw = fields.get("tags", "")
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

        return ImageAnalysisResult(
            title=fields.get("title", image_path.stem),
            style=fields.get("style", ""),
            colors=fields.get("colors", ""),
            components=fields.get("components", ""),
            layout=fields.get("layout", ""),
            tags=tags,
            description=fields.get("description", ""),
            image_path=image_path,
        )
