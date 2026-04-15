"""Output Generator: produce slides, reports, and summaries from vault knowledge."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from vault_rag.store.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SLIDES_PROMPT = (
    "You are a presentation expert. Create a markdown slide deck from the provided context.\n"
    "Use `---` as slide separators. Each slide should have a heading (##) and 3-5 bullet points.\n"
    "Include a title slide and a summary slide.\n"
    "Write in the same language as the context.\n\n"
    "Topic: {topic}\n\n"
    "Context from knowledge base:\n{context}\n\n"
    "Create the slide deck now."
)

_REPORT_PROMPT = (
    "You are a research analyst. Write a comprehensive report from the provided context.\n"
    "Structure: ## Executive Summary, ## Key Findings, ## Analysis, ## Recommendations, ## Sources.\n"
    "Cite sources using [Note Title] format. Write in the same language as the context.\n\n"
    "Topic: {topic}\n\n"
    "Context from knowledge base:\n{context}\n\n"
    "Write the report now."
)

_SUMMARY_PROMPT = (
    "You are an executive briefing writer. Write a concise summary from the provided context.\n"
    "Structure: key points as bullet list, then 1-2 paragraph synthesis.\n"
    "Write in the same language as the context.\n\n"
    "Topic: {topic}\n\n"
    "Context from knowledge base:\n{context}\n\n"
    "Write the briefing now."
)

_CHART_PROMPT = (
    "You are a data visualization expert. Create a Mermaid diagram from the provided context.\n"
    "Choose the most appropriate chart type for the data:\n"
    "- `graph TD` for relationship/dependency maps\n"
    "- `pie` for distribution/proportion\n"
    "- `timeline` for chronological data\n"
    "- `mindmap` for topic hierarchies\n"
    "- `xychart-beta` for numerical comparisons\n\n"
    "Return ONLY the Mermaid code block (```mermaid ... ```), then a brief explanation.\n"
    "Write labels in the same language as the context.\n\n"
    "Topic: {topic}\n\n"
    "Context from knowledge base:\n{context}\n\n"
    "Create the chart now."
)

_MAX_TOKENS = {
    "slides": 2000,
    "report": 4000,
    "summary": 1024,
    "chart": 2000,
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerateResult:
    """Result of an output generation."""

    content: str
    format: str  # "slides" | "report" | "summary"
    sources: list[dict] = field(default_factory=list)
    output_path: Path | None = None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator:
    """Generate structured outputs from vault knowledge."""

    def __init__(
        self,
        client: object,
        model: str,
        embed_fn: Callable[[list[str]], list[list[float]]],
        vector_store: VectorStore,
    ) -> None:
        self._client = client
        self.model = model
        self._embed_fn = embed_fn
        self._store = vector_store

    def _search(self, topic: str, n_results: int) -> list[dict]:
        """Search vector store for relevant context."""
        if self._store.count() == 0:
            return []
        embeddings = self._embed_fn([topic])
        raw = self._store.query(query_embeddings=embeddings, n_results=n_results)
        ids = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]
        return [
            {
                "id": ids[i],
                "document": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
            }
            for i in range(len(ids))
        ]

    def _build_context(self, sources: list[dict]) -> str:
        """Format sources into context string."""
        if not sources:
            return "(no relevant context found)"
        parts = []
        for src in sources:
            title = src["metadata"].get("title", src["id"])
            content = src["document"][:1500]
            parts.append(f"[{title}]\n{content}")
        return "\n\n---\n\n".join(parts)

    def _generate(self, prompt: str, fmt: str, sources: list[dict]) -> GenerateResult:
        """Call LLM and return GenerateResult."""
        message = self._client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS[fmt],
            messages=[{"role": "user", "content": prompt}],
        )
        return GenerateResult(
            content=message.content[0].text,
            format=fmt,
            sources=sources,
        )

    def generate_slides(self, topic: str, n_context: int = 10) -> GenerateResult:
        """Generate a markdown slide deck on the given topic."""
        sources = self._search(topic, n_context)
        context = self._build_context(sources)
        prompt = _SLIDES_PROMPT.format(topic=topic, context=context)
        return self._generate(prompt, "slides", sources)

    def generate_report(self, topic: str, n_context: int = 10) -> GenerateResult:
        """Generate a comprehensive report on the given topic."""
        sources = self._search(topic, n_context)
        context = self._build_context(sources)
        prompt = _REPORT_PROMPT.format(topic=topic, context=context)
        return self._generate(prompt, "report", sources)

    def generate_summary(self, topic: str, n_context: int = 5) -> GenerateResult:
        """Generate a concise executive briefing on the given topic."""
        sources = self._search(topic, n_context)
        context = self._build_context(sources)
        prompt = _SUMMARY_PROMPT.format(topic=topic, context=context)
        return self._generate(prompt, "summary", sources)

    def generate_chart(self, topic: str, n_context: int = 10) -> GenerateResult:
        """Generate a Mermaid chart/diagram on the given topic."""
        sources = self._search(topic, n_context)
        context = self._build_context(sources)
        prompt = _CHART_PROMPT.format(topic=topic, context=context)
        return self._generate(prompt, "chart", sources)

    def save(self, result: GenerateResult, output_dir: Path) -> Path:
        """Save generated output to a markdown file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.format}_{date_str}.md"
        path = output_dir / filename
        path.write_text(result.content, encoding="utf-8")
        return path
