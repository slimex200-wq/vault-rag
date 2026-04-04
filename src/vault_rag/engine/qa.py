"""RAG Q&A Engine: retrieve-rank-generate pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from vault_rag.store.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_QA_PROMPT = (
    "You are a knowledge assistant. Answer based ONLY on the provided context.\n"
    "If the context doesn't contain relevant information, say so.\n"
    "Answer in the same language as the question.\n\n"
    "Context from knowledge base:\n"
    "{context}\n\n"
    "Question: {question}"
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class QAResult:
    """Result of a Q&A query."""

    answer: str
    sources: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# QAEngine
# ---------------------------------------------------------------------------


class QAEngine:
    """Retrieve relevant docs, generate an answer via LLM."""

    def __init__(
        self,
        vector_store: VectorStore,
        embed_fn: Callable[[list[str]], list[list[float]]],
        client: object,
        model: str,
    ) -> None:
        self._store = vector_store
        self._embed_fn = embed_fn
        self._client = client
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, n_results: int = 10) -> list[dict]:
        """Embed query, search vector store, return ranked list of dicts.

        Each dict contains: id, document, metadata, distance.
        Returns empty list when the store is empty.
        """
        if self._store.count() == 0:
            return []

        embeddings = self._embed_fn([query])
        raw = self._store.query(query_embeddings=embeddings, n_results=n_results)

        ids: list[str] = raw["ids"][0]
        documents: list[str] = raw["documents"][0]
        metadatas: list[dict] = raw["metadatas"][0]
        distances: list[float] = raw["distances"][0]

        return [
            {
                "id": ids[i],
                "document": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
            }
            for i in range(len(ids))
        ]

    def find_duplicates(self, text: str, threshold: float = 0.3) -> list[dict]:
        """Find notes semantically similar to text.

        Returns notes with distance strictly less than *threshold*.
        """
        if self._store.count() == 0:
            return []
        results = self.search(text, n_results=5)
        return [r for r in results if r.get("distance") is not None and r["distance"] < threshold]

    def answer(self, question: str, n_context: int = 5) -> QAResult:
        """Full RAG pipeline: search -> format context -> LLM -> QAResult."""
        if self._client is None:
            raise RuntimeError(
                "QAEngine.answer() requires a client. "
                "This instance was created for search/dedup only."
            )
        sources = self.search(question, n_results=n_context)

        if sources:
            context_parts = []
            for src in sources:
                title = src["metadata"].get("title", src["id"])
                content = src["document"][:1000]
                context_parts.append(f"[{title}]\n{content}")
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "(no relevant context found)"

        prompt = _QA_PROMPT.format(context=context, question=question)

        message = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        answer_text: str = message.content[0].text
        return QAResult(answer=answer_text, sources=sources)
