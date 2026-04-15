"""RAG Q&A Engine: retrieve-rank-generate pipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

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

_QA_FOLLOW_UP_PROMPT = (
    "You are a knowledge assistant. Answer based ONLY on the provided context.\n"
    "If the context doesn't contain relevant information, say so.\n"
    "Answer in the same language as the question.\n"
    "After your answer, add a section '## Follow-up Questions' with 3 follow-up questions.\n\n"
    "Context from knowledge base:\n"
    "{context}\n\n"
    "Question: {question}"
)

_EXTRACT_PROMPT = (
    "You are a knowledge extraction expert. Extract and synthesize the key information "
    "relevant to the query from the provided search results.\n"
    "Be concise. Write in the same language as the query.\n\n"
    "Search results:\n{context}\n\n"
    "Query: {question}"
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

    def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        filter_tags: list[str] | None = None,
        keyword: str | None = None,
    ) -> list[dict]:
        """Hybrid search: vector similarity + optional keyword and tag filters."""
        if self._store.count() == 0:
            return []

        embeddings = self._embed_fn([query])
        where_document = {"$contains": keyword} if keyword else None
        raw = self._store.query(
            query_embeddings=embeddings,
            n_results=n_results,
            where_document=where_document,
        )

        ids: list[str] = raw["ids"][0]
        documents: list[str] = raw["documents"][0]
        metadatas: list[dict] = raw["metadatas"][0]
        distances: list[float] = raw["distances"][0]

        results = [
            {
                "id": ids[i],
                "document": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
            }
            for i in range(len(ids))
        ]

        if filter_tags:
            filtered = []
            for r in results:
                note_tags = r["metadata"].get("tags", "")
                if isinstance(note_tags, str):
                    note_tags = [t.strip() for t in note_tags.split(",") if t.strip()]
                if any(ft.lower() in [t.lower() for t in note_tags] for ft in filter_tags):
                    filtered.append(r)
            return filtered

        return results

    def extract(self, results: list[dict], question: str) -> str:
        """Extract key content from search results via LLM."""
        if self._client is None:
            raise RuntimeError("extract() requires a client.")

        context_parts = []
        for src in results:
            title = src["metadata"].get("title", src["id"])
            content = src["document"][:1000]
            context_parts.append(f"[{title}]\n{content}")
        context = "\n\n---\n\n".join(context_parts) if context_parts else "(no results)"

        prompt = _EXTRACT_PROMPT.format(context=context, question=question)
        message = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def find_duplicates(self, text: str, threshold: float = 0.3) -> list[dict]:
        """Find notes semantically similar to text.

        Returns notes with distance strictly less than *threshold*.
        """
        if self._store.count() == 0:
            return []
        results = self.search(text, n_results=5)
        return [r for r in results if r.get("distance") is not None and r["distance"] < threshold]

    def answer(self, question: str, n_context: int = 5, follow_up: bool = False) -> QAResult:
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

        template = _QA_FOLLOW_UP_PROMPT if follow_up else _QA_PROMPT
        prompt = template.format(context=context, question=question)

        message = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        answer_text: str = message.content[0].text
        return QAResult(answer=answer_text, sources=sources)
