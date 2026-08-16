"""Embedding Indexer: batch-upserts ScannedNotes into VectorStore."""

from __future__ import annotations

from collections.abc import Callable

from vault_rag.config import VaultConfig
from vault_rag.ingest.scanner import ScannedNote
from vault_rag.store.vector_store import VectorStore


class Indexer:
    """ScannedNote 목록을 임베딩하여 VectorStore에 upsert한다."""

    def __init__(
        self,
        config: VaultConfig,
        vector_store: VectorStore,
        embed_fn: Callable[[list[str]], list[list[float]]],
        batch_size: int = 50,
    ) -> None:
        self._config = config
        self._store = vector_store
        self._embed_fn = embed_fn
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, notes: list[ScannedNote]) -> int:
        """Incremental upsert. 배치 단위로 처리. 인덱싱된 노트 수 반환."""
        total = 0
        for batch in self._batches(notes):
            self._upsert_batch(batch)
            total += len(batch)
        return total

    def reindex(self, notes: list[ScannedNote]) -> int:
        """Full reindex: 스토어 초기화 후 전체 재인덱싱. 인덱싱된 노트 수 반환."""
        self._store.reset()
        return self.index(notes)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _batches(self, notes: list[ScannedNote]) -> list[list[ScannedNote]]:
        """notes를 batch_size 단위로 분할한다."""
        return [notes[i : i + self._batch_size] for i in range(0, len(notes), self._batch_size)]

    def _upsert_batch(self, batch: list[ScannedNote]) -> None:
        """단일 배치를 임베딩하고 VectorStore에 upsert한다."""
        texts = [self._note_to_text(note) for note in batch]
        embeddings = self._embed_fn(texts)

        self._store.upsert(
            ids=[note.relative_path for note in batch],
            documents=texts,
            metadatas=[self._note_to_metadata(note) for note in batch],
            embeddings=embeddings,
        )

    def _note_to_text(self, note: ScannedNote) -> str:
        """임베딩용 텍스트: title + tags + content[:2000]."""
        tags_str = " ".join(f"#{t}" for t in note.tags) if note.tags else ""
        parts = [note.title]
        if tags_str:
            parts.append(tags_str)
        parts.append(note.content[:2000])
        return "\n".join(parts)

    def _note_to_metadata(self, note: ScannedNote) -> dict:
        """메타데이터 dict: title, tags, links, content_hash, relative_path."""
        return {
            "title": note.title,
            "tags": ",".join(note.tags),
            "links": ",".join(note.links),
            "content_hash": note.content_hash,
            "relative_path": note.relative_path,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_openai_embed_fn(
    model: str = "text-embedding-3-small",
    dimensions: int = 512,
) -> Callable[[list[str]], list[list[float]]]:
    """OpenAI API를 사용하는 임베딩 함수를 생성한다. 종량 과금."""
    from openai import OpenAI

    client = OpenAI()

    def embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(
            input=texts,
            model=model,
            dimensions=dimensions,
        )
        return [item.embedding for item in response.data]

    return embed


def create_local_embed_fn() -> Callable[[list[str]], list[list[float]]]:
    """온디바이스 임베딩 함수. chromadb 번들 onnxruntime 을 쓰므로 과금 없음.

    임베딩 엔드포인트는 어떤 채팅 구독에도 포함되지 않아 OpenAI 를 쓰는 한
    종량 과금을 피할 수 없다. 전체 재인덱싱을 마음껏 돌리려면 로컬이 답이다.
    """
    from chromadb.utils import embedding_functions

    encoder = embedding_functions.DefaultEmbeddingFunction()

    def embed(texts: list[str]) -> list[list[float]]:
        # float() per element, not list(): the encoder yields numpy float32 and
        # chroma's validator only accepts native ints/floats, rejecting the
        # whole batch with "Expected embeddings to be a list of floats...".
        return [[float(value) for value in vector] for vector in encoder(texts)]

    return embed


def create_embed_fn(config: VaultConfig) -> Callable[[list[str]], list[list[float]]]:
    """설정에 따라 임베딩 백엔드를 고른다.

    Raises:
        ValueError: 알 수 없는 provider. 조용히 유료 경로로 폴백하지 않는다.
    """
    provider = config.embedding_provider
    if provider == "local":
        return create_local_embed_fn()
    if provider == "openai":
        return create_openai_embed_fn(config.embedding_model, config.embedding_dimensions)
    raise ValueError(f"Unknown embedding_provider: {provider!r} (expected 'local' or 'openai')")
