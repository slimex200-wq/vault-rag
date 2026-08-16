"""ChromaDB persistent vector store wrapper."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings


class VectorStore:
    """ChromaDB 컬렉션을 감싸는 얇은 래퍼.

    코사인 유사도 기반으로 벡터를 저장하고 검색한다.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "vault_notes",
    ) -> None:
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """문서를 컬렉션에 upsert한다. 동일 id면 업데이트."""
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """벡터 유사도 검색. n_results가 실제 count를 초과하면 자동 조정."""
        actual_count = self.count()
        safe_n = min(n_results, actual_count) if actual_count > 0 else 1

        kwargs: dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": safe_n,
        }
        if where is not None:
            kwargs["where"] = where
        if where_document is not None:
            kwargs["where_document"] = where_document

        return self._collection.query(**kwargs)

    def delete(self, ids: list[str]) -> None:
        """지정 id 문서를 삭제한다."""
        self._collection.delete(ids=ids)

    def get(self, ids: list[str]) -> dict[str, Any]:
        """지정 id 문서를 가져온다."""
        return self._collection.get(ids=ids)

    def content_hashes(self) -> dict[str, str]:
        """저장된 문서의 `relative_path -> content_hash` 맵.

        증분 인덱싱의 기준이 된다. 이게 없으면 매 실행마다 전량 재임베딩이라
        OpenAI 백엔드에서는 실행할 때마다 전체 비용이 다시 나간다.
        """
        stored = self._collection.get(include=["metadatas"])
        result: dict[str, str] = {}
        for doc_id, metadata in zip(stored["ids"], stored["metadatas"] or [], strict=False):
            meta = metadata or {}
            path = meta.get("relative_path") or doc_id
            content_hash = meta.get("content_hash")
            if path and content_hash:
                result[str(path)] = str(content_hash)
        return result

    def count(self) -> int:
        """컬렉션 내 문서 수를 반환한다."""
        return self._collection.count()

    def reset(self) -> None:
        """컬렉션을 삭제하고 빈 상태로 재생성한다."""
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
