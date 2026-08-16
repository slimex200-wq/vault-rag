"""Tests for embedding backend selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vault_rag.config import VaultConfig
from vault_rag.engine.indexer import create_embed_fn


def test_default_provider_is_local() -> None:
    """Re-indexing must not cost money by default; embeddings are never covered
    by a chat subscription, so the free path has to be the default."""
    assert VaultConfig().embedding_provider == "local"


def test_local_provider_uses_the_on_device_encoder() -> None:
    with patch("vault_rag.engine.indexer.create_local_embed_fn") as local:
        embed_fn = create_embed_fn(VaultConfig(embedding_provider="local"))
    local.assert_called_once_with()
    assert embed_fn is local.return_value


def test_openai_provider_passes_model_and_dimensions() -> None:
    config = VaultConfig(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=512,
    )
    with patch("vault_rag.engine.indexer.create_openai_embed_fn") as openai_fn:
        create_embed_fn(config)
    openai_fn.assert_called_once_with("text-embedding-3-small", 512)


def test_unknown_provider_raises_instead_of_falling_back() -> None:
    """A typo must not silently route to the metered backend."""
    with pytest.raises(ValueError, match="Unknown embedding_provider"):
        create_embed_fn(VaultConfig(embedding_provider="opnai"))


def test_local_encoder_converts_numpy_scalars_to_native_floats() -> None:
    """chroma rejects numpy float32; the whole upsert batch fails if we pass it through."""
    import numpy as np

    encoder = MagicMock(return_value=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    with patch("chromadb.utils.embedding_functions.DefaultEmbeddingFunction", return_value=encoder):
        from vault_rag.engine.indexer import create_local_embed_fn

        vectors = create_local_embed_fn()(["a", "b"])

    assert all(isinstance(v, list) for v in vectors)
    assert all(type(value) is float for vector in vectors for value in vector)
