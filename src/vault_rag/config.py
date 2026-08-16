"""VaultConfig: central configuration dataclass for vault-rag."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VaultConfig:
    """Configuration for the vault-rag pipeline."""

    vault_path: Path = field(
        default_factory=lambda: Path("C:/Users/slime/claude-projects/Obsidian Vault")
    )
    # "local" runs all-MiniLM-L6-v2 through the onnxruntime that chromadb
    # already ships, so re-indexing the whole vault costs nothing. "openai"
    # keeps text-embedding-3-small, which is better but metered -- no chat
    # subscription covers the embeddings endpoint.
    embedding_provider: str = "local"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 512
    compile_model: str = "claude-haiku-4-5-20251001"
    qa_model: str = "claude-sonnet-4-6"
    max_context_tokens: int = 4000
    chunk_size: int = 500
    chunk_overlap: int = 50
    excluded_dirs: tuple[str, ...] = (
        ".obsidian",
        "_trash",
        ".git",
        "docs",
        "Templates",
    )
    excluded_dir_prefixes: tuple[str, ...] = (".dedupe-trash-",)
    # Raw source layer: indexed and searchable, but never rewritten. Mixing
    # LLM output into imported originals destroys the point of having them.
    readonly_dirs: tuple[str, ...] = ("Sources",)
    # Operational journals written by the tool itself. Indexing them would let
    # a maintenance pass move the very metrics it is judged by.
    excluded_root_files: tuple[str, ...] = ("log.md",)
    priority_dirs: tuple[str, ...] = (
        "Projects",
        "Knowledge",
        "Research",
        "Reference",
        "Dev",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "vault_path", Path(self.vault_path))

    @property
    def chroma_path(self) -> Path:
        """ChromaDB persistence directory, sibling to the vault."""
        return self.vault_path.parent / "vault-rag" / "data" / "chroma"

    @property
    def graph_path(self) -> Path:
        """Knowledge graph JSON file path."""
        return self.vault_path.parent / "vault-rag" / "data" / "graph.json"

    @property
    def output_path(self) -> Path:
        """Generated output directory."""
        return self.vault_path.parent / "vault-rag" / "output"
