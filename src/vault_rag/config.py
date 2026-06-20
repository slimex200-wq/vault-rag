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
    excluded_dir_prefixes: tuple[str, ...] = (
        ".dedupe-trash-",
    )
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
