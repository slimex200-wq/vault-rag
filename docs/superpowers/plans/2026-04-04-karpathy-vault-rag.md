# Karpathy Vault RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Karpathy의 개인 지식 관리 아키텍처를 Obsidian Vault 위에 구현 — Data Ingest, LLM Engine (Compile/Q&A/Indexing), Knowledge Store, CLI Frontend 4개 레이어를 갖춘 로컬 RAG 시스템

**Architecture:** 4-layer pipeline. Data Ingest(scanner/web clipper/PDF)가 원본을 수집하고, LLM Engine이 compile(요약/태그/링크)+embed+graph 처리하여 Knowledge Store(Obsidian md + ChromaDB + NetworkX graph)에 저장. CLI가 semantic search와 RAG Q&A를 제공하며, Claude Code 훅이 세션마다 자동 컨텍스트 주입.

**Tech Stack:** Python 3.11, ChromaDB (vector store), OpenAI text-embedding-3-small (embeddings), Anthropic Claude (compile/Q&A), NetworkX (knowledge graph), Click (CLI), trafilatura (web clip), PyMuPDF (PDF)

---

## File Structure

```
C:/Users/slime/claude-projects/vault-rag/
├── pyproject.toml                    # Dependencies, project metadata
├── CLAUDE.md                         # Project-specific Claude Code instructions
├── src/
│   └── vault_rag/
│       ├── __init__.py               # Version, package init
│       ├── config.py                 # Paths, API settings, constants
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── scanner.py            # Vault .md file scanner (incremental)
│       │   ├── web_clipper.py        # URL -> markdown note
│       │   └── pdf_reader.py         # PDF -> text extraction
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── compiler.py           # LLM: summarize, auto-tag, auto-link
│       │   ├── indexer.py            # Embeddings generation + ChromaDB upsert
│       │   ├── graph.py              # Wikilink knowledge graph (NetworkX)
│       │   ├── qa.py                 # RAG Q&A: retrieve + rerank + generate
│       │   └── health.py             # Orphan notes, broken links, stale index
│       ├── store/
│       │   ├── __init__.py
│       │   ├── vector_store.py       # ChromaDB wrapper
│       │   └── note_store.py         # Note CRUD with auto-compile trigger
│       └── cli.py                    # Unified Click CLI
├── tests/
│   ├── conftest.py                   # Shared fixtures (tmp vault, mock APIs)
│   ├── test_config.py
│   ├── test_scanner.py
│   ├── test_web_clipper.py
│   ├── test_pdf_reader.py
│   ├── test_compiler.py
│   ├── test_indexer.py
│   ├── test_graph.py
│   ├── test_qa.py
│   ├── test_health.py
│   ├── test_vector_store.py
│   ├── test_note_store.py
│   └── test_cli.py
└── data/                             # Runtime data (gitignored)
    ├── chroma/                       # ChromaDB persistence
    └── graph.json                    # Knowledge graph export
```

**Design Principles:**
- 모든 모듈은 `vault_path`를 주입받음 (테스트 시 tmp dir 사용)
- LLM 호출은 모두 mock 가능하도록 client를 파라미터로 받음
- 기존 `obsidian_rag.py`, `selective_obsidian_indexer.py`, `rag_search.py`는 이 시스템 완성 후 deprecate
- 기존 bash 훅 (`load-vault-context.sh`, `save-to-obsidian.sh`)은 유지하되, CLI에서도 동일 기능 제공

---

## Task 1: Project Scaffold + Config

**Files:**
- Create: `C:/Users/slime/claude-projects/vault-rag/pyproject.toml`
- Create: `C:/Users/slime/claude-projects/vault-rag/CLAUDE.md`
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/__init__.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/config.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/tests/conftest.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/tests/test_config.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "vault-rag"
version = "0.1.0"
description = "Karpathy-style personal knowledge RAG system for Obsidian"
requires-python = ">=3.11"
dependencies = [
    "chromadb>=0.5.0",
    "openai>=2.0.0",
    "anthropic>=0.40.0",
    "networkx>=3.0",
    "click>=8.0",
    "trafilatura>=1.0",
    "PyMuPDF>=1.24",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.4"]

[project.scripts]
vault-rag = "vault_rag.cli:cli"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --tb=short"

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 2: Create CLAUDE.md**

```markdown
# vault-rag

Karpathy 아키텍처 기반 Obsidian Vault RAG 시스템.

## Commands

\```bash
# Test
python -m pytest tests/ -q

# Lint + Format
ruff check . --fix && ruff format .

# Run CLI
python -m vault_rag.cli --help
\```

## Architecture

4-layer: Ingest -> Engine (Compile/Index/Graph/Q&A) -> Store (ChromaDB + Obsidian) -> CLI

## Key Paths

- Vault: C:/Users/slime/claude-projects/Obsidian Vault
- ChromaDB: ./data/chroma/
- Graph: ./data/graph.json
```

- [ ] **Step 3: Write test_config.py (RED)**

```python
# tests/test_config.py
from vault_rag.config import VaultConfig


def test_default_config_has_vault_path():
    cfg = VaultConfig()
    assert cfg.vault_path.exists()


def test_config_with_custom_path(tmp_path):
    cfg = VaultConfig(vault_path=tmp_path)
    assert cfg.vault_path == tmp_path


def test_config_chroma_path(tmp_path):
    cfg = VaultConfig(vault_path=tmp_path)
    assert "chroma" in str(cfg.chroma_path)


def test_config_embedding_model():
    cfg = VaultConfig()
    assert cfg.embedding_model == "text-embedding-3-small"


def test_config_excluded_dirs():
    cfg = VaultConfig()
    assert ".obsidian" in cfg.excluded_dirs
    assert "_trash" in cfg.excluded_dirs
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vault_rag'`

- [ ] **Step 5: Create __init__.py and config.py (GREEN)**

```python
# src/vault_rag/__init__.py
__version__ = "0.1.0"
```

```python
# src/vault_rag/config.py
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VaultConfig:
    vault_path: Path = field(
        default_factory=lambda: Path("C:/Users/slime/claude-projects/Obsidian Vault")
    )
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 512
    compile_model: str = "claude-sonnet-4-20250514"
    qa_model: str = "claude-sonnet-4-20250514"
    max_context_tokens: int = 4000
    chunk_size: int = 500
    chunk_overlap: int = 50
    excluded_dirs: tuple[str, ...] = (".obsidian", "_trash", ".git", "docs", "Templates")
    priority_dirs: tuple[str, ...] = ("Projects", "Knowledge", "Research", "Reference", "Dev")

    @property
    def chroma_path(self) -> Path:
        return self.vault_path.parent / "vault-rag" / "data" / "chroma"

    @property
    def graph_path(self) -> Path:
        return self.vault_path.parent / "vault-rag" / "data" / "graph.json"
```

- [ ] **Step 6: Create conftest.py with shared fixtures**

```python
# tests/conftest.py
from pathlib import Path
import pytest
from vault_rag.config import VaultConfig


@pytest.fixture
def tmp_vault(tmp_path):
    """Create a temporary vault with sample notes."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Sample notes
    (vault / "Projects").mkdir()
    (vault / "Knowledge").mkdir()
    (vault / ".obsidian").mkdir()
    (vault / "_trash").mkdir()

    note1 = vault / "Projects" / "test-project.md"
    note1.write_text(
        "---\ntags: [project, ai]\n---\n# Test Project\n\nThis is about [[AI agents]].\n",
        encoding="utf-8",
    )

    note2 = vault / "Knowledge" / "ai-agents.md"
    note2.write_text(
        "# AI Agents\n\n#ai #agents\n\nAgents use LLMs to perform tasks.\nSee [[test-project]].\n",
        encoding="utf-8",
    )

    note3 = vault / "orphan-note.md"
    note3.write_text("# Orphan\n\nNo links here.\n", encoding="utf-8")

    return vault


@pytest.fixture
def cfg(tmp_vault):
    return VaultConfig(vault_path=tmp_vault)
```

- [ ] **Step 7: Run tests (GREEN)**

Run: `cd C:/Users/slime/claude-projects/vault-rag && pip install -e ".[dev]" && python -m pytest tests/test_config.py -v`
Expected: 5 passed

- [ ] **Step 8: Create .gitignore and init repo**

```bash
cd C:/Users/slime/claude-projects/vault-rag
echo "data/\n__pycache__/\n*.egg-info/\n.ruff_cache/\ndist/" > .gitignore
git init && git add -A && git commit -m "chore: project scaffold with config module"
```

---

## Task 2: Vault Scanner (Data Ingest Layer)

**Files:**
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/ingest/__init__.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/ingest/scanner.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/tests/test_scanner.py`

- [ ] **Step 1: Write test_scanner.py (RED)**

```python
# tests/test_scanner.py
from vault_rag.ingest.scanner import VaultScanner, ScannedNote


def test_scan_finds_md_files(cfg, tmp_vault):
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    paths = [n.path for n in notes]
    assert any("test-project.md" in str(p) for p in paths)
    assert any("ai-agents.md" in str(p) for p in paths)


def test_scan_excludes_obsidian_dir(cfg, tmp_vault):
    # Create a file in .obsidian
    (tmp_vault / ".obsidian" / "config.md").write_text("# Config", encoding="utf-8")
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    paths = [str(n.path) for n in notes]
    assert not any(".obsidian" in p for p in paths)


def test_scan_excludes_trash(cfg, tmp_vault):
    (tmp_vault / "_trash" / "deleted.md").write_text("# Deleted", encoding="utf-8")
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    paths = [str(n.path) for n in notes]
    assert not any("_trash" in p for p in paths)


def test_scanned_note_extracts_title(cfg, tmp_vault):
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    project_note = next(n for n in notes if "test-project" in str(n.path))
    assert project_note.title == "Test Project"


def test_scanned_note_extracts_tags_from_frontmatter(cfg, tmp_vault):
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    project_note = next(n for n in notes if "test-project" in str(n.path))
    assert "project" in project_note.tags
    assert "ai" in project_note.tags


def test_scanned_note_extracts_inline_tags(cfg, tmp_vault):
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    agent_note = next(n for n in notes if "ai-agents" in str(n.path))
    assert "ai" in agent_note.tags
    assert "agents" in agent_note.tags


def test_scanned_note_extracts_wikilinks(cfg, tmp_vault):
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    project_note = next(n for n in notes if "test-project" in str(n.path))
    assert "AI agents" in project_note.links


def test_scanned_note_has_content_hash(cfg, tmp_vault):
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    assert all(n.content_hash for n in notes)


def test_incremental_scan_skips_unchanged(cfg, tmp_vault):
    scanner = VaultScanner(cfg)
    first = scanner.scan()
    hashes = {n.relative_path: n.content_hash for n in first}

    second = scanner.scan(known_hashes=hashes)
    assert len(second) == 0  # Nothing changed


def test_incremental_scan_detects_changes(cfg, tmp_vault):
    scanner = VaultScanner(cfg)
    first = scanner.scan()
    hashes = {n.relative_path: n.content_hash for n in first}

    # Modify a note
    (tmp_vault / "Projects" / "test-project.md").write_text(
        "# Updated Project\n\nNew content.\n", encoding="utf-8"
    )

    second = scanner.scan(known_hashes=hashes)
    assert len(second) == 1
    assert "test-project" in str(second[0].path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_scanner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vault_rag.ingest'`

- [ ] **Step 3: Implement scanner.py (GREEN)**

```python
# src/vault_rag/ingest/__init__.py
```

```python
# src/vault_rag/ingest/scanner.py
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
import re

from vault_rag.config import VaultConfig

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_YAML_TAGS_RE = re.compile(r"tags:\s*\[([^\]]*)\]")
_INLINE_TAG_RE = re.compile(r"(?:^|\s)#([a-zA-Z가-힣][\w가-힣-]*)", re.MULTILINE)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_HEADING_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)


@dataclass(frozen=True)
class ScannedNote:
    path: Path
    relative_path: str
    title: str
    content: str
    tags: list[str]
    links: list[str]
    modified: float
    content_hash: str


class VaultScanner:
    def __init__(self, config: VaultConfig):
        self.config = config

    def scan(self, known_hashes: dict[str, str] | None = None) -> list[ScannedNote]:
        notes = []
        for md_file in self.config.vault_path.rglob("*.md"):
            if self._is_excluded(md_file):
                continue
            relative = str(md_file.relative_to(self.config.vault_path)).replace("\\", "/")
            content = md_file.read_text(encoding="utf-8", errors="replace")
            content_hash = md5(content.encode()).hexdigest()

            if known_hashes and known_hashes.get(relative) == content_hash:
                continue

            notes.append(self._parse(md_file, relative, content, content_hash))
        return notes

    def _is_excluded(self, path: Path) -> bool:
        parts = path.relative_to(self.config.vault_path).parts
        return any(part in self.config.excluded_dirs for part in parts)

    def _parse(
        self, path: Path, relative: str, content: str, content_hash: str
    ) -> ScannedNote:
        body = content
        tags: list[str] = []

        # Extract frontmatter tags
        fm_match = _FRONTMATTER_RE.match(content)
        if fm_match:
            fm_text = fm_match.group(1)
            body = content[fm_match.end() :]
            tag_match = _YAML_TAGS_RE.search(fm_text)
            if tag_match:
                tags.extend(t.strip().strip("'\"") for t in tag_match.group(1).split(",") if t.strip())

        # Extract inline tags
        tags.extend(_INLINE_TAG_RE.findall(body))
        tags = list(dict.fromkeys(tags))  # dedupe, preserve order

        # Extract title from first heading
        heading_match = _HEADING_RE.search(body)
        title = heading_match.group(1).strip() if heading_match else path.stem

        # Extract wikilinks
        links = list(dict.fromkeys(_WIKILINK_RE.findall(content)))

        return ScannedNote(
            path=path,
            relative_path=relative,
            title=title,
            content=body,
            tags=tags,
            links=links,
            modified=path.stat().st_mtime,
            content_hash=content_hash,
        )
```

- [ ] **Step 4: Run tests (GREEN)**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_scanner.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: vault scanner with incremental scanning support"
```

---

## Task 3: Vector Store (ChromaDB Wrapper)

**Files:**
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/store/__init__.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/store/vector_store.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/tests/test_vector_store.py`

- [ ] **Step 1: Write test_vector_store.py (RED)**

```python
# tests/test_vector_store.py
import pytest
from vault_rag.store.vector_store import VectorStore


@pytest.fixture
def vs(tmp_path):
    return VectorStore(persist_dir=tmp_path / "chroma")


def test_upsert_and_query(vs):
    vs.upsert(
        ids=["note1"],
        documents=["AI agents use LLMs to perform autonomous tasks"],
        metadatas=[{"title": "AI Agents", "tags": "ai,agents"}],
        embeddings=[[0.1] * 512],
    )
    results = vs.query(query_embeddings=[[0.1] * 512], n_results=1)
    assert results["ids"][0][0] == "note1"


def test_upsert_updates_existing(vs):
    vs.upsert(
        ids=["note1"],
        documents=["Original content"],
        metadatas=[{"title": "Note 1"}],
        embeddings=[[0.1] * 512],
    )
    vs.upsert(
        ids=["note1"],
        documents=["Updated content"],
        metadatas=[{"title": "Note 1 Updated"}],
        embeddings=[[0.2] * 512],
    )
    results = vs.query(query_embeddings=[[0.2] * 512], n_results=1)
    assert results["documents"][0][0] == "Updated content"


def test_delete(vs):
    vs.upsert(
        ids=["note1", "note2"],
        documents=["Doc 1", "Doc 2"],
        metadatas=[{"title": "N1"}, {"title": "N2"}],
        embeddings=[[0.1] * 512, [0.9] * 512],
    )
    vs.delete(ids=["note1"])
    results = vs.query(query_embeddings=[[0.1] * 512], n_results=10)
    assert "note1" not in results["ids"][0]


def test_count(vs):
    assert vs.count() == 0
    vs.upsert(
        ids=["a", "b"],
        documents=["x", "y"],
        metadatas=[{}, {}],
        embeddings=[[0.1] * 512, [0.2] * 512],
    )
    assert vs.count() == 2


def test_get_by_ids(vs):
    vs.upsert(
        ids=["note1"],
        documents=["Content here"],
        metadatas=[{"title": "Test"}],
        embeddings=[[0.1] * 512],
    )
    result = vs.get(ids=["note1"])
    assert result["documents"][0] == "Content here"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_vector_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Install chromadb + implement vector_store.py (GREEN)**

```bash
pip install chromadb
```

```python
# src/vault_rag/store/__init__.py
```

```python
# src/vault_rag/store/vector_store.py
from pathlib import Path
from typing import Any

import chromadb


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = "vault_notes"):
        self._client = chromadb.PersistentClient(path=str(persist_dir))
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
        where: dict | None = None,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": min(n_results, self._collection.count() or 1),
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)

    def get(self, ids: list[str]) -> dict:
        return self._collection.get(ids=ids)

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
```

- [ ] **Step 4: Run tests (GREEN)**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_vector_store.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: ChromaDB vector store wrapper"
```

---

## Task 4: Embedding Indexer (LLM Engine - Indexing)

**Files:**
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/engine/__init__.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/engine/indexer.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/tests/test_indexer.py`

- [ ] **Step 1: Write test_indexer.py (RED)**

```python
# tests/test_indexer.py
from unittest.mock import MagicMock
from vault_rag.config import VaultConfig
from vault_rag.engine.indexer import Indexer
from vault_rag.ingest.scanner import ScannedNote
from vault_rag.store.vector_store import VectorStore
from pathlib import Path
import time


def _make_note(title="Test", content="Hello world", tags=None, links=None, path="test.md"):
    return ScannedNote(
        path=Path(path),
        relative_path=path,
        title=title,
        content=content,
        tags=tags or [],
        links=links or [],
        modified=time.time(),
        content_hash="abc123",
    )


def test_index_notes_calls_embed_and_upsert(tmp_path):
    mock_embed = MagicMock(return_value=[[0.1] * 512])
    vs = VectorStore(persist_dir=tmp_path / "chroma")
    cfg = VaultConfig(vault_path=tmp_path)
    indexer = Indexer(config=cfg, vector_store=vs, embed_fn=mock_embed)

    notes = [_make_note()]
    indexer.index(notes)

    mock_embed.assert_called_once()
    assert vs.count() == 1


def test_index_batches_large_sets(tmp_path):
    call_count = 0
    def mock_embed(texts):
        nonlocal call_count
        call_count += 1
        return [[0.1] * 512] * len(texts)

    vs = VectorStore(persist_dir=tmp_path / "chroma")
    cfg = VaultConfig(vault_path=tmp_path)
    indexer = Indexer(config=cfg, vector_store=vs, embed_fn=mock_embed, batch_size=2)

    notes = [_make_note(title=f"Note {i}", path=f"note{i}.md") for i in range(5)]
    indexer.index(notes)

    assert call_count == 3  # ceil(5/2) = 3 batches
    assert vs.count() == 5


def test_index_stores_metadata(tmp_path):
    mock_embed = MagicMock(return_value=[[0.1] * 512])
    vs = VectorStore(persist_dir=tmp_path / "chroma")
    cfg = VaultConfig(vault_path=tmp_path)
    indexer = Indexer(config=cfg, vector_store=vs, embed_fn=mock_embed)

    notes = [_make_note(title="AI Agents", tags=["ai", "agents"], links=["test-project"])]
    indexer.index(notes)

    result = vs.get(ids=["test.md"])
    assert result["metadatas"][0]["title"] == "AI Agents"
    assert "ai" in result["metadatas"][0]["tags"]


def test_full_reindex_clears_and_rebuilds(tmp_path):
    mock_embed = MagicMock(return_value=[[0.1] * 512])
    vs = VectorStore(persist_dir=tmp_path / "chroma")
    cfg = VaultConfig(vault_path=tmp_path)
    indexer = Indexer(config=cfg, vector_store=vs, embed_fn=mock_embed)

    indexer.index([_make_note(path="old.md")])
    assert vs.count() == 1

    indexer.reindex([_make_note(path="new.md")])
    assert vs.count() == 1
    result = vs.get(ids=["new.md"])
    assert result["ids"][0] == "new.md"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_indexer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement indexer.py (GREEN)**

```python
# src/vault_rag/engine/__init__.py
```

```python
# src/vault_rag/engine/indexer.py
import math
from collections.abc import Callable

from vault_rag.config import VaultConfig
from vault_rag.ingest.scanner import ScannedNote
from vault_rag.store.vector_store import VectorStore


class Indexer:
    def __init__(
        self,
        config: VaultConfig,
        vector_store: VectorStore,
        embed_fn: Callable[[list[str]], list[list[float]]],
        batch_size: int = 50,
    ):
        self.config = config
        self.vs = vector_store
        self.embed_fn = embed_fn
        self.batch_size = batch_size

    def index(self, notes: list[ScannedNote]) -> int:
        """Index notes incrementally (upsert). Returns count indexed."""
        total = 0
        for i in range(0, len(notes), self.batch_size):
            batch = notes[i : i + self.batch_size]
            texts = [self._note_to_text(n) for n in batch]
            embeddings = self.embed_fn(texts)
            self.vs.upsert(
                ids=[n.relative_path for n in batch],
                documents=[n.content for n in batch],
                metadatas=[self._note_to_metadata(n) for n in batch],
                embeddings=embeddings,
            )
            total += len(batch)
        return total

    def reindex(self, notes: list[ScannedNote]) -> int:
        """Full reindex: clear store and rebuild."""
        self.vs.reset()
        return self.index(notes)

    def _note_to_text(self, note: ScannedNote) -> str:
        """Create embedding text: title + tags + content."""
        parts = [note.title]
        if note.tags:
            parts.append(" ".join(note.tags))
        parts.append(note.content[:2000])
        return "\n".join(parts)

    def _note_to_metadata(self, note: ScannedNote) -> dict:
        return {
            "title": note.title,
            "tags": ",".join(note.tags),
            "links": ",".join(note.links),
            "content_hash": note.content_hash,
            "relative_path": note.relative_path,
        }
```

- [ ] **Step 4: Run tests (GREEN)**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_indexer.py -v`
Expected: 4 passed

- [ ] **Step 5: Add OpenAI embedding helper**

```python
# src/vault_rag/engine/indexer.py 하단에 추가

def create_openai_embed_fn(
    model: str = "text-embedding-3-small", dimensions: int = 512
) -> Callable[[list[str]], list[list[float]]]:
    """Create an embedding function using OpenAI API."""
    from openai import OpenAI

    client = OpenAI()

    def embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(
            input=texts, model=model, dimensions=dimensions
        )
        return [item.embedding for item in response.data]

    return embed
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: embedding indexer with OpenAI + batch support"
```

---

## Task 5: Knowledge Graph (LLM Engine - Graph)

**Files:**
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/engine/graph.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/tests/test_graph.py`

- [ ] **Step 1: Write test_graph.py (RED)**

```python
# tests/test_graph.py
from pathlib import Path
import time
from vault_rag.engine.graph import KnowledgeGraph
from vault_rag.ingest.scanner import ScannedNote


def _note(title, path, links=None, tags=None):
    return ScannedNote(
        path=Path(path), relative_path=path, title=title,
        content="", tags=tags or [], links=links or [],
        modified=time.time(), content_hash="x",
    )


def test_build_graph_from_notes():
    notes = [
        _note("A", "a.md", links=["B"]),
        _note("B", "b.md", links=["A", "C"]),
        _note("C", "c.md"),
    ]
    g = KnowledgeGraph()
    g.build(notes)
    assert g.node_count() == 3
    assert g.edge_count() == 3  # A->B, B->A, B->C


def test_orphan_detection():
    notes = [
        _note("A", "a.md", links=["B"]),
        _note("B", "b.md"),
        _note("Orphan", "orphan.md"),
    ]
    g = KnowledgeGraph()
    g.build(notes)
    orphans = g.find_orphans()
    assert "orphan.md" in orphans


def test_hub_nodes():
    notes = [
        _note("Hub", "hub.md", links=["A", "B", "C", "D", "E"]),
        _note("A", "a.md"), _note("B", "b.md"), _note("C", "c.md"),
        _note("D", "d.md"), _note("E", "e.md"),
    ]
    g = KnowledgeGraph()
    g.build(notes)
    hubs = g.find_hubs(min_degree=3)
    assert "hub.md" in hubs


def test_neighbors():
    notes = [
        _note("A", "a.md", links=["B", "C"]),
        _note("B", "b.md"),
        _note("C", "c.md"),
    ]
    g = KnowledgeGraph()
    g.build(notes)
    neighbors = g.neighbors("a.md")
    assert set(neighbors) == {"b.md", "c.md"}


def test_save_and_load(tmp_path):
    notes = [_note("A", "a.md", links=["B"]), _note("B", "b.md")]
    g = KnowledgeGraph()
    g.build(notes)

    path = tmp_path / "graph.json"
    g.save(path)

    g2 = KnowledgeGraph()
    g2.load(path)
    assert g2.node_count() == 2
    assert g2.edge_count() == 1


def test_clusters():
    notes = [
        _note("A", "a.md", links=["B"]),
        _note("B", "b.md", links=["A"]),
        _note("X", "x.md", links=["Y"]),
        _note("Y", "y.md", links=["X"]),
    ]
    g = KnowledgeGraph()
    g.build(notes)
    clusters = g.find_clusters()
    assert len(clusters) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_graph.py -v`
Expected: FAIL

- [ ] **Step 3: Implement graph.py (GREEN)**

```python
# src/vault_rag/engine/graph.py
import json
from pathlib import Path

import networkx as nx

from vault_rag.ingest.scanner import ScannedNote


class KnowledgeGraph:
    def __init__(self):
        self._graph = nx.DiGraph()
        self._title_to_path: dict[str, str] = {}

    def build(self, notes: list[ScannedNote]) -> None:
        self._graph.clear()
        self._title_to_path.clear()

        for note in notes:
            self._graph.add_node(
                note.relative_path,
                title=note.title,
                tags=note.tags,
            )
            self._title_to_path[note.title.lower()] = note.relative_path
            # Also map filename stem
            stem = Path(note.relative_path).stem.lower()
            self._title_to_path[stem] = note.relative_path

        for note in notes:
            for link in note.links:
                target = self._resolve_link(link)
                if target and target != note.relative_path:
                    self._graph.add_edge(note.relative_path, target)

    def _resolve_link(self, link_text: str) -> str | None:
        key = link_text.lower().strip()
        if key in self._title_to_path:
            return self._title_to_path[key]
        # Try matching with hyphens/spaces normalized
        normalized = key.replace(" ", "-")
        return self._title_to_path.get(normalized)

    def find_orphans(self) -> list[str]:
        return [n for n in self._graph.nodes if self._graph.degree(n) == 0]

    def find_hubs(self, min_degree: int = 5) -> list[str]:
        return [n for n in self._graph.nodes if self._graph.degree(n) >= min_degree]

    def neighbors(self, node_id: str) -> list[str]:
        if node_id not in self._graph:
            return []
        return list(self._graph.successors(node_id))

    def find_clusters(self) -> list[set[str]]:
        undirected = self._graph.to_undirected()
        return [c for c in nx.connected_components(undirected) if len(c) > 1]

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def save(self, path: Path) -> None:
        data = nx.node_link_data(self._graph)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self._graph = nx.node_link_graph(data)
```

- [ ] **Step 4: Run tests (GREEN)**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_graph.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: knowledge graph with orphan/hub/cluster detection"
```

---

## Task 6: LLM Compiler (LLM Engine - Compile)

**Files:**
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/engine/compiler.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/tests/test_compiler.py`

- [ ] **Step 1: Write test_compiler.py (RED)**

```python
# tests/test_compiler.py
from unittest.mock import MagicMock, patch
from pathlib import Path
import time
from vault_rag.engine.compiler import Compiler, CompileResult
from vault_rag.ingest.scanner import ScannedNote


def _note(title="Test", content="AI agents use LLMs.", tags=None, links=None):
    return ScannedNote(
        path=Path("test.md"), relative_path="test.md", title=title,
        content=content, tags=tags or [], links=links or [],
        modified=time.time(), content_hash="x",
    )


def _mock_anthropic_response(text):
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=text)]
    mock_client.messages.create.return_value = mock_msg
    return mock_client


def test_compile_returns_summary():
    response_json = '{"summary": "AI agents overview", "tags": ["ai"], "suggested_links": ["LLM Guide"]}'
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="claude-sonnet-4-20250514")
    result = compiler.compile(_note())
    assert result.summary == "AI agents overview"


def test_compile_returns_tags():
    response_json = '{"summary": "test", "tags": ["ai", "agents"], "suggested_links": []}'
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="claude-sonnet-4-20250514")
    result = compiler.compile(_note())
    assert "ai" in result.tags
    assert "agents" in result.tags


def test_compile_returns_suggested_links():
    response_json = '{"summary": "test", "tags": [], "suggested_links": ["LLM Guide", "Agentic Systems"]}'
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="claude-sonnet-4-20250514")
    result = compiler.compile(_note())
    assert "LLM Guide" in result.suggested_links


def test_compile_handles_malformed_json():
    client = _mock_anthropic_response("not json at all")
    compiler = Compiler(client=client, model="claude-sonnet-4-20250514")
    result = compiler.compile(_note())
    assert result.summary == ""
    assert result.tags == []
    assert result.suggested_links == []


def test_compile_batch():
    response_json = '{"summary": "s", "tags": ["t"], "suggested_links": ["l"]}'
    client = _mock_anthropic_response(response_json)
    compiler = Compiler(client=client, model="claude-sonnet-4-20250514")
    notes = [_note(title=f"N{i}") for i in range(3)]
    results = compiler.compile_batch(notes)
    assert len(results) == 3
    assert client.messages.create.call_count == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_compiler.py -v`
Expected: FAIL

- [ ] **Step 3: Implement compiler.py (GREEN)**

```python
# src/vault_rag/engine/compiler.py
import json
from dataclasses import dataclass, field

from vault_rag.ingest.scanner import ScannedNote

_COMPILE_PROMPT = """You are a knowledge management assistant. Analyze this note and return JSON:

{{
  "summary": "2-3 sentence summary in the note's language",
  "tags": ["lowercase", "relevant", "tags"],
  "suggested_links": ["titles of related concepts that should be linked"]
}}

Note title: {title}
Note tags: {tags}
Note content:
{content}

Return ONLY valid JSON, no markdown fences."""


@dataclass(frozen=True)
class CompileResult:
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    suggested_links: list[str] = field(default_factory=list)


class Compiler:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def compile(self, note: ScannedNote) -> CompileResult:
        prompt = _COMPILE_PROMPT.format(
            title=note.title,
            tags=", ".join(note.tags),
            content=note.content[:3000],
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        return self._parse_response(text)

    def compile_batch(self, notes: list[ScannedNote]) -> list[CompileResult]:
        return [self.compile(note) for note in notes]

    def _parse_response(self, text: str) -> CompileResult:
        try:
            data = json.loads(text)
            return CompileResult(
                summary=data.get("summary", ""),
                tags=data.get("tags", []),
                suggested_links=data.get("suggested_links", []),
            )
        except (json.JSONDecodeError, KeyError):
            return CompileResult()
```

- [ ] **Step 4: Run tests (GREEN)**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_compiler.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: LLM compiler for auto-summarize/tag/link"
```

---

## Task 7: RAG Q&A (LLM Engine - Q&A)

**Files:**
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/engine/qa.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/tests/test_qa.py`

- [ ] **Step 1: Write test_qa.py (RED)**

```python
# tests/test_qa.py
from unittest.mock import MagicMock
from vault_rag.engine.qa import QAEngine


def _mock_embed(texts):
    return [[0.1] * 512] * len(texts)


def _mock_anthropic(answer_text):
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=answer_text)]
    client.messages.create.return_value = msg
    return client


def test_answer_retrieves_and_generates(tmp_path):
    from vault_rag.store.vector_store import VectorStore
    vs = VectorStore(persist_dir=tmp_path / "chroma")
    vs.upsert(
        ids=["note1.md"],
        documents=["AI agents are autonomous programs that use LLMs."],
        metadatas=[{"title": "AI Agents", "tags": "ai", "relative_path": "note1.md"}],
        embeddings=[[0.1] * 512],
    )

    client = _mock_anthropic("AI agents are programs that use LLMs to act autonomously.")
    qa = QAEngine(
        vector_store=vs,
        embed_fn=_mock_embed,
        client=client,
        model="claude-sonnet-4-20250514",
    )

    result = qa.answer("What are AI agents?")
    assert result.answer
    assert len(result.sources) > 0
    assert result.sources[0]["id"] == "note1.md"


def test_answer_with_no_results(tmp_path):
    from vault_rag.store.vector_store import VectorStore
    vs = VectorStore(persist_dir=tmp_path / "chroma")

    client = _mock_anthropic("No relevant information found.")
    qa = QAEngine(
        vector_store=vs,
        embed_fn=_mock_embed,
        client=client,
        model="claude-sonnet-4-20250514",
    )

    result = qa.answer("Completely unrelated query")
    assert result.answer
    assert len(result.sources) == 0


def test_answer_passes_context_to_llm(tmp_path):
    from vault_rag.store.vector_store import VectorStore
    vs = VectorStore(persist_dir=tmp_path / "chroma")
    vs.upsert(
        ids=["note1.md"],
        documents=["Python is a programming language."],
        metadatas=[{"title": "Python", "tags": "python", "relative_path": "note1.md"}],
        embeddings=[[0.1] * 512],
    )

    client = _mock_anthropic("Python answer")
    qa = QAEngine(
        vector_store=vs, embed_fn=_mock_embed,
        client=client, model="claude-sonnet-4-20250514",
    )
    qa.answer("What is Python?")

    call_args = client.messages.create.call_args
    user_msg = call_args.kwargs["messages"][0]["content"]
    assert "Python is a programming language" in user_msg


def test_search_only_returns_ranked_results(tmp_path):
    from vault_rag.store.vector_store import VectorStore
    vs = VectorStore(persist_dir=tmp_path / "chroma")
    vs.upsert(
        ids=["a.md", "b.md"],
        documents=["AI agents", "Cooking recipes"],
        metadatas=[
            {"title": "AI", "tags": "ai", "relative_path": "a.md"},
            {"title": "Cook", "tags": "food", "relative_path": "b.md"},
        ],
        embeddings=[[0.1] * 512, [0.9] * 512],
    )

    qa = QAEngine(
        vector_store=vs, embed_fn=_mock_embed,
        client=MagicMock(), model="claude-sonnet-4-20250514",
    )
    results = qa.search("AI agents", n_results=5)
    assert len(results) == 2
    assert results[0]["id"] in ("a.md", "b.md")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_qa.py -v`
Expected: FAIL

- [ ] **Step 3: Implement qa.py (GREEN)**

```python
# src/vault_rag/engine/qa.py
from collections.abc import Callable
from dataclasses import dataclass, field

from vault_rag.store.vector_store import VectorStore

_QA_PROMPT = """You are a knowledge assistant. Answer based ONLY on the provided context.
If the context doesn't contain relevant information, say so.
Answer in the same language as the question.

Context from knowledge base:
{context}

Question: {question}"""


@dataclass
class QAResult:
    answer: str
    sources: list[dict] = field(default_factory=list)


class QAEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        embed_fn: Callable[[list[str]], list[list[float]]],
        client,
        model: str,
    ):
        self.vs = vector_store
        self.embed_fn = embed_fn
        self.client = client
        self.model = model

    def search(self, query: str, n_results: int = 10) -> list[dict]:
        if self.vs.count() == 0:
            return []
        embeddings = self.embed_fn([query])
        results = self.vs.query(query_embeddings=embeddings, n_results=n_results)
        return [
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            }
            for i in range(len(results["ids"][0]))
        ]

    def answer(self, question: str, n_context: int = 5) -> QAResult:
        sources = self.search(question, n_results=n_context)
        if not sources:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": f"Answer: {question}"}],
            )
            return QAResult(answer=response.content[0].text, sources=[])

        context = "\n\n---\n\n".join(
            f"[{s['metadata'].get('title', s['id'])}]\n{s['document'][:1000]}"
            for s in sources
        )
        prompt = _QA_PROMPT.format(context=context, question=question)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return QAResult(answer=response.content[0].text, sources=sources)
```

- [ ] **Step 4: Run tests (GREEN)**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_qa.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: RAG Q&A engine with retrieve-rank-generate pipeline"
```

---

## Task 8: Health Checks (LLM Engine - Health)

**Files:**
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/engine/health.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/tests/test_health.py`

- [ ] **Step 1: Write test_health.py (RED)**

```python
# tests/test_health.py
from pathlib import Path
import time
from vault_rag.engine.health import HealthChecker
from vault_rag.ingest.scanner import ScannedNote


def _note(title, path, tags=None, links=None):
    return ScannedNote(
        path=Path(path), relative_path=path, title=title,
        content="x", tags=tags or [], links=links or [],
        modified=time.time(), content_hash="x",
    )


def test_detect_broken_links():
    notes = [
        _note("A", "a.md", links=["NonExistent"]),
        _note("B", "b.md"),
    ]
    hc = HealthChecker(notes)
    broken = hc.broken_links()
    assert len(broken) == 1
    assert broken[0]["source"] == "a.md"
    assert broken[0]["target"] == "NonExistent"


def test_detect_orphan_notes():
    notes = [
        _note("A", "a.md", links=["B"]),
        _note("B", "b.md"),
        _note("Orphan", "orphan.md"),
    ]
    hc = HealthChecker(notes)
    orphans = hc.orphan_notes()
    assert "orphan.md" in orphans


def test_detect_untagged_notes():
    notes = [
        _note("Tagged", "tagged.md", tags=["ai"]),
        _note("Untagged", "untagged.md"),
    ]
    hc = HealthChecker(notes)
    untagged = hc.untagged_notes()
    assert "untagged.md" in untagged
    assert "tagged.md" not in untagged


def test_detect_empty_notes():
    notes = [
        ScannedNote(
            path=Path("empty.md"), relative_path="empty.md", title="Empty",
            content="", tags=[], links=[], modified=time.time(), content_hash="x",
        ),
        _note("Full", "full.md"),
    ]
    hc = HealthChecker(notes)
    empty = hc.empty_notes()
    assert "empty.md" in empty


def test_full_report():
    notes = [
        _note("A", "a.md", links=["B", "Missing"], tags=["ai"]),
        _note("B", "b.md"),
        _note("Orphan", "orphan.md"),
    ]
    hc = HealthChecker(notes)
    report = hc.report()
    assert "total_notes" in report
    assert report["total_notes"] == 3
    assert "broken_links" in report
    assert "orphan_notes" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_health.py -v`
Expected: FAIL

- [ ] **Step 3: Implement health.py (GREEN)**

```python
# src/vault_rag/engine/health.py
from pathlib import Path

from vault_rag.ingest.scanner import ScannedNote


class HealthChecker:
    def __init__(self, notes: list[ScannedNote]):
        self.notes = notes
        self._titles = {n.title.lower() for n in notes}
        self._stems = {Path(n.relative_path).stem.lower() for n in notes}
        self._paths = {n.relative_path for n in notes}
        self._all_targets = self._titles | self._stems

    def broken_links(self) -> list[dict]:
        broken = []
        for note in self.notes:
            for link in note.links:
                key = link.lower().strip()
                if key not in self._all_targets and key.replace(" ", "-") not in self._all_targets:
                    broken.append({"source": note.relative_path, "target": link})
        return broken

    def orphan_notes(self) -> list[str]:
        linked_targets: set[str] = set()
        for note in self.notes:
            for link in note.links:
                linked_targets.add(link.lower().strip())
                linked_targets.add(link.lower().strip().replace(" ", "-"))

        link_sources = {n.relative_path for n in self.notes if n.links}

        orphans = []
        for note in self.notes:
            stem = Path(note.relative_path).stem.lower()
            title = note.title.lower()
            is_linked_to = stem in linked_targets or title in linked_targets
            is_linking = note.relative_path in link_sources
            if not is_linked_to and not is_linking:
                orphans.append(note.relative_path)
        return orphans

    def untagged_notes(self) -> list[str]:
        return [n.relative_path for n in self.notes if not n.tags]

    def empty_notes(self) -> list[str]:
        return [n.relative_path for n in self.notes if not n.content.strip()]

    def report(self) -> dict:
        return {
            "total_notes": len(self.notes),
            "broken_links": self.broken_links(),
            "orphan_notes": self.orphan_notes(),
            "untagged_notes": self.untagged_notes(),
            "empty_notes": self.empty_notes(),
            "broken_link_count": len(self.broken_links()),
            "orphan_count": len(self.orphan_notes()),
            "untagged_count": len(self.untagged_notes()),
        }
```

- [ ] **Step 4: Run tests (GREEN)**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_health.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: health checker for broken links, orphans, untagged notes"
```

---

## Task 9: Web Clipper + PDF Reader (Data Ingest)

**Files:**
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/ingest/web_clipper.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/ingest/pdf_reader.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/tests/test_web_clipper.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/tests/test_pdf_reader.py`

- [ ] **Step 1: Write test_web_clipper.py (RED)**

```python
# tests/test_web_clipper.py
from unittest.mock import patch, MagicMock
from vault_rag.ingest.web_clipper import WebClipper


def test_clip_creates_markdown_note(tmp_path):
    mock_html = "<html><head><title>Test Article</title></head><body><p>Hello world content.</p></body></html>"
    with patch("vault_rag.ingest.web_clipper.trafilatura") as mock_traf:
        mock_traf.extract.return_value = "Hello world content."
        mock_traf.extract_metadata.return_value = MagicMock(title="Test Article", date="2026-04-04")

        clipper = WebClipper(vault_path=tmp_path)
        path = clipper.clip(url="https://example.com/article", html=mock_html, folder="Reference")

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Test Article" in content
    assert "Hello world content." in content
    assert "example.com" in content


def test_clip_sanitizes_filename(tmp_path):
    with patch("vault_rag.ingest.web_clipper.trafilatura") as mock_traf:
        mock_traf.extract.return_value = "Content"
        mock_traf.extract_metadata.return_value = MagicMock(title="What/Is:This?", date=None)

        clipper = WebClipper(vault_path=tmp_path)
        path = clipper.clip(url="https://example.com", html="<p>x</p>", folder="Reference")

    assert "/" not in path.stem
    assert ":" not in path.stem
    assert "?" not in path.stem


def test_clip_adds_frontmatter(tmp_path):
    with patch("vault_rag.ingest.web_clipper.trafilatura") as mock_traf:
        mock_traf.extract.return_value = "Content here"
        mock_traf.extract_metadata.return_value = MagicMock(title="Title", date="2026-04-04")

        clipper = WebClipper(vault_path=tmp_path)
        path = clipper.clip(url="https://example.com", html="<p>x</p>")

    content = path.read_text(encoding="utf-8")
    assert content.startswith("---")
    assert "source:" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_web_clipper.py -v`
Expected: FAIL

- [ ] **Step 3: Implement web_clipper.py (GREEN)**

```bash
pip install trafilatura
```

```python
# src/vault_rag/ingest/web_clipper.py
import re
from datetime import datetime
from pathlib import Path

import trafilatura


class WebClipper:
    def __init__(self, vault_path: Path):
        self.vault_path = vault_path

    def clip(self, url: str, html: str | None = None, folder: str = "Reference") -> Path:
        if html is None:
            html = trafilatura.fetch_url(url)

        text = trafilatura.extract(html) or ""
        meta = trafilatura.extract_metadata(html)
        title = (meta.title if meta and meta.title else url.split("/")[-1]) or "Untitled"
        date = (meta.date if meta and meta.date else None) or datetime.now().strftime("%Y-%m-%d")

        safe_title = self._sanitize(title)[:80]
        target_dir = self.vault_path / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{safe_title}.md"

        frontmatter = f"---\nsource: {url}\nclipped: {date}\ntags: [clipped]\n---\n\n"
        content = f"{frontmatter}# {title}\n\n{text}\n"
        path.write_text(content, encoding="utf-8")
        return path

    def _sanitize(self, name: str) -> str:
        return re.sub(r'[<>:"/\\|?*]', "-", name).strip(". ")
```

- [ ] **Step 4: Write test_pdf_reader.py (RED)**

```python
# tests/test_pdf_reader.py
from pathlib import Path
from vault_rag.ingest.pdf_reader import PDFReader


def test_extract_text_from_pdf(tmp_path):
    # Create a minimal PDF for testing using PyMuPDF
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PDF World", fontsize=12)
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()

    reader = PDFReader()
    result = reader.extract(pdf_path)
    assert "Hello PDF World" in result.text
    assert result.page_count == 1


def test_extract_creates_note(tmp_path):
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Research Paper Content", fontsize=12)
    pdf_path = tmp_path / "paper.pdf"
    doc.save(str(pdf_path))
    doc.close()

    reader = PDFReader()
    vault = tmp_path / "vault"
    vault.mkdir()
    note_path = reader.extract_to_note(pdf_path, vault_path=vault, folder="Research")

    assert note_path.exists()
    content = note_path.read_text(encoding="utf-8")
    assert "Research Paper Content" in content
    assert "paper" in note_path.stem.lower()
```

- [ ] **Step 5: Implement pdf_reader.py (GREEN)**

```python
# src/vault_rag/ingest/pdf_reader.py
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PDFResult:
    text: str
    page_count: int
    metadata: dict


class PDFReader:
    def extract(self, pdf_path: Path) -> PDFResult:
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        metadata = dict(doc.metadata) if doc.metadata else {}
        page_count = len(pages)
        doc.close()
        return PDFResult(text="\n\n".join(pages), page_count=page_count, metadata=metadata)

    def extract_to_note(
        self, pdf_path: Path, vault_path: Path, folder: str = "Research"
    ) -> Path:
        result = self.extract(pdf_path)
        title = result.metadata.get("title") or pdf_path.stem

        target_dir = vault_path / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        note_path = target_dir / f"{pdf_path.stem}.md"

        frontmatter = f"---\nsource: {pdf_path.name}\ntype: pdf\npages: {result.page_count}\ntags: [pdf, research]\n---\n\n"
        content = f"{frontmatter}# {title}\n\n{result.text}\n"
        note_path.write_text(content, encoding="utf-8")
        return note_path
```

- [ ] **Step 6: Run all ingest tests (GREEN)**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_web_clipper.py tests/test_pdf_reader.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: web clipper (trafilatura) and PDF reader (PyMuPDF)"
```

---

## Task 10: Note Store (Knowledge Store Layer)

**Files:**
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/store/note_store.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/tests/test_note_store.py`

- [ ] **Step 1: Write test_note_store.py (RED)**

```python
# tests/test_note_store.py
from vault_rag.store.note_store import NoteStore
from vault_rag.config import VaultConfig


def test_create_note(tmp_path):
    cfg = VaultConfig(vault_path=tmp_path)
    store = NoteStore(cfg)
    path = store.create(title="New Note", content="Hello", folder="Knowledge", tags=["test"])

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "# New Note" in text
    assert "Hello" in text
    assert "test" in text


def test_read_note(tmp_path):
    cfg = VaultConfig(vault_path=tmp_path)
    store = NoteStore(cfg)
    store.create(title="Read Me", content="Body text", folder="Knowledge")

    result = store.read("Knowledge/Read Me.md")
    assert result is not None
    assert "Body text" in result


def test_list_notes(tmp_path):
    cfg = VaultConfig(vault_path=tmp_path)
    store = NoteStore(cfg)
    store.create(title="A", content="x", folder="Projects")
    store.create(title="B", content="y", folder="Projects")

    notes = store.list_folder("Projects")
    assert len(notes) == 2


def test_update_note(tmp_path):
    cfg = VaultConfig(vault_path=tmp_path)
    store = NoteStore(cfg)
    path = store.create(title="Update Me", content="Old", folder="Knowledge")
    store.update(path, content="New content")

    text = path.read_text(encoding="utf-8")
    assert "New content" in text
    assert "Old" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_note_store.py -v`
Expected: FAIL

- [ ] **Step 3: Implement note_store.py (GREEN)**

```python
# src/vault_rag/store/note_store.py
from datetime import datetime
from pathlib import Path

from vault_rag.config import VaultConfig


class NoteStore:
    def __init__(self, config: VaultConfig):
        self.vault = config.vault_path

    def create(
        self, title: str, content: str, folder: str = "Knowledge", tags: list[str] | None = None
    ) -> Path:
        target_dir = self.vault / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{title}.md"

        tag_str = ", ".join(tags) if tags else ""
        frontmatter = (
            f"---\n"
            f"created: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"tags: [{tag_str}]\n"
            f"---\n\n"
        )
        path.write_text(f"{frontmatter}# {title}\n\n{content}\n", encoding="utf-8")
        return path

    def read(self, relative_path: str) -> str | None:
        path = self.vault / relative_path
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def list_folder(self, folder: str) -> list[Path]:
        target = self.vault / folder
        if not target.exists():
            return []
        return sorted(target.glob("*.md"))

    def update(self, path: Path, content: str) -> None:
        existing = path.read_text(encoding="utf-8")
        # Preserve frontmatter, replace body
        if existing.startswith("---"):
            end = existing.index("---", 3) + 3
            frontmatter = existing[: end + 1]
            # Find the heading
            lines = existing[end:].strip().split("\n")
            heading = lines[0] if lines and lines[0].startswith("#") else ""
            path.write_text(f"{frontmatter}\n{heading}\n\n{content}\n", encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")
```

- [ ] **Step 4: Run tests (GREEN)**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_note_store.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: note store CRUD with frontmatter preservation"
```

---

## Task 11: CLI Frontend (Unified Click CLI)

**Files:**
- Create: `C:/Users/slime/claude-projects/vault-rag/src/vault_rag/cli.py`
- Create: `C:/Users/slime/claude-projects/vault-rag/tests/test_cli.py`

- [ ] **Step 1: Write test_cli.py (RED)**

```python
# tests/test_cli.py
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from vault_rag.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


import pytest


def test_cli_has_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "vault-rag" in result.output.lower() or "Usage" in result.output


def test_scan_command(runner, tmp_path):
    with patch("vault_rag.cli._get_config") as mock_cfg:
        mock_cfg.return_value = MagicMock(
            vault_path=tmp_path,
            chroma_path=tmp_path / "chroma",
            excluded_dirs=(".obsidian", "_trash"),
            priority_dirs=("Projects",),
            embedding_model="text-embedding-3-small",
            embedding_dimensions=512,
        )
        (tmp_path / "test.md").write_text("# Test\n\nContent", encoding="utf-8")
        result = runner.invoke(cli, ["scan"])
        assert result.exit_code == 0
        assert "1" in result.output  # 1 file scanned


def test_health_command(runner, tmp_path):
    with patch("vault_rag.cli._get_config") as mock_cfg:
        mock_cfg.return_value = MagicMock(
            vault_path=tmp_path,
            excluded_dirs=(".obsidian", "_trash"),
            priority_dirs=("Projects",),
        )
        (tmp_path / "orphan.md").write_text("# Orphan\n\nNo links", encoding="utf-8")
        result = runner.invoke(cli, ["health"])
        assert result.exit_code == 0
        assert "orphan" in result.output.lower() or "total" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implement cli.py (GREEN)**

```python
# src/vault_rag/cli.py
import json

import click

from vault_rag.config import VaultConfig
from vault_rag.engine.health import HealthChecker
from vault_rag.ingest.scanner import VaultScanner


def _get_config() -> VaultConfig:
    return VaultConfig()


@click.group()
@click.version_option(package_name="vault-rag")
def cli():
    """vault-rag: Karpathy-style personal knowledge RAG system."""


@cli.command()
def scan():
    """Scan vault and show statistics."""
    cfg = _get_config()
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    click.echo(f"Scanned {len(notes)} notes from {cfg.vault_path}")
    for note in notes[:10]:
        tags = ", ".join(note.tags[:5]) if note.tags else "(no tags)"
        click.echo(f"  {note.relative_path} [{tags}]")
    if len(notes) > 10:
        click.echo(f"  ... and {len(notes) - 10} more")


@cli.command()
def health():
    """Run health checks on vault."""
    cfg = _get_config()
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    hc = HealthChecker(notes)
    report = hc.report()
    click.echo(f"Total notes: {report['total_notes']}")
    click.echo(f"Broken links: {report['broken_link_count']}")
    click.echo(f"Orphan notes: {report['orphan_count']}")
    click.echo(f"Untagged: {report['untagged_count']}")
    if report["broken_links"]:
        click.echo("\nBroken links:")
        for bl in report["broken_links"][:10]:
            click.echo(f"  {bl['source']} -> {bl['target']}")


@cli.command()
@click.argument("query")
@click.option("-n", "--num", default=10, help="Number of results")
def search(query: str, num: int):
    """Semantic search across vault."""
    cfg = _get_config()
    from vault_rag.engine.indexer import create_openai_embed_fn
    from vault_rag.engine.qa import QAEngine
    from vault_rag.store.vector_store import VectorStore

    vs = VectorStore(persist_dir=cfg.chroma_path)
    if vs.count() == 0:
        click.echo("Index is empty. Run 'vault-rag index' first.")
        return

    embed_fn = create_openai_embed_fn(cfg.embedding_model, cfg.embedding_dimensions)
    qa = QAEngine(vector_store=vs, embed_fn=embed_fn, client=None, model="")
    results = qa.search(query, n_results=num)

    for i, r in enumerate(results, 1):
        title = r["metadata"].get("title", r["id"])
        tags = r["metadata"].get("tags", "")
        dist = f"{r['distance']:.3f}" if r.get("distance") is not None else "?"
        click.echo(f"  {i}. [{dist}] {title} ({r['id']})")
        if tags:
            click.echo(f"     tags: {tags}")


@cli.command()
@click.option("--full", is_flag=True, help="Full reindex (clear + rebuild)")
def index(full: bool):
    """Build/update embedding index."""
    cfg = _get_config()
    from vault_rag.engine.indexer import Indexer, create_openai_embed_fn
    from vault_rag.store.vector_store import VectorStore

    scanner = VaultScanner(cfg)
    vs = VectorStore(persist_dir=cfg.chroma_path)
    embed_fn = create_openai_embed_fn(cfg.embedding_model, cfg.embedding_dimensions)
    indexer = Indexer(config=cfg, vector_store=vs, embed_fn=embed_fn)

    notes = scanner.scan()
    click.echo(f"Scanned {len(notes)} notes")

    if full:
        count = indexer.reindex(notes)
    else:
        count = indexer.index(notes)

    click.echo(f"Indexed {count} notes. Total in store: {vs.count()}")


@cli.command()
@click.argument("question")
def ask(question: str):
    """Ask a question (RAG Q&A)."""
    cfg = _get_config()
    from anthropic import Anthropic
    from vault_rag.engine.indexer import create_openai_embed_fn
    from vault_rag.engine.qa import QAEngine
    from vault_rag.store.vector_store import VectorStore

    vs = VectorStore(persist_dir=cfg.chroma_path)
    embed_fn = create_openai_embed_fn(cfg.embedding_model, cfg.embedding_dimensions)
    client = Anthropic()
    qa = QAEngine(vector_store=vs, embed_fn=embed_fn, client=client, model=cfg.qa_model)

    result = qa.answer(question)
    click.echo(result.answer)
    if result.sources:
        click.echo(f"\n--- Sources ({len(result.sources)}) ---")
        for s in result.sources[:5]:
            click.echo(f"  - {s['metadata'].get('title', s['id'])}")


@cli.command()
@click.argument("url")
@click.option("--folder", default="Reference", help="Target folder")
def clip(url: str, folder: str):
    """Clip a web page into vault."""
    cfg = _get_config()
    from vault_rag.ingest.web_clipper import WebClipper

    clipper = WebClipper(vault_path=cfg.vault_path)
    path = clipper.clip(url=url, folder=folder)
    click.echo(f"Clipped to {path.relative_to(cfg.vault_path)}")


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--folder", default="Research", help="Target folder")
def ingest_pdf(pdf_path: str, folder: str):
    """Ingest a PDF into vault."""
    cfg = _get_config()
    from pathlib import Path
    from vault_rag.ingest.pdf_reader import PDFReader

    reader = PDFReader()
    note_path = reader.extract_to_note(Path(pdf_path), vault_path=cfg.vault_path, folder=folder)
    click.echo(f"Ingested to {note_path.relative_to(cfg.vault_path)}")


@cli.command()
@click.argument("path")
def compile(path: str):
    """Compile a note (auto-summarize, tag, link)."""
    cfg = _get_config()
    from anthropic import Anthropic
    from vault_rag.engine.compiler import Compiler
    from vault_rag.ingest.scanner import VaultScanner

    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    target = next((n for n in notes if path in n.relative_path), None)
    if not target:
        click.echo(f"Note not found: {path}")
        return

    client = Anthropic()
    compiler = Compiler(client=client, model=cfg.compile_model)
    result = compiler.compile(target)

    click.echo(f"Summary: {result.summary}")
    click.echo(f"Tags: {', '.join(result.tags)}")
    click.echo(f"Suggested links: {', '.join(result.suggested_links)}")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run tests (GREEN)**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_cli.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: unified Click CLI with scan/index/search/ask/clip/health/compile"
```

---

## Task 12: Integration Test (End-to-End Pipeline)

**Files:**
- Create: `C:/Users/slime/claude-projects/vault-rag/tests/test_integration.py`

- [ ] **Step 1: Write integration test (RED)**

```python
# tests/test_integration.py
"""End-to-end: scan -> index -> search -> Q&A -> health."""
from unittest.mock import MagicMock
from vault_rag.config import VaultConfig
from vault_rag.ingest.scanner import VaultScanner
from vault_rag.engine.indexer import Indexer
from vault_rag.engine.graph import KnowledgeGraph
from vault_rag.engine.health import HealthChecker
from vault_rag.engine.qa import QAEngine
from vault_rag.store.vector_store import VectorStore


def _mock_embed(texts):
    """Deterministic mock: hash-based pseudo-embeddings."""
    results = []
    for t in texts:
        h = hash(t) % 1000
        results.append([(h + i) / 1000.0 for i in range(512)])
    return results


def test_full_pipeline(tmp_vault, cfg, tmp_path):
    # 1. Scan
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    assert len(notes) >= 2  # test-project.md + ai-agents.md + orphan-note.md

    # 2. Index
    vs = VectorStore(persist_dir=tmp_path / "chroma")
    indexer = Indexer(config=cfg, vector_store=vs, embed_fn=_mock_embed)
    count = indexer.index(notes)
    assert count == len(notes)
    assert vs.count() == len(notes)

    # 3. Search
    qa = QAEngine(
        vector_store=vs, embed_fn=_mock_embed,
        client=MagicMock(), model="test",
    )
    results = qa.search("AI agents")
    assert len(results) > 0

    # 4. Graph
    graph = KnowledgeGraph()
    graph.build(notes)
    assert graph.node_count() == len(notes)
    assert graph.edge_count() >= 1  # at least A->B links

    # 5. Health
    hc = HealthChecker(notes)
    report = hc.report()
    assert report["total_notes"] == len(notes)

    # 6. Incremental scan (no changes)
    hashes = {n.relative_path: n.content_hash for n in notes}
    unchanged = scanner.scan(known_hashes=hashes)
    assert len(unchanged) == 0


def test_compile_and_index_pipeline(tmp_vault, cfg, tmp_path):
    """Compile + index pipeline: note -> compile -> index with metadata."""
    from vault_rag.engine.compiler import Compiler

    scanner = VaultScanner(cfg)
    notes = scanner.scan()

    # Mock compiler
    response_json = '{"summary": "Test summary", "tags": ["ai", "test"], "suggested_links": ["other"]}'
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=response_json)]
    client.messages.create.return_value = msg

    compiler = Compiler(client=client, model="test")
    result = compiler.compile(notes[0])
    assert result.summary == "Test summary"

    # Index
    vs = VectorStore(persist_dir=tmp_path / "chroma2")
    indexer = Indexer(config=cfg, vector_store=vs, embed_fn=_mock_embed)
    indexer.index(notes)
    assert vs.count() == len(notes)
```

- [ ] **Step 2: Run integration test**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/test_integration.py -v`
Expected: 2 passed

- [ ] **Step 3: Run full test suite with coverage**

Run: `cd C:/Users/slime/claude-projects/vault-rag && python -m pytest tests/ -v --cov=vault_rag --cov-report=term-missing`
Expected: All tests pass, 80%+ coverage

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "test: integration tests for full scan-index-search-qa-health pipeline"
```

---

## Task 13: Install Dependencies + Verify

- [ ] **Step 1: Install all dependencies**

```bash
cd C:/Users/slime/claude-projects/vault-rag
pip install chromadb trafilatura
pip install -e ".[dev]"
```

- [ ] **Step 2: Run full test suite**

```bash
python -m pytest tests/ -v --cov=vault_rag
```

Expected: All tests pass

- [ ] **Step 3: Test CLI manually against real vault**

```bash
# Scan real vault
python -m vault_rag.cli scan

# Health check
python -m vault_rag.cli health

# Build index (requires OPENAI_API_KEY)
python -m vault_rag.cli index

# Search
python -m vault_rag.cli search "AI agents"

# Ask a question
python -m vault_rag.cli ask "What projects am I working on?"
```

- [ ] **Step 4: Final commit**

```bash
git add -A && git commit -m "chore: verify full pipeline against real vault"
```

---

## Summary

| Layer | Component | Status |
|-------|-----------|--------|
| **Data Ingest** | Scanner (incremental) | Task 2 |
| **Data Ingest** | Web Clipper (trafilatura) | Task 9 |
| **Data Ingest** | PDF Reader (PyMuPDF) | Task 9 |
| **LLM Engine** | Indexer (OpenAI embeddings + ChromaDB) | Task 4 |
| **LLM Engine** | Knowledge Graph (NetworkX) | Task 5 |
| **LLM Engine** | Compiler (Claude auto-summarize/tag/link) | Task 6 |
| **LLM Engine** | Q&A (RAG retrieve+rank+generate) | Task 7 |
| **LLM Engine** | Health Checks (orphans/broken/untagged) | Task 8 |
| **Knowledge Store** | Vector Store (ChromaDB wrapper) | Task 3 |
| **Knowledge Store** | Note Store (CRUD + frontmatter) | Task 10 |
| **IDE Frontend** | Unified CLI (Click) | Task 11 |
| **Integration** | E2E pipeline test | Task 12 |

**Karpathy 원본 대비 미포함 (FUTURE):**
- Slides/Charts/Dashboards (Obsidian 플러그인으로 해결)
- Synthetic data gen (flashcards, Q&A 생성) — v2에서 추가
- Video/Audio transcript ingest — v2에서 Whisper 연동
