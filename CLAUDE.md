# Shared Harness

Read `AGENTS.md`, `PROJECT_STATE.md`, `CHECKS.md`, and `DECISIONS.md` before making changes. The details below remain authoritative for repo-specific commands, architecture, and design decisions.

# vault-rag

Karpathy-style personal knowledge RAG system for Obsidian vaults.  
Indexes markdown notes via ChromaDB embeddings + NetworkX link graph, then answers questions with a two-stage Anthropic pipeline (compile → qa).

## Commands

```bash
# Test
python -m pytest tests/ -q

# Test with coverage
python -m pytest tests/ --cov=vault_rag --cov-report=term-missing -q

# Lint + fix
ruff check . --fix

# Format
ruff format .

# Install (editable + dev deps)
pip install -e ".[dev]"

# Run CLI
vault-rag --help
```

## Architecture

```
src/vault_rag/
  config.py       — VaultConfig dataclass (single source of truth for paths/models)
  scanner.py      — Vault traversal, front-matter parsing, wikilink extraction
  embedder.py     — OpenAI text-embedding-3-small, upsert to ChromaDB
  graph.py        — NetworkX wikilink graph, PageRank, BFS context expansion
  compiler.py     — Anthropic "compile" pass: distill chunks into dense context
  qa.py           — Anthropic Q&A pass: answer from compiled context
  store.py        — ChromaDB CRUD helpers
  clipper.py      — Web (trafilatura) + PDF (PyMuPDF) ingest
  health.py       — Sanity checks: vault exists, chroma reachable, API keys set
  cli.py          — Click CLI entry point
```

## Key Paths (default)

| Resource | Path |
|----------|------|
| Vault | `C:/Users/slime/claude-projects/Obsidian Vault/` |
| ChromaDB | `../vault-rag/data/chroma/` |
| Graph JSON | `../vault-rag/data/graph.json` |

## Design Decisions

- `VaultConfig` is a plain dataclass (no pydantic) — import cost matters for CLI startup.
- `chroma_path` / `graph_path` are properties so they follow `vault_path` changes (useful in tests with `tmp_path`).
- Excluded dirs (`.obsidian`, `_trash`, `.git`, `docs`, `Templates`) are tuples — immutable and hashable.
- Priority dirs drive PageRank seed weighting: `Projects > Knowledge > Research > Reference > Dev`.
- Two-model pipeline: cheap embedding model for retrieval, expensive `claude-sonnet-4` for compile+qa.

## Environment Variables

```bash
OPENAI_API_KEY=...        # required for embeddings
ANTHROPIC_API_KEY=...     # required for compile + qa
```
