# Project State

## Purpose

Karpathy-style personal knowledge RAG system for Obsidian vaults.

## Current Status

- Active Python package with tests, coverage config, Ruff, and CLI entry point.
- AI harness files are committed and pushed on `master`.
- `CLAUDE.md` is a thin Claude/OMC adapter; architecture and default path assumptions live in `README.md`, `DECISIONS.md`, `CHECKS.md`, and this file.

## Next Work Queue

- Keep retrieval, graph, compile, and QA changes tested.
- Keep import-time dependencies light.
- Update docs when CLI behavior or default paths change.

## Known Blockers

- Live API behavior requires OpenAI/Anthropic credentials and cost-aware testing.

## Last Verified

- 2026-08-16: `python -m pytest tests/ -q` (232 passed), `ruff check .`, and `ruff format --check .` passed.
- 2026-08-16: live Anthropic call performed through the OAuth path in three configurations — store only, store with a deliberately stale `ANTHROPIC_OAUTH_TOKEN`, and env token alone — all returned normally.
- Known gap: live OpenAI embedding calls were not performed (`embedding_provider` defaults to `local`).

## Related Vault Notes

- `C:/Users/slime/claude-projects/Obsidian Vault/Projects/vault-rag/`

## Handoff Rule

When work changes indexing, retrieval, graph behavior, model usage, default paths, or CLI commands, update this file with the new status and next concrete action.
