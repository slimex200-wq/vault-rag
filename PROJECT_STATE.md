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

- 2026-06-20: LLM 인증을 Claude 구독 OAuth 로 전환(`llm_client.make_anthropic_client`).
  실호출 검증 완료 — OAuth 직접 호출 OK, `ask` end-to-end OK(구독 과금, API 키 미사용).
  `qa_model` 을 죽은 `claude-sonnet-4-20250514`(404) → `claude-sonnet-4-6` 로 교체.
  ChromaDB 전체 재인덱싱(4,013 → 4,084 노트, 61s). pytest 155 passed, ruff clean.
- 2026-04-30: `python -m pytest tests/ -q`, `ruff check .`, and `ruff format --check .` passed.

## Related Vault Notes

- `C:/Users/slime/claude-projects/Obsidian Vault/Projects/vault-rag/`

## Handoff Rule

When work changes indexing, retrieval, graph behavior, model usage, default paths, or CLI commands, update this file with the new status and next concrete action.
