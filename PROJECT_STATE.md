# Project State

## Purpose

Karpathy-style personal knowledge RAG system for Obsidian vaults.

## Current Status

- Active Python package with tests, coverage config, Ruff, and CLI entry point.
- `.omc/` is currently untracked; do not clean it up unless explicitly asked.
- `CLAUDE.md` contains the current architecture and default path assumptions.

## Next Work Queue

- Keep retrieval, graph, compile, and QA changes tested.
- Keep import-time dependencies light.
- Update docs when CLI behavior or default paths change.

## Handoff Rule

When work changes indexing, retrieval, graph behavior, model usage, default paths, or CLI commands, update this file with the new status and next concrete action.
