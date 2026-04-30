# Shared AI Entry Point

This repo is used by Codex/OMX and Claude/OMC. Follow your local runtime rules first; this file only points to project-specific context.

Before changing files, read:

1. `PROJECT_STATE.md` - current status, next work, blockers, and last verification.
2. `CHECKS.md` - repo-specific verification commands and risk checks.
3. `DECISIONS.md` - retrieval, model, path, and architecture decisions.
4. `GITHUB_WORKFLOW.md` - default branch, issue, PR, and sync rules.
5. `README.md` - public setup and usage details.

Project facts in those files override generic assumptions. Keep this file thin; put durable project facts in the dedicated harness files.

## Local Context

- Keep CLI startup light.
- Retrieval, indexing, graph, and model changes need focused tests or explicit live-API gaps.
