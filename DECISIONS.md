# Decisions

## Architecture

- `VaultConfig` stays a plain dataclass to keep CLI startup light.
- `chroma_path` and `graph_path` stay derived properties so tests can redirect `vault_path` cleanly.
- Default priority order is `Projects > Knowledge > Research > Reference > Dev`.
- Default excluded directories include `.obsidian`, `_trash`, `.git`, `docs`, and `Templates`.
- Excluded directory configuration should stay immutable/hashable unless there is a concrete need to change it.

## Model Usage

- Keep cheap embeddings separate from more expensive compile/QA model calls.
- Do not add automatic large-scale vault processing without explicit cost and safety notes.
- Default model/API changes must document required environment variables and expected cost impact.
- Anthropic auth resolves per call, not per process: GJC credential store (`~/.gjc/agent/agent.db`, read-only) first, then `ANTHROPIC_OAUTH_TOKEN`, then `ANTHROPIC_API_KEY`. The store wins over the environment variable because its access token rotates roughly every eight hours, so any copy held in a shell or a long-lived process is stale by the next day.
- `llm_client` never writes to that store. Refresh, rotation, and disabling belong to the agent runtime; this repo only reads the current access token and falls back silently when the store is missing, unreadable, disabled, or expired.
