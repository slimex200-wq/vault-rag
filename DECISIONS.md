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
