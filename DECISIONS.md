# Decisions

## Architecture

- `VaultConfig` stays a plain dataclass to keep CLI startup light.
- Default priority order is `Projects > Knowledge > Research > Reference > Dev`.
- Default excluded directories include `.obsidian`, `_trash`, `.git`, `docs`, and `Templates`.

## Model Usage

- Keep cheap embeddings separate from more expensive compile/QA model calls.
- Do not add automatic large-scale vault processing without explicit cost and safety notes.
