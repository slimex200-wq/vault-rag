# Checks

## Standard Commands

```bash
python -m pytest tests/ -q
python -m pytest tests/ --cov=vault_rag --cov-report=term-missing -q
ruff check .
ruff format --check .
```

## Risk-Based Checks

- Retrieval/indexing changes: include a test using a small fixture vault.
- CLI changes: run `vault-rag --help` or the affected command path.
- Model/API changes: document required environment variables and cost implications.

## Before Delivery

- Report which commands ran.
- Report any check that could not be run.
- For model/API work, mention whether live API calls were avoided or performed.
