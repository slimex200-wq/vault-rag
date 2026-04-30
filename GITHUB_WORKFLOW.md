# GitHub Workflow

## Current Branch

- Active branch: `master`
- Remote branch: `origin/master`

## Issues

- Use issues for CLI milestones, retrieval quality, cost-risk work, API changes, and multi-step refactors.
- Link PRs with `Refs #N` or `Closes #N` depending on whether the issue is fully resolved.

## Pull Requests

- Prefer PRs for architecture, retrieval, model, or CLI behavior changes.
- Small docs/harness/style fixes may go directly to `master` when checks pass.

## Sync Rule

```bash
git fetch --all --prune
git status -sb
git pull --rebase origin master
```

Stage files explicitly when generated vault data or local outputs are present.
