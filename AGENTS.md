# AI Harness

Before making changes, read:

1. `PROJECT_STATE.md` — current status, active goal, blockers, and next work.
2. `CLAUDE.md` — repository-specific architecture, commands, and design decisions.
3. `CHECKS.md` — commands and manual checks that prove a change is safe.
4. `DECISIONS.md` — decisions that should not be re-litigated casually.
5. `GITHUB_WORKFLOW.md` — issue, branch, PR, and sync rules.
6. `README.md` — public setup and usage details.

## Operating Rules

- Preserve user changes and untracked work. Check `git status --short` before edits.
- Keep CLI startup light; avoid adding heavy import-time dependencies.
- Do not change default vault paths, model choices, or exclusion rules without updating docs and tests.
- After meaningful changes, run the smallest relevant check from `CHECKS.md` and report any known gap.
- Update `PROJECT_STATE.md` at handoff when status, next work, blockers, or verification changes.
