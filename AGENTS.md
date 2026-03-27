# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Branch Policy

- Never create or use side branches for work in this repo.
- Do all work directly on `main`.
- If a user explicitly asks for branch-based work, merge that branch back into `main` immediately after the work is complete and before handoff.
- If you ever find local commits on any branch other than `main`, merge or fast-forward them back into `main` immediately and return to `main`.
- Before starting work, switch to `main` and keep it current with `git pull --rebase`.
- Before handing off, make sure `main` contains the latest local commits.
- Do not edit git configuration, remotes, tracking branches, or credential settings unless the user explicitly asks.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Artifact Policy

- Never commit `.pt` artifacts of any kind. This includes model checkpoints,
  feature caches, and serialized tensors used for local experiments.
- Treat large generated experiment exports as disposable unless the user
  explicitly asks to version them. In particular, do not commit oversized
  `reports/plots/*_decisions.csv`-style artifacts.
- If a forbidden artifact is ever committed locally, purge it from history
  before handoff and retry the push.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
