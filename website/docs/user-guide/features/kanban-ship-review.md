---
sidebar_position: 13
title: "Ship Review (kanban review)"
description: "Create durable review graphs for code changes with safe triage, ready dispatch, and REVIEW-ONLY contracts"
---

# Ship Review — Kanban Review Graphs

`hermes kanban review create` builds a durable 5-card review graph for any git change. It replaces ad-hoc "hey can someone review this?" messages with a structured, tracked, and replayable workflow.

## The graph shape

```
Parent review card (organisational umbrella)
├─ [REVIEW] Code quality    ─┐
├─ [REVIEW] Security         │  parallel reviewers
└─ [REVIEW] Test coverage   ─┘
         │
         ▼
[SYNTHESIS] GO/NO-GO decision  ← gated on all three reviewers
```

1. **Parent card** — holds the base..head context and the REVIEW-ONLY contract.
2. **Three reviewers** — run in parallel, each with a role-specific checklist.
3. **Synthesis** — auto-promotes to `ready` once all reviewers finish. It reads their handoffs and produces a GO/NO-GO decision.

## Safe triage by default

By default every card is created in `triage`:

```bash
hermes kanban review create "Review PR #42" \
  --base nousmain \
  --head feat/auth \
  --repo /home/me/Projects/myapp \
  --assignee miki
```

Nothing dispatches until a human explicitly promotes cards. This is the safe pattern for reviews that need scheduling or human triage.

## Ready dispatch

If you want the reviewers to start immediately:

```bash
hermes kanban review create "Review PR #42" \
  --base nousmain \
  --head feat/auth \
  --repo /home/me/Projects/myapp \
  --assignee miki \
  --ready
```

With `--ready`:
- Parent + reviewer cards start in `ready` (dispatcher picks them up on next tick).
- Synthesis card starts in `todo` because its parents (the reviewers) are not yet `done`.
- As each reviewer completes, `kanban_db` auto-runs `recompute_ready`.
- When the third reviewer finishes, the synthesis auto-promotes from `todo` → `ready`.

## Local Miki example

A concrete invocation on the Hermes repo itself, using `--json` for scripting:

```bash
hermes kanban review create "Ship kanban review orchestration" \
  --base nousmain \
  --head feat/kanban-ship-review-orchestration \
  --repo /home/nicolas/Projects/hermes-agent \
  --assignee miki \
  --triage \
  --json
```

Output:
```json
{
  "parent_id": "t_a1b2c3d4",
  "reviewer_ids": ["t_e5f6g7h8", "t_i9j0k1l2", "t_m3n4o5p6"],
  "synthesis_id": "t_q7r8s9t0",
  "created": true
}
```

Rerun the same command and you get the **same IDs** — the graph is idempotent by `sha256(repo realpath) + base + head + role`.

## Review-only limitation

Every generated body contains a **REVIEW-ONLY v1** contract:

> Do NOT modify source code. Report findings as structured metadata only.

This is intentional. Reviewer workers are scoped to read, analyse, and report. They do not patch, commit, or push. If a reviewer finds a bug, it records the finding in `kanban_complete(metadata={"findings": [...]})` and the synthesis task decides whether to spawn a separate remediation task.

The contract exists because:
- **Auditability** — a review that silently fixes its own findings is indistinguishable from a no-op.
- **Separation of concerns** — reviewers judge; other agents (or humans) remediate.
- **Safety** — a reviewer with write access could introduce new issues while fixing old ones, especially when running autonomously.

## JSON CLI output

Pass `--json` to get machine-readable output:

```bash
hermes kanban review create "Review PR #42" \
  --base main --head feat/x --repo . --json
```

Keys:
- `parent_id` — the organisational umbrella card
- `reviewer_ids` — list of 3 reviewer task ids
- `synthesis_id` — the synthesis task id
- `created` — `true` if new cards were created, `false` if all existed already

## Idempotency

The graph is keyed by the **resolved repo path** + **base** + **head** + **role**. Changing any of `base`, `head`, or the absolute repo path creates a new graph. Moving the repo directory (e.g., symlinks that resolve differently) also creates a new graph — use stable absolute paths in automation.

## Skills

Attach skills to every card with `--skill` (repeatable):

```bash
hermes kanban review create "Review auth PR" \
  --base main --head feat/auth --repo . \
  --assignee reviewer \
  --skill github-code-review \
  --skill security-pr-audit
```

These are force-loaded into the worker alongside the built-in `kanban-worker` skill.

## Body templates

Each role gets a hardened body with:
- Exact `git diff` commands to run
- Severity labels (**Critical**, **Important**, **Optional/Nit**)
- Role-specific checklist (code-quality, security, test-coverage)
- `kanban_complete` / `kanban_block` contract with expected metadata shape

The synthesis body expects:
- GO/NO-GO decision with rationale
- Blockers, recommended fixes, acknowledged risks
- Rollback plan and evidence reviewed

Bodies are self-contained — a worker can execute the review without conversation history or external context.
