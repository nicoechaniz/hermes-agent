# Fork Workflow — nicoechaniz/hermes-agent

This repository is Nicolás' Hermes Agent fork. It intentionally carries local features that may or may not be upstreamed. Keep this file current whenever the branch topology changes.

## Branch roles

| Branch | Role | Rules |
|--------|------|-------|
| `upstream/main` | NousResearch upstream | Read-only reference. Fetch from `https://github.com/NousResearch/hermes-agent.git`. |
| `nousmain` | Local mirror of upstream main | Must match `upstream/main` exactly after each upstream sync. Do not commit directly. |
| `main` | Integration branch for our runtime | Merge `nousmain` plus all active canonical feature branches. This is what gets deployed to `~/.hermes/hermes-agent`. |
| `feat/*` / `fix/*` | Canonical fork features/fixes | Branch from `nousmain`, not from `main`, unless explicitly creating a short-lived integration/docs cleanup branch. Each branch must apply cleanly over `nousmain`. |
| `backup/*` / `*-legacy` | Safety archives | Do not merge. Keep only while useful for recovery. |

## Maintained patch lanes

The active set is explicit. It is not inferred from `git branch`, merge ancestry,
or the presence of an old remote ref. Each lane must be a clean patch series over
`nousmain` before it can participate in an upstream sync:

| Branch | Purpose |
|--------|---------|
| `feat/altermundi` | Generic Altermundi/Hermes changes. |
| `feat/kimi` | Kimi OAuth/provider behavior, including Kimi Coding streaming and truncation fixes. |
| `feat/kimi-webbridge` | Kimi real-browser toolset. |
| `feat/video-gen-minimax` | MiniMax video generation provider. |
| `fix/tui-history-nav-requires-empty` | Ink TUI history navigation behavior. |
| `feat/daemoncraft` | The single DaemonCraft/Hermes lane: platform registration, gateway, adapter, event/session plumbing, embodied tools, and focused tests. |

`backup/*`, `safepoint-*`, `main-rebuild`, `*-legacy`, and recovery branches are
archives, not composition inputs. Do not drop an active lane because a similar
change exists in `main`; normalize its patch series and compare it deliberately.

The live topology and clean-apply audit are recorded in
`~/wiki/projects/hermes-agent/notes/fork-topology-2026-07-15.md`.

## Sync workflow

Before any push or pull, verify remotes:

```bash
git remote -v
```

Update the upstream mirror:

```bash
git fetch upstream main
git checkout nousmain
git reset --hard upstream/main
```

Rebuild each maintained patch lane on top of the new `nousmain`. Before merging,
prove that each one applies cleanly:

```bash
git checkout <lane>
git rebase nousmain
git merge-tree --write-tree nousmain <lane>
# exit code 0 is required for every active lane
```

Run focused tests after each rebase. Do not rebase `main` onto feature branches;
`main` is rebuilt only after every maintained lane has passed the clean-apply gate.

Rebuild integration `main`:

```bash
git checkout main
git reset --hard nousmain
git merge --no-ff feat/altermundi
git merge --no-ff feat/kimi
git merge --no-ff feat/kimi-webbridge
git merge --no-ff feat/video-gen-minimax
git merge --no-ff fix/tui-history-nav-requires-empty
git merge --no-ff feat/daemoncraft
```

Run tests before pushing. Use the wrapper, never raw `pytest`:

```bash
scripts/run_tests.sh
```

For provider-specific fixes, add an end-to-end smoke test when credentials are available. Example for Kimi OAuth:

```bash
hermes chat --provider kimi-coding -m kimi-k2.6 -q 'Say OK only.' -Q --yolo
```

Push and deploy:

```bash
git push origin main
git -C ~/.hermes/hermes-agent fetch origin main
git -C ~/.hermes/hermes-agent reset --hard origin/main
```

If the gateway is running and the change affects gateway/runtime behavior, restart the relevant service after deployment.

## Rebirth sync coupling

Fork sync and CompAII rebirth sync go together. After a meaningful Hermes fork sync or deploy, also run the rebirth sync from the CompAII state repository:

```bash
python ~/Projects/compaii-state/sync.py
```

## Verification checklist

Before declaring a branch/workflow update complete:

1. `git remote -v` confirms `origin` and `upstream`.
2. `git status --short --branch` is clean.
3. Every maintained patch lane passes `git merge-tree --write-tree nousmain <lane>`.
4. `main` is composed from every maintained patch lane, in the documented order.
5. Superseded remote branches are deleted or clearly documented as archives.
6. Tests relevant to the changed areas pass via `scripts/run_tests.sh`.
7. Deployed checkout `~/.hermes/hermes-agent` matches `origin/main` when runtime behavior changed.
