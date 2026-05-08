# Hermes Agent Fork — Project Memory

## Repository
- **Path:** `~/Projects/hermes-agent/`
- **Fork:** `github.com:nicoechaniz/hermes-agent.git` (origin)
- **Upstream:** `github.com:NousResearch/hermes-agent.git` (upstream)
- **Deploy target:** `~/.hermes/hermes-agent/` (NEVER edit directly; update via `hermes update`)

## Upstream Velocity Context
Hermes is one of the most actively developed open-source projects on GitHub — among the fastest-growing and most pull-requested. **~150 commits/day is normal**, with peaks of 250+. A gap of 800-1000 commits is NOT months of neglect — it is roughly **one week** of upstream development. Do not interpret "N commits behind" as a crisis; it is the baseline reality of tracking this repo. We sync when it makes sense, not out of alarm.

## Active Branches
| Branch | Purpose | Base |
|--------|---------|------|
| `main` | Integration branch — upstream v2026.5.7 + all our features | `upstream/main` (2026-05-08) |
| `feat/daemoncraft` | Consolidated DaemonCraft branch (DC-99, 112, 123, 132, 134) | `main` |
| `nousmain` | Mirror of `upstream/main` (synced 2026-05-08) | — |
| `integration/upstream-sync-2026-05-08` | Reference: merge commit from May 8 sync | `upstream/main` |

## Consolidation Policy
All DaemonCraft work lives under `feat/daemoncraft`. We do NOT keep separate per-DC branches. When a DC feature is done, it gets merged into `feat/daemoncraft`. No `feat/dc-NNN-*` branches survive after merge.

## Our Feature Set (merged into main → feat/daemoncraft)
1. **feat/kimi-oauth-clean** — Kimi OAuth refresh, header fixes
2. **feat/altermundi-tui** — TUI scrollbar, max lines config
3. **feat/altermundi-cli** — Ctrl+C priority config
4. **feat/minimax-defaults** — MiniMax provider defaults
5. **feat/compression-config-reboot** — Configurable compression protect_first_n
6. **feat/dc-112-daemoncraft-gateway** — Gateway adapter wiring, tool_choice propagation
7. **DC-99** — Profile system prompt override per platform
8. **DC-123** — TTS fixes + wake-up logging, CycleDetector
9. **DC-132** — Contextvars-based endpoint resolution for minecraft tools, turn metrics
10. **DC-134** — Configurable turn wall-clock timeout + per-profile max_iterations

## Sync History
| Date | Upstream Version | Commits | Notes |
|------|-----------------|---------|-------|
| 2026-04-30 | v2026.4.30 | base | Initial sync |
| 2026-05-08 | v2026.5.7 | +993 | Full rebase, 7 conflicts resolved, all features preserved |

## Current Milestone
- **Branch:** `main` = `feat/daemoncraft` = `upstream/main` v2026.5.7 + our features
- **Status:** Clean, deployed, gateway running

## Known Pending Work
- **feat/dc-105-unified-social-routing** — Social routing (unmerged, may need cleanup)
- **feat/dc-94-gateway** — Gateway feature (unmerged, may need cleanup)
- **debug/dc-99-log** — Debug branch, can be deleted
- **fix/dc-123-dc-132-temp** — Contaminated with autoresearch, can be deleted (commits already in main)

## Deploy Verification
- Syntax check: `python3 -m py_compile gateway/run.py tools/minecraft_tools.py ...`
- Gateway restart: `systemctl --user restart hermes-gateway.service`
- Health check: DaemonCraft WebSocket connected + Telegram polling
- Steve bot: port 3001, managed by DaemonCraft launcher

## Files We Touch Regularly
| File | What it does |
|------|-------------|
| `gateway/run.py` | Gateway runner — conflicts with upstream on sync |
| `gateway/platforms/daemoncraft.py` | DaemonCraft platform adapter |
| `tools/minecraft_tools.py` | Minecraft tool implementations |
| `model_tools.py` | Tool dispatch, threads session_id |
| `toolsets.py` | Toolset registration |
| `hermes_cli/tools_config.py` | CONFIGURABLE_TOOLSETS |
| `~/.hermes/config.yaml` | Runtime config (written by DaemonCraft launcher) |

## Test Command
```bash
scripts/run_tests.sh
```

## Lattice
- Project initialized in repo root (`.lattice/`)
- Ignored in git via `.git/info/exclude`
