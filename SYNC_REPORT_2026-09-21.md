# Hermes Fork Sync Report — 2026-09-21

## Executive summary

This sync rebases the AlterMundi Hermes fork from the previous upstream mirror to NousResearch `main` at `0ddeaf9334ff232a01612a51520a043db2cab77b` (Hermes Agent `0.21.3`). The upstream delta contains 13,986 commits. Fork-owned behavior was audited as independent lanes instead of replaying the historical integration branch wholesale.

The candidate branch is `sync/2026-09-21/main-candidate`. All maintained lanes are integrated. Canonical branches and deployments remain unchanged until the full-suite gate passes.

## Safety receipts

The pre-sync state was preserved before refs moved:

- Hermes repository backup: `/home/nicolas/Projects/hermes-sync-backups/20260921T175244Z-pre-sync`
  - verified bundle size: 807,620,892 bytes
  - five checksums verified
- Local rebirth repository backup: `/home/nicolas/Projects/compaii-state-backups/20260921T175244Z-pre-sync`
- Daimonmatrix rebirth repository backup: `/home/debian/Projects/compaii-state-backups/20260921T175244Z-pre-sync`
- Git safety refs: `backup/pre-upstream-sync-20260921-175244-*`

The staged `compaii-state` work on daimonmatrix was not reset, merged, or overwritten.

### Concurrent daimonmatrix publication

After the initial inventory, `compaii@daimonmatrix` advanced both remote refs:

- `origin/feat/altermundi` → `8008a8084953b03c3d74d6be70af099292b42cac`
- `origin/main` → `8c3f2447ff4d0d88c8159237c080082e9c9f6a2f`

Those tips were preserved locally before further work as:

- `backup/daimonmatrix-push-20260921T200019Z-origin-feat-altermundi`
- `backup/daimonmatrix-push-20260921T200019Z-origin-main`

The remote delta adds Alibaba China, Coding Plan China, and Token Plan provider/runtime/catalog/documentation coverage. Three of its four commits are patch-equivalent to current upstream; the remaining structural follow-up has the same net behavior in `nousmain`. The composed candidate already contains all four provider profiles and their runtime/docs coverage. `tests/providers/test_provider_profiles.py` plus `tests/hermes_cli/test_api_key_providers.py` pass together (**137 passed, 0 failed**). No remote work was overwritten and no additional Alibaba port is required.

## Upstream change profile

The previous mirror was 13,986 commits behind current Nous `main`. The largest commit-subject clusters are overlapping by design:

| Area | Matching upstream commits |
|---|---:|
| Memory, context, sessions, compression | 2,477 |
| Tools, plugins, skills, MCP | 2,226 |
| Gateway and messaging platforms | 2,035 |
| Desktop and dashboard UI | 1,724 |
| Providers and models | 1,286 |
| Security, authentication, credentials | 1,072 |
| TUI and classic CLI | 976 |
| Update, backup, deployment, fleet | 922 |
| Cron and Kanban | 786 |
| Delegation and subagents | 230 |

### Operationally important upstream changes

- Desktop, dashboard, TUI, and gateway are now distinct surfaces over shared transports; Desktop no longer depends on the dashboard frontend.
- Gateway execution was split into focused modules and hardened for multi-profile isolation, per-profile secrets, turn pools, delivery ownership, and fleet updates.
- Compression gained bounded workers, race handling, provider-aware auxiliary routing, commit fences, and improved telemetry.
- Providers are plugin profiles with centralized runtime credential resolution and model metadata.
- The update pipeline now inventories, snapshots, applies, restarts per deployment kind, verifies live gateway code versions, and writes receipts.
- Cron became host-owned and profile-aware; Kanban gained a durable dispatcher/worker architecture.
- Tool and plugin discovery became more declarative, with a narrow core tool surface and service-gated capabilities.

## Fork lane audit

| Lane | Sync result | Decision |
|---|---|---|
| AlterMundi integration | `8fc960801696e6b7d6a80e11b8b0bda850cade74` + `127bda6a09aa26b5ab8f829754ad6b26703684e4` | Consolidates collective memory, authenticated plugin-command context, completion ownership/lineage fences, and fork workflow documentation in the canonical `feat/altermundi` lane. The follow-up tolerates sparse pre-guard events without weakening the e-stop. |
| AlterMundi classic CLI | `12846a733d31110f323f7eee1c8878280fb27bbc` | Ported `ctrl_c_priority` to current CLI mixins; stale monolithic CLI/TUI code discarded. |
| AlterMundi Ink TUI | `869fddb064633c7e9e5e75bb2cd86413abc62ba5` | Preserved empty-input history navigation behavior and coverage; final follow-up satisfies the complete workspace lint contract. |
| Kimi WebBridge | `21fa394e9cdf46d1d37044b8c465d5c1b0d5cb69` | Preserved opt-in browser bridge, current tool config integration, and safe screenshot path validation. |
| MiniMax video generation | `0879e2c83d3a5aabb700095ba2e3d94ff9d1543b` | Preserved as a video-generation provider plugin; upstream tool owns model selection. |
| DaemonCraft | `bfb6aa8a3467a55b2d865673b1c69c56c4ba7ff3` | Ported to modular gateway turn/context architecture; role-master turn options retained. |
| Kanban review orchestration | `7280d41ddffdd8310d67077b1b3a02be38f3ef44` | Preserved durable review graph and CLI integration in the normalized merge tip. |
| HMK native-memory retirement | upstream `0ddeaf9334` | Retired as a lane: upstream `5a5d6b966d` now gates the native memory tool in the registry without stale availability caching. Four runtime E2E contracts pass on upstream; only tests that required the obsolete helper failed. |
| Kimi official CLI OAuth | historical lane only | Retired from the active composition by operator decision. The branch/history remains recoverable for future reassessment; no fork-owned OAuth delta remains against `nousmain`. |

## Candidate verification

Verified code HEAD before the report-only commit: `e2d4216c4b8086194e1f4973c295cd82588737be`.

Current candidate delta: 63 commits across 88 files, with 13,221 insertions and 112 deletions relative to `nousmain`.

Verified together on the composed candidate:

- `scripts/run_tests.sh` over all 22 changed Python test files: **414 passed, 0 failed**.
- Ink TUI full workspace check (`build`, `typecheck`, Vitest, ESLint): **1,784 passed, 0 failed**, lint clean apart from two upstream deprecation warnings.
- DaemonCraft plus disabled-native-memory contract set: **50 passed, 0 failed** across 7 files.
- Kimi WebBridge remains the only fork-owned Kimi delta; comparison against `nousmain` lists only `tools/kimi_webbridge.py` and `tests/tools/test_kimi_webbridge.py`.
- Post-retirement Kimi/provider gate: **715 passed, 0 failed** across 9 files; Kimi WebBridge passed **28/28**.
- AlterMundi command-context/plugin-dispatch set on the composed candidate: **203 passed, 0 failed** across 4 files.
- AlterMundi ownership/notification/lineage set: **145 passed, 0 failed** across 5 files; the historical 22 named ownership contracts now execute as 60 parametrized cases on the current async architecture.
- xAI paid/public media storage remains opt-in: runtime storage contracts pass **4/4** and provider-picker contracts pass **57 passed, 6 skipped, 0 failed** (`a66d6d4f`).
- The six files touched by complete-suite follow-ups pass together: **360 passed, 0 failed**, with retries disabled (`e88b7527`).
- Five pinned-mtime replacement scenarios now use real atomic replacement instead of same-inode `copy2`; the three sibling cache contracts pass without retries (`7302668f`).
- Individual lane verification included plugin discovery, toolset resolution, compile smoke, CLI version smoke, and the focused behavioral tests for each lane.

The earlier context-compression incident is no longer a release blocker. A
read-only SQLite probe of session `20260921_135851_6d1868` found 2,746 messages,
of which 2,668 were durably marked compacted. No message content was inspected.

The full Python sweep ran every discovered test file in the synchronized `--all-extras` environment: **52,494 tests passed** and 20 tests failed across 19 files. Those failures exposed stale fixtures and integration gaps in xAI storage defaults, generated model metadata, updater dependency probes, MiniMax seed capability metadata, DaemonCraft registry schemas, and docs links. Kimi OAuth-specific failures became irrelevant when that lane was retired. Every remaining failed file, its current equivalent, and touched sibling suites were rerun together with retries disabled: **1,324 passed, 4 skipped, 0 failed across 24 files** before lane retirement; a focused post-retirement gate is recorded below.

The provider-cost audit also found that upstream's new xAI Imagine storage helper enabled permanent public storage when configuration was absent and warned that xAI may bill for storage/hosting. Fork commit `a66d6d4f4125f08b395be0df0193f46759bde78d` restores the previous no-side-effect default: storage is disabled unless explicitly selected, and the interactive picker defaults to the disabled choice. The active default-profile config already has `video_gen.xai.storage.enabled: false`; no provider or model selection was changed.

### Former deployed-main direct-commit gate

Walking the former deployed `main` by first parent found exactly one non-merge
commit outside lane merges: `3fdce5d4206abf9bbff0383caef56846b1dd63af`
(`fix(tui): reconcile duplicate config field after lane merge`). Its complete
delta removed a duplicate `tui?: ConfigTuiConfig` declaration. The candidate's
`ui-tui/src/gatewayTypes.ts` has exactly one such declaration, so this direct
commit is classified as already preserved. All other old-main-only patch IDs
belong to the maintained lane merges audited above.

Four parallel auditors also inspected Kimi, DaemonCraft, operational fixes,
and HMK. They exhausted their time limits before emitting final summaries, so
their partial traces were used only as evidence and never treated as a green
gate. The direct-history gate above was recomputed mechanically against the
current candidate.

## Rebirth reconciliation

The daimonmatrix working tree was copied without modifying its checkout and
preserved exactly as commit `02fba61ba2590c519b60ee9a3c9c2e25cf3b4cf2`:
240 tracked paths plus 103 untracked paths, with zero content, executable-bit,
or presence differences and zero safety findings.

It was reconciled with current `compaii-state` as two-parent merge
`23700efec32a15c2a7b155e1ff167bcf71929e5b` and published for review at
https://github.com/nicoechaniz/compaii-state/pull/30. The generation contains
3,627 artifact records with exact index/manifest binding; all four HMK archives
pass checksum, SQLite quick-check, and foreign-key checks; 52 core tests and
the Wiki protocol integration pass. Six HMK-native records unique to the
remote snapshot remain recoverable from the second parent but were not
activated automatically because HMK forbids blind database merges and several
encode superseded policy.

## Publication and deployment plan

1. Finish and audit the complete Python suite.
2. Commit this report and `FORK_CHANGELOG.md`, then update canonical feature branches from their verified sync branches while retaining the `backup/pre-upstream-sync-*` refs.
3. Advance and push `main` only after the final candidate receipt is green.
4. Update the local deploy and verify CLI/gateway runtime version and health.
5. Update daimonmatrix with a fast-forward/fetch-based deployment; verify the gateway and preserve its independent staged `compaii-state` work.
6. Let `compaii-state` PR #30 complete review/checks; it does not block the Hermes code deployment because it changes only portable rebirth state.
7. Run the final rebirth sync only after the Hermes fork and both deployments are verified.

## Known risks

- The upstream delta is unusually large; focused lane tests are necessary but not sufficient, so broad final validation remains mandatory.

- Completion ownership now aligns with the async session store and lineage abstractions; one high-contention run hit the per-file timeout after partial progress, while subsequent isolated and four-worker gates passed 60/60 and 115/115. This is recorded as load sensitivity, not hidden as an assertion failure.
- Daimonmatrix still contains its original staged and untracked
  `compaii-state` checkout. It was treated as read-only source material and
  remains untouched; PR #30 carries the independently verified reconciliation.
