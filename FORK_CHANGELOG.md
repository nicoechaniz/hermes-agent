# AlterMundi Hermes Fork Changelog

This file records fork-owned integration changes. Upstream Hermes release history remains owned by NousResearch.

## 2026-09-21 — Upstream 0.21.3 synchronization

### Upstream baseline

- Advanced `nousmain` to NousResearch `main` at `0ddeaf9334ff232a01612a51520a043db2cab77b`.
- Rebuilt the integration branch as independent, auditable lanes instead of replaying the former monolithic merge history.
- Preserved pre-sync repository bundles, checksums, lane refs, and concurrent daimonmatrix publications before composing the candidate.

### Preserved fork capabilities

- AlterMundi collective-memory plugin and authenticated plugin-command context.
- Explicit completion ownership, provenance, lineage, anti-replay, and post-compaction boundaries across current async gateway/session modules.
- Classic CLI busy-input interrupt priority and Ink TUI empty-input history navigation.
- Kimi WebBridge browser integration with safe screenshot path handling.
- MiniMax video-generation provider support.
- DaemonCraft per-session routing, turn options, cycle detection, macros, and embodied planning on the modular gateway architecture.
- Kanban durable review orchestration.

### Retired fork work

- Retired the HMK native-memory availability patch because upstream now provides the required live registry behavior.
- Retired the fork-owned Kimi CLI OAuth lane. It remains recoverable in Git history but is no longer an active branch merged over `nousmain`; Kimi WebBridge remains active.
- Discarded stale monolithic `cli.py` and `gateway/run.py` implementations in favor of current upstream mixins and focused modules.

### Reliability and test isolation

- Kept xAI Imagine public media storage explicitly opt-in; missing configuration no longer enables permanent public retention or possible storage billing.
- Removed host/filesystem timing dependence from gateway readiness and atomic configuration-replacement tests.
- Updated idle slash-command fixtures for authenticated plugin provenance.
- Hardened AlterMundi's pre-e-stop provenance path so sparse synthetic events are normalized without weakening the stop boundary.
- Normalized the TUI input-handler test to satisfy the complete workspace lint contract.
- Preserved the former deployed-main fix that removes a duplicate TUI config declaration.
- Normalized all DaemonCraft embodiment tool schemas to the current flat registry contract.
- Replaced hard-coded Kimi WebBridge temporary-output guidance with profile-aware scratch paths.
- Declared MiniMax seed support consistently with its implementation and rebuilt the generated model catalog after lane reconciliation.
- Made configured-platform dependency probes fail closed per platform instead of silently discarding probe exceptions.
- Corrected route-style documentation links and removed same-inode/timestamp assumptions from config-cache tests.

### Verification

- All changed Python test files, lane-specific behavioral suites, compile/version smokes, and the complete Ink TUI workspace check pass.
- The complete sweep plus post-fix rerun receipt is recorded in `SYNC_REPORT_2026-09-21.md`: 52,494 tests passed in the full sweep, followed by 1,324 passed and 0 failed across every affected file on the final code HEAD.

See `SYNC_REPORT_2026-09-21.md` for commit IDs, backup receipts, detailed test counts, known risks, and the deployment sequence.