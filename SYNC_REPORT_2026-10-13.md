# Upstream reconciliation — 2026-10-13

Previous integration: `8a13834d6e7e2792ce4b5d8bf17568c8a467acdb`.
Upstream: `63d0b94120027d487c9a5235b9ec64b696af1b26` (11,892 upstream commits).

All seven canonical lanes in `FORK_WORKFLOW.md` are preserved and composed in
its documented order. Historical main-only behavioral fixes were traced and
ported to their owners; archived Kimi OAuth and HMK retirement lanes remain
retired. The preserved worker provenance fence intentionally differs from
upstream worker-to-parent completion forwarding. Upstream retry/admission and
compression contracts are exercised with foreground-owned completion pins;
malformed provenance fails closed unless the human already selected that row.

The native PM replaces the old editable Python environment: Python 3.14.7,
locked dependencies and native source launchers. Existing optional runtime SDKs
are preserved. Operator provider/model choices are preserved.

Preservation evidence lives outside the repository in
`~/Projects/hermes-sync-backups/20261013-pre-sync`: verified all-refs bundle,
all linked-worktree patches/untracked archives, branch tips, eleven native
profile snapshots, service definition and configuration hashes. User `research/`
files remain untracked and intact.

## Validation

The immutable reconnaissance candidate was `515f2b34ff6af9744fd8d1a7417a885e722f98ba`.
The resumed broad gate completed 3,729 files: 41,196 tests passed, 36 failed,
759 skipped; two files timed out. Earlier green files were retained rather than
repeated. All 26 red files were classified and rerun after the focused corrections.
The final clean-environment rerun qualified 185 tests with two remaining failures.
No single green full-suite run is claimed.

The remaining failures are in unchanged upstream implementations:

- `tests/plugins/test_telegram_update_admission.py`: cancellation before entry of
  a nonblocking PTB error callback leaves its admission claim retained. The local
  configured adapter does not register this callback variant. Ordinary PTB
  dispatch, replay and cancellation cases pass.
- `tests/pm/test_cold_runtime_e2e.py`: its reduced SDK checkout lacks `agent/`;
  post-update maintenance fails, and its Python 3.11 probe remains on the old
  interpreter. The real checkout contains `agent/`; this deploy starts on the
  managed Python 3.14 interpreter explicitly. Native PM construction and all
  nine retained runtime extras were qualified, including local Olm construction.

All maintained-lane focused tests passed. TUI: 179 files, 1,633 tests passed,
2 skipped, successful typecheck and lint. New upstream static ratchets flag
retained fork lint/complexity debt; those findings remain explicit and no ratchet
was weakened. Selected undefined-name checks passed.

Disk-backed scratch stalled under host I/O load. The broad reconnaissance used
reversible tmpfs scratch; path-sensitive tests were rerun on native disk paths.
The original scratch directory was restored. Production databases remain on disk.
The test environment used for native launch fixtures excludes an editable project
installation: installing this checkout in the test driver contaminated simulated
checkouts. No product workaround was added for that fixture environment.

## Operational closure

Deployment uses the same seven canonical lanes and primary daily checkout.
`nousmain` is the local mirror; its historical remote is not an update target.
Final source identity, service status, profile/configuration preservation and
rebirth synchronization are captured in the local release receipt alongside the
verified preservation backup. The source report intentionally does not claim a
clean full-suite verdict or an unqualified cold-bootstrap acceptance.
