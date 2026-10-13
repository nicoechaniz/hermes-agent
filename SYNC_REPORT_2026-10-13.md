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

Focused retained-lane tests and TUI checks passed (179 TUI files, 1,633 passed,
2 skipped). The broad Python reconnaissance and final red-file validation are
recorded in the local verification receipt. No full-suite pass is claimed until
that receipt closes. Static checks flag retained fork lint/complexity debt; those
findings do not establish runtime failures and the ratchets were not weakened.

Disk-backed test scratch suffered reproducible SQLite stalls under host I/O
pressure. Temporary tmpfs scratch allowed reconnaissance; path-sensitive tests
must be rerun with their native disk paths. Production databases remain on disk.

## Deployment

Pending final verification. The old runtime remains active until the candidate
passes its functional gates. See the local verification receipt for final Git
and service identities, configuration preservation and rebirth sync.
