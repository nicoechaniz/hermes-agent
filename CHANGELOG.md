# Changelog — nicoechaniz/hermes-agent fork

Team-facing summary of changes to our fork. For branch topology and sync commands, see `FORK_WORKFLOW.md`.

> Provider note: profile configs reference DeepSeek, Kimi, and MiniMax providers because that is our stack. Team members using different providers should adapt `model.provider`, `model.default`, and `model.base_url` in each profile's `config.yaml`. API keys go in each profile's `.env` or a symlinked shared `.env`.

## 2026-05-14 — Kimi OAuth architecture consolidation

### Centralized resolution

- `resolve_api_key_provider_credentials()` now automatically tries Kimi CLI OAuth (`~/.kimi/credentials/kimi-code.json`) when no `KIMI_API_KEY` env var is present.
- Removed 33 lines of scattered ad-hoc OAuth fallback code from `runtime_provider.py` (`_try_kimi_oauth_credentials()` and both call sites).
- All consumers of the canonical resolver (runtime provider, auxiliary client, etc.) get OAuth automatically without manual fallback code.

### Branch sync

- Cherry-picked missing Kimi commits from `main` into `feat/kimi`:
  - `_try_refresh_kimi_client_credentials()` in `run_agent.py` (mid-conversation 401 refresh)
  - `kimi_coding_default_headers()` applied consistently across all provider paths
- `feat/kimi` is now a complete clean patch stack with no Kimi gaps vs `main`.
- Merged back to `main`, resolved one minor merge conflict in `agent/auxiliary_client.py`.

### Test fixes

- Fixed `test_resolve_kimi_prefers_cli_oauth_without_api_key`: mock now accepts `**kwargs` to match the new `allow_api_key_fallback=False` call signature.
- Fixed two pre-existing broken tests with mismatched mock values:
  - `test_resolve_runtime_provider_kimi_uses_oauth_chat_mode`
  - `test_resolve_runtime_provider_lmstudio_uses_token_when_present`

### Verification

- All 287 provider/auth tests pass: `scripts/run_tests.sh tests/hermes_cli/test_runtime_provider_resolution.py tests/hermes_cli/test_api_key_providers.py -q`

## 2026-05-13 — Kimi branch consolidation

### Branch workflow

- Canonical Kimi branch is now `feat/kimi`.
- `feat/kimi` is a clean patch stack over `nousmain`.
- Superseded branches removed from `origin`:
  - `feat/kimi-oauth-clean`
  - `feat/kimi-oauth-clean-v3`
  - `fix/kimi-context-length-resolution`
- `main` is the integration branch: `nousmain` plus all active canonical feature branches.
- Runtime deploy target `~/.hermes/hermes-agent` tracks `origin/main`.

### Kimi support in `feat/kimi`

- Kimi CLI OAuth credentials are read from `~/.kimi/credentials/kimi-code.json` when `KIMI_API_KEY` is not set.
- `KIMI_API_KEY` takes precedence over OAuth credentials when present.
- `kimi-coding` supports both endpoints:
  - `https://api.kimi.com/coding` via Anthropic Messages
  - `https://api.kimi.com/coding/v1` via OpenAI Chat Completions
- Kimi CLI-compatible `X-Msh-*` headers and user-agent are applied to Kimi requests.
- Runtime provider resolution falls back to Kimi OAuth instead of silently producing `no-key-required`.
- Auxiliary Kimi calls refresh OAuth credentials on 401.
- Kimi K2.x context lengths are pinned to 262144 where needed and stale OpenRouter underreports are ignored for known providers.

### Verification

- Focused tests: `scripts/run_tests.sh tests/hermes_cli/test_runtime_provider_resolution.py tests/agent/test_model_metadata.py -q --tb=short`
- Kimi OAuth smoke: `hermes chat --provider kimi-coding -m kimi-k2.6 -q 'Say OK only.' -Q --yolo`

## 2026-05-11 — Kimi K2.6 context window bug

### Problem

Hermes rejected `kimi-k2.6` with a 32768-token context window even though the real context is 262144.

### Root cause

1. Provider mapping was incomplete for `kimi` / `moonshot` aliases.
2. Provider-unaware OpenRouter metadata could be consulted before curated Hermes defaults.

### Fix

- Added Kimi aliases to models.dev provider mapping.
- Added explicit Kimi K2.x context-length defaults.
- Kept provider-specific/curated context data ahead of provider-unaware OpenRouter fallback.

## 2026-05-09 — Multi-agent coding roster and Kanban hardening

### Profiles

- `riqui`: DeepSeek v4-flash, surgical Kanban worker.
- `miki`: Kimi K2.6 via `kimi-coding` OAuth.
- `maxi`: MiniMax provider, Anthropic Messages endpoint.
- `compaii`: architecture/research profile.

### Kanban fixes

- Kanban worker tasks require explicit `toolsets` in task specs.
- `kanban review create` flow validated end-to-end.
- Auto-specify triage was added for malformed task specs.
- Dispatcher self-spawn risk remains important: tasks assigned to the gateway's own profile must stay in `todo`/`triage` until intentionally claimed.

## 2026-05-08 — Upstream sync v2026.5.7

- Rebased fork work onto upstream main at the time.
- Preserved local fork features.
- Split gateway service handling between CompAII and profile-specific gateway instances.
