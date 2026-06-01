# Changelog — nicoechaniz/hermes-agent fork

> **Provider note:** Profile configs reference DeepSeek, Kimi, and MiniMax providers because that's our stack. Team members using different providers (OpenRouter, Anthropic, Nous, etc.) should adapt `model.provider`, `model.default`, and `model.base_url` in each profile's `config.yaml`. API keys go in each profile's `.env` (or symlink to shared `.env`). The `max_turns` and `reasoning_effort` values are provider-agnostic and should work across backends.

## 2026-05-16 — mc_bit Tool Fix

### Synchronous mc_bit Handler

The `mc_bit` Hermes tool was broken since deploy: `async def _handler(...)` returned
a coroutine object, which surfaced as `object of type 'coroutine' has no len()` in
live tool calls. Replaced with a synchronous `httpx.get` wrapper.

**Branch:** `feat/daemoncraft`
**Commit:** `a16bc0c5b fix(daemoncraft): make mc_bit tool synchronous`

Tests: `scripts/run_tests.sh tests/tools/test_mc_bit_tool.py -q --tb=short` → 3 passed.

### mBit Context in Embodied Service (DaemonCraft side)

See DaemonCraft CHANGELOG for the full mBit context integration. The hermes-agent
side only needed the mc_bit tool fix above — the world_state injection lives in
the embodied service composer on the DaemonCraft repo.

## 2026-05-09 — Multi-Agent Coding Roster + Kanban Hardening

### New Profiles
- **riqui** (deepseek-v4-flash, max_turns=30, reasoning=minimal): Surgical coding Kanban worker. Fixed protocol violation (was max_turns=15 + reasoning=none → iteration exhaustion before kanban_complete).
- **miki** (kimi-k2.6, kimi-coding OAuth via ~/.kimi/, max_turns=30, reasoning=high): Coding agent. Tested working.
- **maxi** (MiniMax-M2.7, minimax provider, Anthropic endpoint, max_turns=30, reasoning=high): Coding agent. Config created but blocked by CLI api_mode detection bug (404 — hardcoded chat_completions vs anthropic_messages).
- **claudio** (planned): Proxy profile → Claude Code CLI
- **gepeto** (planned): Proxy profile → Codex CLI

### Kanban System
- **Protocol violation root cause:** max_turns too low + reasoning=none on weak models → iteration exhaustion → model writes kanban_complete as text (not function call) → clean exit without transition → effective_limit=1 → auto-blocked
- **Fix:** max_turns ≥ 25 + reasoning ≥ minimal for all Kanban coding workers
- **Self-spawn guard:** Dispatcher DOES spawn tasks assigned to gateway's own profile (compaii). Tasks must stay in `todo`/`triage` until manually claimed.
- **Smoke test pattern:** t_4631001e (17s, riqui) validated the fix

### RTK Plugin
- **FIXED** by Riqui (t_ad89b059): Replaced corrupted `rtk_hermes/__init__.py` (circular self-import) with 332-line source from GitHub
- Binary symlinked for gateway PATH
- Plugin loads cleanly on gateway restart (no WARNING)

### Memory Infrastructure
- HMK chapters 9-11 seeded: dispatcher guard, profile roster, maxi api_mode debug
- Project MEMORY.md updated with full profile roster and dispatcher critical rule

### Known Issues
- **maxi:** `hermes -p maxi chat` returns 404. CLI hardcodes api_mode=chat_completions. Provider transport=anthropic_messages is ignored. curl confirms endpoint works.
- **Upstream:** ~90 commits behind (v2026.5.7+), needs sync

## 2026-05-08 — Upstream Sync v2026.5.7

- Full rebase onto upstream/main (993 commits, 7 conflicts resolved)
- All 10 custom features preserved
- Gateway split: hermes-gateway.service (CompAII) + hermes-gateway@steve.service
- RTK plugin installed (but init.py was corrupted — fixed May 9)
- Kanban migration from Lattice (64+ tasks)
- CompAII hardening: max_turns=40, reasoning=high, compression=0.50
- HMK memory kit: library.db seeded, engram_pack prefetch

## Custom Features (all branches merged into main)

1. feat/kimi-oauth-clean — Kimi OAuth refresh, header fixes
2. feat/altermundi-tui — TUI scrollbar, max lines config
3. feat/altermundi-cli — Ctrl+C priority config
4. feat/minimax-defaults — MiniMax provider defaults
5. feat/compression-config-reboot — Configurable compression protect_first_n
6. feat/dc-112-daemoncraft-gateway — Gateway adapter wiring, tool_choice propagation
7. DC-99 — Profile system prompt override per platform
8. DC-123 — TTS fixes + wake-up logging, CycleDetector
9. DC-132 — Contextvars-based endpoint resolution, turn metrics
10. DC-134 — Configurable turn wall-clock timeout + per-profile max_iterations
