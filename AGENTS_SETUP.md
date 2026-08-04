# Agent Setup Guide

How to set up and run the multi-agent Kanban coding roster on your machine.

## Prerequisites

- Hermes Agent installed and working (`hermes chat -q "hello"`)
- API keys for your preferred providers in `~/.hermes/.env`
- Git access to this repo

## Quick Start

```bash
# 1. Pull latest
cd ~/Projects/hermes-agent
git pull origin main

# 2. Sync deploy target (if using gateway)
cd ~/.hermes/hermes-agent
git pull local-project main

# 3. Create profiles (one-time)
hermes profile create riqui
hermes profile create miki
hermes profile create maxi

# 4. Copy configs from repo
cp ~/Projects/hermes-agent/profiles/riqui/config.yaml ~/.hermes/profiles/riqui/
cp ~/Projects/hermes-agent/profiles/miki/config.yaml ~/.hermes/profiles/miki/
cp ~/Projects/hermes-agent/profiles/maxi/config.yaml ~/.hermes/profiles/maxi/

# 5. ADAPT PROVIDERS TO YOUR STACK (IMPORTANT)
# Edit each profile's config.yaml:
#   - model.provider: your provider (openrouter, anthropic, nous, etc.)
#   - model.default: your model name
#   - model.base_url: your provider's endpoint (if needed)
#   - model.api_key or symlink .env
$EDITOR ~/.hermes/profiles/riqui/config.yaml
$EDITOR ~/.hermes/profiles/miki/config.yaml
$EDITOR ~/.hermes/profiles/maxi/config.yaml

# 6. Copy SOUL.md files
cp ~/Projects/hermes-agent/profiles/riqui/SOUL.md ~/.hermes/profiles/riqui/
cp ~/Projects/hermes-agent/profiles/miki/SOUL.md ~/.hermes/profiles/miki/
cp ~/Projects/hermes-agent/profiles/maxi/SOUL.md ~/.hermes/profiles/maxi/

# 7. Symlink .env and agent-memory
ln -sf ~/.hermes/.env ~/.hermes/profiles/riqui/.env
ln -sf ~/.hermes/.env ~/.hermes/profiles/miki/.env
ln -sf ~/.hermes/.env ~/.hermes/profiles/maxi/.env
ln -sf ~/.hermes/agent-memory ~/.hermes/profiles/riqui/agent-memory
ln -sf ~/.hermes/agent-memory ~/.hermes/profiles/miki/agent-memory
ln -sf ~/.hermes/agent-memory ~/.hermes/profiles/maxi/agent-memory

# 8. Test each profile
hermes -p riqui chat -q "hello" --quiet
hermes -p miki chat -q "hello" --quiet
hermes -p maxi chat -q "hello" --quiet   # ⚠ known issue: maxi needs api_mode fix
```

## Profile Reference

| Profile | Purpose | Key config | Status |
|---------|---------|-----------|--------|
| riqui | Fast surgical coding | max_turns=30, reasoning=minimal | ✓ Working |
| miki | Deep-thinking coding (Kimi) | max_turns=30, reasoning=high | ✓ Working |
| maxi | Deep-thinking coding (MiniMax) | max_turns=30, reasoning=high, Anthropic endpoint | ⚠ API mode bug |

## Provider Adaptation

The profiles assume our stack (DeepSeek, Kimi OAuth, MiniMax API key). To use different providers:

### Using OpenRouter
```yaml
model:
  default: openai/gpt-5.4  # or anthropic/claude-sonnet-4-6, etc.
  provider: openrouter
```

### Using Anthropic Direct
```yaml
model:
  default: claude-sonnet-4-6-20250514
  provider: anthropic
```

### Using Nous Portal
```yaml
model:
  default: anthropic/claude-sonnet-4-6
  provider: nous
```

The `agent.max_turns` and `agent.reasoning_effort` settings are provider-agnostic.

## Kanban Worker Rules (CRITICAL)

- All coding profiles MUST have `max_turns >= 25` and `reasoning_effort >= minimal`
- Lower values cause protocol violations (exhausted iterations before kanban_complete)
- Kanban dispatcher spawns `hermes -p <profile> --skills kanban-worker chat -q "work kanban task <id>"`
- Workers MUST end with `kanban_complete()` or `kanban_block()` — text-only exit is a violation
- Dispatcher auto-blocks after 1 protocol violation (effective_limit=1)
