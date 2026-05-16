# Grok Web Bridge as Hermes Provider — Status

**Date:** 2026-05-16  
**Branch:** `feat/daemoncraft` (proxy code lives in `~/.hermes/proxies/grok-bridge/`)  
**Assignee:** compaii

---

## What Works

We have a working **paste-loop** prototype:

1. User asks something that requires system access (e.g., "what user am I running as?")
2. Proxy sends message to Grok web with a "developer automation tool" framing
3. Grok responds with `CMD: whoami` (shell snippet, not JSON)
4. Proxy detects `CMD:` prefix, executes via `subprocess.run` locally
5. Proxy sends the raw output back to Grok as a follow-up message
6. Grok responds naturally: "You are running as user nicolas."

**Grok NEVER executes in its remote sandbox.** It only generates the command text.

---

## Key Discovery

Grok web **rejects** explicit instructions to generate JSON tool calls (detects as jailbreak).  
Grok web **accepts** framing as a "developer automation tool" that generates "shell snippets."

This is the working system prompt:

```
You are assisting with a developer automation tool. Your role is to
generate shell snippets or answer questions.

When the user wants you to check or do something on the system,
generate the exact shell command as a snippet. Prefix it with CMD:
and write only the command.

When you have all the information you need, respond to the user normally.
```

---

## Files

| File | Purpose |
|------|---------|
| `~/.hermes/proxies/grok-bridge/proxy/proxy_kimi_only.py` | Source of truth for the proxy |
| `~/.hermes/proxies/grok-bridge/proxy/proxy.py` | Copy used by systemd service |
| `~/.hermes/proxies/grok-bridge/proxy/grok_system_prompt.txt` | Developer automation tool prompt |
| `~/.hermes/proxies/grok-bridge/proxy/tool_executor.py` | JSON tool call fallback parser |

---

## Open Questions / Next Steps

1. **Grok Skills integration**
   Grok web has native "Skills" (Create skill). We should create a permanent Skill
   with the developer automation prompt so we don't need to inject it on every request.
   The user captured the Skills menu; next session we should either automate creation
   via Kimi Bridge or have the user create it manually.

2. **Tool definitions from Hermes -> Grok**
   Hermes sends `tools` in the OpenAI request body. The proxy currently ignores them.
   We need to translate Hermes tool definitions into Grok's snippet format so Grok
   knows what commands it can generate.

3. **Hermes provider config**
   Need to add `grok_web` provider entry in `~/.hermes/config.yaml` pointing to
   `http://127.0.0.1:8802`.

4. **Streaming support**
   The CMD: loop is only tested with `stream=false`. Need to verify it works with
   Hermes' default streaming mode.

5. **Multi-turn conversation state**
   Grok web maintains conversation state in the browser. The proxy reuses the same
   grok.com tab. Need to ensure this doesn't leak context between unrelated Hermes
   sessions.

6. **Repo hygiene**
   The proxy code lives in `~/.hermes/proxies/` (outside the hermes-agent repo).
   Need to decide if this should be tracked in the fork or kept separate.

---

## Infrastructure

| Service | Endpoint | Status |
|---------|----------|--------|
| Kimi Web Bridge | `http://127.0.0.1:10086` | Running |
| Grok Bridge Proxy | `http://127.0.0.1:8802/v1/chat/completions` | Running (systemd) |
| Grok web | `https://grok.com` | Active tab via Kimi Bridge |

---

## Related Sessions

- User has SuperGrok plan (approved for this automation use case per his
  conversation with xAI).
- Previous attempts with JSON tool calls and Chrome extension were abandoned
  in favor of the Kimi Web Bridge + paste loop approach.
