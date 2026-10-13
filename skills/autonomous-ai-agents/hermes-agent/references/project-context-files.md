# Project Context Files

Hermes injects project-level instructions into the system prompt by reading context files from the working directory. Context-source families use **first match wins**; within the selected `AGENTS.md` family, Hermes merges applicable files from the Git root down to the current working directory.

| File (in priority order) | Discovery | Use when |
|---|---|---|
| `.hermes.md` / `HERMES.md` | Walks parents up to the Git root, stops at Git root | You want Hermes-specific hierarchical project rules |
| `AGENTS.override.md` / `AGENTS.md` / `agents.md` | Inside a Git repo, checks every directory from Git root to cwd; first filename wins per directory, deeper sections appear later and take precedence. Outside a Git repo, checks cwd only. | You want portable agent instructions shared across Hermes, Claude Code, Codex, and other agents |
| `CLAUDE.md` / `claude.md` | Cwd only | Same as AGENTS.md, Claude-flavored |
| `.cursorrules` / `.cursor/rules/*.mdc` | Cwd only | Migrating from Cursor |

`SOUL.md` (in `$HERMES_HOME`) is independent and always loaded when present — it sets the agent's identity, not project rules.

### Pick the right one

- **Use `.hermes.md`** for Hermes-specific rules discovered through its own parent walk.
- **Use `AGENTS.md`** for portable repository rules. Root guidance applies throughout the repository; nested directories may add or override guidance. `AGENTS.override.md` has per-directory precedence without modifying the tracked file.
- **Don't put project rules in `$HERMES_HOME/AGENTS.md` as a cross-project mechanism.** It only participates when that directory is the resolved project path. Use `SOUL.md` for identity and skills for reusable procedures.

### Size and truncation

An explicit positive `context_file_max_chars` setting wins. Otherwise, when the model context length is known, Hermes scales the cap with a 20,000-character floor and 500,000-character ceiling; if it is unknown, the fallback is 20,000 characters. Long content is head + tail truncated with a marker. For an `AGENTS.md` chain, each section is capped and the merged chain is capped again.

### Security

All context files pass through the threat-pattern scanner before reaching the system prompt. Patterns matching prompt injection or promptware are replaced with a `[BLOCKED: ...]` placeholder. This means an `AGENTS.md` containing obvious injection attempts won't reach the model — the scanner blocks the content, not the file, so the rest of the file still loads.

### Disable for one session

`hermes --ignore-rules` skips auto-injection of all project context files (`.hermes.md`, `AGENTS.md`, `CLAUDE.md`, `.cursorrules`) **and** `SOUL.md` identity, plus user config, plugins, and MCP servers. Use it to isolate whether a problem is your setup or Hermes itself.

### Example: a small `.hermes.md`

```markdown
# My Project

Hermes: when working in this repo, follow these rules.

## Build
- Always run `make test` before declaring a change done.
- Use `uv run` for Python, not `pip install`.

## Style
- Prefer `pathlib.Path` over `os.path`.
- No `print()` in production code — use the `logger`.
```

That file at `/home/me/projects/myrepo/.hermes.md` is auto-loaded when Hermes runs in any subdirectory of `/home/me/projects/myrepo`, but not when it runs in `/home/me/other-project`.
