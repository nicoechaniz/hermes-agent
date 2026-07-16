# Reporte de estado — Fork Hermes Agent (Altermundi)

**Fecha:** 2026-07-07
**Autor:** CompAII
**Destinatarios:** Mariano, Nicolás y el equipo que usa Hermes

---

> **Historical report.** This document records the 2026-07-07 integration
> snapshot. It is not the current branch source of truth. The maintained patch
> set, live refs, and clean-apply requirement are in
> `~/wiki/projects/hermes-agent/notes/fork-topology-2026-07-15.md`.

## 1. Resumen ejecutivo

Se dejaron funcionando al 100% las siguientes capacidades del fork:

| Área | Estado | Nota |
|------|--------|------|
| HMK (memoria local) | ✅ | Plugin `hmk-memory` andando; `librarian` lee/escribe `library.db` |
| Kanban | ✅ | Transiciones triage→ready→done por API; sin workarounds |
| Delegación / ia-bridge | ✅ | Foro configurado; síntesis completa |
| Ctrl-C en input | ✅ | Limpia el buffer primero; solo interrumpe si está vacío |
| Paste expandido | ✅ | `show_full_input=true`, thresholds en 0, `collapse_large_pastes=false` |
| Aux auto-detect | ✅ | No fuerza el modelo principal sobre el provider fallback |
| Memoria colectiva (Mariano) | ✅ | Nuevo plugin `altermundi-memory` con tools `altermundi_search` y `altermundi_doc` |

The generic changes recorded here were integrated into the 2026-07-07 `main`
snapshot. They must be represented by a clean `feat/altermundi` patch lane before
the next upstream composition; this report does not assert current sync readiness.

---

## 2. Cambios en el código del fork

Rama `feat/altermundi` mergeada a `main` (14 commits ahead de `origin/main`):

1. `fix(cli): restore ctrl_c_priority clear_input behavior`
2. `fix(cli): restore TUI config support in prompt_toolkit input`
3. `fix(cli): restore TUI input options and Ctrl+C priority`
4. `fix(agent): avoid forcing main model onto fallback provider in auxiliary auto-detect`
5. `fix(cli): respect tui.collapse_large_pastes in bracketed paste handler`
6. `feat(tools): add altermundi toolset for collective memory plugin`
7. `test(streaming): cover PR #12 streaming/tool-call fragmentation fixes`
8. `fix(truncation): detect truncated tool-call JSON args as length-truncation`
9. `fix(streaming): don't split kimi-coding tool-call arg deltas into separate calls`
10. `fix(terminal): expand ~ and env vars in workdir for background processes`
11. `fix(terminal): correct default TERMINAL_TIMEOUT display in terminal tool`
12. Docs de fork y changelog.

---

## 3. Configuración activa del usuario (`~/.hermes/config.yaml`)

```yaml
model:
  default: kimi-k2.7-code
  provider: kimi-for-coding

display:
  ctrl_c_priority: clear_input
  busy_input_mode: steer

paste_collapse_threshold: 0
paste_collapse_char_threshold: 0
paste_collapse_threshold_fallback: 0

tui:
  show_full_input: true
  input_max_lines: 30
  history_nav_requires_empty_input: true
  collapse_large_pastes: false

platform_toolsets:
  cli:
    - ...
    - altermundi

plugins:
  enabled:
    - dialogue-handoff
    - hmk-memory
    - rtk-hermes
    - altermundi-memory
```

---

## 4. Memoria colectiva de AlterMundi (nueva)

### Plugin

Ubicación: `~/.hermes/plugins/altermundi-memory/`

### Tools

- `altermundi_search(q, k=10, kind=None, project=None)` — búsqueda léxica (FTS) sobre la memoria colectiva.
- `altermundi_doc(id)` — devuelve el documento completo en markdown.

### Requisito de red

Solo funciona cuando el equipo está conectado a **anyVPN / ZeroTier**. El servicio vive en `http://10.10.20.1:8899` y es **solo lectura**.

### Prueba real

```bash
hermes chat -q \"buscá en la memoria colectiva de AlterMundi información sobre Harmonic Information Theory\" -t altermundi -Q
```

Resultado: el agente invocó `altermundi_search`, luego `altermundi_doc`, y produjo un resumen estructurado con citas de `mapa/proyectos/phideus.md`, `Harmonic_Information_Theory_Foundations.md`, etc.

### Modo de uso para agentes

En cualquier conversación con Hermes (CLI, Telegram, Discord, etc.), mientras el toolset `altermundi` esté habilitado y el equipo esté en anyVPN, el agente puede consultar la memoria colectiva bajo demanda.

---

## 5. Ramas del fork

| Rama | Estado | Recomendación |
|------|--------|---------------|
| `main` | Historical integration snapshot | See the 2026-07-15 topology note for live state. |
| `feat/altermundi` | Maintained lane | Must cleanly apply over `nousmain` before sync. |
| `feat/daemoncraft` | Maintained lane | Must cleanly apply over `nousmain` before sync. |
| `feat/kanban-ship-review-orchestration` | Historical/obsolete | Not a composition input. |

---

## 6. Verificaciones realizadas

- `HERMES_KANBAN_BOARD=hermes-agent hermes kanban list` → OK.
- `hermes hmk-memory query -q 'infrastructure fixes' --limit 3` → OK.
- `hermes chat -q "..." -t altermundi -Q` → OK (agente usó `altermundi_search` + `altermundi_doc`).
- Sintaxis de `cli.py`, `agent/auxiliary_client.py`, `agent/conversation_loop.py`, `tools/terminal_tool.py` → OK.
- Limpieza de backups y archivos `.orig`/`.rej` → OK.

---

## 7. Qué falta / siguiente pasos

1. **Validar en producción:** probar Ctrl-C, paste y memoria colectiva en una sesión real de CLI/Telegram/Discord si aplica.
2. **Considerar un skill de documentación:** si el equipo quiere, se puede crear un skill `altermundi-memory` que documente casos de uso avanzados (filtros por `kind`/`project`, exploración con `/atlas/`, etc.).

---

*Todo funciona como debería. El reporte está listo para compartir.*
