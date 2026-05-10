#!/usr/bin/env python3
"""embodied_plan — single-tool body orchestration delegate.

The Hermes-side counterpart to the DaemonCraft embodied service v1.

Hermes' cloud LLM (Kimi/MiniMax/etc.) calls this **one tool** when it
needs the body to do something. The embodied service handles:

  1. Reading world_state from bot/server.js
  2. Filtering allowed_tools by executor_supported
  3. Composing a canonical Gemma-Andy v2 payload
  4. Calling Ollama (gemma-andy:e4b-v2-2-3-q8_0)
  5. Parsing the response (with <think> strip + bracket fallback)
  6. Dispatching each tool_call to bot/server.js
  7. Returning the assembled {plan, execution_results}

Hermes never has to know about the granular Mineflayer mc_* tools — that
is Gemma-Andy's job. Path B canonical per team architectural decision
2026-05-08 (see vault/concepts/gemma-andy-embodied-service.md and
vault/epics/E002-body-protocol-wireup.md).

Environment:
    EMBODIED_SERVICE_URL  Base URL of the embodied service
                          (default: http://localhost:7790)
    EMBODIED_PLAN_TIMEOUT Per-request timeout in seconds
                          (default: 60 — Ollama + dispatch can be slow)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from tools.registry import registry

logger = logging.getLogger(__name__)


def _service_url() -> str:
    return os.environ.get("EMBODIED_SERVICE_URL", "http://localhost:7790").rstrip("/")


def _timeout() -> float:
    try:
        return float(os.environ.get("EMBODIED_PLAN_TIMEOUT", "60"))
    except ValueError:
        return 60.0


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

EMBODIED_PLAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "embodied_plan",
        "description": (
            "Delegate a body task in Minecraft to the embodied service "
            "(Gemma-Andy via Ollama). Use this when the user wants the "
            "agent's Minecraft character to DO something — gather, build, "
            "fight, navigate, craft, etc. The service handles world-state "
            "perception, tool selection, and execution against bot/server.js. "
            "You only describe the high-level intent in natural language. "
            "DO NOT use granular mc_* tools when this tool is available — "
            "this one collapses what would be 5-15 LLM rounds into a single "
            "delegation backed by a fine-tuned local model.\n\n"
            "USE WHEN:\n"
            "- The user asks the bot to do something physical in Minecraft\n"
            "- A multi-step body task (gather → craft → place)\n"
            "- A movement / navigation request\n"
            "- A combat / defensive action\n\n"
            "NOT FOR:\n"
            "- Conversation, narrative, education (handle yourself)\n"
            "- Reading/explaining game state to the user (handle yourself)\n"
            "- Tasks outside body orchestration (writing code, web research, etc.)\n\n"
            "INTENT COMPOSITION RULES — these are validated by primitives_lab\n"
            "experiments 001-007 against gemma-andy:e4b-v2-2-3-q8_0. Following\n"
            "them is the difference between a task succeeding and the body\n"
            "model emitting an empty plan or the wrong tool:\n\n"
            "1. ENGLISH IMPERATIVE ALWAYS. Compose the intent in English\n"
            "   imperative form regardless of the user's surface language.\n"
            "   Spanish conversational ('dame X', 'seguime') makes the model\n"
            "   pick the wrong tool semantics. The model is a body, not a\n"
            "   conversation partner.\n\n"
            "2. PLACEMENT INTENTS NEED EXPLICIT NON-BOT COORDINATES. When the\n"
            "   intent is to place block(s), supply each (x, y, z) explicitly\n"
            "   and ENSURE no coordinate equals the bot's current position.\n"
            "   Phrases like 'stack upward', 'build a wall here', 'place at\n"
            "   your spot' make the body model pick the bot's own [x, y, z],\n"
            "   which fails with bot_action_failed (can't place a block in\n"
            "   the space the bot occupies). Read bot_position from the\n"
            "   most recent world snapshot, then compose target coords that\n"
            "   are adjacent. Example for a 4-block vertical wall starting\n"
            "   one block in front of the player: 'Place 4 oak_planks at\n"
            "   coordinates (X, Y, Z), (X, Y+1, Z), (X, Y+2, Z), (X, Y+3, Z)'\n"
            "   with the actual integer values substituted in.\n\n"
            "3. MULTI-STEP INTENTS NEED NUMBERED STAGES. The body model only\n"
            "   produces a true gather→craft→place plan when the intent\n"
            "   enumerates stages: 'Step 1: scan for X. Step 2: mine N X.\n"
            "   Step 3: craft into Y. Step 4: place at <coords>.' Free-form\n"
            "   prose ('build a wall using wood from nearby trees, then...')\n"
            "   collapses to empty plans. If you need >2 stages, enumerate.\n\n"
            "4. DON'T DELEGATE CONDITIONALS. The body model does not honor\n"
            "   if/then/else against world_state. NEVER write 'if you have X\n"
            "   then Y else Z'. Read world state first (call this tool with\n"
            "   a get_inventory-style intent OR consult prior tool results),\n"
            "   decide the branch yourself, then issue an unconditional\n"
            "   imperative.\n\n"
            "5. PLAYER-AS-TARGET INTENTS NEED EXPLICIT USERNAME. 'Toss N X to\n"
            "   the player named <username>' / 'Follow the player named\n"
            "   <username>' / 'Stand next to player <username>' all hit 100%\n"
            "   success. Pronoun forms ('come to me', 'follow me') do not."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": (
                        "Natural-language description of what the bot should do. "
                        "Compose in English imperative form (rule 1 in the parent "
                        "tool description). Be CONCRETE — include 'what', 'where' "
                        "(exact coordinates when placement or movement is "
                        "involved), and 'why' when relevant.\n\n"
                        "Good examples:\n"
                        "- 'Mine 4 oak_log from the tree at (17, 68, 30), then "
                        "  return to coordinates (5, 65, 38).'\n"
                        "- 'Place 4 oak_planks at coordinates (5, 65, 37), "
                        "  (5, 66, 37), (5, 67, 37), (5, 68, 37) — a vertical "
                        "  pillar starting one block east of the player.'\n"
                        "- 'Toss 16 oak_planks to the player named Fede3043.'\n"
                        "- 'Follow the player named Fede3043 wherever they go.'\n\n"
                        "Bad examples that the body model misinterprets:\n"
                        "- 'Build a tall wall' → no coords, model picks bot's own\n"
                        "  position and fails with bot_action_failed.\n"
                        "- 'Stack oak_planks upward until materials run out' → "
                        "  same problem; the model has no implicit notion of an\n"
                        "  adjacent build face.\n"
                        "- 'Dame 2 oak_planks' (Spanish 'give') → model crafts\n"
                        "  instead of tossing.\n"
                        "- 'If you have planks then build, otherwise gather' → "
                        "  model emits one branch regardless of inventory.\n\n"
                        "Ambiguous intents are okay only when the ambiguity is "
                        "about the user's preference (not about geometry or "
                        "available materials). The embodied service responds "
                        "with an ask_clarification tool_call which surfaces a "
                        "question to ask the user."
                    ),
                },
                "autonomy_level": {
                    "type": "integer",
                    "description": (
                        "Guardian autonomy. 0=observer / 1=assistant / "
                        "2=supervised builder (DEFAULT, safe for kids+adults) / "
                        "3=autonomous companion / 4=advanced operator (risky)."
                    ),
                    "default": 2,
                },
                "allowed_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional override of the tool subset Gemma-Andy may use. "
                        "Names must be canonical v2 tool names. When omitted, the "
                        "service uses its default safe set. The service further "
                        "filters by executor_supported, so passing tools the bot "
                        "server doesn't implement is harmless — they're dropped."
                    ),
                },
                "guardian_constraints": {
                    "type": "object",
                    "description": (
                        "Optional override of the safety constraints. Recognized "
                        "fields include no_tnt, no_protected_zone_edit, "
                        "protected_zone_owner, plus any no_<verb> bool flags. "
                        "Defaults are sane (no_tnt=true, no_protected_zone_edit=true)."
                    ),
                },
                "previous_error": {
                    "type": "object",
                    "description": (
                        "Optional. Pass when the previous embodied_plan call's "
                        "execution_results contained a failure and you want "
                        "Gemma-Andy to compose a recovery plan. Shape: "
                        "{tool: <name>, error_type: 'stuck'|'no_path'|'tool_timeout'|"
                        "'hazard_detected'|'missing_material'|'other', "
                        "details: <string>}.\n\n"
                        "Implementation note: Gemma-Andy v2-2-3 currently ignores "
                        "the structured previous_error field 100% of the time "
                        "(primitives_lab experiment 003+007). The Hermes-side "
                        "handler works around this by prepending a narrative "
                        "reformulation of previous_error to the intent text, "
                        "which the model honors 100% of the time (experiment "
                        "007 in_intent_directive at n=10). The structured field "
                        "is still forwarded for forward-compat with a future "
                        "model retrain."
                    ),
                },
                "deadline_seconds": {
                    "type": "integer",
                    "description": (
                        "Wall-clock budget for the WHOLE call (compose + Ollama + "
                        "dispatch). Default 30. Set higher for long execution "
                        "sequences."
                    ),
                    "default": 30,
                },
            },
            "required": ["intent"],
        },
    },
}


# ---------------------------------------------------------------------------
# Replan composition
# ---------------------------------------------------------------------------


def _compose_replan_intent(intent: str, prev_err: dict[str, Any]) -> str:
    """Embed `previous_error` as a recovery directive trailing the intent.

    Why: `gemma-andy:e4b-v2-2-3-q8_0` ignores the structured `previous_error`
    field 100% of the time (verified by primitives_lab experiment 003 + 007,
    n=10 each). The same failure context embedded in the intent text shifts
    the plan 100% of the time (experiment 007 `in_intent_directive`).

    Composition shape — the order is load-bearing. Put the original intent
    first, then the failure narration, then the recovery directive **last**.
    The model has a strong last-instruction bias; in our first iteration the
    narrative was prepended and the original (uncorrected) intent landed at
    the tail — Andy re-emitted the failing tool. Trailing recovery shifts
    the plan as designed in experiment 007.

    This is a Hermes-side workaround — once Andy is retrained to honor the
    structured field, the rewrite can be removed without breaking anything.
    """
    tool = prev_err.get("tool") or "the previous action"
    error_type = prev_err.get("error_type") or "failed"
    details = (prev_err.get("details") or "").strip()
    # Past-tense framing: the original (failing) intent never appears as a
    # live imperative — only as something we *tried* and that failed. This
    # matches experiment 007's winning `in_intent_narrative` shape (9/10).
    # The closing directive is what the model picks up as the active task.
    parts = [
        f"We tried to {intent}, but {tool} failed with error_type={error_type}.",
    ]
    if details:
        parts.append(details)
    parts.append(
        "Compose a new plan that achieves the same outcome using the "
        "actually-available state. Do not re-emit the failing action."
    )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _handler(args: dict[str, Any], **_kw: Any) -> str:
    intent = (args or {}).get("intent", "")
    if not intent or not isinstance(intent, str):
        return json.dumps({
            "ok": False,
            "error": {"error_type": "missing_intent",
                      "details": "embodied_plan requires a non-empty 'intent' string"},
        })

    body: dict[str, Any] = {"intent": intent}
    for k in (
        "autonomy_level",
        "allowed_tools",
        "guardian_constraints",
        "previous_error",
        "deadline_seconds",
    ):
        if k in args and args[k] is not None:
            body[k] = args[k]

    # Multi-bot: include bot_api_url so the embodied service dispatches
    # to the correct bot/server.js instance. Read from the agent's own
    # environment (each DaemonCraft agent has BOT_API_URL in its systemd
    # unit pointing to its own Mineflayer bot). Also accept explicit
    # override from tool args (for future per-call routing).
    bot_api_url = args.get("bot_api_url") or os.environ.get("BOT_API_URL")
    if bot_api_url:
        body["bot_api_url"] = bot_api_url

    prev_err = body.get("previous_error")
    if isinstance(prev_err, dict) and prev_err:
        body["intent"] = _compose_replan_intent(intent, prev_err)
        # Once we've embedded the failure narrative into the intent, forwarding
        # the structured field is redundant — and worse, the daemoncraft
        # `recovery_naive_retry` mitigation compares only tool names, so it
        # flags `place_block` re-emission as a regression even when the model
        # correctly swapped the block argument. Drop the structured field to
        # avoid that false positive. Re-enable forwarding once Andy is
        # retrained to honor previous_error directly (then this rewrite path
        # can be retired entirely).
        body.pop("previous_error", None)

    url = f"{_service_url()}/intent"
    timeout = _timeout()

    try:
        resp = httpx.post(url, json=body, timeout=timeout)
    except httpx.TimeoutException as exc:
        logger.warning("embodied_plan timed out after %.1fs: %s", timeout, exc)
        return json.dumps({
            "ok": False,
            "error": {
                "error_type": "embodied_service_timeout",
                "details": f"timed out after {timeout}s waiting for {url}",
            },
        })
    except httpx.RequestError as exc:
        logger.warning("embodied_plan request failed: %s", exc)
        return json.dumps({
            "ok": False,
            "error": {
                "error_type": "embodied_service_unreachable",
                "details": f"{type(exc).__name__}: {exc}",
            },
        })

    try:
        result = resp.json()
    except json.JSONDecodeError as exc:
        return json.dumps({
            "ok": False,
            "error": {
                "error_type": "embodied_service_bad_response",
                "details": f"non-JSON body (status {resp.status_code}): {resp.text[:200]}",
            },
        })

    # Auto-recovery (Pipeline 2): if the execution failed with a usable
    # `details` string, retry once with a narrative-recovery intent. This
    # generalises lesson 7 of primitives_lab and prevents the cloud LLM
    # from pivoting to "give materials to player" on the first hiccup.
    #
    # Recurrence guard: skip if the caller already supplied previous_error
    # (cloud LLM-driven recovery — let it own the loop) OR if details is
    # missing (nothing to embed).
    caller_drove_recovery = isinstance(args, dict) and bool(args.get("previous_error"))
    if (not caller_drove_recovery
            and isinstance(result, dict)
            and result.get("ok") is False
            and isinstance(result.get("execution_results"), list)
            and len(result["execution_results"]) > 0):
        first_failure = result["execution_results"][0]
        if (isinstance(first_failure, dict)
                and first_failure.get("ok") is False
                and isinstance(first_failure.get("details"), str)
                and first_failure["details"]):
            recovery_prev_err = {
                "tool": first_failure.get("tool", "unknown"),
                "error_type": first_failure.get("error_type", "other"),
                "details": first_failure["details"],
            }
            # Compose a fresh body with the rewritten intent. Drop
            # previous_error from the wire payload (existing behavior:
            # avoids the daemoncraft `recovery_naive_retry` mitigation
            # false-positive).
            retry_body = dict(body)
            retry_body["intent"] = _compose_replan_intent(intent, recovery_prev_err)
            retry_body.pop("previous_error", None)
            try:
                retry_resp = httpx.post(url, json=retry_body, timeout=timeout)
                retry_result = retry_resp.json()
                if isinstance(retry_result, dict) and retry_result.get("ok") is True:
                    return json.dumps(retry_result)
            except (httpx.TimeoutException, httpx.RequestError, json.JSONDecodeError) as exc:
                logger.warning("embodied_plan auto-retry failed: %s", exc)
                # Fall through to return the original failure.

    # Pass the (possibly retried) service response through verbatim. Hermes'
    # AIAgent gets the full {ok, plan, execution_results, ...} envelope.
    return json.dumps(result)


def _check_service_available() -> bool:
    """Light availability check — does NOT call /health (would block tool
    discovery on a slow service). The check_fn is invoked at toolset
    enumeration time; an unreachable service still lets the tool register
    and produce a clean error at call time. We just verify the URL parses."""
    try:
        url = _service_url()
        return url.startswith("http://") or url.startswith("https://")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

# AST check in tools/registry.py only recognizes `registry.register(...)`
# at module scope, not inside loops or conditionals.
registry.register(
    name="embodied_plan",
    toolset="embodiment",
    schema=EMBODIED_PLAN_SCHEMA,
    handler=_handler,
    check_fn=_check_service_available,
    emoji="🤖",
    description=EMBODIED_PLAN_SCHEMA["function"]["description"],
)
