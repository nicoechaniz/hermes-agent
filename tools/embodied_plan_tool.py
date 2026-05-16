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
# Policy integration (optional — graceful degradation if gemma_policy missing)
# ---------------------------------------------------------------------------

try:
    from tools.gemma_policy import GemmaPolicy

    _GemmaPolicy = GemmaPolicy
except Exception as _import_err:  # pragma: no cover
    logger.debug("gemma_policy not available: %s", _import_err)
    _GemmaPolicy = None


def _policy_mode_default() -> str:
    """Platform-aware default: 'auto' when DaemonCraft bot context is detected,
    'raw' for CLI and other platforms (backward compatibility)."""
    return "auto" if os.environ.get("BOT_API_URL") else "raw"


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _post_intent(body: dict[str, Any]) -> dict[str, Any]:
    """POST *body* to the embodied-service ``/intent`` endpoint and return the
    parsed JSON response as a Python dict.

    All network and decode errors are caught and returned as
    ``{"ok": False, "error": {...}}`` so callers never raise."""
    url = f"{_service_url()}/intent"
    timeout = _timeout()
    try:
        resp = httpx.post(url, json=body, timeout=timeout)
    except httpx.TimeoutException:
        return {
            "ok": False,
            "error": {
                "error_type": "embodied_service_timeout",
                "details": f"timed out after {timeout}s waiting for {url}",
            },
        }
    except httpx.RequestError as exc:
        return {
            "ok": False,
            "error": {
                "error_type": "embodied_service_unreachable",
                "details": f"{type(exc).__name__}: {exc}",
            },
        }
    try:
        return resp.json()
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": {
                "error_type": "embodied_service_bad_response",
                "details": f"non-JSON body (status {resp.status_code}): {resp.text[:200]}",
            },
        }


# ---------------------------------------------------------------------------
# Raw passthrough handler (debugging escape hatch)
# ---------------------------------------------------------------------------

def _raw_handler(args: dict[str, Any]) -> str:
    """Forward the intent verbatim to the embodied service."""
    intent = args.get("intent", "")
    if not intent or not isinstance(intent, str):
        return json.dumps({
            "ok": False,
            "error": {
                "error_type": "missing_intent",
                "details": "embodied_plan requires a non-empty 'intent' string",
            },
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

    bot_api_url = args.get("bot_api_url") or os.environ.get("BOT_API_URL")
    if bot_api_url:
        body["bot_api_url"] = bot_api_url

    result = _post_intent(body)
    
    # Tier 2a recovery: deterministic synthesis for spatial failures.
    # If the service failed with a known spatial error (target_occupied,
    # no_solid_neighbor, bot_in_target, block_not_in_inventory), try
    # a simple coordinate offset or material substitute before giving up.
    failed = [r for r in (result.get("execution_results") or []) if not r.get("ok")]
    if failed and not body.get("previous_error"):
        first_failure = failed[0]
        error_type = first_failure.get("error_type", "")
        retry_body = dict(body)
        if error_type in ("target_occupied", "bot_in_target", "no_solid_neighbor"):
            # Simple spatial recovery: add a hint to offset coordinates.
            retry_body["previous_error"] = {
                "tool": first_failure.get("tool"),
                "error_type": error_type,
                "details": first_failure.get("details", ""),
            }
            retry_body["_recovery_hint"] = "spatial_retry_adjacent"
            result = _post_intent(retry_body)
    
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Policy-wrapped handler
# ---------------------------------------------------------------------------

def _policy_handler(args: dict[str, Any]) -> str:
    """Run the GemmaPolicy L2→L3→L5→(L1+L4) pipeline before calling the
    embodied service.  Sub-intents are POSTed sequentially to ``/intent``;
    ``previous_error`` is threaded between calls."""
    intent = args.get("intent", "")
    if not intent or not isinstance(intent, str):
        return json.dumps({
            "ok": False,
            "error": {
                "error_type": "missing_intent",
                "details": "embodied_plan requires a non-empty 'intent' string",
            },
        })

    if _GemmaPolicy is None:
        logger.warning("policy_mode=auto but gemma_policy unavailable; falling back to raw")
        return _raw_handler(args)

    policy = _GemmaPolicy()
    policy_result = policy.execute(intent)

    # L2 / L3 cut — handled upstream, do NOT call /intent
    if policy_result["policy_handled"]:
        return json.dumps({
            "ok": True,
            "outcome": "policy_handled_upstream",
            "policy_handled": True,
            "policy_layer": policy_result["policy_layer"],
            "policy_reason": policy_result["policy_reason"],
            "plan": None,
            "execution_results": [],
            "mitigation": {
                "policy_layer": policy_result["policy_layer"],
                "policy_reason": policy_result["policy_reason"],
                "intent_original": intent,
                "sub_intents_count": 0,
                "normalized_chain": [],
                "category_chain": [],
                "allowed_tools_chain": [],
                "sub_intent_outcomes": [],
            },
        })

    sub_intents = policy_result["sub_intents"]
    categories = policy_result["categories"]
    allowed_tools_chain = policy_result["allowed_tools"]

    all_execution_results: list[dict] = []
    aggregated_plan = None
    previous_error = None
    sub_intent_outcomes: list[str] = []

    for idx, (sub_intent, category, policy_tools) in enumerate(zip(
        sub_intents, categories, allowed_tools_chain
    )):
        body: dict[str, Any] = {"intent": sub_intent}
        for k in ("autonomy_level", "guardian_constraints", "deadline_seconds"):
            if k in args and args[k] is not None:
                body[k] = args[k]

        # Use caller's allowed_tools if explicitly provided, else policy-narrowed
        if "allowed_tools" in args and args["allowed_tools"] is not None:
            body["allowed_tools"] = args["allowed_tools"]
        else:
            body["allowed_tools"] = policy_tools

        if previous_error is not None:
            body["previous_error"] = previous_error

        bot_api_url = args.get("bot_api_url") or os.environ.get("BOT_API_URL")
        if bot_api_url:
            body["bot_api_url"] = bot_api_url

        # Pass verification metadata so the embodied service can log the full pipeline
        body["_verification_meta"] = {
            "intent_original": intent,
            "policy_layer": "decomposition" if len(sub_intents) > 1 else (policy_result.get("policy_layer") or "normalization"),
            "category": category,
            "sub_intent_index": idx,
            "sub_intents_total": len(sub_intents),
        }

        result = _post_intent(body)

        if result.get("ok"):
            if result.get("execution_results"):
                all_execution_results.extend(result["execution_results"])
            if result.get("plan") and aggregated_plan is None:
                aggregated_plan = result["plan"]
            previous_error = None
            sub_intent_outcomes.append("embodied_succeeded")
        else:
            previous_error = {
                "tool": result.get("error", {}).get("error_type", "unknown"),
                "error_type": result.get("error", {}).get("error_type", "other"),
                "details": result.get("error", {}).get("details", "unknown error"),
            }
            all_execution_results.append({
                "ok": False,
                "tool": "embodied_plan",
                "error_type": previous_error["error_type"],
                "details": previous_error["details"],
            })
            sub_intent_outcomes.append("embodied_failed")

    return json.dumps({
        "ok": True,
        "outcome": "embodied_ready",
        "policy_handled": False,
        "plan": aggregated_plan,
        "execution_results": all_execution_results,
        "mitigation": {
            "policy_layer": "decomposition" if len(sub_intents) > 1 else "normalization",
            "policy_reason": f"processed into {len(sub_intents)} sub-intent(s)",
            "intent_original": intent,
            "sub_intents_count": len(sub_intents),
            "normalized_chain": sub_intents,
            "category_chain": categories,
            "allowed_tools_chain": allowed_tools_chain,
            "sub_intent_outcomes": sub_intent_outcomes,
        },
    })


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
            "- Tasks outside body orchestration (writing code, web research, etc.)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": (
                        "Natural-language description of what the bot should do. "
                        "Be CONCRETE. Include 'what', 'where', and 'why' when "
                        "relevant. Examples: 'Help the player gather 12 oak logs "
                        "before night.' / 'Go to coordinates [120, 64, -33] but "
                        "avoid the ravine.' / 'Build a small shelter using planks "
                        "from the inventory.' Ambiguous intents are okay — the "
                        "embodied service will respond with an ask_clarification "
                        "tool_call which surfaces a question to ask the user."
                    ),
                },
                "policy_mode": {
                    "type": "string",
                    "enum": ["auto", "raw"],
                    "description": (
                        "Policy wrapping mode. 'auto' enables the GemmaPolicy "
                        "layer (scope filter, ambiguity detection, decomposition, "
                        "normalization, tool narrowing). 'raw' forwards the intent "
                        "verbatim for debugging. Default is platform-aware: 'auto' "
                        "when BOT_API_URL is set (DaemonCraft gateway), 'raw' "
                        "otherwise (CLI backward compatibility)."
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
                        "details: <string>}."
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
# Main handler
# ---------------------------------------------------------------------------

def _handler(args: dict[str, Any] | None = None, **_kw: Any) -> str:
    args = args or {}
    policy_mode = args.get("policy_mode")
    if policy_mode is None:
        policy_mode = _policy_mode_default()

    if policy_mode == "raw":
        return _raw_handler(args)
    return _policy_handler(args)


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
