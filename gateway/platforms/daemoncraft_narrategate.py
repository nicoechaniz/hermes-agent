#!/usr/bin/env python3
"""
L4 narrate-gate for DaemonCraft — verify-before-narrate enforcement.

Tracks the most recent mc_* tool result and the most recent body state
(queried via /status on the bot server). On each heartbeat, if the L4's
last assistant message narrates a past-tense action that contradicts
the most recent tool result, logs a warning and exposes a `should_remind`
flag the daemoncraft platform can use to inject a system reminder.

This is Opción B of t_0fa2c6dc (2026-06-02). The SOUL already has
"Verify Before Narrate" (Opción A) but the LLM still confabulates when
the body state is stale. This module provides a runtime check.

Architecture:
- NarrateGateTracker is a stateful observer owned by the DaemonCraft
  adapter. Touched only from the gateway event loop coroutine.
- record_tool_result(action, args, result) is called from the
  transform_tool_result hook for mc_action_result events. It captures
  the last tool's claimed outcome (done, cancelled, stuck, no_progress).
- record_body_status(status) is called from the heartbeat path with
  the latest bot /status response (compact summary: position, health,
  holding, last_action).
- detect_narrate_mismatch(assistant_text) is called after each LLM
  response. It returns None if the narration is consistent with the
  last tool result, or a mismatch dict the platform can use to inject
  a reminder.

Heuristics (deliberately conservative — false positives are worse than
missed mismatches because the LLM can correct itself; bad reminders
undermine trust in the system):

- PAST_TENSE_PATTERNS: regex matching common Minecraft narrations in
  English and Spanish (the bot's two primary languages).
- Tool result categories: "did_move" (mc_move done with non-trivial
  position delta), "did_not_move" (mc_move cancelled/stuck/no_progress),
  "did_place" (mc_build place done), "did_not_place" (place failed).
- Mismatch examples:
  - Tool said "did_not_move" but LLM narrates "I walked/avancé/recorrí"
  - Tool said "did_not_place" but LLM narrates "I built/construí/puse"
  - Tool said success but LLM narrates a different action category

Limitations (known, will refine if mismatch rate is high):
- Only checks simple past-tense patterns. Misses: "I have arrived"
  (present perfect), "Llegué" past tense forms in other conjugations,
  idioms like "I should now be at..." (conditional, not past).
- Does not check spatial claims (e.g. "I am at X,Y,Z" vs actual
  position). Could add later.
- Does not check temporal claims ("just now", "earlier"). Could add.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional


# Past-tense narration patterns. Group them by what they imply:
#   moved_past:  the LLM claims it moved
#   placed_past: the LLM claims it placed/built
#   arrived_past: the LLM claims it arrived somewhere
#   failed_past:  the LLM claims a tool failed
PAST_TENSE_MOVED = re.compile(
    r"\b("
    r"i walked|i moved|i went|i ran|i traveled|i travelled|"
    r"i stepped|i headed|i advanced|i proceeded|i navigated|"
    r"avancé|avanzó|avanzamos|avanzaron|"
    r"caminé|caminó|caminamos|caminaron|"
    r"fui|fue|fuimos|fueron|"
    r"corrí|corrió|corrimos|corrieron|"
    r"recorrí|recorrió|recorrimos|recorrieron|"
    r"me moví|se movió|me desplacé|se desplazó"
    r")\b",
    re.IGNORECASE,
)

PAST_TENSE_PLACED = re.compile(
    r"\b("
    r"i placed|i built|i constructed|i put|i set|"
    r"i dropped|i laid|i stacked|"
    r"colocoqué|coloqué|colocamos|colocaron|"
    r"construí|construyó|construimos|construyeron|"
    r"puse|puso|pusimos|pusieron|"
    r"edifiqué|edificó|edificamos|edificaron|"
    r"hice una?|hizo una? hice un|hizo un"
    r")\b",
    re.IGNORECASE,
)

PAST_TENSE_ARRIVED = re.compile(
    r"\b("
    r"i arrived|i got to|i reached|i made it|"
    r"i'm at|i am at|i'm now at|i am now at|"
    r"llegué|llegó|llegamos|llegaron|"
    r"ya estoy|ya está|ya estamos|ya están"
    r")\b",
    re.IGNORECASE,
)

# Categorical outcomes from server.js typed_result schema (canonical).
# These are the typed outcome strings the server emits. See
# agents/bot/lib/typed_result.js for the canonical list.
# Categorized as: did the action actually accomplish its goal, or did it fail/stall?
MOVEMENT_SUCCESS_OUTCOMES = {"success", "displaced"}  # bot moved (success) or fell (displaced, still moved)
MOVEMENT_FAILURE_OUTCOMES = {"cancelled", "stuck", "no_progress", "preempted", "error", "unknown"}
BUILD_SUCCESS_OUTCOMES = {"success", "displaced"}      # block placed (displaced means fell after, but place succeeded)
BUILD_FAILURE_OUTCOMES = {"cancelled", "stuck", "no_progress", "preempted", "error", "unknown"}


@dataclass
class ToolResultRecord:
    """Compact record of the most recent tool result.

    Built from the server's typed_result fields. outcome and category
    are typed enums (not string-matched from a result blob).
    """
    tool_name: str
    claimed_outcome: str   # "success" | "no_progress" | "cancelled" | "stuck" | "preempted" | "error" | "displaced" | "unknown"
    action_category: str   # "movement" | "build" | "mine" | "interact" | "craft" | "other"
    position_before: Optional[tuple] = None  # (x, y, z) before the tool
    position_after: Optional[tuple] = None   # (x, y, z) after the tool
    timestamp: float = 0.0


@dataclass
class BodyStateSnapshot:
    """Compact record of the most recent body state, captured from /status."""
    position: Optional[tuple] = None  # (x, y, z)
    health: Optional[float] = None
    holding: Optional[str] = None
    last_action: Optional[str] = None  # last_action.action from body_session
    timestamp: float = 0.0


@dataclass
class NarrateMismatch:
    """Returned by detect_narrate_mismatch when narration contradicts reality."""
    kind: str          # "moved_but_didnt" | "placed_but_didnt" | "arrived_but_didnt"
    pattern: str       # the past-tense pattern that matched
    snippet: str       # the offending text snippet
    tool_name: str
    actual_outcome: str
    timestamp: float = 0.0

    def reminder_text(self) -> str:
        """Format a system reminder the gateway can inject."""
        return (
            f"[Verify-Before-Narrate] Last assistant message narrated "
            f"'{self.snippet}' but the most recent tool result ({self.tool_name}) "
            f"had outcome '{self.actual_outcome}'. The narration contradicts "
            f"the verified tool result. Correct the narration in your next turn "
            f"based on the actual outcome. If you are uncertain, call "
            f"mc_perceive(type='status') before narrating."
        )


@dataclass
class NarrateGateTracker:
    """Stateful observer. Touched only from the gateway event loop."""
    last_tool: Optional[ToolResultRecord] = None
    last_body: Optional[BodyStateSnapshot] = None
    _last_reminded_at: float = 0.0
    reminder_cooldown_seconds: float = 30.0

    def record_tool_result(
        self,
        *,
        tool_name: str,
        outcome: str,
        action_category: str = "other",
        position_before: Optional[tuple] = None,
        position_after: Optional[tuple] = None,
        now: float = 0.0,
    ) -> None:
        """Capture a tool result for future narration comparison.

        outcome and action_category are TYPED fields (canonical enums
        from agents/bot/lib/typed_result.js), not strings to parse. The
        server emits these at the top level of every tool result. See
        lib/typed_result.js for the canonical enum values.
        """
        # Validate that the outcome is a known enum value. If the server
        # emits an unknown outcome, treat as "unknown" — never crash.
        if outcome not in _VALID_OUTCOMES:
            outcome = "unknown"
        if action_category not in _VALID_CATEGORIES:
            action_category = "other"
        self.last_tool = ToolResultRecord(
            tool_name=tool_name,
            claimed_outcome=outcome,
            action_category=action_category,
            position_before=position_before,
            position_after=position_after,
            timestamp=now or time.time(),
        )

    def record_body_status(
        self,
        *,
        position: Optional[tuple] = None,
        health: Optional[float] = None,
        holding: Optional[str] = None,
        last_action: Optional[str] = None,
        now: float = 0.0,
    ) -> None:
        """Capture a body state snapshot from the heartbeat."""
        self.last_body = BodyStateSnapshot(
            position=position,
            health=health,
            holding=holding,
            last_action=last_action,
            timestamp=now or time.time(),
        )

    def detect_narrate_mismatch(
        self,
        assistant_text: str,
        now: float = 0.0,
    ) -> Optional[NarrateMismatch]:
        """Return a NarrateMismatch if the LLM's narration contradicts
        the most recent tool result, else None.

        Conservative: only fires on clear past-tense claims that the
        tool result contradicts. Misses implicit claims and conditional
        language — those are not worth the false-positive cost.
        """
        if not assistant_text or not self.last_tool:
            return None

        text = assistant_text.strip()
        if not text:
            return None

        tool = self.last_tool
        ac = tool.action_category

        # Check moved-claim against move-outcome
        if ac == "movement":
            if tool.claimed_outcome in MOVEMENT_FAILURE_OUTCOMES:
                m = PAST_TENSE_MOVED.search(text)
                if m:
                    return NarrateMismatch(
                        kind="moved_but_didnt",
                        pattern=m.group(0),
                        snippet=_snippet(text, m.start()),
                        tool_name=tool.tool_name,
                        actual_outcome=tool.claimed_outcome,
                        timestamp=now or time.time(),
                    )
                m2 = PAST_TENSE_ARRIVED.search(text)
                if m2:
                    return NarrateMismatch(
                        kind="arrived_but_didnt",
                        pattern=m2.group(0),
                        snippet=_snippet(text, m2.start()),
                        tool_name=tool.tool_name,
                        actual_outcome=tool.claimed_outcome,
                        timestamp=now or time.time(),
                    )

        # Check placed-claim against place-outcome
        if ac == "build":
            if tool.claimed_outcome in BUILD_FAILURE_OUTCOMES:
                m = PAST_TENSE_PLACED.search(text)
                if m:
                    return NarrateMismatch(
                        kind="placed_but_didnt",
                        pattern=m.group(0),
                        snippet=_snippet(text, m.start()),
                        tool_name=tool.tool_name,
                        actual_outcome=tool.claimed_outcome,
                        timestamp=now or time.time(),
                    )

        return None

    def should_inject_reminder(
        self,
        mismatch: NarrateMismatch,
        now: float = 0.0,
    ) -> bool:
        """Throttle reminders so we don't spam the LLM."""
        now = now or time.time()
        if (now - self._last_reminded_at) < self.reminder_cooldown_seconds:
            return False
        self._last_reminded_at = now
        return True

    def reset(self) -> None:
        self.last_tool = None
        self.last_body = None


def _classify_outcome_legacy() -> None:
    """Legacy outcome classifier. REMOVED in t_a2c3facb refactor.

    The server now emits a typed `outcome` field directly (see
    agents/bot/lib/typed_result.js). Consumers consume the typed field
    directly via record_tool_result(outcome=...). No string matching.

    This stub remains as a marker — if you see code calling this, it
    predates the typed_result refactor and should be migrated.
    """
    raise NotImplementedError(
        "Legacy _classify_outcome removed. Use the typed 'outcome' field "
        "from agents/bot/lib/typed_result.js. See t_a2c3facb in kanban."
    )


# Canonical outcome + category sets (mirror agents/bot/lib/typed_result.js).
# Validated at record_tool_result() time. Unknown values are coerced to
# "unknown" / "other" respectively.
_VALID_OUTCOMES = {
    "success", "no_progress", "cancelled", "preempted",
    "stuck", "error", "displaced", "unknown",
}
_VALID_CATEGORIES = {"movement", "build", "mine", "interact", "craft", "other"}


def _snippet(text: str, center: int, width: int = 80) -> str:
    """Return a context window around `center` in `text`."""
    start = max(0, center - width // 2)
    end = min(len(text), center + width // 2)
    return text[start:end].replace("\n", " ").strip()


# Action-to-category mapping mirrors agents/bot/lib/typed_result.js. Kept
# locally so this module can compute the category from the raw action
# name without a server round-trip. The two definitions must stay in
# sync; if you change one, change both.
_MOVEMENT_ACTIONS_FOR_NARRATE = frozenset({
    "goto", "gotonear", "goto_near", "follow", "flee", "bg_goto",
    "pathfind", "stop", "come", "navigate",
})
_BUILD_ACTIONS_FOR_NARRATE = frozenset({
    "place", "fill", "build", "interact",
})
_MINE_ACTIONS_FOR_NARRATE = frozenset({
    "dig", "mine", "collect", "tunnel", "spiral",
})
_INTERACT_ACTIONS_FOR_NARRATE = frozenset({
    "chat", "equip", "use", "eat", "drink", "sleep", "attack", "shoot",
    "sneak", "shield", "toss", "pickup", "equip_item",
})
_CRAFT_ACTIONS_FOR_NARRATE = frozenset({
    "craft", "smelt", "brew", "furnace_smelt", "view_craftable",
})


def _narrate_action_category(action: str) -> str:
    """Categorize a tool action for the NarrateGateTracker.

    Returns one of the typed categories: "movement", "build", "mine",
    "interact", "craft", "other". Used by detect_narrate_mismatch to
    pick the right past-tense pattern. Mirrors
    lib/typed_result.js::categoryForAction — keep in sync.
    """
    a = (action or "").lower().split("@")[0].strip()
    if not a:
        return "other"
    if a in _MOVEMENT_ACTIONS_FOR_NARRATE or any(x in a for x in ("goto", "follow", "flee", "path", "navigat")):
        return "movement"
    if a in _MINE_ACTIONS_FOR_NARRATE:
        return "mine"
    if a in _BUILD_ACTIONS_FOR_NARRATE:
        return "build"
    if a in _INTERACT_ACTIONS_FOR_NARRATE:
        return "interact"
    if a in _CRAFT_ACTIONS_FOR_NARRATE:
        return "craft"
    return "other"
