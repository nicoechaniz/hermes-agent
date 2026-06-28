"""L4 stuck detection for DaemonCraft — judge + spatial objective buckets.

Tracks per-heartbeat whether the L4 is making progress on its current
objective. When the bot is stuck on the same target area for N consecutive
heartbeats AND position is not changing AND recent judges report
no_progress/error, returns a pivot reason string that the gateway uses
to interrupt the active L4 turn and queue a system message telling the L4
to radically change its category of action.

Not a replacement for the iteration cap (gateway/run.py) or the prompt
rules (McCompaii SOUL). It's the per-heartbeat enforcement that catches
"same target, no movement, just trying different action aliases" — the
pattern we observed empirically in the 100-iter thrash turn on 2026-06-01.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


# Movement-ish action names from server.js judgeIntents + mc_move aliases.
# These are the "trying to navigate" actions that get aliased into one
# another when the pathfinder gives up: goto / goto_near / follow / flee.
_MOVEMENT_ACTIONS = frozenset({
    "goto", "gotonear", "goto_near", "follow", "flee", "bg_goto",
    "pathfind", "stop", "come", "navigate",
})

_COORD_RE = re.compile(r"(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)")


def _bucket(v, size: int = 5) -> int:
    return int(math.floor(float(v) / size))


def _pos_bucket(pos, size: int = 5) -> Tuple[int, int, int]:
    if not pos:
        return (0, 0, 0)
    return (
        _bucket(pos.get("x", 0), size),
        _bucket(pos.get("y", 0), size),
        _bucket(pos.get("z", 0), size),
    )


def _cells_within(buckets: List[Tuple[int, int, int]], max_diff: int = 1) -> bool:
    """True if all buckets are within max_diff of each other on every axis.

    Used by StuckPivotTracker's Path B to detect thrash that hops between
    adjacent cells (e.g. bucket_size=3 with target rotating between
    (190,40,-110) and (190,40,-111)) without the L4 actually moving the bot.
    """
    if not buckets:
        return True
    xs = [b[0] for b in buckets]
    ys = [b[1] for b in buckets]
    zs = [b[2] for b in buckets]
    return (max(xs) - min(xs) <= max_diff
            and max(ys) - min(ys) <= max_diff
            and max(zs) - min(zs) <= max_diff)


def _parse_target_from_task(task) -> Optional[Tuple[int, int, int]]:
    if not task:
        return None
    action = str(task.get("action") or "")
    m = _COORD_RE.search(action)
    if m:
        try:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
        except (ValueError, IndexError):
            return None
    for key in ("target", "goal", "dest"):
        t = task.get(key)
        if isinstance(t, dict) and "x" in t:
            try:
                return int(t["x"]), int(t["y"]), int(t["z"])
            except (ValueError, KeyError, TypeError):
                return None
    return None


def _parse_target_from_judge(j: dict) -> Optional[Tuple[int, int, int]]:
    intent = j.get("intent")
    if isinstance(intent, dict) and intent.get("x") is not None:
        try:
            return int(intent["x"]), int(intent["y"]), int(intent["z"])
        except (ValueError, KeyError, TypeError):
            return None
    act = str(j.get("action") or "")
    m = _COORD_RE.search(act)
    if m:
        try:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
        except (ValueError, IndexError):
            return None
    return None


def _movement_class(action: str) -> str:
    a = (action or "").lower().split("@")[0].strip()
    if a in _MOVEMENT_ACTIONS or any(x in a for x in ("goto", "follow", "flee", "path")):
        return "movement"
    if a in ("dig", "mine", "collect", "tunnel", "spiral"):
        return "mining"
    if a in ("place", "fill", "build"):
        return "build"
    return a or "other"


@dataclass
class StuckPivotTracker:
    """Detect same-objective / no-progress loops using judges + position.

    Call record_heartbeat() once per heartbeat. Returns a pivot reason
    string when the threshold is met, else None. Stateful across calls
    (tracks the last position bucket and a short objective streak).
    Threading: a single instance is owned by the DaemonCraftAdapter and
    is touched only from the gateway event loop coroutine, no lock needed.

    Detection signatures (refined sub-fix 4, 2026-06-02):
    - bucket_size=3 (was 5) — finer spatial granularity catches thrash in
      3-5 block cells (the goto_near+place loop observed on 2026-06-01 06:36)
    - obj_key includes action class — "place → goto_near → place" with
      different coordinates but same action_class counts as the same
      objective, so a thrash that changes target every call still fires
    - Path B (sub-fix 4): position pinned to the same cell + all stuck +
      at least one bad L4 judge (no_progress/error/preempted/blocked) in
      recent window. Catches the rotate-action-class thrash.
    - cooldown reduced to 60s (was 120s) — repeated thrash on a related
      objective in the same minute should re-fire
    """
    threshold: int = 3
    bucket_size: int = 3
    cooldown_seconds: float = 60.0
    _objective_streak: Deque[Tuple] = field(default_factory=lambda: deque(maxlen=8))
    _last_pos_bucket: Optional[Tuple[int, int, int]] = None
    _fired_objective: Optional[Tuple] = None
    _fired_at: float = 0.0

    def reset_turn(self) -> None:
        """Call at the start of a new L4 turn to clear the streak.

        Doesn't clear _fired_objective / _fired_at — those are cooldown
        state that should survive across heartbeats in the same wall-clock
        window.
        """
        self._objective_streak.clear()
        self._last_pos_bucket = None

    def record_heartbeat(
        self,
        *,
        body_session: dict,
        status: dict,
        pending_judges: Optional[List[dict]] = None,
        now: float = 0.0,
    ) -> Optional[str]:
        """Record a heartbeat and return a pivot reason if threshold met.

        Args:
            body_session: latest body_session dict from the heartbeat payload.
            status: latest status dict from the heartbeat payload.
            pending_judges: list of judge entries pending for the L4 session.
            now: current time.time() (for cooldown check).
        Returns:
            Pivot reason string if threshold met, else None.
        """
        pending_judges = pending_judges or []
        pos = (body_session or {}).get("position") or (status or {}).get("position") or {}
        pos_b = _pos_bucket(pos, self.bucket_size)

        task = (status or {}).get("task") or {}
        target = _parse_target_from_task(task)
        action_class = _movement_class(str(task.get("action") or ""))
        if target is None:
            for j in reversed(pending_judges):
                if j.get("initiator") == "l4_agent":
                    target = _parse_target_from_judge(j)
                    if target:
                        action_class = action_class or _movement_class(str(j.get("action") or ""))
                        break

        if target is not None:
            tgt_b = (
                _bucket(target[0], self.bucket_size),
                _bucket(target[1], self.bucket_size),
                _bucket(target[2], self.bucket_size),
            )
            obj_key: Tuple = ("nav", action_class) + tgt_b
        else:
            obj_key = ("local", action_class) + pos_b

        moved = (
            self._last_pos_bucket is not None
            and self._last_pos_bucket != pos_b
        )
        self._last_pos_bucket = pos_b

        # Recent L4 judges: no_progress / error at same place
        l4 = [j for j in pending_judges if j.get("initiator") == "l4_agent"]
        bad_outcome = False
        if l4:
            last = l4[-1]
            bad_outcome = last.get("outcome") in ("no_progress", "error") or (
                last.get("reason_code") in ("NO_MOVEMENT", "EXCEPTION")
            )
            act = _movement_class(str(last.get("action") or ""))
            if act == "movement" and bad_outcome:
                obj_key = obj_key + ("movement_fail",)

        stuck_here = not moved or bad_outcome
        self._objective_streak.append(obj_key + ("stuck" if stuck_here else "moved",))

        if len(self._objective_streak) < self.threshold:
            return None

        recent = list(self._objective_streak)[-self.threshold:]
        # Strip the trailing 'stuck'/'moved' flag to compare objectives
        base_keys = [r[:-1] if r[-1] in ("stuck", "moved") else r for r in recent]

        # Path A: all base_keys identical AND all stuck. The canonical pattern.
        all_same_obj = len({b for b in base_keys}) == 1
        all_stuck = all(r[-1] == "stuck" for r in recent)
        if not (all_same_obj and all_stuck):
            # Path B (sub-fix 4): position is pinned to a small area AND
            # all heartbeats are stuck AND at least one bad L4 judge in the
            # recent window. Catches goto_near+place+goto_near thrash where
            # the bot barely moves but the action class rotates and the
            # target hops between adjacent buckets.
            spatial_keys = [b[-3:] for b in base_keys]
            all_in_one_cell = _cells_within(spatial_keys, max_diff=1)
            l4_judges_recent = [j for j in pending_judges
                                if j.get("initiator") == "l4_agent"]
            has_bad_judge = any(
                j.get("outcome") in ("no_progress", "error", "preempted", "blocked")
                for j in l4_judges_recent[-self.threshold:]
            )
            if not (all_in_one_cell and all_stuck and has_bad_judge):
                return None

        # Cooldown: don't re-fire on the same objective within cooldown window
        if obj_key == self._fired_objective:
            if now and (now - self._fired_at) < self.cooldown_seconds:
                return None
        # Also dedupe across the two paths: if we already fired for this cell
        # recently, suppress.
        spatial_only = obj_key[-3:] if len(obj_key) >= 5 else pos_b
        if spatial_only == self._fired_objective:
            if now and (now - self._fired_at) < self.cooldown_seconds:
                return None
        self._fired_objective = obj_key
        self._fired_at = now

        if target is not None:
            tgt_str = f"({target[0]:.0f},{target[1]:.0f},{target[2]:.0f})"
        else:
            tgt_str = f"cell {pos_b}"
        return (
            f"STUCK_PIVOT: {self.threshold} heartbeats with no progress on "
            f"target {tgt_str}. Radically change category of action — try "
            f"mining, building, exploring elsewhere, crafting, or "
            f"documenting. Do NOT retry movement to the same target area."
        )
