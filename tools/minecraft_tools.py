#!/usr/bin/env python3

"""
HermesCraft — Embodied Hermes agents for Minecraft

Copyright (c) 2026 bigph00t

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
HermesCraft Minecraft Tools — Consolidated Toolset

Native Hermes toolset that wraps the Mineflayer bot HTTP API.

Instead of 77 individual mc_* tools (which bloat context window and cause
decision paralysis), this consolidated set exposes 8 high-level tools.
Each tool uses an 'action' or 'type' parameter to route to the correct
bot API endpoint.

Environment:
    MC_API_URL  - Bot server URL (default: http://localhost:3001)
"""

import json
import os
import re
import threading
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error


MC_API_URL = os.getenv("MC_API_URL", "http://localhost:3001")

# Global cancel event — set by agent_loop.py when chat arrives during a turn
_cancel_event: Optional[threading.Event] = None


def set_cancel_event(event: Optional[threading.Event]):
    """Wire the cancel event from agent_loop so tool calls can be interrupted mid-flight."""
    global _cancel_event
    _cancel_event = event


def _api_get(path: str, timeout: int = 15) -> dict:
    url = f"{MC_API_URL}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
            return body
        except Exception:
            return {"ok": False, "error": f"Bot server error: {e.code} {e.reason}"}
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"Bot server not responding at {MC_API_URL}: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _cancel_bot_action():
    """Tell the bot server to stop whatever it's doing (mining, moving, etc.)."""
    try:
        req = urllib.request.Request(
            f"{MC_API_URL}/task/cancel",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            pass
    except Exception:
        pass


def _api_post(path: str, data: Optional[dict] = None, timeout: int = 300) -> dict:
    """POST to the bot server. Runs in a thread so it can be cancelled mid-flight."""
    url = f"{MC_API_URL}{path}"
    payload = json.dumps(data or {}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    result_container: dict = {}
    exception_container: dict = {}

    def do_request():
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result_container["result"] = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            exception_container["error"] = e

    t = threading.Thread(target=do_request)
    t.start()

    # Poll every 0.5s — if cancel_event fires, abort the server action and return
    poll_interval = 0.5
    elapsed = 0.0
    while t.is_alive() and elapsed < timeout:
        t.join(timeout=poll_interval)
        elapsed += poll_interval
        if _cancel_event is not None and _cancel_event.is_set():
            _cancel_bot_action()
            return {"ok": False, "error": "Interrupted by new chat message — action cancelled."}

    if t.is_alive():
        # Still running after timeout — abandon it
        return {"ok": False, "error": f"Request timed out after {timeout}s"}

    if "error" in exception_container:
        e = exception_container["error"]
        if isinstance(e, urllib.error.HTTPError):
            try:
                body = json.loads(e.read().decode("utf-8"))
                return body
            except Exception:
                return {"ok": False, "error": f"Bot server error: {e.code} {e.reason}"}
        elif isinstance(e, urllib.error.URLError):
            return {"ok": False, "error": f"Bot server not responding at {MC_API_URL}: {e}"}
        else:
            return {"ok": False, "error": str(e)}

    return result_container.get("result", {})


def _fmt(resp: dict) -> str:
    if not resp.get("ok", True):
        return f"Error: {resp.get('error', 'Unknown error')}"
    parts = []
    if "result" in resp:
        parts.append(f"Result: {resp['result']}")
    if "task_id" in resp:
        parts.append(f"Task {resp['task_id']} started ({resp.get('status', 'running')})")
    if "task" in resp and isinstance(resp.get("task"), dict):
        t = resp["task"]
        parts.append(f"Task: {t.get('action')} | status: {t.get('status')} | elapsed: {t.get('elapsed_s', '?')}s")
        if t.get("error"):
            parts.append(f"Task error: {t['error']}")
    state = resp.get("state")
    if state:
        for k, v in state.items():
            if k not in ("new_chat", "task"):
                parts.append(f"{k}: {v}")
    data = resp.get("data")
    if data and isinstance(data, dict):
        if "summary" in data:
            parts.append(data["summary"])
        elif "messages" in data:
            for m in data["messages"][-10:]:
                w = " [whisper]" if m.get("whisper") else ""
                parts.append(f"<{m['from']}> {m['message']}{w}")
        elif "map" in data:
            parts.append(data["map"])
            parts.append(f"Center: {data.get('center', '?')}  Scale: {data.get('scale', '?')}")
        else:
            for k, v in list(data.items())[:15]:
                parts.append(f"{k}: {v}")
    if "locations" in resp:
        for loc in resp["locations"][:10]:
            parts.append(f"  ({loc.get('x', '?')}, {loc.get('y', '?')}, {loc.get('z', '?')}) — {loc.get('distance', '?')}m")
    return "\n".join(parts) if parts else json.dumps(resp, indent=2)


def check_minecraft_available() -> bool:
    try:
        _api_get("/health", timeout=3)
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. mc_perceive — Observation and state gathering
# ═══════════════════════════════════════════════════════════════════════════════

_PERCEIVE_GET_ENDPOINTS = {
    "status": "/status",
    "inventory": "/inventory",
    "nearby": "/nearby",
    "look": "/look",
    "scene": "/scene",
    "screenshot": "/screenshot",
    "map": "/map",
    "read_chat": "/chat",
    "overhear": "/overhear",
    "sounds": "/sounds",
    "stats": "/stats",
    "health": "/health",
    "deaths": "/deaths",
    "commands": "/commands",
    "furnaces": "/furnaces",
    "task_status": "/task",
    "social": "/social",
}

_PERCEIVE_POST_ENDPOINTS = {
    "team_status": "/action/team_status",
    "report": "/action/report",
    "fair_play": "/action/set_fair_play",
}


def _handle_mc_perceive(args: dict, **kwargs) -> str:
    """Observe the Minecraft world: status, inventory, surroundings, chat, etc."""
    ptype = args.get("type", "status")

    if ptype in _PERCEIVE_GET_ENDPOINTS:
        path = _PERCEIVE_GET_ENDPOINTS[ptype]
        if ptype == "nearby":
            path += f'?radius={args.get("radius", 32)}'
        elif ptype == "scene":
            path += f'?range={args.get("range", 16)}'
        elif ptype == "map":
            path += f'?radius={args.get("radius", 16)}'
        elif ptype in ("read_chat", "overhear"):
            path += f'?count={args.get("count", 20)}'
        elif ptype == "screenshot":
            w = args.get("width", 1280)
            h = args.get("height", 720)
            path += f'?width={w}&height={h}'
        return _fmt(_api_get(path))

    if ptype in _PERCEIVE_POST_ENDPOINTS:
        endpoint = _PERCEIVE_POST_ENDPOINTS[ptype]
        payload = {}
        if ptype == "report":
            if "message" not in args:
                return "Error: message is required for report"
            payload["message"] = args["message"]
        elif ptype == "fair_play":
            payload["enabled"] = args.get("enabled", True)
        return _fmt(_api_post(endpoint, payload))

    return f"Error: unknown perceive type '{ptype}'"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. mc_move — Navigation and locomotion
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_mc_move(args: dict, **kwargs) -> str:
    """Move the bot: goto coordinates, follow a player, stop, etc."""
    action = args.get("action", "stop")
    payload: Dict[str, Any] = {}

    if action == "goto":
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for goto"
        payload = {"x": args["x"], "y": args["y"], "z": args["z"]}
        return _fmt(_api_post("/action/goto", payload))

    if action == "goto_near":
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for goto_near"
        payload = {"x": args["x"], "y": args["y"], "z": args["z"], "range": args.get("range", 2)}
        return _fmt(_api_post("/action/goto_near", payload))

    if action == "follow":
        if "player" not in args:
            return "Error: player is required for follow"
        return _fmt(_api_post("/action/follow", {"player": args["player"]}))

    if action == "stop":
        return _fmt(_api_post("/action/stop"))

    if action == "deathpoint":
        return _fmt(_api_post("/action/deathpoint"))

    return f"Error: unknown move action '{action}'"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. mc_mine — Resource gathering and block interaction
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_mc_mine(args: dict, **kwargs) -> str:
    """Mine, dig, collect, and find resources in the world."""
    action = args.get("action", "pickup")
    payload: Dict[str, Any] = {}

    if action == "collect":
        if "block" not in args:
            return "Error: block is required for collect"
        payload = {"block": args["block"], "count": args.get("count", 1)}
        return _fmt(_api_post("/action/collect", payload))

    if action == "dig":
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for dig"
        payload = {"x": args["x"], "y": args["y"], "z": args["z"]}
        return _fmt(_api_post("/action/dig", payload))

    if action == "pickup":
        return _fmt(_api_post("/action/pickup"))

    if action == "find_blocks":
        if "block" not in args:
            return "Error: block is required for find_blocks"
        payload = {"block": args["block"], "radius": args.get("radius", 32), "count": args.get("count", 10)}
        return _fmt(_api_post("/action/find_blocks", payload))

    if action == "find_entities":
        payload = {"radius": args.get("radius", 32)}
        if args.get("type"):
            payload["type"] = args["type"]
        return _fmt(_api_post("/action/find_entities", payload))

    return f"Error: unknown mine action '{action}'"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. mc_build — Construction, placement, and block interaction
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_mc_build(args: dict, **kwargs) -> str:
    """Build, place blocks, fill areas, interact with blocks, and utility actions."""
    action = args.get("action", "use")
    payload: Dict[str, Any] = {}

    if action == "place":
        if "block" not in args:
            return "Error: block is required for place"
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for place"
        payload = {"block": args["block"], "x": args["x"], "y": args["y"], "z": args["z"]}
        return _fmt(_api_post("/action/place", payload))

    if action == "fill":
        if "block" not in args:
            return "Error: block is required for fill"
        for coord in ("x1", "y1", "z1", "x2", "y2", "z2"):
            if coord not in args:
                return f"Error: {coord} is required for fill"
        payload = {
            "block": args["block"],
            "x1": args["x1"], "y1": args["y1"], "z1": args["z1"],
            "x2": args["x2"], "y2": args["y2"], "z2": args["z2"],
            "hollow": args.get("hollow", False),
        }
        return _fmt(_api_post("/action/place_fill", payload))

    if action == "interact":
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for interact"
        payload = {"x": args["x"], "y": args["y"], "z": args["z"]}
        return _fmt(_api_post("/action/interact", payload))

    if action == "till":
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for till"
        payload = {"x": args["x"], "y": args["y"], "z": args["z"]}
        return _fmt(_api_post("/action/till", payload))

    if action == "bonemeal":
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for bonemeal"
        payload = {"x": args["x"], "y": args["y"], "z": args["z"]}
        return _fmt(_api_post("/action/bonemeal", payload))

    if action == "flatten":
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for flatten"
        payload = {"x": args["x"], "y": args["y"], "z": args["z"]}
        return _fmt(_api_post("/action/flatten", payload))

    if action == "ignite":
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for ignite"
        payload = {"x": args["x"], "y": args["y"], "z": args["z"]}
        return _fmt(_api_post("/action/ignite", payload))

    if action == "fish":
        return _fmt(_api_post("/action/fish"))

    if action == "close":
        return _fmt(_api_post("/action/close_screen"))

    if action == "use":
        return _fmt(_api_post("/action/use"))

    if action == "toss":
        if "item" not in args:
            return "Error: item is required for toss"
        payload = {"item": args["item"]}
        if args.get("count") is not None:
            payload["count"] = args["count"]
        return _fmt(_api_post("/action/toss", payload))

    if action == "sleep":
        return _fmt(_api_post("/action/sleep_bed"))

    if action == "wait":
        payload = {"seconds": args.get("seconds", 5)}
        return _fmt(_api_post("/action/wait", payload))

    if action == "connect":
        return _fmt(_api_post("/connect"))

    return f"Error: unknown build action '{action}'"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. mc_craft — Crafting, smelting, and recipes
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_mc_craft(args: dict, **kwargs) -> str:
    """Craft items, look up recipes, and manage furnaces."""
    action = args.get("action", "craft")
    payload: Dict[str, Any] = {}

    if action == "craft":
        if "item" not in args:
            return "Error: item is required for craft"
        payload = {"item": args["item"], "count": args.get("count", 1)}
        return _fmt(_api_post("/action/craft", payload))

    if action == "recipes":
        if "item" not in args:
            return "Error: item is required for recipes"
        payload = {"item": args["item"]}
        return _fmt(_api_post("/action/recipes", payload))

    if action == "smelt":
        if "input" not in args:
            return "Error: input is required for smelt"
        payload = {"input": args["input"], "count": args.get("count", 1)}
        if args.get("fuel"):
            payload["fuel"] = args["fuel"]
        return _fmt(_api_post("/action/smelt", payload))

    if action == "smelt_start":
        if "input" not in args:
            return "Error: input is required for smelt_start"
        payload = {"input": args["input"], "count": args.get("count", 1)}
        if args.get("fuel"):
            payload["fuel"] = args["fuel"]
        return _fmt(_api_post("/action/smelt_start", payload))

    if action in ("furnace_check", "furnace_take"):
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for {action}"
        payload = {"x": args["x"], "y": args["y"], "z": args["z"]}
        endpoint = "/action/furnace_check" if action == "furnace_check" else "/action/furnace_take"
        return _fmt(_api_post(endpoint, payload))

    return f"Error: unknown craft action '{action}'"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. mc_combat — Combat, equipment, and survival actions
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_mc_combat(args: dict, **kwargs) -> str:
    """Fight, flee, equip gear, eat, and execute combat maneuvers."""
    action = args.get("action", "eat")
    payload: Dict[str, Any] = {}

    if action == "attack":
        payload = {}
        if args.get("target"):
            payload["target"] = args["target"]
        return _fmt(_api_post("/action/attack", payload))

    if action == "fight":
        payload = {"retreat_health": args.get("retreat_health", 6), "duration": args.get("duration", 30)}
        if args.get("target"):
            payload["target"] = args["target"]
        return _fmt(_api_post("/action/fight", payload))

    if action == "flee":
        payload = {"distance": args.get("distance", 16)}
        return _fmt(_api_post("/action/flee", payload))

    if action == "eat":
        return _fmt(_api_post("/action/eat"))

    if action == "equip":
        if "item" not in args:
            return "Error: item is required for equip"
        payload = {"item": args["item"], "slot": args.get("slot", "hand")}
        return _fmt(_api_post("/action/equip", payload))

    if action == "sneak":
        payload = {"enable": args.get("enable", True)}
        return _fmt(_api_post("/action/sneak", payload))

    if action == "shield":
        payload = {"duration": args.get("duration", 3)}
        return _fmt(_api_post("/action/shield_block", payload))

    if action == "shoot":
        payload = {"predict": args.get("predict", True)}
        if args.get("target"):
            payload["target"] = args["target"]
        return _fmt(_api_post("/action/shoot", payload))

    if action == "sprint_attack":
        payload = {}
        if args.get("target"):
            payload["target"] = args["target"]
        return _fmt(_api_post("/action/sprint_attack", payload))

    if action == "crit":
        payload = {}
        if args.get("target"):
            payload["target"] = args["target"]
        return _fmt(_api_post("/action/critical_hit", payload))

    if action == "strafe":
        payload = {"direction": args.get("direction", "random"), "duration": args.get("duration", 5)}
        if args.get("target"):
            payload["target"] = args["target"]
        return _fmt(_api_post("/action/strafe", payload))

    if action == "combo":
        payload = {"style": args.get("style", "aggressive")}
        if args.get("target"):
            payload["target"] = args["target"]
        return _fmt(_api_post("/action/combo", payload))

    return f"Error: unknown combat action '{action}'"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. mc_chat — Communication and team coordination
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_mc_chat(args: dict, **kwargs) -> str:
    """Send messages: public chat, whispers, team chat, rally points, etc."""
    action = args.get("action", "chat")
    payload: Dict[str, Any] = {}

    if action == "chat":
        if "message" not in args:
            return "Error: message is required for chat"
        return _fmt(_api_post("/action/chat", {"message": args["message"]}))

    if action == "whisper":
        if "player" not in args or "message" not in args:
            return "Error: player and message are required for whisper"
        return _fmt(_api_post("/action/whisper", {"player": args["player"], "message": args["message"]}))

    if action == "chat_to":
        if "player" not in args or "message" not in args:
            return "Error: player and message are required for chat_to"
        return _fmt(_api_post("/action/chat_to", {"player": args["player"], "message": args["message"]}))

    if action == "team_chat":
        if "message" not in args:
            return "Error: message is required for team_chat"
        return _fmt(_api_post("/action/team_chat", {"message": args["message"]}))

    if action == "rally":
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for rally"
        payload = {"x": args["x"], "y": args["y"], "z": args["z"]}
        if args.get("message"):
            payload["message"] = args["message"]
        return _fmt(_api_post("/action/rally", payload))

    if action == "set_team":
        if "team" not in args:
            return "Error: team is required for set_team"
        payload = {"team": args["team"], "role": args.get("role", "warrior")}
        if args.get("teammates"):
            payload["teammates"] = args["teammates"].split(",")
        return _fmt(_api_post("/action/set_team", payload))

    if action == "complete_command":
        payload = {"index": args.get("index", 0)}
        return _fmt(_api_post("/action/complete_command", payload))

    return f"Error: unknown chat action '{action}'"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. mc_manage — Containers, waypoints, and background tasks
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_mc_manage(args: dict, **kwargs) -> str:
    """Manage containers, saved locations, and background tasks."""
    action = args.get("action", "marks")
    payload: Dict[str, Any] = {}

    if action == "chest":
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for chest"
        payload = {"x": args["x"], "y": args["y"], "z": args["z"]}
        return _fmt(_api_post("/action/list_container", payload))

    if action == "deposit":
        if "item" not in args:
            return "Error: item is required for deposit"
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for deposit"
        payload = {"item": args["item"], "x": args["x"], "y": args["y"], "z": args["z"], "count": args.get("count", 0)}
        return _fmt(_api_post("/action/deposit", payload))

    if action == "withdraw":
        if "item" not in args:
            return "Error: item is required for withdraw"
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for withdraw"
        payload = {"item": args["item"], "x": args["x"], "y": args["y"], "z": args["z"], "count": args.get("count", 0)}
        return _fmt(_api_post("/action/withdraw", payload))

    if action == "mark":
        if "name" not in args:
            return "Error: name is required for mark"
        payload = {"name": args["name"], "note": args.get("note", "")}
        return _fmt(_api_post("/action/mark", payload))

    if action == "marks":
        return _fmt(_api_post("/action/marks"))

    if action == "go_mark":
        if "name" not in args:
            return "Error: name is required for go_mark"
        return _fmt(_api_post("/action/go_mark", {"name": args["name"]}))

    if action == "unmark":
        if "name" not in args:
            return "Error: name is required for unmark"
        return _fmt(_api_post("/action/unmark", {"name": args["name"]}))

    if action == "bg_goto":
        for coord in ("x", "y", "z"):
            if coord not in args:
                return f"Error: {coord} is required for bg_goto"
        payload = {"x": args["x"], "y": args["y"], "z": args["z"]}
        return _fmt(_api_post("/task/goto", payload))

    if action == "bg_collect":
        if "block" not in args:
            return "Error: block is required for bg_collect"
        payload = {"block": args["block"], "count": args.get("count", 1)}
        return _fmt(_api_post("/task/collect", payload))

    if action == "bg_fight":
        payload = {"retreat_health": args.get("retreat_health", 6), "duration": args.get("duration", 30)}
        if args.get("target"):
            payload["target"] = args["target"]
        return _fmt(_api_post("/task/fight", payload))

    if action == "bg_combo":
        payload = {"style": args.get("style", "aggressive")}
        if args.get("target"):
            payload["target"] = args["target"]
        return _fmt(_api_post("/task/combo", payload))

    if action == "bg_strafe":
        payload = {"direction": args.get("direction", "random"), "duration": args.get("duration", 10)}
        if args.get("target"):
            payload["target"] = args["target"]
        return _fmt(_api_post("/task/strafe", payload))

    if action == "cancel":
        return _fmt(_api_post("/task/cancel"))

    if action == "task_status":
        return _fmt(_api_get("/task"))

    return f"Error: unknown manage action '{action}'"


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Schemas
# ═══════════════════════════════════════════════════════════════════════════════

MC_PERCEIVE_SCHEMA = {
    "name": "mc_perceive",
    "description": "Observe the Minecraft world. Use 'status' for full state, 'inventory' for items, 'nearby' for blocks/entities, 'look' for a narrative description, 'scene' for fair-play view, 'map' for ASCII top-down, 'read_chat' for recent messages, 'social' for interaction summary, 'sounds' for audio events, 'health' for quick vitals, 'deaths' for death log, 'commands' for pending orders, 'furnaces' for active furnaces, 'task_status' for background tasks, 'team_status' for teammates, 'report' to send intel, 'fair_play' to toggle fairness mode.",
    "parameters": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["status", "inventory", "nearby", "look", "scene", "map", "read_chat", "overhear", "sounds", "stats", "health", "deaths", "commands", "furnaces", "task_status", "social", "team_status", "report", "fair_play"],
                "description": "What to observe",
            },
            "radius": {"type": "number", "description": "Scan radius for nearby/map"},
            "range": {"type": "number", "description": "View range for scene"},
            "count": {"type": "number", "description": "Message count for read_chat/overhear"},
            "message": {"type": "string", "description": "Intel message for report action"},
            "enabled": {"type": "boolean", "description": "Toggle fair play mode on/off"},
        },
        "required": ["type"],
    },
}

MC_MOVE_SCHEMA = {
    "name": "mc_move",
    "description": "Navigate the bot. 'goto' walks to exact coordinates. 'goto_near' stops within a range. 'follow' trails a player. 'stop' halts all movement. 'deathpoint' returns to last death location.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["goto", "goto_near", "follow", "stop", "deathpoint"],
                "description": "Movement action",
            },
            "x": {"type": "number", "description": "X coordinate"},
            "y": {"type": "number", "description": "Y coordinate"},
            "z": {"type": "number", "description": "Z coordinate"},
            "player": {"type": "string", "description": "Player name to follow"},
            "range": {"type": "number", "description": "Acceptable distance for goto_near"},
        },
        "required": ["action"],
    },
}

MC_MINE_SCHEMA = {
    "name": "mc_mine",
    "description": "Gather resources. 'collect' mines N blocks of a type. 'dig' breaks a specific block. 'pickup' grabs nearby drops. 'find_blocks' locates block positions. 'find_entities' scans for mobs/players.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["collect", "dig", "pickup", "find_blocks", "find_entities"],
                "description": "Mining action",
            },
            "block": {"type": "string", "description": "Block type (e.g. oak_log, iron_ore)"},
            "x": {"type": "number", "description": "X coordinate for dig"},
            "y": {"type": "number", "description": "Y coordinate for dig"},
            "z": {"type": "number", "description": "Z coordinate for dig"},
            "count": {"type": "number", "description": "How many blocks to mine or max results"},
            "radius": {"type": "number", "description": "Search radius"},
            "entity_type": {"type": "string", "description": "Entity filter for find_entities"},
        },
        "required": ["action"],
    },
}

MC_BUILD_SCHEMA = {
    "name": "mc_build",
    "description": "Build and interact with the world. 'place' a single block. 'fill' a volume. 'interact' right-clicks a block (chests, doors, furnaces). 'till' hoes grass_block/dirt into farmland. 'bonemeal' grows crops/saplings. 'flatten' shovels grass/dirt into dirt_path. 'ignite' lights netherrack/TNT/campfires with flint_and_steel. 'fish' casts a fishing rod. 'close' any open screen. 'use' activates held item. 'toss' drops items. 'sleep' finds a bed. 'wait' pauses. 'connect' reconnects the bot.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["place", "fill", "interact", "till", "bonemeal", "flatten", "ignite", "fish", "close", "use", "toss", "sleep", "wait", "connect"],
                "description": "Build/interaction action",
            },
            "block": {"type": "string", "description": "Block type for place/fill"},
            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
            "x1": {"type": "number"}, "y1": {"type": "number"}, "z1": {"type": "number"},
            "x2": {"type": "number"}, "y2": {"type": "number"}, "z2": {"type": "number"},
            "hollow": {"type": "boolean", "description": "Fill hollow for fill action"},
            "item": {"type": "string", "description": "Item for toss"},
            "count": {"type": "number", "description": "Item count for toss"},
            "seconds": {"type": "number", "description": "Seconds to wait"},
        },
        "required": ["action"],
    },
}

MC_CRAFT_SCHEMA = {
    "name": "mc_craft",
    "description": "Craft items and manage furnaces. 'craft' creates an item. 'recipes' looks up requirements. 'smelt' cooks in furnace and waits. 'smelt_start' loads furnace and leaves. 'furnace_check' inspects a furnace. 'furnace_take' collects output.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["craft", "recipes", "smelt", "smelt_start", "furnace_check", "furnace_take"],
                "description": "Crafting action",
            },
            "item": {"type": "string", "description": "Item name for craft/recipes"},
            "input": {"type": "string", "description": "Input material for smelting"},
            "fuel": {"type": "string", "description": "Fuel for smelting (optional)"},
            "count": {"type": "number", "description": "Quantity"},
            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
        },
        "required": ["action"],
    },
}

MC_COMBAT_SCHEMA = {
    "name": "mc_combat",
    "description": "Combat and survival. 'attack' a target. 'fight' sustained combat with retreat threshold. 'flee' from hostiles. 'eat' best food. 'equip' an item. 'sneak' toggle. 'shield' block. 'shoot' bow. 'sprint_attack' for knockback. 'crit' for jump-attack. 'strafe' while fighting. 'combo' executes a style sequence.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["attack", "fight", "flee", "eat", "equip", "sneak", "shield", "shoot", "sprint_attack", "crit", "strafe", "combo"],
                "description": "Combat action",
            },
            "target": {"type": "string", "description": "Target mob or player"},
            "retreat_health": {"type": "number", "description": "HP threshold to retreat during fight"},
            "duration": {"type": "number", "description": "Duration in seconds for fight/strafe"},
            "distance": {"type": "number", "description": "Flee distance"},
            "item": {"type": "string", "description": "Item to equip"},
            "slot": {"type": "string", "description": "Equipment slot (hand, head, chest, legs, feet, off-hand)"},
            "enable": {"type": "boolean", "description": "Enable/disable sneak"},
            "predict": {"type": "boolean", "description": "Predict target movement for shoot"},
            "direction": {"type": "string", "description": "Strafe direction: left, right, random"},
            "style": {"type": "string", "description": "Combo style: aggressive, defensive, balanced"},
        },
        "required": ["action"],
    },
}

MC_CHAT_SCHEMA = {
    "name": "mc_chat",
    "description": "Communication. 'chat' public message. 'whisper' private to one player. 'chat_to' alternative private message. 'team_chat' to teammates. 'rally' sets a team rally point. 'set_team' assigns team/role. 'complete_command' marks a pending order done.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["chat", "whisper", "chat_to", "team_chat", "rally", "set_team", "complete_command"],
                "description": "Chat action",
            },
            "message": {"type": "string", "description": "Message content"},
            "player": {"type": "string", "description": "Target player for whisper/chat_to"},
            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
            "team": {"type": "string", "description": "Team name for set_team"},
            "role": {"type": "string", "description": "Role for set_team (default: warrior)"},
            "teammates": {"type": "string", "description": "Comma-separated teammate names for set_team"},
            "index": {"type": "number", "description": "Command index to complete"},
        },
        "required": ["action"],
    },
}

MC_MANAGE_SCHEMA = {
    "name": "mc_manage",
    "description": "Manage containers, waypoints, and background tasks. 'chest' lists contents. 'deposit'/'withdraw' items. 'mark' saves current location. 'marks' lists waypoints. 'go_mark' navigates to one. 'unmark' deletes. 'bg_goto'/'bg_collect'/'bg_fight' background tasks. 'bg_combo'/'bg_strafe' background combat. 'cancel' stops background task. 'task_status' checks progress.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["chest", "deposit", "withdraw", "mark", "marks", "go_mark", "unmark", "bg_goto", "bg_collect", "bg_fight", "bg_combo", "bg_strafe", "cancel", "task_status"],
                "description": "Management action",
            },
            "item": {"type": "string", "description": "Item name for deposit/withdraw"},
            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
            "count": {"type": "number", "description": "Item count for deposit/withdraw or block count for bg_collect"},
            "name": {"type": "string", "description": "Waypoint name for mark/go_mark/unmark"},
            "note": {"type": "string", "description": "Optional note for mark"},
            "block": {"type": "string", "description": "Block type for bg_collect"},
            "target": {"type": "string", "description": "Target for bg_fight/bg_combo/bg_strafe"},
            "retreat_health": {"type": "number"},
            "duration": {"type": "number"},
            "style": {"type": "string", "description": "Combo style for bg_combo"},
            "direction": {"type": "string", "description": "Strafe direction for bg_strafe"},
        },
        "required": ["action"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# 9. mc_plan — Persistent goal & task planning
# ═══════════════════════════════════════════════════════════════════

def _handle_mc_plan(args: dict, **kwargs) -> str:
    """Manage persistent goals and tasks. Bots use this to remember multi-step projects across turns."""
    action = args.get("action", "get_plan")
    payload: Dict[str, Any] = {}

    if action == "set_goal":
        if "goal" not in args:
            return "Error: goal is required for set_goal"
        payload = {
            "action": "set_goal",
            "goal": args["goal"],
            "tasks": args.get("tasks", []),
        }
        return _fmt(_api_post("/action/plan", payload))

    if action == "get_plan":
        return _fmt(_api_post("/action/plan", {"action": "get_plan"}))

    if action == "update_task":
        if "task_id" not in args:
            return "Error: task_id is required for update_task"
        payload = {
            "action": "update_task",
            "task_id": args["task_id"],
            "status": args.get("status"),
            "result": args.get("result"),
            "attempt": args.get("attempt"),
        }
        return _fmt(_api_post("/action/plan", payload))

    if action == "add_task":
        if "goal" not in args:
            return "Error: goal (task description) is required for add_task"
        payload = {
            "action": "add_task",
            "goal": args["goal"],
            "status": args.get("status", "pending"),
        }
        return _fmt(_api_post("/action/plan", payload))

    if action == "remove_task":
        if "task_id" not in args:
            return "Error: task_id is required for remove_task"
        payload = {
            "action": "remove_task",
            "task_id": args["task_id"],
        }
        return _fmt(_api_post("/action/plan", payload))

    if action == "clear_goal":
        return _fmt(_api_post("/action/plan", {"action": "clear_goal"}))

    return f"Error: unknown plan action '{action}'"


MC_PLAN_SCHEMA = {
    "name": "mc_plan",
    "description": "Persistent goal and task management. Use this to plan multi-step projects that survive across turns. 'set_goal' creates a goal with tasks. 'get_plan' reads current progress. 'update_task' marks tasks done/in_progress/blocked. 'add_task' appends a task. 'remove_task' deletes one. 'clear_goal' resets everything.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set_goal", "get_plan", "update_task", "add_task", "remove_task", "clear_goal"],
                "description": "Planning action",
            },
            "goal": {"type": "string", "description": "Goal description (for set_goal) or task description (for add_task)"},
            "tasks": {
                "type": "array",
                "description": "List of tasks for set_goal",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "done", "blocked"]},
                        "attempts": {"type": "number"},
                    },
                },
            },
            "task_id": {"type": "number", "description": "Zero-based task index for update/remove"},
            "status": {"type": "string", "enum": ["pending", "in_progress", "done", "blocked"], "description": "New status for update_task"},
            "result": {"type": "string", "description": "Optional result note for update_task"},
            "attempt": {"type": "boolean", "description": "If true, increments attempt counter for update_task"},
        },
        "required": ["action"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# 10. mc_screenshot — Ray-traced world capture
# ═══════════════════════════════════════════════════════════════════

def _handle_mc_screenshot(args: dict, **kwargs) -> str:
    """Take a screenshot of the Minecraft world from the bot's first-person perspective.

    Uses prismarine-viewer (Three.js WebGL renderer) + puppeteer headless Chrome.
    The image is saved as PNG to the bot server and the path is returned.
    """
    payload: Dict[str, Any] = {}
    if "width" in args:
        payload["width"] = args["width"]
    if "height" in args:
        payload["height"] = args["height"]
    if "file_name" in args:
        fname = args["file_name"]
        if not fname.endswith(".png"):
            fname += ".png"
        payload["file_name"] = fname

    resp = _api_post("/action/screenshot", payload, timeout=300)
    if not resp.get("ok", True):
        return f"Error: {resp.get('error', 'Screenshot failed')}"

    path = resp.get("path", "unknown")
    width = resp.get("width", "?")
    height = resp.get("height", "?")
    return f"Screenshot saved to {path} ({width}x{height})"


MC_SCREENSHOT_SCHEMA = {
    "name": "mc_screenshot",
    "description": "Take a screenshot of the Minecraft world from the bot's eyes. Uses a WebGL renderer (prismarine-viewer) served on a local port and captured via headless Chrome. Produces a PNG image. Specify width/height (default 1280x720, max 1920x1080) and optionally a custom file_name. The returned path is an absolute PNG file path. If you need to SEE what is in the image, call vision_analyze with the returned path.",
    "parameters": {
        "type": "object",
        "properties": {
            "width": {"type": "number", "description": "Image width in pixels (default: 1280, max: 1920)"},
            "height": {"type": "number", "description": "Image height in pixels (default: 720, max: 1080)"},
            "file_name": {"type": "string", "description": "Custom filename for the screenshot (optional). Will be saved as a .png file."},
        },
    },
}


# ═══════════════════════════════════════════════════════════════════
# 11. mc_command — Execute Minecraft server commands
# ═══════════════════════════════════════════════════════════════════

def _handle_mc_command(args: dict, **kwargs) -> str:
    """Execute a Minecraft server command via the bot's chat interface.
    
    The bot must have operator privileges for most commands.
    Commands are sent as chat messages starting with '/' and are executed
    by the server without appearing in public chat.
    """
    command = args.get("command", "")
    if not command:
        return "Error: command is required"
    if not command.startswith("/"):
        command = "/" + command

    # ═─ Intercept /godmode toggle ─══════════════════════════════════════
    stripped = command.strip().lower()
    if stripped == "/godmode on" or stripped == "/godmode":
        _gm_path = Path.home() / ".local" / "share" / "daemoncraft" / "rolemaster" / "godmode"
        _gm_path.parent.mkdir(parents=True, exist_ok=True)
        _gm_path.write_text("on")
        return "Godmode ENABLED. The Daemon Guardian will keep you in creative mode with invulnerability effects."
    if stripped == "/godmode off":
        _gm_path = Path.home() / ".local" / "share" / "daemoncraft" / "rolemaster" / "godmode"
        _gm_path.parent.mkdir(parents=True, exist_ok=True)
        _gm_path.write_text("off")
        return "Godmode DISABLED. The Daemon Guardian is paused. You can now take damage, drown, or switch gamemodes. Say '/godmode on' to restore protection."

    return _fmt(_api_post("/chat/send", {"message": command}))


MC_COMMAND_SCHEMA = {
    "name": "mc_command",
    "description": "Execute any Minecraft server command. The bot must have operator privileges. Examples: /weather thunder, /time set midnight, /summon zombie ~ ~ ~, /give @p diamond 1, /effect give @p blindness 10, /playsound ambient.cave ambient @p, /tellraw @p {\"text\":\"Hello\"}, /setblock ~ ~ ~ stone, /fill x1 y1 z1 x2 y2 z2 water. This is the primary tool for world manipulation in Role Master mode.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Minecraft command to execute. Must start with / or it will be added automatically.",
            },
        },
        "required": ["command"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# 12. mc_story — Narrative state tracker for Role Master mode
# ═══════════════════════════════════════════════════════════════════

import os
from pathlib import Path

_STORY_PATH = Path(os.getenv("DAEMONCRAFT_STORY_PATH", Path.home() / ".local" / "share" / "daemoncraft" / "story.json"))
_BLUEPRINT_PATH = Path(os.getenv("DAEMONCRAFT_BLUEPRINT_PATH", Path.home() / ".local" / "share" / "daemoncraft" / "blueprint.json"))
# Shared blueprints directory used by the dashboard and mc_story
_BLUEPRINTS_DIR = Path(__file__).parent.parent / "blueprints"


def _load_story() -> dict:
    if _STORY_PATH.exists():
        try:
            return json.loads(_STORY_PATH.read_text())
        except Exception:
            pass
    return {
        "title": None,
        "phase": None,
        "phase_started_at": None,
        "phase_timeout_minutes": None,
        "last_player_activity": None,
        "day": 1,
        "flags": {},
        "objectives": [],
        "events": [],
        "player_choices": {},
        "active_sensors": [],
        "active_blueprint": None,
        "active_blueprint_tag": None,
    }


def _save_story(story: dict) -> None:
    _STORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORY_PATH.write_text(json.dumps(story, indent=2))


def _handle_mc_story(args: dict, **kwargs) -> str:
    """Track narrative state for Role Master adventures. Pure Python — no bot server needed."""
    action = args.get("action", "get_state")
    story = _load_story()

    if action == "get_state":
        import datetime as _dt
        lines = [
            f"Story: {story.get('title') or 'Untitled'}",
            f"Phase: {story.get('phase') or 'none'}",
            f"Day: {story.get('day', 1)}",
            f"Active blueprint: {story.get('active_blueprint', 'none')}",
            f"Active blueprint tag: {story.get('active_blueprint_tag', 'none')}",
            f"Flags: {json.dumps(story.get('flags', {}))}",
            f"Objectives ({len(story.get('objectives', []))}):",
        ]
        for obj in story.get("objectives", []):
            status = obj.get("status", "pending")
            lines.append(f"  [{status}] {obj.get('title', 'Untitled')}: {obj.get('description', '')}")
        # Timeout info
        timeout = story.get("phase_timeout_minutes")
        started = story.get("phase_started_at")
        last_act = story.get("last_player_activity")
        if timeout and started:
            elapsed = (_dt.datetime.now(_dt.timezone.utc) - _dt.datetime.fromisoformat(started)).total_seconds() / 60
            remaining = timeout - elapsed
            lines.append(f"Phase timeout: {max(0, remaining):.1f} minutes remaining")
        if last_act:
            ago = (_dt.datetime.now(_dt.timezone.utc) - _dt.datetime.fromisoformat(last_act)).total_seconds() / 60
            lines.append(f"Last player activity: {ago:.1f} minutes ago")
        lines.append(f"Events ({len(story.get('events', []))}): {story.get('events', [])[-5:]}")
        return "\n".join(lines)

    if action == "set_flag":
        key = args.get("key")
        value = args.get("value")
        if key is None:
            return "Error: key is required for set_flag"
        story["flags"][key] = value
        _save_story(story)
        return f"Flag set: {key} = {value}"

    if action == "advance_phase":
        phase = args.get("phase")
        if not phase:
            return "Error: phase is required for advance_phase"
        import datetime as _dt
        story["phase"] = phase
        story["phase_started_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        timeout = args.get("timeout_minutes")
        if timeout is not None:
            story["phase_timeout_minutes"] = timeout
        story["events"].append(f"Advanced to phase: {phase}")
        _save_story(story)
        return f"Phase advanced to: {phase}"

    if action == "record_activity":
        import datetime as _dt
        story["last_player_activity"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        _save_story(story)
        return "Player activity recorded"

    if action == "check_timeout":
        import datetime as _dt
        phase = story.get("phase")
        timeout = story.get("phase_timeout_minutes")
        started = story.get("phase_started_at")
        last_act = story.get("last_player_activity")
        if not phase or not timeout:
            return "No active phase with timeout"
        # Use last_player_activity if available, otherwise phase_started_at
        ref_time = last_act or started
        if not ref_time:
            return "No reference time for timeout check"
        elapsed = (_dt.datetime.now(_dt.timezone.utc) - _dt.datetime.fromisoformat(ref_time)).total_seconds() / 60
        if elapsed > timeout:
            story["phase"] = None
            story["phase_started_at"] = None
            story["phase_timeout_minutes"] = None
            # Reset objectives of abandoned phase
            for obj in story.get("objectives", []):
                if obj.get("status") == "pending":
                    obj["status"] = "abandoned"
            _save_story(story)
            return f"Phase '{phase}' ABANDONED after {elapsed:.1f} minutes of inactivity. Objectives reset."
        return f"Phase '{phase}' still active. {timeout - elapsed:.1f} minutes remaining."

    if action == "reset_phase":
        phase = args.get("phase")
        if phase:
            story["events"].append(f"Phase reset: {phase}")
        story["phase"] = None
        story["phase_started_at"] = None
        story["phase_timeout_minutes"] = None
        for obj in story.get("objectives", []):
            if obj.get("status") in ("pending", "abandoned"):
                obj["status"] = "pending"
        _save_story(story)
        return f"Phase reset. Current phase: none. Pending objectives restored."

    if action == "advance_day":
        story["day"] = story.get("day", 1) + 1
        story["events"].append(f"Day advanced to {story['day']}")
        _save_story(story)
        return f"Day advanced to {story['day']}"

    if action == "add_objective":
        title = args.get("title")
        if not title:
            return "Error: title is required for add_objective"
        obj = {
            "id": len(story.get("objectives", [])),
            "title": title,
            "description": args.get("description", ""),
            "status": "pending",
            "optional": args.get("optional", False),
        }
        story.setdefault("objectives", []).append(obj)
        story["events"].append(f"Added objective: {title}")
        _save_story(story)
        return f"Objective added: {title}"

    if action == "complete_objective":
        obj_id = args.get("objective_id")
        if obj_id is None:
            return "Error: objective_id is required for complete_objective"
        objectives = story.get("objectives", [])
        if obj_id < 0 or obj_id >= len(objectives):
            return f"Error: objective_id {obj_id} not found"
        objectives[obj_id]["status"] = "done"
        story["events"].append(f"Completed objective: {objectives[obj_id]['title']}")
        _save_story(story)
        return f"Objective completed: {objectives[obj_id]['title']}"

    if action == "log_event":
        event = args.get("event")
        if not event:
            return "Error: event is required for log_event"
        story.setdefault("events", []).append(event)
        _save_story(story)
        return f"Event logged: {event}"

    if action == "get_events":
        count = args.get("count", 10)
        events = story.get("events", [])
        recent = events[-count:] if events else []
        return "Recent events:\n" + "\n".join(f"  {i+1}. {e}" for i, e in enumerate(recent)) if recent else "No events recorded yet."

    if action == "set_title":
        title = args.get("title")
        if not title:
            return "Error: title is required for set_title"
        story["title"] = title
        _save_story(story)
        return f"Story title set: {title}"

    if action == "record_choice":
        player = args.get("player", "unknown")
        choice = args.get("choice")
        if not choice:
            return "Error: choice is required for record_choice"
        story.setdefault("player_choices", {})[player] = choice
        story["events"].append(f"{player} chose: {choice}")
        _save_story(story)
        return f"Choice recorded for {player}: {choice}"

    if action == "reset":
        _save_story({
            "title": None,
            "phase": None,
            "day": 1,
            "flags": {},
            "objectives": [],
            "events": [],
            "player_choices": {},
        })
        return "Story state reset"

    if action == "save_blueprint":
        blueprint = args.get("blueprint")
        name = args.get("name")
        if not blueprint:
            return "Error: blueprint JSON is required for save_blueprint"
        if not isinstance(blueprint, dict):
            return "Error: blueprint must be a JSON object"
        if name:
            target = _BLUEPRINTS_DIR / f"{name}.json"
            _BLUEPRINTS_DIR.mkdir(parents=True, exist_ok=True)
        else:
            target = _BLUEPRINT_PATH
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(blueprint, indent=2))
        return f"Blueprint saved: {blueprint.get('metadata', {}).get('title', 'Untitled')}"

    if action == "load_blueprint":
        name = args.get("name")
        if name:
            target = _BLUEPRINTS_DIR / f"{name}.json"
        else:
            target = _BLUEPRINT_PATH
        if not target.exists():
            return f"No blueprint found: {target.name}"
        try:
            bp = json.loads(target.read_text())
            title = bp.get("metadata", {}).get("title", "Untitled")
            phases = len(bp.get("phases", []))
            entities = len(bp.get("entities", []))
            # Store blueprint tag in story state for cleanup reference
            tag = re.sub(r'[^a-z0-9_]', '_', title.lower())
            story["active_blueprint"] = str(target.name)
            story["active_blueprint_tag"] = f"dc_blueprint_{tag}"
            _save_story(story)
            return f"Blueprint: {title}\nTag: dc_blueprint_{tag}\nPhases: {phases}\nEntities: {entities}\nFlags: {json.dumps(bp.get('flags', {}))}"
        except Exception as e:
            return f"Error loading blueprint: {e}"

    if action == "check_score":
        player = args.get("player")
        objective = args.get("objective")
        if not player or not objective:
            return "Error: player and objective are required for check_score"
        result = _api_get(f"/scoreboard?objective={objective}&player={player}")
        if not result.get("ok"):
            return _fmt(result)
        data = result.get("data", {})
        score = data.get("score", 0)
        note = data.get("note", "")
        return f"Score for {player} on {objective}: {score}" + (f" ({note})" if note else "")

    if action == "set_score":
        player = args.get("player")
        objective = args.get("objective")
        value = args.get("value", 0)
        if not player or not objective:
            return "Error: player and objective are required for set_score"
        result = _api_post("/chat/send", {"message": f"/scoreboard players set {player} {objective} {value}"})
        return _fmt(result)

    if action == "run_function":
        function = args.get("function")
        if not function:
            return "Error: function path is required for run_function"
        result = _api_post("/chat/send", {"message": f"/function {function}"})
        return _fmt(result)

    if action == "setup_sensors":
        sensors = args.get("sensors", [])
        if not sensors:
            return "Error: sensors list required for setup_sensors"
        created = []
        for s in sensors:
            name = s.get("name")
            criterion = s.get("criterion", "dummy")
            poll_command = s.get("poll_command")
            if not name:
                continue
            # Create scoreboard in Minecraft
            _api_post("/chat/send", {"message": f"/scoreboard objectives add {name} {criterion}"})
            # Register/update in story state
            existing = story.get("active_sensors", [])
            existing = [x for x in existing if x.get("name") != name]
            existing.append({"name": name, "criterion": criterion, "poll_command": poll_command})
            story["active_sensors"] = existing
            created.append(name)
        _save_story(story)
        return f"Sensors created and registered: {created}"

    if action == "poll_sensors":
        player = args.get("player", "@a")
        reset = args.get("reset", True)
        sensors = story.get("active_sensors", [])
        if not sensors:
            return "No active sensors"
        results = []
        for s in sensors:
            name = s.get("name")
            poll_command = s.get("poll_command")
            # Execute poll command for dummy sensors (proximity, zone, etc.)
            if poll_command:
                _api_post("/chat/send", {"message": poll_command})
            # Read score via native API
            result = _api_get(f"/scoreboard?objective={name}&player={player}")
            if result.get("ok"):
                score = result.get("data", {}).get("score", 0)
                fired = score > 0
                if fired and reset:
                    _api_post("/chat/send", {"message": f"/scoreboard players set {player} {name} 0"})
                results.append(f"{name}: {score}" + (" (fired)" if fired else ""))
            else:
                results.append(f"{name}: error")
        return "Sensor poll results:\n" + "\n".join(results)

    if action == "cleanup_sensors":
        targets = args.get("sensors", [])
        sensors = story.get("active_sensors", [])
        if not targets:
            # Default: cleanup all
            targets = [s.get("name") for s in sensors]
        removed = []
        for name in targets:
            _api_post("/chat/send", {"message": f"/scoreboard objectives remove {name}"})
            removed.append(name)
        story["active_sensors"] = [s for s in sensors if s.get("name") not in targets]
        _save_story(story)
        return f"Sensors removed: {removed}. Remaining: {[s['name'] for s in story['active_sensors']]}"

    return f"Error: unknown story action '{action}'"


MC_STORY_SCHEMA = {
    "name": "mc_story",
    "description": "Narrative state tracker for Role Master mode. Tracks story phase, day counter, flags, objectives, events, player choices, and active scoreboard sensors across sessions. Supports phase timeouts, activity tracking, and sensor restoration for quest-like progression. All data persists in a JSON file. No bot connection required.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "get_state", "set_flag", "advance_phase", "advance_day",
                    "add_objective", "complete_objective", "log_event", "get_events",
                    "set_title", "record_choice", "reset",
                    "save_blueprint", "load_blueprint",
                    "record_activity", "check_timeout", "reset_phase",
                    "check_score", "set_score", "run_function",
                    "setup_sensors", "poll_sensors", "cleanup_sensors",
                ],
                "description": "Story management action",
            },
            "key": {"type": "string", "description": "Flag key (for set_flag)"},
            "value": {"type": ["string", "number", "boolean"], "description": "Flag value (for set_flag)"},
            "phase": {"type": "string", "description": "Phase name (for advance_phase or reset_phase)"},
            "timeout_minutes": {"type": "number", "description": "Minutes before phase is abandoned if no player activity (for advance_phase)"},
            "title": {"type": "string", "description": "Objective or story title"},
            "description": {"type": "string", "description": "Objective description"},
            "objective_id": {"type": "number", "description": "Objective index to complete"},
            "event": {"type": "string", "description": "Event description to log"},
            "count": {"type": "number", "description": "Number of recent events to retrieve (for get_events; default: 10)"},
            "player": {"type": "string", "description": "Player name (for record_choice or check_score/set_score)"},
            "choice": {"type": "string", "description": "Choice description (for record_choice)"},
            "optional": {"type": "boolean", "description": "Whether objective is optional"},
            "blueprint": {"type": "object", "description": "Full adventure blueprint JSON (for save_blueprint)"},
            "objective": {"type": "string", "description": "Scoreboard objective name (for check_score/set_score)"},
            "sensors": {
                "type": "array",
                "description": "List of sensor objects for setup_sensors or cleanup_sensors. Each object: {name, criterion, poll_command?}",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "criterion": {"type": "string"},
                        "poll_command": {"type": "string", "description": "Optional /execute command for dummy sensors"},
                    },
                },
            },
            "reset": {"type": "boolean", "description": "Whether to reset fired sensor scores to 0 after polling (for poll_sensors; default: true)"},
            "function": {"type": "string", "description": "Datapack function path (for run_function)"},
        },
        "required": ["action"],
    },
}


MC_REGISTRY_SCHEMA = {
    "name": "mc_registry",
    "description": "Query the shared Minecraft validation registry for canonical lists of biomes, entities, items, blocks, effects, and scoreboard criteria. Use this when you need to know valid values for adventure blueprints (e.g., 'what flying passive mobs exist?', 'what biomes are in the overworld?', 'is crow a valid entity?'). Results are sourced from minecraft-data for the configured server version.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["biomes", "entities", "items", "blocks", "effects", "scoreboard_criteria"],
                "description": "Registry category to query",
            },
            "filter": {"type": "string", "description": "Optional substring filter on name or displayName (case-insensitive)"},
            "limit": {"type": "number", "description": "Max results to return (default 20, max 100)"},
            "type_filter": {"type": "string", "description": "For entities: filter by type (e.g. mob, animal, hostile, passive, ambient)"},
            "dimension": {"type": "string", "description": "For biomes: filter by dimension (overworld, nether, end)"},
        },
        "required": ["category"],
    },
}

def _handle_mc_registry(args: dict, **kwargs) -> str:
    category = args.get("category")
    filt = (args.get("filter") or "").lower()
    limit = min(int(args.get("limit") or 20), 100)
    type_filter = (args.get("type_filter") or "").lower()
    dimension = (args.get("dimension") or "").lower()

    registry_path = Path(__file__).parent.parent / "data" / "minecraft-registry.json"
    if not registry_path.exists():
        return "Error: minecraft-registry.json not found. Run scripts/generate-minecraft-registry.js to create it."

    try:
        registry = json.loads(registry_path.read_text())
    except Exception as e:
        return f"Error reading registry: {e}"

    items = registry.get(category)
    if items is None:
        return f"Error: unknown category '{category}'. Valid: biomes, entities, items, blocks, effects, scoreboard_criteria"

    results = []
    for item in items:
        name = item.get("name", "")
        display = item.get("displayName", "")
        if filt and filt not in name.lower() and filt not in display.lower():
            continue
        if category == "entities" and type_filter:
            if type_filter not in (item.get("type") or "").lower():
                continue
        if category == "biomes" and dimension:
            if dimension not in (item.get("dimension") or "").lower():
                continue
        results.append(item)

    if not results:
        return f"No {category} matched the filters."

    lines = [f"{category} ({len(results)} matches, showing first {min(limit, len(results))}):"]
    for item in results[:limit]:
        if category == "entities":
            lines.append(f"  - {item['name']} ({item.get('displayName','')}) type={item.get('type','')}, category={item.get('category','')}")
        elif category == "biomes":
            lines.append(f"  - {item['name']} ({item.get('displayName','')}) dimension={item.get('dimension','')}")
        elif category == "scoreboard_criteria":
            lines.append(f"  - {item['name']} — {item.get('description','')}")
        else:
            lines.append(f"  - {item['name']} ({item.get('displayName','')})")

    if len(results) > limit:
        lines.append(f"  ... and {len(results) - limit} more")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════════════

registry.register(
    name="mc_perceive",
    toolset="minecraft",
    schema=MC_PERCEIVE_SCHEMA,
    handler=lambda args, **kw: _handle_mc_perceive(args, **kw),
    check_fn=check_minecraft_available,
)
registry.register(
    name="mc_move",
    toolset="minecraft",
    schema=MC_MOVE_SCHEMA,
    handler=lambda args, **kw: _handle_mc_move(args, **kw),
    check_fn=check_minecraft_available,
)
registry.register(
    name="mc_mine",
    toolset="minecraft",
    schema=MC_MINE_SCHEMA,
    handler=lambda args, **kw: _handle_mc_mine(args, **kw),
    check_fn=check_minecraft_available,
)
registry.register(
    name="mc_build",
    toolset="minecraft",
    schema=MC_BUILD_SCHEMA,
    handler=lambda args, **kw: _handle_mc_build(args, **kw),
    check_fn=check_minecraft_available,
)
registry.register(
    name="mc_craft",
    toolset="minecraft",
    schema=MC_CRAFT_SCHEMA,
    handler=lambda args, **kw: _handle_mc_craft(args, **kw),
    check_fn=check_minecraft_available,
)
registry.register(
    name="mc_combat",
    toolset="minecraft",
    schema=MC_COMBAT_SCHEMA,
    handler=lambda args, **kw: _handle_mc_combat(args, **kw),
    check_fn=check_minecraft_available,
)
# ── Environment flag: loop mode suppresses mc_chat registration ──
# The gateway (social layer) needs mc_chat. The loop (body layer) does not.
if not os.getenv("DC_LOOP_MODE"):
    registry.register(
        name="mc_chat",
        toolset="minecraft",
        schema=MC_CHAT_SCHEMA,
        handler=lambda args, **kw: _handle_mc_chat(args, **kw),
        check_fn=check_minecraft_available,
    )
else:
    print("[minecraft_tools] DC_LOOP_MODE=1 — mc_chat tool suppressed for body-only mode", flush=True)
registry.register(
    name="mc_manage",
    toolset="minecraft",
    schema=MC_MANAGE_SCHEMA,
    handler=lambda args, **kw: _handle_mc_manage(args, **kw),
    check_fn=check_minecraft_available,
)
registry.register(
    name="mc_plan",
    toolset="minecraft",
    schema=MC_PLAN_SCHEMA,
    handler=lambda args, **kw: _handle_mc_plan(args, **kw),
    check_fn=check_minecraft_available,
)
registry.register(
    name="mc_screenshot",
    toolset="minecraft",
    schema=MC_SCREENSHOT_SCHEMA,
    handler=lambda args, **kw: _handle_mc_screenshot(args, **kw),
    check_fn=check_minecraft_available,
)
registry.register(
    name="mc_command",
    toolset="minecraft",
    schema=MC_COMMAND_SCHEMA,
    handler=lambda args, **kw: _handle_mc_command(args, **kw),
    check_fn=check_minecraft_available,
)
MC_NOOP_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "Optional reason for choosing no action.",
        },
    },
}

def _handle_mc_noop(args: Dict[str, Any], **kw) -> str:
    """No-op tool for wake-up events where the agent chooses not to react."""
    return "No action taken."


registry.register(
    name="mc_story",
    toolset="minecraft",
    schema=MC_STORY_SCHEMA,
    handler=lambda args, **kw: _handle_mc_story(args, **kw),
    check_fn=check_minecraft_available,
)

registry.register(
    name="mc_registry",
    toolset="minecraft",
    schema=MC_REGISTRY_SCHEMA,
    handler=lambda args, **kw: _handle_mc_registry(args, **kw),
    check_fn=check_minecraft_available,
)

registry.register(
    name="mc_no_op",
    toolset="minecraft",
    schema=MC_NOOP_SCHEMA,
    handler=lambda args, **kw: _handle_mc_noop(args, **kw),
    check_fn=check_minecraft_available,
)
