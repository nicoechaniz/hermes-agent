#!/usr/bin/env python3
"""mc_navigate — semantic + geometric perception macros for Hermes.

Calls the bot server's GET /navigate?action=... endpoint. Returns
structured JSON for the LLM to consume (no grid parsing required).

Actions (11 total):

  SEMANTIC (5) — answer high-level questions:
    identify_cave     — am I in a cave? (escape direction, sky access)
    identify_interior — am I inside a structure? (enriched: doors, safety, furni)
    find_doors        — list all doors in radius
    verify_door       — check a specific door's state
    scan_structure    — full structure context

  GEOMETRIC (5) — answer spatial questions:
    walkable          — list of cells the bot can stand on
    path_to           — run pathfinder with timeout
    corners           — 4 corner cells of walkable area
    escape_routes     — cardinal directions with distance + blocker
    structure_outline — bounding boxes of distinct structures

  EXACT (1) — companion to type-based CJK visual:
    verify_block      — exact block name at a position (the 10% case)

  LEGEND (1):
    visual_legend     — canonical block→char mapping (server source of truth)

Type-based CJK mapping: see SOUL_daemoncraft §Perception Macros.
"""
from __future__ import annotations

import json
import sys
from typing import Any

import httpx


def _bot_url() -> str:
    import os
    return (
        os.environ.get("BOT_API_URL")
        or os.environ.get("MC_API_URL")
        or "http://localhost:3003"
    )


def _handler(args: dict[str, Any] | None = None, **_kw: Any) -> str:
    args = args or {}
    action = args.get("action")
    if not action:
        return (
            "Error: mc_navigate requires an 'action' parameter. "
            "Valid actions: identify_cave, identify_interior, find_doors, "
            "verify_door, scan_structure, walkable, path_to, corners, "
            "escape_routes, structure_outline, verify_block, visual_legend."
        )

    # Build query params from args (skip action since it goes in path)
    params: dict[str, Any] = {"action": action}
    for k, v in args.items():
        if k == "action":
            continue
        params[k] = v

    try:
        resp = httpx.get(f"{_bot_url()}/navigate", params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"mc_navigate error: {exc}"

    if not data.get("ok"):
        return f"mc_navigate error: {data.get('error', 'unknown')}"

    d = data.get("data", {})

    # Compact summaries for the most common actions
    if action == "identify_cave":
        lines = ["identify_cave:"]
        lines.append(f"  is_cave: {d.get('is_cave')}")
        if d.get("is_cave"):
            lines.append(f"  ceiling_height: {d.get('ceiling_height')}")
            lines.append(f"  has_sky_access: {d.get('has_sky_access')}")
            lines.append(f"  sky_light: {d.get('sky_light')}")
            lines.append(f"  exit_direction: {d.get('exit_direction')}")
            lines.append(f"  escape_tools: {d.get('escape_tools')}")
        lines.append(f"  depth_blocks: {d.get('depth_blocks')}")
        return "\n".join(lines)

    if action == "identify_interior":
        lines = ["identify_interior:"]
        lines.append(f"  is_interior: {d.get('is_interior')}")
        lines.append(f"  structure_type: {d.get('structure_type')}")
        if d.get("is_interior"):
            lines.append(f"  ceiling_height: {d.get('ceiling_height')}")
            lines.append(f"  wall_count: {d.get('wall_count')}")
            lines.append(f"  volume_blocks: {d.get('volume_blocks')}")
            ap = d.get("access_points", [])
            if ap:
                lines.append(f"  access_points: {len(ap)} (open={sum(1 for x in ap if x.get('is_open'))})")
            else:
                lines.append(f"  access_points: 0")
            mb = d.get("missing_blocks", [])
            lines.append(f"  missing_blocks: {len(mb)}")
            furni = d.get("furni", {})
            if furni:
                lines.append(f"  furni: {furni}")
            lines.append(f"  hostile_presence: {d.get('hostile_presence')}")
            lines.append(f"  is_safe: {d.get('is_safe')}")
            issues = d.get("safety_issues", [])
            if issues:
                lines.append(f"  safety_issues: {issues[:5]}")
        return "\n".join(lines)

    if action == "find_doors":
        doors = d.get("doors", [])
        lines = [f"find_doors: {len(doors)} door(s)"]
        for door in doors[:5]:
            lines.append(f"  {door.get('position')}: {door.get('type')} "
                         f"is_open={door.get('is_open')} blocking={door.get('is_blocking')}")
        return "\n".join(lines)

    if action == "verify_door":
        return (f"verify_door: {d.get('type')} is_open={d.get('is_open')} "
                f"hinge={d.get('hinge_side')} has_top={d.get('has_door_top')}")

    if action == "scan_structure":
        return json.dumps(d, indent=2)[:2000]

    if action == "walkable":
        cells = d.get("cells", [])
        return f"walkable: {len(cells)} cells (capped at 256)"

    if action == "path_to":
        lines = ["path_to:"]
        lines.append(f"  reachable: {d.get('reachable')}")
        if d.get("reachable"):
            wp = d.get("waypoints", [])
            lines.append(f"  waypoints: {len(wp)}")
            lines.append(f"  distance: {d.get('distance', '?')}")
        else:
            lines.append(f"  reason: {d.get('reason', '?')}")
        return "\n".join(lines)

    if action == "corners":
        corners = d.get("corners", [])
        lines = [f"corners: {len(corners)} corner(s)"]
        for c in corners[:4]:
            lines.append(f"  {c}")
        return "\n".join(lines)

    if action == "escape_routes":
        lines = ["escape_routes:"]
        lines.append(f"  best_escape: {d.get('best_escape')}")
        cards = d.get("cardinals", {})
        for k in ("north", "south", "east", "west", "up", "down"):
            v = cards.get(k, {})
            if v:
                lines.append(f"  {k}: free={v.get('free')} distance={v.get('distance', '?')}")
        return "\n".join(lines)

    if action == "structure_outline":
        bboxes = d.get("bounding_boxes", [])
        return f"structure_outline: {len(bboxes)} structure(s)"

    if action == "verify_block":
        return (f"verify_block: {d.get('position')} = {d.get('block')} "
                f"(category={d.get('category')}, walkable={d.get('is_walkable')})")

    if action == "visual_legend":
        # Compact: char → first block name
        mapping = d.get("mapping", {})
        char_to_name = d.get("char_to_names", {})
        lines = [f"visual_legend: {d.get('block_count')} blocks → "
                 f"{d.get('char_count')} distinct chars"]
        for ch, name in list(char_to_name.items())[:8]:
            extras = d.get("mapping", {}).get(name, {}).get("+more", 0)
            if extras:
                lines.append(f"  {ch} = {name} (+{extras} more)")
            else:
                lines.append(f"  {ch} = {name}")
        if len(char_to_name) > 8:
            lines.append(f"  ... and {len(char_to_name) - 8} more")
        return "\n".join(lines)

    # Default: dump JSON
    return json.dumps(d, indent=2)[:3000]


# Registry import
try:
    from tools.registry import registry
except ImportError:
    # Fallback for direct execution / tests
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from hermes_cli.tools import registry


registry.register(
    name="mc_navigate",
    toolset="embodiment",
    schema={
        "type": "function",
        "function": {
            "name": "mc_navigate",
            "description": (
                "Semantic + geometric perception macros for the Minecraft world. "
                "Returns structured JSON instead of text grids. Use this INSTEAD "
                "of parsing mc_bit output for any of these questions:\n\n"
                "Semantic (5):\n"
                "- identify_cave: am I in a cave?\n"
                "- identify_interior: am I inside a structure? (returns access_points, missing_blocks, furni, hostile_presence, is_safe, safety_issues)\n"
                "- find_doors: list all doors in radius with is_open state\n"
                "- verify_door: check a specific door\n"
                "- scan_structure: full structure context (interior + doors + furni + safety)\n\n"
                "Geometric (5):\n"
                "- walkable: list of cells the bot can stand on\n"
                "- path_to: run pathfinder with timeout, returns reachable + waypoints\n"
                "- corners: 4 corner cells of walkable area\n"
                "- escape_routes: cardinal directions with distance + blocker + best_escape\n"
                "- structure_outline: bounding boxes of distinct structures\n\n"
                "Exact (1):\n"
                "- verify_block: exact block name at a position (10% case where the type-based CJK in mbit isn't enough)\n\n"
                "Legend (1):\n"
                "- visual_legend: canonical block→char mapping (server source of truth)\n\n"
                "Required: action. Optional: x, y, z (anchor position), radius, target_x, target_y, target_z.\n\n"
                "See SOUL_daemoncraft §Perception Macros for the decision rule."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "identify_cave", "identify_interior", "find_doors",
                            "verify_door", "scan_structure", "walkable", "path_to",
                            "corners", "escape_routes", "structure_outline",
                            "verify_block", "visual_legend",
                        ],
                        "description": "Which perception macro to run",
                    },
                    "x": {"type": "number", "description": "Anchor X (default: bot position)"},
                    "y": {"type": "number", "description": "Anchor Y (default: bot position)"},
                    "z": {"type": "number", "description": "Anchor Z (default: bot position)"},
                    "radius": {"type": "number", "description": "Scan radius in blocks (default varies by action)"},
                    "target_x": {"type": "number", "description": "Pathfinder target X (path_to)"},
                    "target_y": {"type": "number", "description": "Pathfinder target Y (path_to)"},
                    "target_z": {"type": "number", "description": "Pathfinder target Z (path_to)"},
                },
                "required": ["action"],
            },
        },
    },
    handler=_handler,
    emoji="🧭",
    description="Semantic + geometric perception macros (identify_cave, find_doors, walkable, path_to, etc.)",
)
