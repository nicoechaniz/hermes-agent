#!/usr/bin/env python3
"""mc_bit — mBit chunk perception for Hermes.

Queries the bot server's GET /blocks endpoint with mBit visual encoding.
Returns a text-native spatial representation of a Minecraft volume with
1 unique character per block, no symbol collisions, and a legend showing
which block each character represents.

The visual format distinguishes all 1166 vanilla Minecraft 1.21 blocks:
- yellow_terracotta, brown_terracotta, orange_terracotta, red_terracotta
  each get their own character (the old 'full' format collapsed all 16
  terracotta colors to 'T').
- door types (oak, iron, spruce, etc.) all share '◫' (door = door).
- chest / trapped_chest / ender_chest all share '◰' (chest = chest).
- furnace / blast_furnace / smoker all share '⊡' (furnace = furnace).
- crafting_table / cartography_table / smithing_table / fletching_table / loom
  all share '⊞' (crafting = crafting).
- beds (16 colors) all share '⊏' (bed = bed).
- glass types (18) all share '▢' (glass = glass).
- Mnemonic overrides for super-common blocks: air→' ', water→'~', lava→'!',
  redstone_wire→'R', torch→'†', lantern→'◊'.
- The remaining ~1090 block names get unique CJK Unified Ideographs
  (U+4E00+) assigned alphabetically and deterministically.

For pathfinding ground truth, use `format='binary'` which gives a 0/1
walkability grid (0=walkable, 1=solid, Y-major). The server supports
this as a separate endpoint for performance; it does not use the visual
char mapping.

For quick cardinal clearances, use mc_perceive(type='scene') instead.
For bot state / inventory, use mc_perceive(type='status') or 'nearby'.
"""

from __future__ import annotations

from typing import Any

import httpx

from tools.bot_api_url_ctx import get_bot_api_url
from tools.registry import registry


def _bot_url() -> str:
    return get_bot_api_url().rstrip("/")


def _missing_required(args: dict[str, Any]) -> list[str]:
    return [name for name in ("x1", "y1", "z1", "x2", "y2", "z2") if args.get(name) is None]


def _handler(args: dict[str, Any] | None = None, **_kw: Any) -> str:
    args = args or {}
    missing = _missing_required(args)
    if missing:
        return (
            "Error: mc_bit requires x1, y1, z1, x2, y2, z2 "
            f"(missing: {', '.join(missing)}). Use mc_perceive(type='status') "
            "to get bot position first."
        )

    fmt = str(args.get("format", "visual"))
    if fmt not in ("visual",):
        return (
            f"mc_bit error: unsupported format {fmt!r}. "
            "The only supported format is 'visual' (1 unique char per block, no collisions, with legend). "
            "For walkability ground truth, parse the visual output — walkable blocks are: ' ' (air), "
            "'~' (water), '!' (lava), ',' (short_grass), ';' (tall_grass), '†' (torch), '◊' (lantern), "
            "and the CJK chars mapped to other plants/leaves. Or use mc_perceive(type='scene') for cardinal clearances."
        )

    try:
        params: dict[str, int | str] = {
            "x1": int(args["x1"]),
            "y1": int(args["y1"]),
            "z1": int(args["z1"]),
            "x2": int(args["x2"]),
            "y2": int(args["y2"]),
            "z2": int(args["z2"]),
            "format": fmt,
        }
        if args.get("cx") is not None:
            params["cx"] = int(args["cx"])
        if args.get("cz") is not None:
            params["cz"] = int(args["cz"])
    except (TypeError, ValueError) as exc:
        return f"mc_bit error: coordinates must be integers ({exc})"

    try:
        resp = httpx.get(f"{_bot_url()}/blocks", params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"mc_bit error: {exc}"

    if not data.get("ok"):
        return f"mc_bit error: {data.get('error', 'unknown')}"

    d = data["data"]
    text = d.get("text", "")
    count = d.get("count", 0)
    elapsed = d.get("elapsed_ms", 0)
    return f"mBit {fmt} ({count} blocks, {elapsed}ms):\n{text}"


registry.register(
    name="mc_bit",
    toolset="embodiment",
    schema={
        "type": "function",
        "function": {
            "name": "mc_bit",
            "description": (
                "Perceive a 3D chunk of the Minecraft world as text using the mBit 'visual' format. "
                "Returns a spatial text representation of blocks in the given volume with 1 unique character per block, "
                "no symbol collisions, and a legend at the bottom showing which block each character represents.\n\n"
                "The visual format distinguishes all 1166 vanilla Minecraft 1.21 blocks (yellow_terracotta ≠ brown_terracotta ≠ "
                "orange_terracotta etc.). Door types share '◫', chest types share '◰', furnace types share '⊡', "
                "crafting tables share '⊞', beds share '⊏', glass types share '▢'. Mnemonic chars for super-common blocks: "
                "air→' ', water→'~', lava→'!', redstone_wire→'R', torch→'†', lantern→'◊'. The remaining ~1090 block names "
                "get unique CJK Unified Ideographs.\n\n"
                "Use this ONLY when you need a raw block grid (spatial awareness over a volume) — to understand terrain layout, "
                "plan builds, or verify exact block placement before/after acting. For bot state, inventory, nearby entities, "
                "chat, or quick status checks, use mc_perceive instead.\n\n"
                "Formats:\n"
                "- visual: 1 unique char per block with a legend. The only supported format. "
                "Best for distinguishing block types (yellow_terracotta ≠ brown_terracotta ≠ orange_terracotta etc.).\n\n"
                "Walkable blocks in the visual output: ' ' (air, cave_air, void_air), '~' (water), "
                "'!' (lava), ',' (short_grass), ';' (tall_grass), '†' (torch/wall_torch/soul_torch), "
                "'◊' (lantern/soul_lantern), and the CJK chars mapped to other plants/leaves. "
                "For quick cardinal clearances, use mc_perceive(type='scene').\n\n"
                "TIP: scan a small volume (≤8x8x8 = 512 blocks) when you need exact block-level awareness — "
                "larger volumes return lots of chars to read."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {"type": "integer", "description": "Min X coordinate"},
                    "y1": {"type": "integer", "description": "Min Y coordinate"},
                    "z1": {"type": "integer", "description": "Min Z coordinate"},
                    "x2": {"type": "integer", "description": "Max X coordinate"},
                    "y2": {"type": "integer", "description": "Max Y coordinate"},
                    "z2": {"type": "integer", "description": "Max Z coordinate"},
                    "format": {
                        "type": "string",
                        "enum": ["visual"],
                        "description": "mBit format. 'visual' is the only supported format (1 unique char per block, no collisions, with legend).",
                        "default": "visual",
                    },
                    "cx": {"type": "integer", "description": "Center X for context (optional)"},
                    "cz": {"type": "integer", "description": "Center Z for context (optional)"},
                },
                "required": ["x1", "y1", "z1", "x2", "y2", "z2"],
            },
        },
    },
    handler=_handler,
    emoji="🧊",
    description="Perceive a 3D Minecraft chunk as text (mBit visual format, no collisions)",
)
