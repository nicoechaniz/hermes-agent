#!/usr/bin/env python3
"""mc_bit — mBit chunk perception for Hermes.

Queries the bot server's GET /blocks endpoint with mBit format encoding.
Returns text-native spatial representation of a 16×16×16 chunk.

Formats:
  binary  — walkable (0) / solid (1) per (X,Z), 256 chars
  columns — (UP free, DOWN solid) per column, 256 pairs
  rows    — free blocks in N,S,E,W,Up,Down from center
  surface — ground block type per (X,Z), 256 chars
  full    — every block as single char (4096 chars, Y-major)
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from tools.bot_api_url_ctx import get_bot_api_url
from tools.registry import registry


def _bot_url() -> str:
    return get_bot_api_url().rstrip("/")


async def _handler(args: dict[str, Any] | None = None, **_kw: Any) -> str:
    args = args or {}
    x1 = args.get("x1")
    y1 = args.get("y1")
    z1 = args.get("z1")
    x2 = args.get("x2")
    y2 = args.get("y2")
    z2 = args.get("z2")
    fmt = args.get("format", "binary")

    if None in (x1, y1, z1, x2, y2, z2):
        return "Error: mc_bit requires x1, y1, z1, x2, y2, z2. Use mc_perceive(type='status') to get bot position first."

    url = f"{_bot_url()}/blocks"
    params = {
        "x1": int(x1), "y1": int(y1), "z1": int(z1),
        "x2": int(x2), "y2": int(y2), "z2": int(z2),
        "format": fmt,
    }
    if args.get("cx") is not None:
        params["cx"] = int(args["cx"])
    if args.get("cz") is not None:
        params["cz"] = int(args["cz"])

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return f"mc_bit error: {e}"

    if not data.get("ok"):
        return f"mc_bit error: {data.get('error', 'unknown')}"

    d = data["data"]
    text = d.get("text", "")
    count = d.get("count", 0)
    elapsed = d.get("elapsed_ms", 0)

    return f"mBit {fmt} ({count} blocks, {elapsed}ms):\n{text}"


# Registration
registry.register(
    name="mc_bit",
    toolset="embodiment",
    schema={
        "type": "function",
        "function": {
            "name": "mc_bit",
            "description": (
                "Perceive a 3D chunk of the Minecraft world as text using mBit format. "
                "Returns a spatial text representation of blocks in the given volume. "
                "Use this INSTEAD of mc_perceive(type='nearby') when you need spatial "
                "awareness — to understand terrain, find paths, avoid holes, plan builds, "
                "or verify the bot's surroundings before acting.\n\n"
                "Formats:\n"
                "- binary: walkable (0) / solid (1) map, 256 chars (best for navigation)\n"
                "- columns: (UP free, DOWN solid) per column, 256 pairs (terrain profile)\n"
                "- rows: free distance in N,S,E,W,Up,Down from center (horizon scan)\n"
                "- surface: ground block type per (X,Z), 256 chars (what's on the ground)\n"
                "- full: every block as single char, 4096 chars Y-major (exact world view)\n\n"
                "TIP: Use mc_bit(format='binary') BEFORE any mc_move or mc_build to check "
                "if the destination is walkable and not a pit. Use format='surface' to "
                "check ground type before tilling or placing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {"type": "integer", "description": "Min X coordinate (bot X - 7 for 16-wide chunk)"},
                    "y1": {"type": "integer", "description": "Min Y coordinate (bot Y - 2 for below-feet context)"},
                    "z1": {"type": "integer", "description": "Min Z coordinate (bot Z - 7 for 16-wide chunk)"},
                    "x2": {"type": "integer", "description": "Max X coordinate (bot X + 8 for 16-wide chunk)"},
                    "y2": {"type": "integer", "description": "Max Y coordinate (bot Y + 13 for above-head context)"},
                    "z2": {"type": "integer", "description": "Max Z coordinate (bot Z + 8 for 16-wide chunk)"},
                    "format": {
                        "type": "string",
                        "enum": ["binary", "columns", "rows", "surface", "full"],
                        "description": "mBit format. Default: binary.",
                        "default": "binary",
                    },
                    "cx": {"type": "integer", "description": "Center X for rows format"},
                    "cz": {"type": "integer", "description": "Center Z for rows format"},
                },
                "required": ["x1", "y1", "z1", "x2", "y2", "z2"],
            },
        },
    },
    handler=_handler,
    emoji="🧊",
    description="Perceive a 3D Minecraft chunk as text (mBit format)",
)
