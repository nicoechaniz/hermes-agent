#!/usr/bin/env python3
"""mc_bit — mBit chunk perception for Hermes.

Queries the bot server's GET /blocks endpoint with mBit format encoding.
Returns text-native spatial representation of a Minecraft volume.

Formats:
  binary  — walkable (0) / solid (1) per (X,Z), compact navigation map
  columns — terrain profile per column
  rows    — free blocks in N,S,E,W,Up,Down from center
  surface — ground block type per (X,Z)
  full    — every block as single char, Y-major, best for exact diff checks
"""

from __future__ import annotations

from typing import Any

import httpx

from tools.bot_api_url_ctx import get_bot_api_url
from tools.registry import registry


VALID_FORMATS = {"binary", "columns", "rows", "surface", "full"}


def _bot_url() -> str:
    return get_bot_api_url().rstrip("/")


def _missing_required(args: dict[str, Any]) -> list[str]:
    return [name for name in ("x1", "y1", "z1", "x2", "y2", "z2") if args.get(name) is None]


def _handler(args: dict[str, Any] | None = None, **_kw: Any) -> str:
    """Synchronous registry handler.

    Hermes tool dispatch currently expects a concrete string result from normal
    tool handlers. The first deploy-only version used ``async def`` and returned
    a coroutine object, which surfaced as ``object of type 'coroutine' has no
    len()`` in live tool calls. Keep this handler synchronous unless the tool
    registry grows first-class async support.
    """
    args = args or {}
    missing = _missing_required(args)
    if missing:
        return (
            "Error: mc_bit requires x1, y1, z1, x2, y2, z2 "
            f"(missing: {', '.join(missing)}). Use mc_perceive(type='status') "
            "to get bot position first."
        )

    fmt = str(args.get("format", "binary"))
    if fmt not in VALID_FORMATS:
        return f"mc_bit error: unsupported format {fmt!r}. Valid formats: {', '.join(sorted(VALID_FORMATS))}"

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
                "Perceive a 3D chunk of the Minecraft world as text using mBit format. "
                "Returns a spatial text representation of blocks in the given volume. "
                "Use this INSTEAD of mc_perceive(type='nearby') when you need spatial "
                "awareness — to understand terrain, find paths, avoid holes, plan builds, "
                "or verify the bot's surroundings before acting.\n\n"
                "Formats:\n"
                "- binary: walkable (0) / solid (1) map (best for navigation)\n"
                "- columns: terrain profile per column\n"
                "- rows: free distance in N,S,E,W,Up,Down from center (horizon scan)\n"
                "- surface: ground block type per (X,Z) (what's on the ground)\n"
                "- full: every block as single char, Y-major (exact world view / diff verification)\n\n"
                "TIP: Use mc_bit(format='binary') BEFORE movement to check walkability. "
                "Use format='surface' before tilling/building. Use small format='full' volumes "
                "for exact before/after verification."
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
