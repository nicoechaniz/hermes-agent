"""altermundi-memory plugin — read-only access to the collective memory.

Exposes two tools:
- altermundi_search : FTS search over the collective memory index.
- altermundi_doc    : retrieve a full markdown document by doc_id.

The service lives at http://10.10.20.1:8899 and is reachable only from inside
the AlterMundi anyVPN/ZeroTier network. It is read-only: POST/PUT/DELETE are
not accepted.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_BASE_URL = "http://10.10.20.1:8899"
_TIMEOUT = 30


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only so the plugin has no extra deps)
# ---------------------------------------------------------------------------
def _get_json(path: str) -> Dict[str, Any]:
    """Fetch JSON from the collective-memory API."""
    url = f"{_BASE_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"altermundi-memory HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"altermundi-memory unreachable: {e.reason}") from e


def _check_available() -> bool:
    """Service-gated availability check."""
    try:
        with urllib.request.urlopen(f"{_BASE_URL}/health", timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            n_docs = data.get("n_docs", 0)
            return int(n_docs or 0) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
_SEARCH_SCHEMA = {
    "name": "altermundi_search",
    "description": (
        "Search AlterMundi's collective memory (read-only FTS index) for "
        "documents, source files, maps, and bibliographic entries. Returns "
        "ranked results with doc_id, title, project, kind, and a snippet. "
        "Use the doc_id in altermundi_doc to read the full text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "q": {
                "type": "string",
                "description": "Search query (<=512 characters).",
            },
            "k": {
                "type": "integer",
                "description": "Number of results (max 50, default 10).",
                "default": 10,
            },
            "kind": {
                "type": "string",
                "description": "Optional filter: map, source, biblioteca, fs_doc, fs_code, fs_config, fs_pdf, fs_notebook.",
            },
            "project": {
                "type": "string",
                "description": "Optional project filter, e.g. phideus, SAINet, editorial-altermundi.",
            },
        },
        "required": ["q"],
    },
}

_DOC_SCHEMA = {
    "name": "altermundi_doc",
    "description": (
        "Retrieve the full markdown body of a document from AlterMundi's "
        "collective memory by doc_id. The doc_id comes from altermundi_search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "doc_id returned by altermundi_search (e.g. mapa/proyectos/phideus.md).",
            },
        },
        "required": ["id"],
    },
}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _handle_search(args: Dict[str, Any], **_: Any) -> str:
    q = (args.get("q") or "").strip()
    if not q:
        return json.dumps({"success": False, "error": "q is required"}, ensure_ascii=False)
    k = int(args.get("k") or 10)
    k = max(1, min(50, k))
    params: List[tuple] = [("q", q), ("k", str(k))]
    kind = (args.get("kind") or "").strip()
    if kind:
        params.append(("kind", kind))
    project = (args.get("project") or "").strip()
    if project:
        params.append(("project", project))

    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    data = _get_json(f"/search?{query}")
    return json.dumps(data, ensure_ascii=False, indent=2)


def _handle_doc(args: Dict[str, Any], **_: Any) -> str:
    doc_id = (args.get("id") or "").strip()
    if not doc_id:
        return json.dumps({"success": False, "error": "id is required"}, ensure_ascii=False)
    query = urllib.parse.urlencode({"id": doc_id})
    data = _get_json(f"/doc?{query}")
    return json.dumps(data, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------
def register(ctx) -> None:
    ctx.register_tool(
        name="altermundi_search",
        toolset="altermundi",
        schema=_SEARCH_SCHEMA,
        handler=_handle_search,
        check_fn=_check_available,
        emoji="🌐",
    )
    ctx.register_tool(
        name="altermundi_doc",
        toolset="altermundi",
        schema=_DOC_SCHEMA,
        handler=_handle_doc,
        check_fn=_check_available,
        emoji="📄",
    )
