"""Tests for tools.embodied_plan_tool."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import httpx


def test_tool_registered():
    """Importing the module should register the tool in the global registry."""
    from tools.registry import registry
    import tools.embodied_plan_tool  # noqa: F401

    tool = registry.get_entry("embodied_plan")
    assert tool is not None
    assert tool.toolset == "embodiment"
    assert tool.schema["function"]["name"] == "embodied_plan"


def test_handler_rejects_missing_intent():
    from tools.embodied_plan_tool import _handler

    out = _handler({})
    payload = json.loads(out)
    assert payload["ok"] is False
    assert payload["error"]["error_type"] == "missing_intent"


def test_handler_rejects_non_string_intent():
    from tools.embodied_plan_tool import _handler

    out = _handler({"intent": 42})
    payload = json.loads(out)
    assert payload["ok"] is False
    assert payload["error"]["error_type"] == "missing_intent"


def test_handler_posts_intent_to_service():
    """Standard happy path — handler posts to /intent and returns the
    service's response verbatim."""
    from tools.embodied_plan_tool import _handler

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "ok": True,
        "context_id": "abc-123",
        "plan": {
            "body_plan": ["scan", "mine"],
            "checks": ["time=day"],
            "tool_calls": [{"name": "scan_nearby", "arguments": {"radius": 16}}],
            "failure_policy": "ask the player",
            "operational_risk": "low",
        },
        "execution_results": [{"tool": "scan_nearby", "ok": True, "data": {}}],
        "elapsed_seconds": 1.2,
    }
    fake_response.status_code = 200

    captured = {}
    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        captured["timeout"] = timeout
        return fake_response

    with patch("tools.embodied_plan_tool.httpx.post", side_effect=fake_post):
        out = _handler({
            "intent": "Help the player gather wood before night.",
            "autonomy_level": 2,
            "allowed_tools": ["scan_nearby", "mine_block"],
        })

    assert captured["url"].endswith("/intent")
    assert captured["body"]["intent"] == "Help the player gather wood before night."
    assert captured["body"]["autonomy_level"] == 2
    assert captured["body"]["allowed_tools"] == ["scan_nearby", "mine_block"]
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["plan"]["operational_risk"] == "low"


def test_handler_omits_none_optional_fields():
    """Optional fields that are None must NOT be in the request body — the
    service treats absence as 'use default', not as 'use None'."""
    from tools.embodied_plan_tool import _handler

    captured = {}
    def fake_post(url, json=None, timeout=None):
        captured["body"] = json
        resp = MagicMock()
        resp.json.return_value = {"ok": True}
        resp.status_code = 200
        return resp

    with patch("tools.embodied_plan_tool.httpx.post", side_effect=fake_post):
        _handler({
            "intent": "Do a thing.",
            "previous_error": None,  # explicitly None — should not be forwarded
        })

    assert "intent" in captured["body"]
    assert "previous_error" not in captured["body"]


def test_handler_handles_timeout():
    from tools.embodied_plan_tool import _handler

    with patch("tools.embodied_plan_tool.httpx.post",
               side_effect=httpx.TimeoutException("request timed out")):
        out = _handler({"intent": "test"})
    payload = json.loads(out)
    assert payload["ok"] is False
    assert payload["error"]["error_type"] == "embodied_service_timeout"


def test_handler_handles_connection_error():
    from tools.embodied_plan_tool import _handler

    with patch("tools.embodied_plan_tool.httpx.post",
               side_effect=httpx.ConnectError("connection refused")):
        out = _handler({"intent": "test"})
    payload = json.loads(out)
    assert payload["ok"] is False
    assert payload["error"]["error_type"] == "embodied_service_unreachable"


def test_handler_handles_non_json_response():
    from tools.embodied_plan_tool import _handler

    fake_response = MagicMock()
    fake_response.json.side_effect = json.JSONDecodeError("bad", "", 0)
    fake_response.status_code = 502
    fake_response.text = "<html>bad gateway</html>"

    with patch("tools.embodied_plan_tool.httpx.post", return_value=fake_response):
        out = _handler({"intent": "test"})
    payload = json.loads(out)
    assert payload["ok"] is False
    assert payload["error"]["error_type"] == "embodied_service_bad_response"


def test_check_service_available_validates_url():
    from tools.embodied_plan_tool import _check_service_available

    assert _check_service_available() is True  # default http://localhost:7790


def test_service_url_respects_env(monkeypatch):
    monkeypatch.setenv("EMBODIED_SERVICE_URL", "http://10.10.20.5:7790")
    from tools.embodied_plan_tool import _service_url

    assert _service_url() == "http://10.10.20.5:7790"


# ─── Policy-mode tests ────────────────────────────────────────────────────────

def test_policy_mode_raw_explicit():
    """Explicit 'raw' should bypass the policy layer entirely."""
    from tools.embodied_plan_tool import _handler

    captured = {}
    def fake_post(url, json=None, timeout=None):
        captured["body"] = json
        resp = MagicMock()
        resp.json.return_value = {"ok": True, "plan": {}, "execution_results": []}
        resp.status_code = 200
        return resp

    with patch("tools.embodied_plan_tool.httpx.post", side_effect=fake_post):
        out = _handler({"intent": "Mine 1 oak_log", "policy_mode": "raw"})

    assert captured["body"]["intent"] == "Mine 1 oak_log"
    payload = json.loads(out)
    assert payload["ok"] is True


def test_policy_scope_filter_blocks_joke(monkeypatch):
    """Out-of-scope chat (joke) must be handled upstream — no HTTP call."""
    monkeypatch.setenv("BOT_API_URL", "http://bot:3000")
    from tools.embodied_plan_tool import _handler

    calls = []
    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        return MagicMock()

    with patch("tools.embodied_plan_tool.httpx.post", side_effect=fake_post):
        out = _handler({"intent": "Contame un chiste corto", "policy_mode": "auto"})

    assert len(calls) == 0
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["outcome"] == "policy_handled_upstream"
    assert payload["policy_layer"] == "scope"
    assert payload["mitigation"]["intent_original"] == "Contame un chiste corto"


def test_policy_ambiguity_blocks_vague(monkeypatch):
    """Vague intents must be handled upstream — no HTTP call."""
    monkeypatch.setenv("BOT_API_URL", "http://bot:3000")
    from tools.embodied_plan_tool import _handler

    calls = []
    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        return MagicMock()

    with patch("tools.embodied_plan_tool.httpx.post", side_effect=fake_post):
        out = _handler({"intent": "Hacé algo entretenido", "policy_mode": "auto"})

    assert len(calls) == 0
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["outcome"] == "policy_handled_upstream"
    assert payload["policy_layer"] == "ambiguity"


def test_policy_decomposes_and_normalizes(monkeypatch):
    """Multi-step intents are split, normalized, and posted sequentially."""
    monkeypatch.setenv("BOT_API_URL", "http://bot:3000")
    from tools.embodied_plan_tool import _handler

    calls = []
    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        resp = MagicMock()
        resp.json.return_value = {
            "ok": True,
            "plan": {"body_plan": []},
            "execution_results": [{"tool": "scan_nearby", "ok": True}],
        }
        resp.status_code = 200
        return resp

    with patch("tools.embodied_plan_tool.httpx.post", side_effect=fake_post):
        out = _handler({
            "intent": "Mine 3 oak_log and craft a crafting table",
            "policy_mode": "auto",
        })

    assert len(calls) == 2
    # First sub-intent should be normalized English imperative
    assert calls[0]["intent"].startswith("Mine")
    # Second sub-intent should also be normalized
    assert calls[1]["intent"].startswith("Craft")

    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["outcome"] == "embodied_ready"
    assert len(payload["mitigation"]["normalized_chain"]) == 2
    assert payload["mitigation"]["category_chain"][0] == "mining"
    assert len(payload["execution_results"]) == 2


def test_policy_narrows_tools(monkeypatch):
    """Policy should narrow allowed_tools by intent category."""
    monkeypatch.setenv("BOT_API_URL", "http://bot:3000")
    from tools.embodied_plan_tool import _handler

    calls = []
    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        resp = MagicMock()
        resp.json.return_value = {"ok": True, "plan": {}, "execution_results": []}
        resp.status_code = 200
        return resp

    with patch("tools.embodied_plan_tool.httpx.post", side_effect=fake_post):
        out = _handler({"intent": "Equip torch", "policy_mode": "auto"})

    assert len(calls) == 1
    narrowed = calls[0].get("allowed_tools", [])
    assert "equip_item" in narrowed
    assert "mine_block" not in narrowed

    payload = json.loads(out)
    assert payload["mitigation"]["allowed_tools_chain"][0] == narrowed


def test_policy_threads_previous_error(monkeypatch):
    """If a sub-intent fails, previous_error is threaded into the next call."""
    monkeypatch.setenv("BOT_API_URL", "http://bot:3000")
    from tools.embodied_plan_tool import _handler

    call_idx = 0
    def fake_post(url, json=None, timeout=None):
        nonlocal call_idx
        call_idx += 1
        resp = MagicMock()
        if call_idx == 1:
            resp.json.return_value = {
                "ok": False,
                "error": {
                    "error_type": "missing_material",
                    "details": "no oak_log in inventory",
                },
            }
        else:
            resp.json.return_value = {
                "ok": True,
                "plan": {},
                "execution_results": [{"tool": "scan_nearby", "ok": True}],
            }
        resp.status_code = 200
        return resp

    with patch("tools.embodied_plan_tool.httpx.post", side_effect=fake_post):
        out = _handler({
            "intent": "Mine 3 oak_log and craft a crafting table",
            "policy_mode": "auto",
        })

    payload = json.loads(out)
    assert payload["ok"] is True
    assert len(payload["execution_results"]) == 2
    assert payload["execution_results"][0]["ok"] is False
    assert payload["execution_results"][1]["ok"] is True
