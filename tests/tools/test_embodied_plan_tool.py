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


# ---------------------------------------------------------------------------
# Narrative replan composition (workaround for `previous_error` regression
# verified by primitives_lab experiment 007).
# ---------------------------------------------------------------------------


def test_compose_replan_intent_includes_tool_and_error_type():
    from tools.embodied_plan_tool import _compose_replan_intent

    out = _compose_replan_intent(
        "build a wall with 4 oak_planks at the player",
        {"tool": "place_block", "error_type": "missing_material",
         "details": "no oak_log; have oak_planks(40)"},
    )
    assert "place_block" in out
    assert "missing_material" in out
    assert "no oak_log" in out
    # Original intent appears in past-tense framing
    assert "tried to build a wall with 4 oak_planks at the player" in out


def test_compose_replan_intent_uses_past_tense_framing():
    """The original (failing) intent must appear in past-tense framing — the
    model has a strong last-instruction bias and we need the recovery
    directive to be the active imperative, not the failing instruction."""
    from tools.embodied_plan_tool import _compose_replan_intent

    out = _compose_replan_intent(
        "build a wall with 4 oak_log blocks",
        {"tool": "place_block", "error_type": "missing_material",
         "details": "no oak_log in inventory; have oak_planks(40)."},
    )
    # Original intent is wrapped in "We tried to ...", not stated as a goal
    assert "tried to build a wall with 4 oak_log blocks" in out
    # The closing sentence must be the recovery directive
    sentences = [s.strip() for s in out.split(". ") if s.strip()]
    last = sentences[-1].rstrip(".")
    assert "oak_log" not in last, f"failing material in trailing directive: {last!r}"
    assert any(kw in last.lower() for kw in ("compose", "plan", "re-emit", "avoid"))


def test_compose_replan_intent_handles_missing_fields():
    from tools.embodied_plan_tool import _compose_replan_intent

    out = _compose_replan_intent("retry", {})
    assert "the previous action" in out  # fallback
    assert "failed" in out
    assert "retry" in out


def test_handler_prepends_narrative_when_previous_error_present():
    """When previous_error is set, the handler must rewrite the intent on the
    wire to embed the failure narrative — Gemma-Andy ignores the structured
    field but honors in-intent narration (validated by 007 at n=10)."""
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
            "intent": "Place 4 blocks at the player.",
            "previous_error": {
                "tool": "place_block",
                "error_type": "missing_material",
                "details": "no oak_log in inventory; have oak_planks(40).",
            },
        })

    sent_intent = captured["body"]["intent"]
    # Original intent appears, wrapped in past-tense framing
    assert "tried to Place 4 blocks at the player" in sent_intent
    # Failure narrative embedded
    assert "place_block" in sent_intent
    assert "missing_material" in sent_intent
    assert "no oak_log" in sent_intent
    # Structured field dropped — see handler comment for rationale
    # (avoids daemoncraft `recovery_naive_retry` false-positive)
    assert "previous_error" not in captured["body"]


def test_handler_does_not_modify_intent_when_no_previous_error():
    from tools.embodied_plan_tool import _handler

    captured = {}
    def fake_post(url, json=None, timeout=None):
        captured["body"] = json
        resp = MagicMock()
        resp.json.return_value = {"ok": True}
        resp.status_code = 200
        return resp

    with patch("tools.embodied_plan_tool.httpx.post", side_effect=fake_post):
        _handler({"intent": "Toss 2 oak_planks to the player."})

    assert captured["body"]["intent"] == "Toss 2 oak_planks to the player."
    assert "previous_error" not in captured["body"]


def test_handler_skips_narrative_when_previous_error_is_empty_dict():
    """Defensive: an empty dict shouldn't trigger the narrative prepend."""
    from tools.embodied_plan_tool import _handler

    captured = {}
    def fake_post(url, json=None, timeout=None):
        captured["body"] = json
        resp = MagicMock()
        resp.json.return_value = {"ok": True}
        resp.status_code = 200
        return resp

    with patch("tools.embodied_plan_tool.httpx.post", side_effect=fake_post):
        _handler({"intent": "Do a thing.", "previous_error": {}})

    assert captured["body"]["intent"] == "Do a thing."
    # Empty dict is treated as "no error info" — neither prepended nor forwarded.
    assert "previous_error" not in captured["body"] or captured["body"].get("previous_error") == {}
