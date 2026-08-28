"""TUI tools.show must reflect the model-facing native-memory gate."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {},
        },
    }


@pytest.fixture()
def server(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")

    methods = dict(mod._methods)
    yield mod
    mod._methods.clear()
    mod._methods.update(methods)
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()
    setattr(mod, "_db", None)


def _tool_names(response: dict) -> set[str]:
    return {
        tool["name"]
        for section in response["result"]["sections"]
        for tool in section["tools"]
    }


def test_tui_tools_show_hides_native_memory_for_external_only_session(
    server, monkeypatch
):
    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **kwargs: [_tool("memory"), _tool("librarian")],
    )
    monkeypatch.setattr(model_tools, "get_toolset_for_tool", lambda name: "memory")
    server._sessions["sid"] = {
        "agent": SimpleNamespace(
            enabled_toolsets=["memory"],
            _memory_enabled=False,
            _user_profile_enabled=False,
        )
    }

    response = server._methods["tools.show"](1, {"session_id": "sid"})

    assert _tool_names(response) == {"librarian"}
    assert response["result"]["total"] == 1


def test_tui_tools_show_keeps_native_memory_when_store_enabled(server, monkeypatch):
    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **kwargs: [_tool("memory"), _tool("librarian")],
    )
    monkeypatch.setattr(model_tools, "get_toolset_for_tool", lambda name: "memory")
    server._sessions["sid"] = {
        "agent": SimpleNamespace(
            enabled_toolsets=["memory"],
            _memory_enabled=True,
            _user_profile_enabled=False,
        )
    }

    response = server._methods["tools.show"](1, {"session_id": "sid"})

    assert _tool_names(response) == {"memory", "librarian"}
    assert response["result"]["total"] == 2
