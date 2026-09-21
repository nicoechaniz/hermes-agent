"""Behavior contracts for the bundled AlterMundi memory plugin."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def plugin_module():
    plugin_path = (
        Path(__file__).parents[2]
        / "plugins"
        / "altermundi-memory"
        / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_altermundi_memory_plugin", plugin_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_register_exposes_read_only_tools_in_altermundi_toolset(plugin_module):
    registrations = []
    ctx = SimpleNamespace(
        register_tool=lambda **kwargs: registrations.append(kwargs)
    )

    plugin_module.register(ctx)

    assert [entry["name"] for entry in registrations] == [
        "altermundi_search",
        "altermundi_doc",
    ]
    assert {entry["toolset"] for entry in registrations} == {"altermundi"}
    assert all(entry["check_fn"] is plugin_module._check_available for entry in registrations)


def test_search_encodes_filters_and_clamps_result_count(plugin_module, monkeypatch):
    get_json = MagicMock(return_value={"results": []})
    monkeypatch.setattr(plugin_module, "_get_json", get_json)

    result = json.loads(
        plugin_module._handle_search(
            {"q": "harmonic beacon", "k": 999, "kind": "source", "project": "phideus"}
        )
    )

    assert result == {"results": []}
    get_json.assert_called_once_with(
        "/search?q=harmonic%20beacon&k=50&kind=source&project=phideus"
    )


def test_doc_uses_url_encoded_document_id(plugin_module, monkeypatch):
    get_json = MagicMock(return_value={"content": "body"})
    monkeypatch.setattr(plugin_module, "_get_json", get_json)

    result = json.loads(
        plugin_module._handle_doc({"id": "mapa/proyectos/HIT notes.md"})
    )

    assert result == {"content": "body"}
    get_json.assert_called_once_with(
        "/doc?id=mapa%2Fproyectos%2FHIT+notes.md"
    )


def test_empty_required_arguments_fail_without_network(plugin_module, monkeypatch):
    get_json = MagicMock()
    monkeypatch.setattr(plugin_module, "_get_json", get_json)

    assert json.loads(plugin_module._handle_search({})) == {
        "success": False,
        "error": "q is required",
    }
    assert json.loads(plugin_module._handle_doc({})) == {
        "success": False,
        "error": "id is required",
    }
    get_json.assert_not_called()
