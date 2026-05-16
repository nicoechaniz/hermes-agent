"""Tests for the mc_bit Hermes tool wrapper."""

from __future__ import annotations

import importlib


def test_mc_bit_handler_is_sync_and_formats_response(monkeypatch):
    tool = importlib.import_module("tools.mc_bit_tool")

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "ok": True,
                "data": {
                    "format": "surface",
                    "text": "GG\nTT\n",
                    "count": 4,
                    "elapsed_ms": 2,
                },
            }

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("MC_API_URL", "http://bot.test:3003")
    monkeypatch.setattr(tool.httpx, "get", fake_get)

    result = tool._handler({
        "x1": 1,
        "y1": 2,
        "z1": 3,
        "x2": 4,
        "y2": 5,
        "z2": 6,
        "format": "surface",
    })

    assert isinstance(result, str)
    assert result == "mBit surface (4 blocks, 2ms):\nGG\nTT\n"
    assert captured == {
        "url": "http://bot.test:3003/blocks",
        "params": {"x1": 1, "y1": 2, "z1": 3, "x2": 4, "y2": 5, "z2": 6, "format": "surface"},
        "timeout": 10.0,
    }


def test_mc_bit_missing_coordinates_returns_error():
    tool = importlib.import_module("tools.mc_bit_tool")

    result = tool._handler({"x1": 1})

    assert "requires x1, y1, z1, x2, y2, z2" in result
    assert "missing:" in result


def test_mc_bit_invalid_format_returns_error():
    tool = importlib.import_module("tools.mc_bit_tool")

    result = tool._handler({
        "x1": 1,
        "y1": 2,
        "z1": 3,
        "x2": 4,
        "y2": 5,
        "z2": 6,
        "format": "bad",
    })

    assert "unsupported format" in result
    assert "binary" in result
