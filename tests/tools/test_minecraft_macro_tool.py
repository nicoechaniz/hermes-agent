from tools import minecraft_tools


def _capture_macro_post(monkeypatch):
    captured = {}

    def fake_post(path, body, timeout):
        captured.update(path=path, body=body, timeout=timeout)
        return {
            "ok": True,
            "message": "Done.",
            "steps": 3,
            "finalY": 73,
        }

    monkeypatch.setattr(minecraft_tools, "_api_post", fake_post)
    return captured


def test_mc_macro_schema_exposes_open_sky_override():
    prop = minecraft_tools.MC_MACRO_SCHEMA["properties"]["stop_on_open_sky"]
    assert prop["type"] == "boolean"


def test_spiral_forwards_explicit_open_sky_override(monkeypatch):
    captured = _capture_macro_post(monkeypatch)

    result = minecraft_tools._handle_mc_macro({
        "macro": "spiral",
        "target_y": 80,
        "steps_per_side": 2,
        "stop_on_open_sky": False,
    })

    assert "Done." in result
    assert captured == {
        "path": "/macro",
        "body": {
            "macro": "spiral",
            "target_y": 80,
            "steps_per_side": 2,
            "stop_on_open_sky": False,
        },
        "timeout": 600,
    }


def test_spiral_omits_open_sky_override_by_default(monkeypatch):
    captured = _capture_macro_post(monkeypatch)

    minecraft_tools._handle_mc_macro({
        "macro": "spiral",
        "target_y": 80,
    })

    assert "stop_on_open_sky" not in captured["body"]
