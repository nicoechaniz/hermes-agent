"""Tests for xAI Imagine storage helper behavior."""

from __future__ import annotations

import yaml


def _invalidate_config_cache():
    try:
        import hermes_cli.config as cfg_mod

        if hasattr(cfg_mod, "_invalidate_load_config_cache"):
            cfg_mod._invalidate_load_config_cache()
    except Exception:
        pass


def test_storage_defaults_to_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _invalidate_config_cache()

    from tools.xai_http import build_xai_storage_options

    storage = build_xai_storage_options(
        "image_gen",
        filename_prefix="hermes-xai-image",
        extension="png",
    )

    assert storage is None


def test_explicitly_enabled_storage_defaults_to_permanent_public_urls(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "image_gen": {"xai": {"storage": {"enabled": True}}},
    }))
    _invalidate_config_cache()

    from tools.xai_http import build_xai_storage_options

    storage = build_xai_storage_options(
        "image_gen",
        filename_prefix="hermes-xai-image",
        extension="png",
    )

    assert storage is not None
    assert storage["public_url"] is True
    assert "expires_after" not in storage
    assert storage["filename"].startswith("hermes-xai-image-")
    assert storage["filename"].endswith(".png")


def test_invalid_storage_retention_falls_back_to_bounded_ttl(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "video_gen": {
            "xai": {
                "storage": {
                    "enabled": True,
                    "expires_after": "definitely-not-a-duration",
                },
            },
        },
    }))
    _invalidate_config_cache()

    from tools.xai_http import build_xai_storage_options

    storage = build_xai_storage_options(
        "video_gen",
        filename_prefix="hermes-xai-video",
        extension="mp4",
    )

    assert storage is not None
    assert storage["expires_after"] == 172800


def test_storage_picker_defaults_to_disabled(monkeypatch):
    import hermes_cli.tools_config as tools_config
    import hermes_cli.tools_config_providers as providers

    observed = {}

    def choose(prompt, choices, default):
        observed.update(prompt=prompt, choices=choices, default=default)
        return default

    monkeypatch.setattr(tools_config, "_prompt_choice", choose)
    monkeypatch.setattr(providers, "_print_warning", lambda _message: None)
    monkeypatch.setattr(providers, "_print_success", lambda _message: None)

    config = {}
    providers._configure_xai_imagine_storage("video_gen", config)

    assert observed["default"] == 0
    assert observed["choices"][0] == "Disable stored public URLs (recommended)"
    assert config["video_gen"]["xai"]["storage"] == {"enabled": False}
