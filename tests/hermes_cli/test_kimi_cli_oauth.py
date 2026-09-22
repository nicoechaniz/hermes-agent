"""Kimi official-CLI OAuth compatibility, reconstructed from bec42006c05b0a9f6020e3033901a77e27e39898."""

from __future__ import annotations

import json


def _write_cli_store(home, *, expires_at=4_000_000_000):
    credential_path = home / "credentials" / "kimi-code.json"
    credential_path.parent.mkdir(parents=True)
    credential_path.write_text(json.dumps({
        "access_token": "test-cli-access-token", "refresh_token": "test-cli-refresh-token", "expires_at": expires_at,
    }), encoding="utf-8")
    (home / "device_id").write_text("test-device", encoding="utf-8")
    return credential_path


def test_explicit_kimi_key_wins_over_official_cli_store(monkeypatch, tmp_path):
    from hermes_cli import auth
    from hermes_cli import models

    _write_cli_store(tmp_path)
    monkeypatch.setenv("KIMI_CODE_HOME", str(tmp_path))
    monkeypatch.setattr(auth, "_resolve_api_key_provider_secret", lambda *_: ("explicit-kimi-key", "KIMI_API_KEY"))
    monkeypatch.setattr(auth, "resolve_kimi_cli_oauth_credentials", lambda **_: (_ for _ in ()).throw(AssertionError()))

    resolved = auth.resolve_api_key_provider_credentials("kimi-coding")

    assert resolved["api_key"] == "explicit-kimi-key"
    assert resolved["source"] == "KIMI_API_KEY"
    assert "kimi_cli_oauth" not in resolved
    monkeypatch.setattr(models, "_profile_live_catalog", lambda _provider: ["explicit-key-model"])
    monkeypatch.setattr(auth, "kimi_cli_model_ids", lambda: (_ for _ in ()).throw(
        AssertionError("explicit keys must not read the CLI model catalog")))
    assert models.provider_model_ids("kimi-coding") == ["explicit-key-model"]


def test_cli_store_is_status_and_picker_visible_without_api_key(monkeypatch, tmp_path):
    from hermes_cli import auth
    from hermes_cli.model_switch import list_authenticated_providers
    from hermes_cli.models import provider_model_ids

    _write_cli_store(tmp_path)
    (tmp_path / "config.toml").write_text(
        '[models.primary]\nprovider = "managed:kimi-code"\nmodel = "k3-256k"\n\n'
        'default_model = "primary"\n', encoding="utf-8")
    monkeypatch.setenv("KIMI_CODE_HOME", str(tmp_path))
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_CODING_API_KEY", raising=False)

    status = auth.get_auth_status("kimi-coding")

    assert status["configured"] is True
    assert status["key_source"] == "kimi-cli-oauth"
    assert provider_model_ids("kimi-coding") == ["k3-256k"]
    rows = list_authenticated_providers(current_provider="kimi-coding")
    kimi_row = next(row for row in rows if row["slug"] == "kimi-coding")
    assert kimi_row["models"] == ["k3-256k"]


def test_expired_cli_token_refreshes_and_atomically_persists(monkeypatch, tmp_path):
    from hermes_cli import auth_zai_kimi as kimi

    credential_path = _write_cli_store(tmp_path, expires_at=0)
    monkeypatch.setenv("KIMI_CODE_HOME", str(tmp_path))

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"access_token": "rotated-access-token", "refresh_token": "rotated-refresh-token", "expires_in": 600}

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def post(*_args, **_kwargs):
            return Response()

    monkeypatch.setattr(kimi.httpx, "Client", Client)

    resolved = kimi.resolve_kimi_cli_oauth_credentials(base_url="https://api.kimi.com/coding")
    persisted = json.loads(credential_path.read_text(encoding="utf-8"))

    assert resolved["source"] == "kimi-cli-oauth-refresh"
    assert persisted["access_token"] == "rotated-access-token"
    assert persisted["refresh_token"] == "rotated-refresh-token"
    assert persisted["expires_at"] > 0


def test_runtime_resolves_cli_oauth_to_current_messages_endpoint(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(runtime_provider, "resolve_provider", lambda *_args, **_kwargs: "kimi-coding")
    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {})
    monkeypatch.setattr(runtime_provider, "resolve_api_key_provider_credentials", lambda _provider: {
        "provider": "kimi-coding", "api_key": "cli-token", "base_url": "https://api.kimi.com/coding",
        "source": "kimi-cli-oauth", "kimi_cli_oauth": True,
    })

    resolved = runtime_provider.resolve_runtime_provider(requested="kimi-coding", target_model="k3")

    assert resolved["base_url"] == "https://api.kimi.com/coding"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["source"] == "kimi-cli-oauth"
