"""Kimi CLI OAuth headers and the one-shot 401 recovery rung."""

from __future__ import annotations

from types import SimpleNamespace


def test_official_kimi_headers_are_only_attached_for_cli_origin(monkeypatch):
    import agent.anthropic_adapter as adapter

    captured = []
    monkeypatch.setattr(adapter, "_require_sdk", lambda _label: object())
    monkeypatch.setattr(adapter, "_new_sdk_client", lambda _sdk, _kwargs, headers, **_kw: captured.append(headers) or object())
    monkeypatch.setattr("hermes_cli.auth.kimi_coding_default_headers", lambda: {"X-Msh-Platform": "kimi_cli"})

    adapter.build_anthropic_client("explicit-kimi-key", "https://api.kimi.com/coding", kimi_cli_oauth=False)
    adapter.build_anthropic_client("cli-token", "https://api.kimi.com/coding", kimi_cli_oauth=True)

    assert "X-Msh-Platform" not in captured[0]
    assert captured[1]["X-Msh-Platform"] == "kimi_cli"


def test_kimi_cli_401_refresh_is_exactly_one_rung():
    from agent.turn_recovery import _refresh_credentials_after_401
    from agent.turn_retry_state import TurnRetryState

    calls = []
    agent = SimpleNamespace(
        provider="kimi-coding", api_mode="anthropic_messages", _is_kimi_cli_oauth=True,
        _try_refresh_kimi_cli_client_credentials=lambda: calls.append(True) or True,
        _buffer_vprint=lambda _message: None,
    )
    retry = TurnRetryState()

    assert _refresh_credentials_after_401(agent, Exception("unauthorized"), retry, 401) is True
    assert _refresh_credentials_after_401(agent, Exception("unauthorized"), retry, 401) is False
    assert calls == [True]
