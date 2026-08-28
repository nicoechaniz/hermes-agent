"""Unit tests for TurnRetryState (god-file Phase 1b).

The dataclass holds the inner-retry-loop's one-shot recovery guards + restart
signals. These tests verify its naming and default-semantics invariants without
freezing the complete field enumeration; provider recovery branches may add
new independent guards over time.
"""

from __future__ import annotations

from dataclasses import fields

from agent.turn_retry_state import TurnRetryState


REQUIRED_FIELDS = {
    "codex_auth_retry_attempted",
    "anthropic_auth_retry_attempted",
    "copilot_auth_retry_attempted",
    "kimi_auth_retry_attempted",
    "has_retried_429",
    "restart_with_compressed_messages",
}


def test_fields_follow_one_shot_boolean_contract():
    names = {f.name for f in fields(TurnRetryState)}
    assert REQUIRED_FIELDS <= names

    state = TurnRetryState()
    for name in names:
        assert name.endswith("_attempted") or name.startswith("restart_with_") or name == "has_retried_429"
        assert getattr(state, name) is False




def test_guards_are_independently_mutable():
    s = TurnRetryState()
    s.codex_auth_retry_attempted = True
    s.restart_with_compressed_messages = True
    assert s.codex_auth_retry_attempted is True
    assert s.restart_with_compressed_messages is True
    # untouched guards stay False
    assert s.has_retried_429 is False
    assert s.anthropic_auth_retry_attempted is False


def test_copilot_provider_check_accepts_alias_spellings():
    """`/model` and profile configs can leave `github-copilot` / `github` as
    the provider spelling; the recovery gates must not silently skip them."""
    from agent.conversation_loop import _is_copilot_provider
    from run_agent import AIAgent

    class _Agent:
        # Reuse the real single-owner check unbound; only provider/_base_url
        # state is faked.
        _is_copilot_provider = AIAgent._is_copilot_provider
        _is_copilot_url = AIAgent._is_copilot_url

        def __init__(self, provider, base_url=""):
            self.provider = provider
            self._base_url_lower = base_url.lower()

    assert _is_copilot_provider(_Agent("copilot"))
    assert _is_copilot_provider(_Agent("github-copilot"))
    assert _is_copilot_provider(_Agent("GitHub-Copilot"))
    assert _is_copilot_provider(_Agent("github"))
    # URL fallback: unnormalized provider but a Copilot base URL.
    assert _is_copilot_provider(_Agent("custom", "https://api.githubcopilot.com"))
    assert not _is_copilot_provider(_Agent("openrouter", "https://openrouter.ai/api/v1"))

    class _NoMethod:
        provider = "github-copilot"

    # Fallback path when the agent object lacks the method entirely.
    assert _is_copilot_provider(_NoMethod())
