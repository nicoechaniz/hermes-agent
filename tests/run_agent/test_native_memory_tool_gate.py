"""Fail-closed gating for the built-in MEMORY.md/USER.md tool.

External memory providers share the ``memory`` toolset. Profiles with both
native stores disabled must keep provider tools while removing only the native
``memory`` schema and its system-prompt guidance.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


class _LibrarianProvider:
    name = "hmk-lib"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return [
            {
                "name": "librarian",
                "description": "External HMK librarian tool",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]

    def shutdown(self):
        pass


def _tool_names(agent) -> set[str]:
    return {
        tool["function"]["name"]
        for tool in agent.tools or []
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name")
    }


def _make_agent(memory_config: dict, *, provider=None) -> Any:
    config = {"memory": memory_config, "agent": {}}
    with ExitStack() as stack:
        stack.enter_context(patch("hermes_cli.config.load_config", return_value=config))
        stack.enter_context(
            patch("hermes_cli.config.load_config_readonly", return_value=config)
        )
        stack.enter_context(
            patch("agent.model_metadata.get_model_context_length", return_value=204_800)
        )
        stack.enter_context(
            patch("run_agent.check_toolset_requirements", return_value={})
        )
        stack.enter_context(patch("run_agent.OpenAI"))
        if provider is not None:
            stack.enter_context(
                patch("plugins.memory.load_memory_provider", return_value=provider)
            )

        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            enabled_toolsets=["memory", "terminal"],
        )


def test_native_memory_absent_and_guidance_omitted_when_stores_disabled():
    agent = _make_agent(
        {
            "memory_enabled": False,
            "user_profile_enabled": False,
            "provider": "",
        }
    )

    from agent.prompt_builder import MEMORY_GUIDANCE

    assert "memory" not in _tool_names(agent)
    assert "memory" not in agent.valid_tool_names
    assert MEMORY_GUIDANCE not in agent._build_system_prompt()
    assert agent._memory_store is None


def test_native_memory_present_when_memory_store_enabled():
    agent = _make_agent(
        {
            "memory_enabled": True,
            "user_profile_enabled": False,
            "provider": "",
        }
    )

    assert "memory" in _tool_names(agent)
    assert "memory" in agent.valid_tool_names
    assert agent._memory_store is not None


def test_native_memory_present_when_user_profile_enabled():
    agent = _make_agent(
        {
            "memory_enabled": False,
            "user_profile_enabled": True,
            "provider": "",
        }
    )

    assert "memory" in _tool_names(agent)
    assert "memory" in agent.valid_tool_names
    assert agent._memory_store is not None


def test_provider_tool_survives_when_native_stores_disabled():
    agent = _make_agent(
        {
            "memory_enabled": False,
            "user_profile_enabled": False,
            "provider": "hmk-lib",
        },
        provider=_LibrarianProvider(),
    )

    names = _tool_names(agent)
    assert "memory" not in names
    assert "librarian" in names
    assert agent.valid_tool_names == names
    assert agent._memory_manager is not None
    assert agent._memory_store is None


def test_filter_native_memory_tool_leaves_provider_tools():
    from agent.memory_manager import filter_native_memory_tool

    tools = [
        {"type": "function", "function": {"name": "memory", "parameters": {}}},
        {"type": "function", "function": {"name": "librarian", "parameters": {}}},
        {"type": "function", "function": {"name": "terminal", "parameters": {}}},
    ]
    names = {"memory", "librarian", "terminal"}

    filtered = filter_native_memory_tool(tools, names)

    assert filtered is not None
    assert [tool["function"]["name"] for tool in filtered] == [
        "librarian",
        "terminal",
    ]
    assert names == {"librarian", "terminal"}


def test_gate_is_noop_when_native_store_enabled():
    from agent.memory_manager import apply_native_memory_tool_gate

    agent = SimpleNamespace(
        tools=[
            {"type": "function", "function": {"name": "memory", "parameters": {}}}
        ],
        valid_tool_names={"memory"},
        _memory_enabled=True,
        _user_profile_enabled=False,
    )

    assert apply_native_memory_tool_gate(agent) is False
    assert agent.valid_tool_names == {"memory"}


def test_gate_removes_native_tool_when_both_stores_disabled():
    from agent.memory_manager import apply_native_memory_tool_gate

    agent = SimpleNamespace(
        tools=[
            {"type": "function", "function": {"name": "memory", "parameters": {}}},
            {"type": "function", "function": {"name": "librarian", "parameters": {}}},
        ],
        valid_tool_names={"memory", "librarian"},
        _memory_enabled=False,
        _user_profile_enabled=False,
    )

    assert apply_native_memory_tool_gate(agent) is True
    assert [tool["function"]["name"] for tool in agent.tools] == ["librarian"]
    assert agent.valid_tool_names == {"librarian"}
