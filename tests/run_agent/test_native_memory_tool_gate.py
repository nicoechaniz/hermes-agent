"""Native built-in ``memory`` tool is gated by MEMORY.md / USER.md flags.

HMK-first profiles set both ``memory.memory_enabled`` and
``memory.user_profile_enabled`` false while keeping an external provider.
The memory *toolset* must stay available so provider tools (e.g. librarian)
still inject; only the native MEMORY.md/USER.md tool leaves the model schema.

Exercises real ``AIAgent`` init + tool-definition wiring — not source-text
assertions.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch


class _LibrarianProvider:
    """Minimal external memory provider that exposes a non-native tool."""

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
                    "properties": {
                        "query": {"type": "string"},
                    },
                },
            }
        ]

    def shutdown(self):
        pass


def _tool_names(agent) -> set[str]:
    names = set()
    for tool in agent.tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            names.add(fn["name"])
    return names


def _make_agent(cfg: dict, *, provider=None, **agent_kwargs):
    """Construct a real AIAgent with controlled memory config."""
    with ExitStack() as stack:
        stack.enter_context(patch("hermes_cli.config.load_config", return_value=cfg))
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

        kwargs = {
            "api_key": "test-key-1234567890",
            "base_url": "https://openrouter.ai/api/v1",
            "quiet_mode": True,
            "skip_context_files": True,
            "skip_memory": False,
            # Keep the memory toolset in play without loading every
            # platform toolset — mirrors how a restricted profile still
            # has memory available for provider injection.
            "enabled_toolsets": ["memory", "terminal"],
        }
        kwargs.update(agent_kwargs)
        return AIAgent(**kwargs)


def test_native_memory_tool_absent_when_both_stores_disabled():
    """Both native stores off → built-in ``memory`` not in model tool schema."""
    cfg = {
        "memory": {
            "memory_enabled": False,
            "user_profile_enabled": False,
            "provider": "",
        },
        "agent": {},
    }
    agent = _make_agent(cfg)

    names = _tool_names(agent)
    assert "memory" not in names
    assert "memory" not in agent.valid_tool_names
    assert agent._memory_store is None
    assert agent._memory_enabled is False
    assert agent._user_profile_enabled is False
    # Memory toolset still contributed other non-native tools if any; at
    # minimum the gate must not have disabled tool resolution entirely.
    assert agent.tools is not None


def test_native_memory_tool_present_when_memory_enabled():
    """memory_enabled alone keeps the built-in ``memory`` tool."""
    cfg = {
        "memory": {
            "memory_enabled": True,
            "user_profile_enabled": False,
            "provider": "",
        },
        "agent": {},
    }
    agent = _make_agent(cfg)

    names = _tool_names(agent)
    assert "memory" in names
    assert "memory" in agent.valid_tool_names
    assert agent._memory_store is not None
    assert agent._memory_enabled is True


def test_native_memory_tool_present_when_user_profile_enabled():
    """user_profile_enabled alone (USER.md) still needs the built-in tool."""
    cfg = {
        "memory": {
            "memory_enabled": False,
            "user_profile_enabled": True,
            "provider": "",
        },
        "agent": {},
    }
    agent = _make_agent(cfg)

    names = _tool_names(agent)
    assert "memory" in names
    assert "memory" in agent.valid_tool_names
    assert agent._memory_store is not None
    assert agent._user_profile_enabled is True


def test_provider_tools_inject_when_native_stores_disabled():
    """External provider tools (e.g. librarian) still inject on HMK-first profiles.

    The memory toolset remains enabled; only the native ``memory`` tool is
    stripped. Provider schemas must still appear in the model tool surface.
    """
    cfg = {
        "memory": {
            "memory_enabled": False,
            "user_profile_enabled": False,
            "provider": "hmk-lib",
        },
        "agent": {},
    }
    provider = _LibrarianProvider()
    agent = _make_agent(cfg, provider=provider)

    names = _tool_names(agent)
    assert "memory" not in names
    assert "memory" not in agent.valid_tool_names
    assert "librarian" in names
    assert "librarian" in agent.valid_tool_names
    assert agent._memory_manager is not None
    assert agent._memory_store is None


def test_filter_native_memory_tool_leaves_provider_tools():
    """Unit-level: filter only removes the built-in name, not peers."""
    from agent.memory_manager import filter_native_memory_tool

    tools = [
        {"type": "function", "function": {"name": "memory", "parameters": {}}},
        {"type": "function", "function": {"name": "librarian", "parameters": {}}},
        {"type": "function", "function": {"name": "terminal", "parameters": {}}},
    ]
    names = {"memory", "librarian", "terminal"}
    filtered = filter_native_memory_tool(tools, names)

    assert [t["function"]["name"] for t in filtered] == ["librarian", "terminal"]
    assert "memory" not in names
    assert names == {"librarian", "terminal"}


def test_apply_native_memory_tool_gate_no_op_when_store_enabled():
    """Gate is a no-op when either native store flag is true."""
    from agent.memory_manager import apply_native_memory_tool_gate

    agent = SimpleNamespace(
        tools=[
            {"type": "function", "function": {"name": "memory", "parameters": {}}},
        ],
        valid_tool_names={"memory"},
        _memory_enabled=True,
        _user_profile_enabled=False,
    )
    assert apply_native_memory_tool_gate(agent) is False
    assert agent.tools[0]["function"]["name"] == "memory"
    assert "memory" in agent.valid_tool_names


def test_apply_native_memory_tool_gate_strips_when_both_disabled():
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
    assert [t["function"]["name"] for t in agent.tools] == ["librarian"]
    assert agent.valid_tool_names == {"librarian"}
