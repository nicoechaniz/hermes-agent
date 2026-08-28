"""CLI /tools must reflect the model-facing native-memory gate."""

from types import SimpleNamespace

from cli import HermesCLI


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {},
        },
    }


def _make_cli(*, memory_enabled: bool, user_profile_enabled: bool) -> HermesCLI:
    cli = HermesCLI.__new__(HermesCLI)
    cli.enabled_toolsets = ["memory"]
    cli.config = {
        "memory": {
            "memory_enabled": memory_enabled,
            "user_profile_enabled": user_profile_enabled,
        }
    }
    cli.agent = SimpleNamespace(
        _memory_enabled=memory_enabled,
        _user_profile_enabled=user_profile_enabled,
    )
    return cli


def test_cli_tools_hides_native_memory_for_external_only_profile(monkeypatch, capsys):
    import cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "get_tool_definitions",
        lambda **kwargs: [_tool("memory"), _tool("librarian")],
    )
    monkeypatch.setattr(cli_module, "get_toolset_for_tool", lambda name: "memory")

    _make_cli(memory_enabled=False, user_profile_enabled=False).show_tools()
    output = capsys.readouterr().out

    assert "* memory" not in output
    assert "* librarian" in output
    assert "Total: 1 tools" in output


def test_cli_tools_keeps_native_memory_when_store_enabled(monkeypatch, capsys):
    import cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "get_tool_definitions",
        lambda **kwargs: [_tool("memory"), _tool("librarian")],
    )
    monkeypatch.setattr(cli_module, "get_toolset_for_tool", lambda name: "memory")

    _make_cli(memory_enabled=True, user_profile_enabled=False).show_tools()
    output = capsys.readouterr().out

    assert "* memory" in output
    assert "* librarian" in output
    assert "Total: 2 tools" in output
