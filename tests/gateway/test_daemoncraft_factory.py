"""DaemonCraft registration and current gateway factory contracts."""
from gateway.config import GatewayConfig, Platform, PlatformConfig, _PLATFORM_CONNECTED_CHECKERS
from gateway.platforms.daemoncraft import DaemonCraftAdapter
from gateway.run import GatewayRunner


def test_daemoncraft_is_a_builtin_factory_and_requires_its_connection_fields():
    cfg = PlatformConfig(enabled=True, extra={"bot_api_url": "http://bot.test", "bot_username": "Steve"})
    assert _PLATFORM_CONNECTED_CHECKERS[Platform.DAEMONCRAFT](cfg)
    assert not _PLATFORM_CONNECTED_CHECKERS[Platform.DAEMONCRAFT](PlatformConfig(enabled=True, extra={"bot_api_url": "http://bot.test"}))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    adapter = runner._create_adapter(Platform.DAEMONCRAFT, cfg)
    assert isinstance(adapter, DaemonCraftAdapter)
    assert adapter.platform is Platform.DAEMONCRAFT


def test_daemoncraft_toolset_resolves_body_planner_bit_and_macro_tools():
    import tools.embodied_plan_tool  # noqa: F401
    import tools.mc_bit_tool  # noqa: F401
    import tools.mc_navigate_tool  # noqa: F401
    import tools.minecraft_tools  # noqa: F401
    from toolsets import resolve_toolset

    resolved = set(resolve_toolset("hermes-daemoncraft"))
    assert {
        "embodied_plan",
        "mc_bit",
        "mc_navigate",
        "mc_perceive",
        "mc_macro",
        "mc_interoception",
    } <= resolved
