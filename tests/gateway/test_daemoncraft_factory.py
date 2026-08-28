"""Behavioral coverage for DaemonCraft gateway adapter construction."""

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.daemoncraft import DaemonCraftAdapter
from gateway.run import GatewayRunner


def test_gateway_factory_creates_daemoncraft_adapter():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    config = PlatformConfig(
        enabled=True,
        extra={
            "bot_api_url": "http://localhost:9999",
            "bot_username": "TestBot",
        },
    )

    adapter = runner._create_adapter(Platform.DAEMONCRAFT, config)

    assert isinstance(adapter, DaemonCraftAdapter)
    assert adapter.platform is Platform.DAEMONCRAFT
