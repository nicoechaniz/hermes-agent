"""Tests for CycleDetector and _inject_synthetic_perceive hook in daemoncraft.py."""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy optional deps before importing daemoncraft
# ---------------------------------------------------------------------------

def _stub_module(name: str, **attrs):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

_stub_module("httpx")
_stub_module("aiohttp", WSMsgType=MagicMock(), ClientSession=MagicMock)

from gateway.platforms.daemoncraft import CycleDetector  # noqa: E402


# ===========================================================================
# CycleDetector tests
# ===========================================================================

class TestCycleDetector:
    """5 focused tests for CycleDetector behaviour."""

    def test_no_trigger_below_threshold(self):
        """N-1 identical calls must NOT trigger."""
        cd = CycleDetector(n=4, window=20, action="warn")
        results = [cd.record("tool_x", {"k": "v"}) for _ in range(3)]
        assert not any(r.triggered for r in results)

    def test_trigger_at_nth_identical_call(self):
        """The Nth identical call must trigger."""
        cd = CycleDetector(n=3, window=10, action="warn")
        r = None
        for _ in range(3):
            r = cd.record("loop_tool", {})
        assert r.triggered is True
        assert r.count >= 3

    def test_reset_after_action_no_double_trigger(self):
        """After triggering, subsequent calls with the same sig should NOT re-trigger."""
        cd = CycleDetector(n=3, window=10, action="warn")
        for _ in range(3):
            cd.record("loop_tool", {})
        # 4th and 5th same-sig calls — suppressed
        r4 = cd.record("loop_tool", {})
        r5 = cd.record("loop_tool", {})
        assert r4.triggered is False
        assert r5.triggered is False

    def test_window_size_evicts_old_entries(self):
        """Once the ring buffer (size=window) is filled with other sigs, old counts are gone."""
        # window=3: buffer holds at most 3 entries
        cd = CycleDetector(n=3, window=3, action="warn")
        # Two calls of "old_tool" — not yet triggering
        cd.record("old_tool", {})
        cd.record("old_tool", {})
        # Fill buffer with 3 different sigs, evicting "old_tool" entries
        cd.record("tool_b", {})
        cd.record("tool_c", {})
        cd.record("tool_d", {})
        # Now one more "old_tool" — only 1 in window, should not trigger
        r = cd.record("old_tool", {})
        assert r.triggered is False

    def test_different_sigs_do_not_trigger(self):
        """Calls with different args must not be counted together."""
        cd = CycleDetector(n=3, window=10, action="warn")
        results = [cd.record("tool_a", {"n": i}) for i in range(6)]
        assert not any(r.triggered for r in results)


# ===========================================================================
# _inject_synthetic_perceive hook tests
# ===========================================================================

def _make_adapter():
    from gateway.platforms.daemoncraft import DaemonCraftAdapter
    from gateway.config import PlatformConfig

    cfg = PlatformConfig(
        enabled=True,
        extra={
            "bot_api_url": "http://localhost:9999",
            "bot_username": "TestBot",
            "profile": "test",
        },
    )
    adapter = DaemonCraftAdapter(cfg)
    return adapter


def _wire_adapter(adapter, *, session_id="world-session-1", hook_results=()):
    """Attach a mock session_store and stub invoke_hook."""
    store = MagicMock()
    store.append_to_transcript = MagicMock()
    adapter._session_store = store

    # Stub _get_world_session_id
    adapter._get_world_session_id = MagicMock(return_value=session_id)
    return store


class TestSyntheticPerceiveHook:
    """3 tests covering the transform_tool_result hook path."""

    @pytest.mark.anyio
    async def test_hook_called_before_transcript_append(self):
        """invoke_hook must be called; tool_msg append comes after it."""
        adapter = _make_adapter()
        store = _wire_adapter(adapter)
        call_order = []

        def fake_invoke_hook(event, **kwargs):
            call_order.append("hook")
            return iter([])  # no replacement

        # Capture append_to_transcript calls in order
        original_append = store.append_to_transcript
        def recording_append(sid, msg):
            call_order.append(("append", msg["role"]))
        store.append_to_transcript.side_effect = recording_append

        with patch("gateway.platforms.daemoncraft.invoke_hook", fake_invoke_hook, create=True), \
             patch.dict(sys.modules, {"hermes_cli.plugins": types.SimpleNamespace(invoke_hook=fake_invoke_hook)}):
            # Patch the local import inside _inject_synthetic_perceive
            import importlib
            import gateway.platforms.daemoncraft as dc_mod
            with patch.object(dc_mod, "_inject_synthetic_perceive_hook_module", None, create=True):
                # We patch the from-import by monkeypatching the module namespace
                pass

            # Direct patch: replace hermes_cli.plugins in sys.modules
            fake_plugins = types.ModuleType("hermes_cli.plugins")
            fake_plugins.invoke_hook = fake_invoke_hook
            sys.modules["hermes_cli.plugins"] = fake_plugins
            sys.modules.setdefault("hermes_cli", types.ModuleType("hermes_cli"))

            await adapter._inject_synthetic_perceive({"x": 1})

        # assistant append should come first, then hook, then tool append
        assert ("append", "assistant") in call_order
        assert ("append", "tool") in call_order
        assert call_order.index(("append", "assistant")) < call_order.index("hook")
        assert call_order.index("hook") < call_order.index(("append", "tool"))

    @pytest.mark.anyio
    async def test_hook_receives_mc_perceive_tool_name(self):
        """invoke_hook must be called with tool_name='mc_perceive'."""
        adapter = _make_adapter()
        _wire_adapter(adapter)

        received_kwargs: dict = {}

        def fake_invoke_hook(event, **kwargs):
            received_kwargs.update({"event": event, **kwargs})
            return iter([])

        fake_plugins = types.ModuleType("hermes_cli.plugins")
        fake_plugins.invoke_hook = fake_invoke_hook
        sys.modules["hermes_cli.plugins"] = fake_plugins
        sys.modules.setdefault("hermes_cli", types.ModuleType("hermes_cli"))

        await adapter._inject_synthetic_perceive({"obs": "block"})

        assert received_kwargs.get("event") == "transform_tool_result"
        assert received_kwargs.get("tool_name") == "mc_perceive"

    @pytest.mark.anyio
    async def test_transcript_appended_even_if_hook_raises(self):
        """If invoke_hook raises, transcript append must still happen."""
        adapter = _make_adapter()
        store = _wire_adapter(adapter)

        def exploding_hook(event, **kwargs):
            raise RuntimeError("hook boom")

        fake_plugins = types.ModuleType("hermes_cli.plugins")
        fake_plugins.invoke_hook = exploding_hook
        sys.modules["hermes_cli.plugins"] = fake_plugins
        sys.modules.setdefault("hermes_cli", types.ModuleType("hermes_cli"))

        await adapter._inject_synthetic_perceive({"obs": "fire"})

        # Both assistant_msg and tool_msg must have been appended
        assert store.append_to_transcript.call_count == 2
        roles = [c.args[1]["role"] for c in store.append_to_transcript.call_args_list]
        assert roles == ["assistant", "tool"]
