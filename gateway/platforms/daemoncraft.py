"""
DaemonCraft platform adapter for Hermes Gateway.

Routes Minecraft chat (player whispers + world broadcasts) through the
Hermes AIAgent, while the agent_loop.py handles embodiment (movement,
quest engine, sensors).

The adapter consumes the Bot API WebSocket and HTTP endpoints:
  - WS /ws       : inbound chat events (array snapshot)
  - POST /chat/send   : outbound text
  - POST /tts/play    : outbound TTS relay to dashboards
  - GET  /agent/log   : recent loop turns for context injection
"""

import asyncio
import json
import logging
import os
import random
import time
from typing import Any, Dict, Optional, Set

import aiohttp
from aiohttp import WSMsgType

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

META_NO_CLAMP = "_no_clamp"  # Set in metadata to bypass gateway-side char clamping (used by TTS transcripts)


class DaemonCraftAdapter(BasePlatformAdapter):
    """Gateway adapter for DaemonCraft (Minecraft bot API)."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.DAEMONCRAFT)
        self._bot_api_url: str = (config.extra or {}).get("bot_api_url", "")
        self._bot_username: str = (config.extra or {}).get("bot_username", "")
        self._profile: str = (config.extra or {}).get("profile", "")
        self._allowed_users: Set[str] = set()
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._last_seen_timestamp: int = 0
        self._shutdown_event = asyncio.Event()
        self._world_names: Set[str] = set()  # Track broadcast worlds for send() routing
        self._ws_retry_count: int = 0
        self._voice_mode_default: str = "all"  # DaemonCraft defaults to TTS for all replies
        self._last_tts_time: float = 0.0
        self._tts_queue: list[dict] = []  # Dedup buffer for rapid-fire messages

        # Load allowlist by UUID (preferred) or username fallback.
        raw_allow = os.getenv("DAEMONCRAFT_ALLOWED_USERS", "").strip()
        if raw_allow:
            self._allowed_users = {u.strip().lower() for u in raw_allow.split(",") if u.strip()}

        # Force group sessions per world (broadcasts must share context)
        if config.extra is None:
            config.extra = {}
        config.extra.setdefault("group_sessions_per_user", False)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        if not self._bot_api_url:
            logger.error("[DaemonCraft] bot_api_url missing in platform config extra")
            return False
        if not self._bot_username:
            logger.error("[DaemonCraft] bot_username missing in platform config extra")
            return False

        self._last_seen_timestamp = int(time.time() * 1000)
        self._shutdown_event.clear()
        self._session = aiohttp.ClientSession()
        self._ws_task = asyncio.create_task(self._ws_loop())
        self._mark_connected()
        logger.info("[DaemonCraft] Connected to %s as %s", self._bot_api_url, self._bot_username)
        return True

    async def disconnect(self) -> None:
        self._shutdown_event.set()
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None
        if self._session:
            await self._session.close()
            self._session = None
        self._mark_disconnected()
        logger.info("[DaemonCraft] Disconnected")

    # ------------------------------------------------------------------
    # WebSocket listener
    # ------------------------------------------------------------------

    async def _ws_loop(self) -> None:
        ws_url = self._bot_api_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        while not self._shutdown_event.is_set():
            try:
                async with self._session.ws_connect(ws_url) as ws:
                    self._ws_retry_count = 0
                    logger.info("[DaemonCraft] WebSocket connected")
                    while not self._shutdown_event.is_set():
                        msg = await ws.receive(timeout=30)
                        if msg.type == WSMsgType.TEXT:
                            await self._on_ws_message(msg.data)
                        elif msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                            break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._ws_retry_count += 1
                delay = min(2 ** self._ws_retry_count, 30)
                jitter = random.random()  # 0–1s uniform jitter
                sleep_time = delay + jitter
                logger.warning("[DaemonCraft] WebSocket error: %s — reconnecting in %.1fs", e, sleep_time)
                await asyncio.sleep(sleep_time)

    async def _on_ws_message(self, data: str) -> None:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return

        msg_type = payload.get("type")
        if msg_type == "chat":
            messages = payload.get("data", [])
            if not isinstance(messages, list):
                return
            await self._handle_chat_batch(messages)
        elif msg_type == "quest_event":
            data = payload.get("data", {})
            await self._handle_quest_event(data)
        elif msg_type == "blueprint_updated":
            data = payload.get("data", {})
            await self._handle_blueprint_updated(data)
        elif msg_type == "heartbeat_context":
            data = payload.get("data", {})
            await self._handle_heartbeat_context(data)
        elif msg_type == "interrupt":
            # Loop-to-gateway interrupt acknowledgment — no action needed
            pass
        elif msg_type == "status":
            pass
        else:
            logger.debug("[DaemonCraft] Unknown WS message type: %s", msg_type)

    async def _handle_chat_batch(self, messages: list) -> None:
        """Process a batch of chat messages with bot filtering and @mention classification.

        - Bot messages without @mention are silently dropped.
        - Human @mentions are treated as urgent (interrupts loop + immediate response).
        - All other human messages are queued normally.
        """
        new_messages = [m for m in messages if m.get("time", 0) > self._last_seen_timestamp]
        if not new_messages:
            return

        for entry in new_messages:
            self._last_seen_timestamp = max(self._last_seen_timestamp, entry.get("time", 0))

        # Load known bots from env (same source as agent_loop.py)
        known_bots = set(
            u.strip().lower()
            for u in os.getenv("MC_KNOWN_BOTS", self._bot_username).split(",")
            if u.strip()
        )

        urgent_msgs = []
        accepted_msgs = []
        import re

        # Build a regex that matches @username with word boundaries,
        # tolerating trailing punctuation like @pamplinas, or @pamplinas!
        mention_re = re.compile(rf"\b@{re.escape(self._bot_username.lower())}\b", re.IGNORECASE)

        for entry in new_messages:
            from_user = entry.get("from", "").lower()
            msg_text = entry.get("message", "")
            is_bot = from_user in known_bots
            mentions_bot = bool(mention_re.search(msg_text))

            if is_bot and not mentions_bot:
                continue  # Silently drop bot spam

            accepted_msgs.append(entry)

            # Only human @mentions are urgent (bots never interrupt, even with @mention)
            if mentions_bot and not is_bot:
                urgent_msgs.append(entry)

        # Interrupt the loop for urgent human @mentions before generating response
        if urgent_msgs:
            senders = ", ".join({m.get("from", "Player") for m in urgent_msgs})
            logger.info("[DaemonCraft] Urgent @mention from %s — interrupting loop", senders)
            await self._interrupt_agent("urgent_mention")
        elif accepted_msgs:
            senders = ", ".join({m.get("from", "Player") for m in accepted_msgs})
            logger.info("[DaemonCraft] Chat from %s queued", senders)

        # Process all accepted messages through the gateway
        for entry in accepted_msgs:
            await self._handle_chat_entry(entry)

    async def _interrupt_agent(self, reason: str) -> None:
        """POST /agent/interrupt to abort the loop's in-progress LLM turn."""
        try:
            async with self._session.post(
                f"{self._bot_api_url}/agent/interrupt",
                json={"reason": reason},
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.warning("[DaemonCraft] /agent/interrupt failed: %s %s", resp.status, body)
                else:
                    logger.debug("[DaemonCraft] /agent/interrupt sent (%s)", reason)
        except Exception as e:
            logger.warning("[DaemonCraft] /agent/interrupt exception: %s", e)

    async def _handle_quest_event(self, data: dict) -> None:
        """Process a quest_event from the QuestEngine.

        Builds a narrative message and injects it into the gateway so the
        AIAgent can respond to the player (narrate phase changes, etc.).
        """
        message = data.get("message", "A quest event occurred.")
        event_type = data.get("event_type", "quest_event")
        from_phase = data.get("from_phase")
        to_phase = data.get("to_phase")

        # Build a natural-language description for the gateway AIAgent
        lines = [f"[Quest Event] {message}"]
        if from_phase and to_phase:
            lines.append(f"Phase transition: {from_phase} → {to_phase}")
        elif event_type:
            lines.append(f"Event type: {event_type}")
        event_text = "\n".join(lines)

        logger.info("[DaemonCraft] Quest event: %s", event_text.replace("\n", " | "))

        # Route to the world broadcast session (group chat)
        source = self.build_source(
            chat_id="world",
            chat_name="world",
            chat_type="group",
            user_id="quest_engine",
            user_name="QuestEngine",
            thread_id="world",
        )
        source.profile = self._profile

        event = MessageEvent(
            text=event_text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=data,
        )
        await self.handle_message(event)

    async def _handle_blueprint_updated(self, data: dict) -> None:
        """Process a blueprint_updated event from the dashboard.

        Notifies the gateway AIAgent that a blueprint was modified so it
        can reload or acknowledge the change.
        """
        name = data.get("name", "unknown")
        saved_at = data.get("saved_at", 0)

        event_text = (
            f"[Blueprint Updated] The blueprint '{name}' was edited via the dashboard "
            f"at {time.strftime('%H:%M:%S', time.localtime(saved_at / 1000))}. "
            f"Use mc_story(action='load_blueprint', name='{name}') to reload the latest version."
        )

        logger.info("[DaemonCraft] Blueprint updated: %s", name)

        source = self.build_source(
            chat_id="world",
            chat_name="world",
            chat_type="group",
            user_id="dashboard",
            user_name="Dashboard",
            thread_id="world",
        )
        source.profile = self._profile

        event = MessageEvent(
            text=event_text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=data,
        )
        await self.handle_message(event)

    async def _handle_heartbeat_context(self, data: dict) -> None:
        """Process a heartbeat_context from the agent_loop.

        Injects world-state/perception into the gateway's AIAgent session.
        The AIAgent decides if action is needed. If the response is empty
        or just acknowledges with no substance, it is NOT sent to chat.
        Only responses with actual content or tool-call intent go to Minecraft.
        """
        status = data.get("status") or {}
        nearby = data.get("nearby") or {}
        inventory = data.get("inventory") or {}
        plan = data.get("plan") or {}
        events = data.get("events") or []

        lines = ["[Heartbeat — World Update]"]
        if status:
            pos = status.get("position")
            if pos:
                lines.append(f"Position: x={pos.get('x','?')} y={pos.get('y','?')} z={pos.get('z','?')}")
            lines.append(f"Health: {status.get('health', '?')}/{status.get('max_health', '?')}")
            lines.append(f"Food: {status.get('food', '?')}")
        if nearby:
            ents = nearby.get("entities", [])
            if ents:
                names = []
                for e in ents[:8]:
                    if isinstance(e, dict):
                        names.append(e.get("name", str(e)))
                    else:
                        names.append(str(e))
                lines.append(f"Nearby entities: {', '.join(names)}")
            blocks = nearby.get("blocks", [])
            if blocks:
                lines.append(f"Nearby blocks: {', '.join(str(b) for b in blocks[:5])}")
        if inventory:
            items = inventory.get("items", [])
            if items:
                lines.append(f"Inventory ({len(items)} items)")
        if plan:
            goal = plan.get("goal") or plan.get("title")
            if goal:
                lines.append(f"Active goal: {goal}")
            tasks = plan.get("tasks", [])
            if tasks:
                pending = [t.get("name", str(t)) for t in tasks if t.get("status") in ("pending", "in_progress")]
                if pending:
                    lines.append(f"Pending tasks: {', '.join(pending[:5])}")
        if events:
            for ev in events[:3]:
                lines.append(f"Event: {ev}")

        text = "\n".join(lines)
        if len(text) < 30:
            # Not enough data to bother the AI
            logger.debug("[DaemonCraft] Heartbeat context too sparse, skipping")
            return

        logger.info("[DaemonCraft] Heartbeat context injected (%d chars)", len(text))

        source = self.build_source(
            chat_id="world",
            chat_name="world",
            chat_type="group",
            user_id="heartbeat",
            user_name="Heartbeat",
            thread_id="world",
        )
        source.profile = self._profile

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=data,
        )

        # If a session is already active, do NOT inject heartbeat now —
        # it would fight with the in-progress turn. The heartbeat is best-effort.
        session_key = f"world:world:{self._profile}"
        if session_key in self._active_sessions:
            logger.debug("[DaemonCraft] Session active, heartbeat deferred")
            return

        if self._message_handler is None:
            return

        try:
            response = await self._message_handler(event)
        except Exception as e:
            logger.error("[DaemonCraft] Heartbeat handler error: %s", e, exc_info=True)
            return

        if not response:
            return

        # Filter out empty/no-op responses
        stripped = response.strip()
        if len(stripped) < 10:
            logger.debug("[DaemonCraft] Heartbeat response too short, suppressed: %s", stripped[:50])
            return
        if stripped.lower() in (
            "ok", "okay", "no action", "nothing", "...",
            "no change", "all good", "idle", "waiting",
        ):
            logger.debug("[DaemonCraft] Heartbeat response no-op, suppressed: %s", stripped[:50])
            return

        # Response has substance — send to world chat
        logger.info("[DaemonCraft] Heartbeat response -> chat: %s", stripped[:80])
        await self.send(chat_id="world", content=response)

    async def _handle_chat_entry(self, entry: dict) -> None:
        from_ = entry.get("from", "")
        if not from_:
            return
        if from_.lower() == self._bot_username.lower():
            return  # Ignore self-echo

        # Authorization by UUID (preferred) or username fallback
        sender_uuid = entry.get("uuid")
        if self._allowed_users:
            allowed = False
            if sender_uuid and sender_uuid.lower() in self._allowed_users:
                allowed = True
            if from_.lower() in self._allowed_users:
                allowed = True
            if not allowed:
                logger.debug("[DaemonCraft] Ignored message from unauthorized user: %s", from_)
                return

        text = entry.get("message", "")
        if not text:
            return

        is_whisper = entry.get("whisper", False)
        is_private = entry.get("private", False)
        world = entry.get("world", "world")

        # Session mapping
        if is_whisper or is_private:
            # 1:1 session
            chat_id = from_
            chat_type = "dm"
            thread_id = None
        else:
            # Group session per world
            chat_id = world
            chat_type = "group"
            thread_id = world
            self._world_names.add(world)

        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_id,
            chat_type=chat_type,
            user_id=sender_uuid or from_,
            user_name=from_,
            thread_id=thread_id,
        )
        source.profile = self._profile

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=entry,
        )

        await self.handle_message(event)

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        # _world_names is populated lazily from inbound broadcasts. If the gateway
        # initiates an outbound broadcast before any inbound from that world, this
        # will default to DM (whisper). For now the agent only replies to inbound.
        is_group = chat_id in self._world_names
        payload: dict[str, Any] = {"message": content}

        if is_group:
            payload["target"] = "broadcast"
        else:
            payload["target"] = chat_id

        try:
            async with self._session.post(
                f"{self._bot_api_url}/chat/send",
                json=payload,
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.warning("[DaemonCraft] /chat/send failed: %s %s", resp.status, body)
                    return SendResult(success=False, error=f"HTTP {resp.status}: {body}")
                return SendResult(success=True)
        except Exception as e:
            logger.warning("[DaemonCraft] /chat/send exception: %s", e)
            return SendResult(success=False, error=str(e), retryable=True)

    async def _copy_and_relay_tts(self, audio_path: str, chat_id: str) -> SendResult:
        """Copy audio to shared TTS cache and POST /tts/play to dashboards."""
        try:
            import shutil

            tts_dir = "/tmp/daemoncraft-tts"
            os.makedirs(tts_dir, exist_ok=True)
            filename = os.path.basename(audio_path)
            dest = os.path.join(tts_dir, filename)
            shutil.copy2(audio_path, dest)

            # Build public URL — bot API serves /tts/audio/:filename
            audio_url = f"{self._bot_api_url}/tts/audio/{filename}"

            async with self._session.post(
                f"{self._bot_api_url}/tts/play",
                json={"audio_url": audio_url, "chat_id": chat_id},
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.warning("[DaemonCraft] /tts/play failed: %s %s", resp.status, body)
                    return SendResult(success=False, error=f"HTTP {resp.status}: {body}")
            return SendResult(success=True)
        except Exception as e:
            logger.warning("[DaemonCraft] /tts/play exception: %s", e)
            return SendResult(success=False, error=str(e), retryable=True)

    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        """Relay TTS audio to dashboards and send transcript to Minecraft chat."""
        result = await self._copy_and_relay_tts(audio_path, chat_id)
        if not result.success:
            return result

        # Also send the full text to Minecraft chat so players can read it
        text = kwargs.get("text", "[Voice message]")
        return await self.send(chat_id, text)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        # Minecraft has no typing indicator — no-op
        pass

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Relay TTS audio to dashboards via the bot API."""
        return await self._copy_and_relay_tts(audio_path, chat_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_type = "group" if chat_id in self._world_names else "dm"
        return {"name": chat_id, "type": chat_type, "chat_id": chat_id}


# ------------------------------------------------------------------
# Requirements check
# ------------------------------------------------------------------

def check_daemoncraft_requirements() -> bool:
    """DaemonCraft only needs aiohttp (already a core dep)."""
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        return False
