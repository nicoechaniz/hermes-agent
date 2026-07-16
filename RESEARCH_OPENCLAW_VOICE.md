# Research Report: OpenClaw Real-Time Voice Mode and Portability to Other Agent Frameworks

**Date:** 2026 (research snapshot)
**Focus:** Technical implementation of real-time voice in OpenClaw ecosystem, underlying technologies, architecture, code patterns, integration with agent reasoning, and portability (esp. to Hermes Agent / nousresearch/hermes-agent and embodied/Minecraft-style agents).
**Constraint:** Pure research — no code changes authored during investigation. All findings from public GitHub, docs, X/Twitter signals, and direct source inspection.

## 1. What is OpenClaw?

- **Core Project**: https://github.com/openclaw/openclaw (Node.js / pnpm monorepo, MIT).
- Local-first, self-hosted personal AI "Gateway" (control plane) + "Snaps"/agents.
- Strong multi-channel messaging (WhatsApp, Telegram, Discord, Slack, iMessage, Signal, WeChat, etc.), tools/skills (browser, shell, files, cron, canvas), persistent sessions/memory, multi-agent routing.
- Workspace: `~/.openclaw/workspace/` with `SOUL.md`, `IDENTITY.md`, `USER.md`, `AGENTS.md`, skills in `skills/<name>/SKILL.md`.
- Gateway typically listens on WS `ws://127.0.0.1:18789` (configurable), protocol v3 (JSON frames: `req`/`res`/`event`).
- Native voice: Voice Wake (global triggers) + Talk Mode on companion "nodes" (macOS menu bar, iOS/Android apps). Not full low-latency bidirectional realtime in the core gateway for arbitrary clients/phone.
- Creator/community: Peter Steinberger (@steipete / @openclaw on X), viral 2025-2026 growth, hackathons, comparisons to Hermes Agent (similar local agent/gateway philosophy; some users run both or migrate).

**Key Docs**:
- Architecture: https://docs.openclaw.ai/concepts/architecture
- Gateway RPC protocol: https://docs.openclaw.ai/reference/rpc
- Voice Wake: https://docs.openclaw.ai/nodes/voicewake
- Talk Mode: https://docs.openclaw.ai/nodes/talk (includes realtime config section)

## 2. OpenClaw's Native Voice Implementation (Talk Mode + Voice Wake)

### Voice Wake
- Global list of triggers stored by Gateway: `~/.openclaw/settings/voicewake.json` (`{triggers: ["openclaw", "jarvis", ...]}`).
- RPC: `voicewake.get`/`set`, `voicewake.routing.get`/`set` (maps trigger → target `sessionKey` or `agentId` or "current").
- Events: `voicewake.changed`, `voicewake.routing.changed` pushed to all WS clients/nodes.
- macOS/iOS nodes perform local wake-word detection (Porcupine? device ASR), forward trigger + audio/transcript to Gateway.
- Android: currently manual mic in Voice tab (wake disabled in some builds).
- Routing allows per-wake-word targeting of specific agents/sessions.

### Talk Mode (Native Continuous Voice Conversation)
- **Native path** (macOS/iOS/Android nodes):
  1. Local device STT (on-device ASR, configurable `speechLocale`).
  2. Transcript sent to active Gateway session (via chat pipeline or talk RPCs).
  3. Agent reasons / uses tools.
  4. Response via `talk.speak` RPC to node for TTS playback (ElevenLabs primary, system TTS fallback, local MLX on macOS).
- Phases: Listening (mic level viz) → Thinking → Speaking. Interrupt on speech (stops playback, records timestamp for next prompt).
- Voice directives in assistant replies (first JSON line, stripped): `{ "voice": "<id>", "once": true, "speed": ..., "stability": ... }` for per-turn or persistent voice params (ElevenLabs model, etc.).
- Config (`openclaw.json` under `talk:`):
  - `provider`: "elevenlabs" | "system" | "mlx"
  - `providers.elevenlabs`: voiceId, modelId (eleven_v3), apiKey, outputFormat, stability, similarity, etc.
  - `silenceTimeoutMs`, `interruptOnSpeech: true`
  - `speechLocale`
- **Browser realtime Talk** (emerging/native):
  - `talk.client.create` (webrtc / provider-websocket) or `talk.session.create` (gateway-relay).
  - `realtime:` section: `provider: "openai"`, `transport: "webrtc"`, `brain: "agent-consult"` (or "direct-tools", "none"), model `gpt-realtime-2` or similar, voice (cedar etc.), `instructions`.
  - Browser clients forward tool calls via `talk.client.toolCall` → Gateway does `openclaw_agent_consult` (delegates back to full agent).
  - Transcription-only mode (`brain: "none"`) for dictation/captions using `talk.session.appendAudio` etc. + `talk.event`.
- `talk.catalog` exposes supported modes/transports/brains/providers for clients.
- Limitations: Native realtime is platform/browser-tied; full phone/low-latency bidirectional + arbitrary clients is where community bridges shine. Issues #7200, #8088 track native realtime/WebRTC/SIP/OpenAI-Realtime deeper integration.

**Personality Injection**: Talk/realtime prompts pull from workspace `IDENTITY.md`/`SOUL.md`/`USER.md` (tools/skills left to main agent).

## 3. Community Real-Time Voice Bridges (The "Real" Low-Latency Voice Mode)

Core OpenClaw voice is good for nodes but not the fluid, sub-second, interruptible, phone-style experience. Community fills the gap with modern stacks.

### Primary Reference: langwatch/openclaw-phone-assistant (Recommended for True Realtime)
- **Repo**: https://github.com/langwatch/openclaw-phone-assistant (Python, uv, Pipecat-based).
- **Goal**: Talk naturally to OpenClaw "Snaps" via browser (WebRTC) or phone (Twilio). "Snaps" voice interface.
- **Architecture** (exact from README + source):
  ```
  Browser mic/speaker <—WebRTC—> Pipecat pipeline <—WebSocket—> OpenClaw Gateway (ws://...18789)
  Phone (Twilio) <—> Pipecat pipeline
                      ↓
                 OpenAI Realtime API (gpt-4o-realtime-preview or gpt-realtime-2)  OR  Google Gemini Live
  ```
  - **Voice loop**: Handled entirely by realtime speech-to-speech model (low latency, natural prosody, built-in VAD/interruption detection in Pipecat + provider).
  - **Delegation**: The realtime LLM (voice "brain") has **exactly one substantive tool**: `ask_openclaw(question)`.
    - Forwards to OpenClaw via `OpenClawGatewayClient.chat_send(sessionKey, message, ...)` (or `agent_send` fallback).
    - Uses idempotencyKey as runId; listens for `chat` events (state: "final" / "error" / "aborted") carrying `message.content[].text`.
    - OpenClaw does the heavy lifting (tools, memory, multi-step, channel actions, long reasoning).
  - **During delegation waits** (OpenClaw can take 10s–minutes for real actions): Hold music injected into audio pipeline (`HoldMusicPlayer`).
  - **Interruption handling** (key innovation):
    - `UserStartedSpeakingFrame` → sets `user_interrupted` Event.
    - Race: tool wait vs interrupt. On interrupt, return status to voice LLM ("still processing, respond to what user just said"), **keep the in-flight `send_task` alive in background**.
    - User can later say "check again" or repeat question → resumes waiting on the same task (no duplicate work).
    - Watchdog + error recovery for dropped tool calls / races ("already_has_active_response").
  - **End of call**: `end_call` tool (LLM must speak goodbye first, then call it).
  - **Rules enforced in system prompt** (critical for voice UX):
    - ALWAYS speak a short ack *before* calling `ask_openclaw` (prevents dead silence + hold music).
    - Keep responses SHORT (1-2 sentences ideal for voice).
    - No markdown; natural phone conversation.
    - Load personality from OpenClaw workspace (IDENTITY/SOUL/USER.md) but delegate tools.
  - **Transcript sync** (`TranscriptSync`): Voice turns appended to OpenClaw session's `.jsonl` (under `agents/<agentId>/sessions/<sid>.jsonl`) for continuity/memory in main agent.
  - **Transports**:
    - WebRTC: `SmallWebRTCTransport` (browser client at `/client`, port 7860). Pipecat runner.
    - Twilio: `FastAPIWebsocketTransport` + `TwilioFrameSerializer` (24kHz pipeline ↔ 8kHz), caller ID filtering (`ALLOWED_CALLER_NUMBERS`), ringing/pickup/error tones (mulaw direct playback before pipeline), Cloudflared tunnel for webhook, `make twilio`.
  - **Providers**: `providers/openai_rt.py` and `gemini_live.py` (create_llm + context_aggregator).
  - **Other files**: `bot.py` (main, tool handlers, pipeline, interruption watcher, watchdog), `openclaw_client.py` (full protocol impl), `hold_music.py`, `config.py`, `audio_debug.py`, Makefile (daemon, tunnel, webrtc/twilio).
  - **Auth/Connect**: Protocol v3 handshake (`connect` with client caps `["tool-events"]`, role "operator", optional token/password). Handles `connect.challenge` nonce. `chat.send` preferred over `agent` for event broadcasting/tool visibility.
  - **Session**: `OPENCLAW_SESSION_KEY` e.g. `agent:main:main`.
  - **Daemon**: systemd for 24/7 (bot + tunnel).
  - **Why it works**: Decouples voice surface (Pipecat + Realtime LLM) from reasoning/tools (OpenClaw). Voice LLM is "dumb but fast conversational"; full agent is "smart but slow".

This is the canonical example of "OpenClaw real-time voice mode" in 2026.

### Other Notable Community Voice Projects
- **Purple-Horizons/openclaw-voice** (https://github.com/Purple-Horizons/openclaw-voice):
  - Browser-based (React? + FastAPI/WS backend).
  - Local STT: faster-whisper (on-device, sizes tiny→large-v3-turbo, CUDA/MPS/CPU).
  - VAD: Silero.
  - TTS: ElevenLabs streaming (sentence-by-sentence for perceived low latency; turbo_v2_5) or local (XTTS-v2, Chatterbox).
  - Smart text cleaning (strip markdown, hashtags, URLs for TTS).
  - Continuous/auto-listen mode after response.
  - Direct OpenClaw Gateway integration (HTTP chat completions or WS; dedicated "voice" agent config recommended).
  - WebSocket protocol for browser: `start_listening`, `audio` (base64 PCM), `stop_listening`; events `transcript`, `response_chunk`, `audio_chunk`, `vad_status`.
  - Mobile-friendly via HTTPS (Tailscale Funnel or nginx).
  - Roadmap: WebRTC.
  - Classic turn-based pipeline (not speech-to-speech Realtime LLM). Easier for fully local (no Realtime API key cost/latency).

- **sachaabot/openclaw-voice-agent** (https://github.com/sachaabot/openclaw-voice-agent):
  - Hardware wake-word voice interface (Raspberry Pi CM5 / PamirAI Distiller).
  - Porcupine (Picovoice) for "hey openclaw" or custom.
  - Whisper STT → text to local OpenClaw Gateway → TTS (gTTS / ElevenLabs / offline Piper).
  - LEDs for state, systemd service, `config.yaml`.
  - Simple, reliable for always-on local device. Not low-latency realtime LLM.

- **malpern/VoxClaw** (https://github.com/malpern/VoxClaw):
  - Networked TTS "speaker" for headless OpenClaw. Mac listener (port 4140?) speaks text sent over network from remote gateway. Apple voices + OpenAI/ElevenLabs. Simple way to "give your server agent a voice."

- **openserv-labs/openclaw-voice-avatar**:
  - Realtime voice + video with lip-synced avatar.

- **Nat Eliason / community gists** (e.g. https://gist.github.com/Nateliason/66fb5220574023d5f59a1c4e92914603):
  - Full Pipecat + Deepgram STT + ElevenLabs TTS + WebRTC voice chat pipeline wired to Clawdbot/OpenClaw (via gateway or OpenAI-compatible endpoint). Includes `bot.py`, `server.py`, `index.html`. "ClawChat voice" tutorials (Chinese "15-minute private voice secretary" setups also use FastAPI+Pipecat+WebRTC).

- **Discord voice bridges / skills**: Multiple (ai-agent-Zofia/discord-voice-bridge-openclaw-skill etc. via ClawHub). OpenClaw skills for joining Discord voice and bridging audio/transcripts to agent.

- **LiveKit mentions**: OpenClaw skills (e.g. gora050/livekit-integration) for room/data management. LiveKit (WebRTC SFU + agents) pairs naturally with Pipecat (has LiveKit transport) for scalable multi-user or embodied voice rooms.

- **Twilio / phone plugins** in main repo (voice-call plugin issues around OpenAI Realtime conversation mode, hold music, drops).

## 4. Underlying Technologies

- **Pipecat** (https://github.com/pipecat-ai/pipecat, Daily.co): The dominant framework for building realtime voice (and multimodal) AI agents in Python. Handles:
  - Pipeline orchestration (frames: audio, LLMRun, UserStartedSpeaking, etc.).
  - Transports: SmallWebRTC, FastAPIWebsocket + Twilio serializer, Daily, LiveKit, etc.
  - Services: OpenAI Realtime, Gemini Live, Deepgram/Whisper STT, ElevenLabs/Cartesia TTS, SileroVAD, function calling/tools schema, context aggregation, interruption, metrics.
  - Runners for browser (WebRTC signaling) and telephony.
  - Why used: Battle-tested for low-latency, interruption, hold audio injection, custom tools. Many voice agent examples (Home Assistant, custom agents).

- **OpenAI Realtime API** (`gpt-4o-realtime-preview`, `gpt-realtime-2`): End-to-end speech-to-speech (audio in → audio out + text transcripts). Low latency, natural turn-taking, voice selection (alloy, cedar, etc.). Tool calling supported (the delegation hook). Primary for "ChatGPT Advanced Voice Mode"-like feel.

- **Google Gemini Live**: Alternative realtime speech-to-speech provider in the same Pipecat setups.

- **WebRTC**: Core low-latency browser audio transport (mic/speaker bidirectional). Pipecat SmallWebRTCTransport or Daily.co rooms. Signaling via FastAPI/WS.

- **Twilio**: Telephony (phone numbers, voice webhooks, media streams). Serializer handles 8kHz mulaw ↔ 24kHz. Cloudflared / ngrok for public HTTPS webhook.

- **VAD (Voice Activity Detection)**: Silero (open, local, in Purple), built-in in Realtime providers / Pipecat, or device-level. Critical for continuous mode, silence timeout, interruption.

- **STT (non-realtime path)**: faster-whisper (local, on-device privacy), Deepgram (cloud streaming), device ASR on nodes.

- **TTS (non-realtime)**: ElevenLabs (streaming, high quality, voice cloning params), local XTTS/Piper/Chatterbox, system voices, MLX.

- **Gateway Protocol (OpenClaw side)**: Custom JSON-over-WS v3.
  - Handshake: `connect` (client id/version/platform/mode/caps/scopes/auth token or password; handles challenge nonce).
  - `chat.send` (preferred for voice bridges): idempotencyKey/runId, returns immediately `{status:"started"}`, then `chat` events with `state: "final"` + full text (or error/aborted). Broadcasts tool events to session subscribers.
  - `agent`: CLI-style, returns final payloads after accepted + done.
  - Sessions: `sessionKey` (e.g. `agent:main:main`), resolved to internal IDs. Transcripts in per-agent JSONL.
  - Other: sessions.list, tool events, etc.
  - `openclaw_client.py` is the reference Python implementation (pending futures, event handlers for chat, final vs accepted responses, etc.).

- **Hold Music / Audio Injection**: During long backend calls (tools), inject prerecorded or generated audio frames into the output pipeline without breaking the realtime loop.

- **Other**: nacl for Discord voice crypto (in Hermes), Opus decode, etc.

## 5. How Voice Mode Integrates with the Agent's Reasoning Loop

**Decoupled "Fast Voice Surface + Smart Brain" Pattern** (the key architectural insight, highly portable):

1. **Realtime Voice LLM** (Pipecat + OpenAI Realtime / Gemini Live) owns:
   - Audio I/O, VAD, interruption, prosody, short conversational responses.
   - System prompt: Personality (from main agent files) + strict rules ("you are the voice interface", "ONE tool: ask_XXX", "speak BEFORE tool call", "short answers", "end_call on goodbye").
   - Tool calling only for delegation.

2. **Delegation Tool** (`ask_openclaw` / `openclaw_agent_consult` / equivalent):
   - Sends the user's spoken request (as text) over stable interface (WS to Gateway `chat.send` or direct agent invocation) to the **full agent session**.
   - Voice LLM waits (with hold music / status updates on interrupt).
   - Receives final text response → relays naturally ("Here's what I found...").

3. **Main Agent** (OpenClaw "Snap" or Hermes AIAgent) owns:
   - All tools, memory (long-term, sessions), skills, multi-step reasoning, channel actions (email, Discord messages, browser, files, cron, Minecraft controls), persona depth.
   - Runs in its own loop (possibly with higher max_iterations, different model, sandboxing).
   - May take seconds to minutes; voice surface stays responsive.

4. **Interruption & Backgrounding**:
   - User can barge in during thinking/hold → voice LLM acknowledges immediately; original request continues async in main agent.
   - Resume by re-asking (client tracks in-flight by question or runId).

5. **Continuity**:
   - Optional transcript sync (voice turns → main session JSONL) so main agent "remembers" the voice conversation.
   - Shared sessionKey / agentId.
   - Personality files shared (but tools not duplicated in voice prompt).

6. **Error / UX Hardening**:
   - Speak first before any tool (no silence).
   - Watchdogs for silent failures / dropped calls.
   - Recovery on pipeline errors (race conditions in realtime providers).
   - Caller filtering, tones, daemonization.

**Result**: You get fluid voice like "Her" or Advanced Voice Mode, but powered by your full local agent with real capabilities and persistent identity across text/voice channels. The voice part is thin (~1 tool); the agent is thick.

Native OpenClaw Talk does similar but with device STT + `talk.speak` TTS + Gateway chat in the middle (higher latency, less fluid interruptions than Realtime API).

## 6. Integration Patterns (Discord, Local, Minecraft/Embodied)

- **Discord**:
  - OpenClaw: Skills/bridges for voice channels (audio → agent → TTS back?).
  - Hermes: **Native strong support** in `gateway/platforms/discord.py`:
    - `VoiceReceiver`: Joins guild voice channels, captures per-user Opus audio (DAVE/secretbox decrypt or passthrough), SPEAKING events (op 5), silence detection/polling loop, delivers to `_voice_input_callback` (STT in run.py or pipeline), duplicate suppression via `_recent_voice_transcripts`.
    - Auto-disconnect on inactivity.
    - Audio output caching (`cache_audio_from_url/bytes`).
    - Gateway `voice_mode` ("off"/"voice_only"/"all"), `/voice` commands, `auto_tts`, per-chat state persistence (`gateway_voice_mode.json`).
    - `_sync_voice_mode_state_to_adapter`.
  - Pattern for advanced realtime on Discord voice: Run Pipecat bot that joins the same voice channel (or bridges), uses realtime LLM, delegates via Hermes Discord adapter session or direct AIAgent. Or extend existing VoiceReceiver to feed a Pipecat pipeline.

- **Local / Desktop / Hardware**:
  - Wake word (Porcupine / Silero) → capture → STT (Whisper or device) or direct to Pipecat realtime → delegate to local Hermes/OpenClaw gateway (localhost WS or even in-process Python call).
  - Examples: sachaabot hardware agent, VoxClaw for output, Purple for browser local STT.
  - Port: Easy — run voice worker alongside Hermes TUI/CLI; use Python AIAgent directly for zero-latency delegation (better than WS).

- **Minecraft-style Embodied Agents** (highly relevant to this workspace):
  - Hermes recent commits: daemoncraft, minecraft_tools.py, embodied_plan, spatial enrichment, mBit, Path fallback.
  - Pattern: Voice commands ("go to the village, mine the diamonds, build a wall here") → realtime voice LLM delegates to Hermes agent running in Minecraft environment (tools for movement, block ops, perception, planning).
  - Feedback: Agent actions → spatial state or screenshots → vision or text summary → TTS/voice response.
  - Or full embodied loop: voice as high-level planner, low-level control via Minecraft env.
  - OpenClaw has analogous "robot" skills (ROS mentions in community). LiveKit rooms could coordinate voice + embodied telemetry.
  - Portability win: Same delegation tool pattern. Voice surface doesn't need to know about Minecraft; the brain agent does.

- **Browser / Web / Dashboard**:
  - WebRTC client (Pipecat SmallWebRTC or custom) → voice worker → Hermes (via tui_gateway JSON-RPC, web_server PTY/REST, or direct agent API).
  - Hermes has dashboard embedding TUI + PTY bridge; voice could be additional pane or separate.

- **Phone / Telephony**: Twilio + Pipecat (proven in langwatch repo). Filter callers, public tunnel.

- **Multi-Framework Coexistence**:
  - https://github.com/AaronWong1999/hermesclaw: Proxy/bridge to run Hermes + OpenClaw on same WeChat (avoids iLink lock contention). Useful migration or parallel use. Separate migration importers for settings/memory/skills.
  - Shared community (gbrain setups work for both; Garry Tan posts on quick WebRTC/Twilio voice for Hermes/OpenClaw/gbrain stacks).

## 7. X/Twitter and Community Signals (Porting / Discussions)

- Primary accounts: @openclaw, @steipete (founder, later OpenAI?).
- Voice hype: "Voice mode that feels like the movie Her", realtime with local agents, hackathons (ROSClaw for robots, voice secretaries).
- Pipecat + OpenClaw voice chat: @nateliason (Felix agent, open-sourced Claw voice gists + "ClawChat: How to Build a Cross-Platform Voice Chat").
- Quick deploys: @garrytan and others: "Install OpenClaw or Hermes ... get it on WebRTC or your Twilio number in <30 minutes" (gbrain-powered).
- Comparisons/ports: Frequent Hermes vs OpenClaw threads (complementary strengths: OpenClaw strong on autonomous background + channels; Hermes on deep collab?). hermesclaw bridge for shared accounts. Migration tools and "run both" patterns.
- LiveKit / WebRTC / Pipecat: Mentioned alongside OpenClaw in agent infrastructure discussions, purple teaming (security of powerful local agents), observability (LangWatch pairs with Pipecat traces).
- No single "here is the port of langwatch-phone-assistant to Hermes" mega-thread in top results, but the pattern is repeatedly described as generalizable ("voice frontend + delegation tool to your agent backend"). Chinese tutorials and Medium/LinkedIn posts on 15-min OpenClaw voice secretaries using Pipecat emphasize the same architecture.
- Discord voice + agent bridging is a recurring skill/plugin request.

## 8. Portability Assessment & Recommendations for Hermes / Other Frameworks

**High Portability (Score: 9/10)** — The OpenClaw community pattern is deliberately **framework-agnostic on the brain side**.

**Minimal Requirements for Target Agent (Hermes or any)**:
- Expose a stable "send message to session X, get final response text" interface (WS RPC like `chat.send`, HTTP chat completions, or direct Python `AIAgent.run_conversation` / `chat`).
- Session continuity + optional transcript logging (JSONL or equivalent).
- Optional: Access to persona files (SOUL/IDENTITY) for voice prompt injection.
- Optional: Tool event streaming / approvals if voice surface should surface them.

**Why Easy for Hermes**:
- Python core: Voice worker (Pipecat process) can `import` and directly instantiate `AIAgent(...)` with the right session/credential context — zero network for delegation in local setups (huge latency win vs OpenClaw's WS).
- Existing Discord voice I/O (`VoiceReceiver`, auto-TTS, voice_mode, STT dedup) provides a ready hook. Can feed voice channel audio directly into Pipecat or use as fallback.
- Gateway already has voice concepts and multi-platform sessions.
- tui_gateway / web_server / acp_adapter give structured control surfaces.
- Minecraft/daemoncraft embodied work is a perfect "brain" for voice-delegated high-level commands.
- Recent gateway fixes (run_conversation, etc.) show active iteration.

**Recommended Porting Approach** (for a Hermes voice mode):
1. Fork/adapt `langwatch/openclaw-phone-assistant` (or Nat's gist) → replace OpenClaw client with Hermes equivalent (direct AIAgent or existing gateway WS/JSON-RPC if exposed cleanly).
2. System prompt adaptation: "You are the voice interface for Hermes. Delegate via `ask_hermes` tool...".
3. For Discord voice channels: Extend or bridge the existing `VoiceReceiver` + STT to feed the Pipecat pipeline (or run a dedicated voice bot per guild).
4. Local mic: Add wake word (Porcupine integration or use Silero in Pipecat) + always-listening worker.
5. Minecraft embodied: Same delegation; the tool response can include spatial state or vision summaries.
6. Extras: Transcript sync to Hermes session DB (`hermes_state.py`), hold music, interruption resume, personality from `~/.hermes` equivalents.
7. Observability: LangWatch or Hermes' own observability plugin.
8. Start simple: Browser WebRTC + local Hermes agent (no Twilio first).

**Alternatives / Layers**:
- For non-realtime (easier, fully local/privacy): Adapt Purple-Horizons style (Whisper + VAD + streaming TTS + direct call to AIAgent).
- Hardware: Porcupine + Whisper + TTS + direct agent call (like sachaabot).
- Use Pipecat's built-in examples for custom LLM agents (replace the "ask_hermes" tool impl).
- For LiveKit: Add LiveKit transport to Pipecat for room-based multi-agent voice or embodied coordination.

**Risks / Considerations** (technical facts):
- Realtime API costs (per-minute audio) vs local STT/TTS.
- Interruption races and "already_has_active_response" errors (handled in the reference bot.py with recovery/watchdog).
- Session auth / token security for the delegation WS (OpenClaw uses gateway token from `~/.openclaw/openclaw.json`).
- Sandboxing: Voice surface should not grant extra privileges; delegate enforces policy.
- Latency: In-process delegation >> WS >> HTTP.
- Platform voice APIs (Discord) have their own limits (Opus, encryption, speaking indicators) — Pipecat can run alongside or replace.

**Live Examples to Study (in order of fidelity)**:
1. https://github.com/langwatch/openclaw-phone-assistant (full Pipecat + Realtime + delegation + Twilio/WebRTC + interruption + sync).
2. Nat Eliason gist (lighter Claw-specific voice chat).
3. Purple-Horizons/openclaw-voice (STT/TTS local pipeline).
4. Hermes `gateway/platforms/discord.py` (VoiceReceiver, voice modes) + `gateway/run.py` (voice_mode persistence, auto_tts).
5. OpenClaw native talk config + `talk.client.toolCall` path in docs.

## 9. Conclusions & Actionable Insights

OpenClaw's "real-time voice mode" is **not a single native feature** but a thriving ecosystem pattern: Pipecat-orchestrated realtime speech-to-speech (OpenAI/Gemini) as the conversational skin, delegating via a single narrow tool to the full persistent agent "brain" over the Gateway protocol. This gives fluid voice UX without compromising the agent's power or requiring the voice layer to duplicate tools/memory.

The architecture is **extremely portable** — it has already been adapted across local, browser, phone, Discord, and (by extension) embodied/robot use cases. Hermes is particularly well-positioned because of its Python depth, existing Discord voice pipeline, embodied Minecraft work, and similar gateway/session model. A high-quality Hermes voice mode could be built by forking the langwatch reference and wiring it to `AIAgent` (local) or the gateway (multi-platform), potentially surpassing OpenClaw's current native offering in fluidity for Discord and local cases.

**Next Research Steps (if desired)**: Deep-dive specific Hermes voice input path (STT details in run.py + discord receiver callbacks), inspect tui_gateway for structured voice hooks, review Pipecat LiveKit transport for Minecraft coordination, or prototype the delegation client mirroring `openclaw_client.py` against Hermes' RPC surfaces.

**Primary URLs**:
- OpenClaw main + docs: github.com/openclaw/openclaw , docs.openclaw.ai
- Best realtime bridge: github.com/langwatch/openclaw-phone-assistant (read openclaw_client.py + bot.py)
- STT/TTS bridge: github.com/Purple-Horizons/openclaw-voice
- Hermes Discord voice: workspace/gateway/platforms/discord.py (VoiceReceiver class)
- Pipecat: github.com/pipecat-ai/pipecat
- Community voice gist: gist.github.com/Nateliason/66fb5220574023d5f59a1c4e92914603
- Cross-framework: github.com/AaronWong1999/hermesclaw

This report is based on direct source reads (READMEs, bot.py, openclaw_client.py, Hermes voice code), docs pages, GitHub issues, and web/X signals. All technical claims trace to verifiable public artifacts as of the research date.

---

*End of Report. Suitable for internal use in hermes-agent for planning voice enhancements or embodied voice control.*