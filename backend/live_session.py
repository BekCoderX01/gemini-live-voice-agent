"""Bidirectional Gemini Live API session manager with non-blocking tool execution."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Protocol

from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

from backend.config import (
    DEFAULT_API_VERSION,
    SessionSettings,
    build_live_connect_config,
)
from backend.tools import NonBlockingToolExecutor, WorkspaceState, get_function_declarations

logger = logging.getLogger(__name__)


class ClientFactory(Protocol):
    """Protocol for creating a google.genai.Client (injectable for testing)."""

    def __call__(self, api_key: str) -> Any: ...


def default_client_factory(api_key: str) -> genai.Client:
    """Create an official Google GenAI client configured for the Live API."""
    return genai.Client(
        api_key=api_key,
        http_options={"api_version": DEFAULT_API_VERSION},
    )


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


class VoiceAgentSession:
    """Bridges a browser WebSocket client and a Gemini Live API AsyncSession."""

    def __init__(
        self,
        *,
        websocket: WebSocket,
        api_key: str,
        settings: SessionSettings,
        workspace: WorkspaceState | None = None,
        client_factory: ClientFactory | None = None,
        simulate_tool_latency: bool = True,
        max_reconnect_attempts: int = 3,
    ) -> None:
        self.websocket = websocket
        self.api_key = api_key
        self.settings = settings
        self.workspace = workspace or WorkspaceState()
        self.client_factory = client_factory or default_client_factory
        self.max_reconnect_attempts = max_reconnect_attempts

        self.tool_executor = NonBlockingToolExecutor(
            workspace=self.workspace,
            scheduling_overrides=settings.tool_scheduling,
            simulate_latency=simulate_tool_latency,
        )

        self._ws_send_lock = asyncio.Lock()
        self._gemini_send_lock = asyncio.Lock()
        self._session_ready = asyncio.Event()
        self._current_session: Any | None = None
        self._running = False
        self._reconnect_requested = False
        self.resumption_handle: str | None = None

    # ------------------------------------------------------------------ #
    # Browser output helpers
    # ------------------------------------------------------------------ #
    async def send_json(self, payload: dict[str, Any]) -> None:
        """Send a JSON control/telemetry event to the browser safely."""
        async with self._ws_send_lock:
            try:
                await self.websocket.send_json(payload)
            except Exception:
                pass  # Browser socket already closed

    async def send_bytes(self, data: bytes) -> None:
        """Send raw 24kHz PCM audio bytes to the browser safely."""
        async with self._ws_send_lock:
            try:
                await self.websocket.send_bytes(data)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        """Run the full duplex session loop with automatic session resumption."""
        self._running = True
        client = self.client_factory(self.api_key)
        reconnect_attempts = 0

        await self.send_json({"type": "workspace_state", "state": self.workspace.snapshot()})

        browser_task = asyncio.create_task(self._browser_input_loop(), name="browser-input-loop")

        try:
            while self._running:
                self._reconnect_requested = False
                config = build_live_connect_config(
                    settings=self.settings,
                    function_declarations=get_function_declarations(),
                    resumption_handle=self.resumption_handle,
                )

                try:
                    async with client.aio.live.connect(
                        model=self.settings.model,
                        config=config,
                    ) as session:
                        self._current_session = session
                        self._session_ready.set()
                        reconnect_attempts = 0

                        await self.send_json(
                            {
                                "type": "session_ready",
                                "model": self.settings.model,
                                "resumed": bool(self.resumption_handle),
                                "extended_thinking": self.settings.is_extended_thinking,
                                "vad_mode": self.settings.vad_mode,
                            }
                        )

                        await self._gemini_receive_loop(session)

                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if not self._running:
                        break
                    logger.warning("Gemini Live connection interrupted: %s", exc)
                    if self.resumption_handle and reconnect_attempts < self.max_reconnect_attempts:
                        reconnect_attempts += 1
                        await self.send_json(
                            {
                                "type": "session_reconnecting",
                                "attempt": reconnect_attempts,
                                "reason": str(exc),
                            }
                        )
                        await asyncio.sleep(0.4)
                        continue
                    await self.send_json({"type": "error", "message": str(exc)})
                    break
                finally:
                    self._session_ready.clear()
                    self._current_session = None

                if not self._running:
                    break

                if self._reconnect_requested and self.resumption_handle:
                    await self.send_json(
                        {
                            "type": "session_reconnecting",
                            "attempt": 1,
                            "reason": "GoAway received - resuming session",
                        }
                    )
                    continue
                break
        finally:
            self._running = False
            self._session_ready.clear()
            if not browser_task.done():
                browser_task.cancel()
                await asyncio.gather(browser_task, return_exceptions=True)
            await self.tool_executor.shutdown()
            await self.send_json({"type": "session_closed"})

    # ------------------------------------------------------------------ #
    # Browser -> Gemini
    # ------------------------------------------------------------------ #
    async def _browser_input_loop(self) -> None:
        """Read incoming audio/control frames from the browser and forward to Gemini."""
        try:
            while self._running:
                message = await self.websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    self._running = False
                    await self._close_current_session()
                    break

                if message.get("bytes") is not None:
                    await self._forward_audio_chunk(message["bytes"])
                elif message.get("text") is not None:
                    await self.handle_client_json(message["text"])
        except WebSocketDisconnect:
            self._running = False
            await self._close_current_session()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Browser input loop ended: %s", exc)
            self._running = False
            await self._close_current_session()

    async def _close_current_session(self) -> None:
        session = self._current_session
        if session is not None:
            try:
                await session.close()
            except Exception:
                pass

    async def _forward_audio_chunk(self, pcm_bytes: bytes) -> None:
        session = self._current_session
        if session is None or not pcm_bytes:
            return
        async with self._gemini_send_lock:
            await session.send_realtime_input(
                audio=types.Blob(data=pcm_bytes, mime_type="audio/pcm;rate=16000")
            )

    async def handle_client_json(self, raw_text: str) -> None:
        """Handle JSON control messages coming from the browser."""
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            return

        kind = data.get("type")
        if kind == "ping":
            await self.send_json({"type": "pong"})
            return

        if kind == "update_scheduling":
            tool_name = str(data.get("tool_name") or "")
            scheduling = str(data.get("scheduling") or "")
            if tool_name and scheduling:
                self.tool_executor.update_scheduling(tool_name, scheduling)
                await self.send_json(
                    {
                        "type": "scheduling_updated",
                        "tool_name": tool_name,
                        "scheduling": self.tool_executor.get_scheduling(tool_name),
                    }
                )
            return

        if kind == "stop":
            self._running = False
            await self._close_current_session()
            return

        session = self._current_session
        if session is None:
            return

        async with self._gemini_send_lock:
            if kind == "text":
                text = str(data.get("text") or "").strip()
                if text:
                    await session.send_realtime_input(text=text)
            elif kind == "client_content":
                text = str(data.get("text") or "").strip()
                if text:
                    await session.send_client_content(
                        turns=types.Content(role="user", parts=[types.Part(text=text)]),
                        turn_complete=bool(data.get("turn_complete", True)),
                    )
            elif kind == "audio_stream_end":
                await session.send_realtime_input(audio_stream_end=True)
            elif kind == "activity_start":
                await session.send_realtime_input(activity_start=types.ActivityStart())
            elif kind == "activity_end":
                await session.send_realtime_input(activity_end=types.ActivityEnd())

    # ------------------------------------------------------------------ #
    # Gemini -> Browser
    # ------------------------------------------------------------------ #
    async def _gemini_receive_loop(self, session: Any) -> None:
        """Continuously receive events from Gemini across multiple turns.

        `session.receive()` ends its iterator when an interaction completes
        (turn_complete, or interaction_status == IDLE for Extended Thinking),
        so we keep re-entering it for the lifetime of the connection.
        """
        while self._running and not self._reconnect_requested:
            received_any = False
            async for message in session.receive():
                received_any = True
                await self.process_server_message(message)
                if not self._running or self._reconnect_requested:
                    break
            if not received_any:
                break

    async def process_server_message(self, message: types.LiveServerMessage) -> None:
        """Process one LiveServerMessage: audio, transcripts, async tools, lifecycle."""
        if message.server_content is not None:
            await self._process_server_content(message.server_content)

        # Non-blocking tool calls: spawn background tasks and return immediately,
        # so audio keeps streaming while tools run.
        if message.tool_call and message.tool_call.function_calls:
            self.tool_executor.dispatch(
                list(message.tool_call.function_calls),
                # Function scheduling is not supported by Extended Thinking
                allow_scheduling=not self.settings.is_extended_thinking,
                on_event=self.send_json,
                on_complete=self._send_tool_response,
            )

        if message.tool_call_cancellation and message.tool_call_cancellation.ids:
            await self.tool_executor.cancel(list(message.tool_call_cancellation.ids), self.send_json)

        if message.session_resumption_update is not None:
            upd = message.session_resumption_update
            if upd.resumable and upd.new_handle:
                self.resumption_handle = upd.new_handle
                await self.send_json({"type": "resumption_handle", "resumable": True})

        if message.go_away is not None:
            await self.send_json(
                {"type": "go_away", "time_left": str(getattr(message.go_away, "time_left", ""))}
            )
            if self.resumption_handle:
                self._reconnect_requested = True

        if message.usage_metadata is not None:
            usage = message.usage_metadata
            await self.send_json(
                {
                    "type": "usage_metadata",
                    "total_tokens": usage.total_token_count,
                    "prompt_tokens": usage.prompt_token_count,
                    "response_tokens": usage.response_token_count,
                    "thoughts_tokens": usage.thoughts_token_count,
                }
            )

    async def _process_server_content(self, sc: types.LiveServerContent) -> None:
        if sc.interrupted is True:
            await self.send_json({"type": "interrupted"})

        # Process ALL parts: a single event may carry audio, thoughts and text.
        if sc.model_turn and sc.model_turn.parts:
            for part in sc.model_turn.parts:
                if part.inline_data and part.inline_data.data:
                    raw = part.inline_data.data
                    audio = raw if isinstance(raw, (bytes, bytearray)) else base64.b64decode(raw)
                    await self.send_bytes(bytes(audio))
                if part.text:
                    if getattr(part, "thought", False):
                        await self.send_json({"type": "thought_summary", "text": part.text})
                    else:
                        await self.send_json({"type": "model_text", "text": part.text})

        if sc.input_transcription and sc.input_transcription.text:
            await self.send_json({"type": "input_transcription", "text": sc.input_transcription.text})

        if sc.interim_input_transcription and sc.interim_input_transcription.text:
            await self.send_json(
                {"type": "interim_input_transcription", "text": sc.interim_input_transcription.text}
            )

        if sc.output_transcription and sc.output_transcription.text:
            await self.send_json({"type": "output_transcription", "text": sc.output_transcription.text})

        if sc.generation_complete is True:
            await self.send_json({"type": "generation_complete"})

        status = _enum_name(sc.interaction_status)
        if status == "INTERACTION_STATUS_UNSPECIFIED":
            status = None

        if sc.turn_complete is True:
            reason = _enum_name(sc.turn_complete_reason)
            if reason == "TURN_COMPLETE_REASON_UNSPECIFIED":
                reason = None
            await self.send_json(
                {"type": "turn_complete", "reason": reason, "interaction_status": status}
            )

        if status is not None:
            await self.send_json({"type": "interaction_status", "status": status})

    async def _send_tool_response(
        self, fn_response: types.FunctionResponse, _record: dict[str, Any]
    ) -> None:
        """Send a FunctionResponse, waiting briefly if a reconnect is in progress."""
        if not self._running:
            return

        if self._current_session is None:
            try:
                await asyncio.wait_for(self._session_ready.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                logger.warning("No active session for tool response %s", fn_response.id)
                return

        session = self._current_session
        if session is None or not self._running:
            return

        async with self._gemini_send_lock:
            await session.send_tool_response(function_responses=[fn_response])
