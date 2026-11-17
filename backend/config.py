"""Configuration models and LiveConnectConfig builder for the Gemini Live voice agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google.genai import types
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_LIVE = "gemini-3.8-live"
MODEL_EXTENDED_THINKING = "gemini-3.8-live-extended-thinking"
SUPPORTED_MODELS = (MODEL_LIVE, MODEL_EXTENDED_THINKING)

AVAILABLE_VOICES = (
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Aoede",
    "Zephyr",
    "Leda",
    "Orus",
)

VALID_THINKING_LEVELS = ("LOW", "MEDIUM", "HIGH")
VALID_SCHEDULING = ("INTERRUPT", "WHEN_IDLE", "SILENT")

DEFAULT_API_VERSION = os.getenv("GEMINI_API_VERSION", "v1beta")

DEFAULT_SYSTEM_INSTRUCTION = """You are Aria, a fast, perceptive real-time voice concierge and research assistant.

Conversational rules:
1. Keep spoken responses natural, concise, and direct — usually 1 to 3 sentences unless the user asks for deep detail. Do not read raw JSON, markdown symbols, or long URLs aloud.
2. You have asynchronous (non-blocking) tools that run in the background while conversation continues:
   - Use `search_flights` whenever the user asks about flights, fares, or flight availability between cities.
   - Use `get_weather_forecast` whenever the user asks about weather, temperature, or forecast conditions for a city.
   - Use `deep_research_topic` when the user asks for an in-depth briefing, comparison, or multi-angle research summary.
   - Use `set_timer_or_reminder` when the user asks to set a countdown timer or reminder.
   - Use `control_workspace_device` when the user asks to adjust lights, thermostat temperature, ambient music, or do-not-disturb mode, or check workspace status.
3. Because tools execute asynchronously without blocking the call:
   - When starting a longer lookup (`search_flights`, `deep_research_topic`, `get_weather_forecast`), give a brief natural acknowledgment (e.g., "Looking up flights to Tokyo now — feel free to keep talking") and let the conversation flow naturally.
   - Never invent tool results before the function response arrives.
   - If a tool call fails or is cancelled due to interruption, acknowledge it clearly without guessing data.
"""


class SessionSettings(BaseModel):
    """Per-session configuration options sent by the browser client."""

    model: str = MODEL_LIVE
    voice_name: str = "Puck"
    thinking_level: str = "LOW"
    include_thoughts: bool = True
    enable_google_search: bool = True
    vad_mode: Literal["automatic", "hybrid", "manual"] = "automatic"
    silence_duration_ms: int = Field(default=600, ge=200, le=3000)
    tool_scheduling: dict[str, str] = Field(default_factory=dict)
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION
    api_key: str | None = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if value not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{value}'. Choose from: {', '.join(SUPPORTED_MODELS)}"
            )
        return value

    @field_validator("voice_name")
    @classmethod
    def validate_voice(cls, value: str) -> str:
        if value not in AVAILABLE_VOICES:
            raise ValueError(
                f"Unsupported voice '{value}'. Choose from: {', '.join(AVAILABLE_VOICES)}"
            )
        return value

    @field_validator("thinking_level")
    @classmethod
    def validate_thinking_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in VALID_THINKING_LEVELS:
            raise ValueError(
                f"Unsupported thinking_level '{value}'. Choose from: {', '.join(VALID_THINKING_LEVELS)}"
            )
        return normalized

    @field_validator("tool_scheduling")
    @classmethod
    def validate_tool_scheduling(cls, mapping: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for tool_name, sched in mapping.items():
            norm = sched.strip().upper()
            if norm not in VALID_SCHEDULING:
                raise ValueError(
                    f"Unsupported scheduling '{sched}' for '{tool_name}'. Choose from: {', '.join(VALID_SCHEDULING)}"
                )
            cleaned[tool_name] = norm
        return cleaned

    @property
    def is_extended_thinking(self) -> bool:
        return self.model == MODEL_EXTENDED_THINKING


def get_server_api_key() -> str | None:
    """Return configured Gemini API key from environment if present."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key or key == "your_gemini_api_key_here":
        return None
    return key.strip()


def resolve_api_key(override_key: str | None = None) -> str | None:
    """Resolve explicit session API key override or server environment key."""
    if override_key and override_key.strip():
        return override_key.strip()
    return get_server_api_key()


def build_live_connect_config(
    settings: SessionSettings,
    function_declarations: list[types.FunctionDeclaration],
    resumption_handle: str | None = None,
) -> types.LiveConnectConfig:
    """Build a LiveConnectConfig tailored to Gemini 3.8 Live or Extended Thinking."""
    tools: list[types.Tool] = [
        types.Tool(function_declarations=function_declarations)
    ]
    if settings.enable_google_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))

    if settings.vad_mode == "manual":
        realtime_input_config = types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True,
            )
        )
    else:
        realtime_input_config = types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                prefix_padding_ms=100,
                silence_duration_ms=settings.silence_duration_ms,
            )
        )

    thinking_config: types.ThinkingConfig | None = None
    if settings.is_extended_thinking:
        # Gemini 3.8 Live Extended Thinking supports LOW, MEDIUM, HIGH (not MINIMAL)
        thinking_config = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel(settings.thinking_level),
            include_thoughts=settings.include_thoughts,
        )

    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=settings.voice_name,
                )
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part(text=settings.system_instruction)]
        ),
        tools=tools,
        thinking_config=thinking_config,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=realtime_input_config,
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow(),
        ),
        session_resumption=types.SessionResumptionConfig(
            handle=resumption_handle,
        ),
    )
