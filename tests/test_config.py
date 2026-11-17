"""
Unit tests for backend.config LiveConnectConfig builder.
Verifies configurations for both gemini-3.8-live and gemini-3.8-live-extended-thinking models.
"""

import pytest
from google.genai import types
from backend.config import (
    build_live_connect_config,
    SessionSettings,
    MODEL_LIVE,
    MODEL_EXTENDED_THINKING,
    SUPPORTED_MODELS,
    AVAILABLE_VOICES,
    VALID_THINKING_LEVELS,
    VALID_SCHEDULING
)
from backend.tools import get_function_declarations


def test_supported_models_and_voices():
    assert MODEL_LIVE in SUPPORTED_MODELS
    assert MODEL_EXTENDED_THINKING in SUPPORTED_MODELS
    assert "Puck" in AVAILABLE_VOICES
    assert "Charon" in AVAILABLE_VOICES


def test_build_config_standard_live():
    """Verify standard gemini-3.8-live config does NOT include thinking_config."""
    decls = get_function_declarations()
    settings = SessionSettings(
        model=MODEL_LIVE,
        voice_name="Puck",
        enable_google_search=True
    )
    config = build_live_connect_config(
        settings=settings,
        function_declarations=decls
    )

    assert isinstance(config, types.LiveConnectConfig)
    assert config.response_modalities == [types.Modality.AUDIO]
    assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Puck"
    assert config.thinking_config is None
    assert config.tools is not None
    assert len(config.tools[0].function_declarations) == 5


def test_build_config_extended_thinking():
    """Verify extended-thinking model sets thinking_config with thinking_level."""
    decls = get_function_declarations()
    settings = SessionSettings(
        model=MODEL_EXTENDED_THINKING,
        voice_name="Charon",
        thinking_level="MEDIUM",
        vad_mode="manual"
    )
    config = build_live_connect_config(
        settings=settings,
        function_declarations=decls
    )

    assert isinstance(config, types.LiveConnectConfig)
    assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Charon"
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert config.realtime_input_config.automatic_activity_detection.disabled is True


def test_build_config_session_resumption():
    """Verify resumption handle is included when provided."""
    decls = get_function_declarations()
    settings = SessionSettings(model=MODEL_LIVE)
    config = build_live_connect_config(
        settings=settings,
        function_declarations=decls,
        resumption_handle="test_resume_token_xyz_123"
    )
    assert config.session_resumption is not None
    assert config.session_resumption.handle == "test_resume_token_xyz_123"


def test_invalid_settings_rejection():
    """Verify validation errors for invalid model or voice."""
    with pytest.raises(ValueError):
        SessionSettings(model="unsupported-model-v1")

    with pytest.raises(ValueError):
        SessionSettings(voice_name="UnknownVoiceXYZ")

    with pytest.raises(ValueError):
        SessionSettings(thinking_level="SUPER_HIGH")
