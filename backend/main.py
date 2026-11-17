"""FastAPI server: serves the browser UI and bridges WebSocket audio to Gemini Live."""

from __future__ import annotations

import json
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from backend.config import (
    AVAILABLE_VOICES,
    MODEL_EXTENDED_THINKING,
    MODEL_LIVE,
    PROJECT_ROOT,
    VALID_SCHEDULING,
    VALID_THINKING_LEVELS,
    SessionSettings,
    get_server_api_key,
    resolve_api_key,
)
from backend.live_session import VoiceAgentSession
from backend.tools import get_tools_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("voice-agent")

FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(title="Gemini Live Voice Agent", version="0.1.0")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "server_api_key_configured": get_server_api_key() is not None})


@app.get("/api/options")
async def options() -> JSONResponse:
    return JSONResponse(
        {
            "models": [
                {"id": MODEL_LIVE, "label": "Gemini 3.8 Live (low latency)"},
                {"id": MODEL_EXTENDED_THINKING, "label": "Gemini 3.8 Live Extended Thinking"},
            ],
            "voices": list(AVAILABLE_VOICES),
            "thinking_levels": list(VALID_THINKING_LEVELS),
            "scheduling_modes": list(VALID_SCHEDULING),
            "tools": get_tools_metadata(),
            "server_api_key_configured": get_server_api_key() is not None,
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """First text frame must be {"type": "start", "settings": {...}}."""
    await websocket.accept()
    try:
        first = await websocket.receive_text()
        payload = json.loads(first)
        if payload.get("type") != "start":
            await websocket.send_json({"type": "error", "message": "First message must be type 'start'."})
            await websocket.close(code=1008)
            return

        try:
            settings = SessionSettings(**(payload.get("settings") or {}))
        except ValidationError as exc:
            await websocket.send_json({"type": "error", "message": f"Invalid settings: {exc.errors()}"})
            await websocket.close(code=1008)
            return

        api_key = resolve_api_key(settings.api_key)
        if not api_key:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "No Gemini API key. Set GEMINI_API_KEY in .env or enter a key in the UI.",
                }
            )
            await websocket.close(code=1008)
            return

        logger.info("Starting session model=%s voice=%s vad=%s", settings.model, settings.voice_name, settings.vad_mode)
        session = VoiceAgentSession(websocket=websocket, api_key=api_key, settings=settings)
        await session.run()
    except WebSocketDisconnect:
        logger.info("Browser disconnected")
    except json.JSONDecodeError:
        await websocket.close(code=1003)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Session failed")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def main() -> None:
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
