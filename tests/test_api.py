"""
Unit tests for FastAPI endpoints (/api/health, /api/options).
"""

import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.config import MODEL_LIVE, MODEL_EXTENDED_THINKING, AVAILABLE_VOICES


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_options_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/options")
        assert response.status_code == 200
        data = response.json()

        model_ids = [m["id"] if isinstance(m, dict) else m for m in data["models"]]
        assert MODEL_LIVE in model_ids
        assert MODEL_EXTENDED_THINKING in model_ids

        for voice in AVAILABLE_VOICES:
            assert voice in data["voices"]
        assert len(data["tools"]) == 5
