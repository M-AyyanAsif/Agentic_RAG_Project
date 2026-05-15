"""Basic API smoke tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_sessions_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/sessions")
    assert response.status_code == 200
    assert "sessions" in response.json()
