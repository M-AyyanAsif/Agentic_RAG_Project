"""Basic API smoke tests with optimized fixtures."""

from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from backend.main import app

# Senior Tip: Use a fixture so the app doesn't reload for every single test
@pytest.fixture(scope="module")
def client():
    """Create a test client that persists across the module."""
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client: TestClient) -> None:
    """Verify the API is alive."""
    response = client.get("/health")
    assert response.status_code == 200
    # Ensuring it matches the standard health check format
    data = response.json()
    assert data.get("status") == "ok"
    assert "version" in data  # Good practice to track version in health

def test_sessions_endpoint(client: TestClient) -> None:
    """Verify the session listing logic works."""
    response = client.get("/sessions")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data.get("sessions"), list)
    
def test_root_not_found(client: TestClient) -> None:
    """Ensure undefined routes return 404 cleanly."""
    response = client.get("/undefined_route_xyz")
    assert response.status_code == 404