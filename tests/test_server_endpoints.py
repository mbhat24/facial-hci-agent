"""Tests for server API endpoints."""
import pytest
from fastapi.testclient import TestClient
from dashboard.server import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "active_sessions" in data
    assert "llm_enabled" in data
    assert "version" in data


def test_index_endpoint(client):
    """Test the index endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    assert b"Facial HCI Agent" in response.content


def test_metrics_endpoint(client):
    """Test the Prometheus metrics endpoint."""
    response = client.get("/metrics")
    
    assert response.status_code == 200
    # Prometheus metrics should be in the response
    assert b"process_" in response.content or b"python_" in response.content


def test_consent_endpoint(client):
    """Test the consent grant endpoint."""
    response = client.post(
        "/api/consent",
        json={"user_agent": "test-agent"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "consented_at" in data


def test_data_export_endpoint(client):
    """Test the GDPR data export endpoint."""
    # First grant consent to get a session
    consent_response = client.post(
        "/api/consent",
        json={"user_agent": "test-agent"}
    )
    session_id = consent_response.json()["session_id"]
    
    # Try to export data (will fail since session not active, but should return 404)
    response = client.post(
        "/api/data/export",
        json={"session_id": session_id}
    )
    
    # Session not active, should return 404
    assert response.status_code == 404


def test_data_delete_endpoint(client):
    """Test the GDPR data delete endpoint."""
    response = client.post(
        "/api/data/delete",
        json={"session_id": "nonexistent-session"}
    )
    
    # Should return 500 or handle gracefully
    assert response.status_code in [500, 404]
