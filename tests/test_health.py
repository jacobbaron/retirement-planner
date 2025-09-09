"""Tests for the health check endpoint."""

import json

from app import create_app


def test_healthz_ok():
    """Test that the health endpoint returns 200 with correct JSON."""
    app = create_app()
    client = app.test_client()

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.content_type == "application/json"

    data = json.loads(response.data)
    assert data == {"status": "ok"}
